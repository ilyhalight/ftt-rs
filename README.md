# fasttext-turbo (Rust)

A fast, memory-mapped inference engine for [fastText](https://fasttext.cc/)
models, written from scratch in Rust. It loads ordinary `.bin` and `.ftz`
files produced by fastText (or
[fastText-turbo](https://github.com/CommonGlot/fastText-turbo)) and returns the
same labels, with probabilities within a few millionths on x86-64.

**Measured on one core:** 4.9–16× faster than the stock fastText Python
package on the models below; **25–76×** on all cores. A 2,102-label, 256-dim
softmax model runs at **20,000 lines/s on one core** and **95,000 lines/s on
eight**, with no Python in the hot path.

```text
$ ftt predict-prob lid.176.ftz
Hello, how are you today?
__label__en 0.974427 __label__es 0.003582 __label__ur 0.001750
```

## Why it is fast

The reference implementation already does the obvious things. This engine
attacks the two costs that dominate fastText prediction: reading random
embedding rows and multiplying the hidden vector by the output matrix.

- **Zero-copy model.** The file is `mmap`ed and matrices are viewed in place,
  so the embedding table is not copied to the heap.
- **Per-word row sums.** Character n-grams make a word contribute dozens of
  embedding rows, but that sum never changes. For frequent vocabulary words it
  is precomputed once at load: a sentence that would gather ~650 rows becomes
  one vector add per word plus rows for OOV words and word n-grams. The default
  cache budget is 32 MiB (`FASTTEXT_TURBO_WORD_CACHE_MB`); this is exact up to
  floating-point summation order and adds roughly 40–55% throughput on
  n-gram-heavy models.
- **On-demand features, exact hash.** Character n-grams are computed straight
  from the input bytes with fastText's signed-byte FNV-1a hash and its UTF-8
  boundary rules. OOV words and cache misses take this path; nothing is built
  from an approximate tokenizer.
- **Cache-blocked batch scoring.** Scoring is a skinny GEMM:
  `scores[batch, labels] = hidden[batch, dim] * W[labels, dim]ᵀ`. The batch
  kernel walks labels in the outer loop and eight lines at a time in the inner
  loop, so each 1 KB label row is read from memory once per batch instead of
  once per line. Batches are sized to keep the score matrix in L2.
- **AVX2 + FMA, with runtime dispatch.** Eight independent accumulators keep the
  FMA pipeline busy; a NEON path covers Apple silicon and other aarch64
  targets; a portable scalar path is always available. There is no
  `-march=native`, so the same binary runs everywhere.
- **Vectorized softmax denominator.** Only labels whose contribution is above
  single precision are summed, and the vectorized exp keeps that pass off the
  profile. Selected labels are re-scored with `exp` for accuracy.
- **Prefetched row gathers.** A sentence averages hundreds of scattered 1 KB
  rows. The average kernel issues software prefetches several rows ahead,
  which roughly doubles the throughput of this memory-bound step.
- **All cores, in order.** Batched prediction splits work across scoped
  threads and reassembles results in input order. No external thread-pool or
  Rayon dependency.
- **Norm-bound label skipping.** For softmax models the row norms of the
  output matrix are precomputed. Blocks of eight labels are visited in
  descending norm order and skipped when Cauchy-Schwarz says they cannot beat
  the k-th score and are below the `exp(-20)` softmax cutoff. Scores feed a
  fused online softmax/top-k, so the full label-score matrix is never
  materialized. The engine measures the skip ratio on the first batches and
  automatically falls back to the dense batch kernel if fewer than 25% of
  blocks are skippable. This is exact within the existing denominator cutoff.
- **Single-pass feature extraction.** Tokenization, cache lookup, word
  n-grams and feature-row collection happen in one token pass. In-vocabulary
  words contribute a cached sum directly, without materializing subword ids,
  and no per-token allocation occurs under the hood.
- **Zero-copy, allocation-light CLI.** Input files are `mmap`ed and split into
  borrowed line slices. Predictions are written into one flat reusable buffer
  with per-line lengths, and top-k uses a reusable small-`k` array (with a
  heap fallback for large `k`, no heap per document), so the CLI stays within
  about 10% of the library's raw throughput.

The softmax denominator cutoff changes probabilities by less than 1e-6.
Everything else is the same arithmetic as fastText, in a different summation
order. The crate reproduces fastText's less obvious behaviours too: the
`log(p + 1e-5)` bias in returned probabilities, the 512-entry sigmoid lookup
table used by one-vs-all/negative-sampling prediction, hierarchical-softmax
tree construction from label counts, the sign-extension of 32-bit token hashes
when chaining word n-grams, and the pruning map in quantized models.

## Quick start

```bash
cargo build --release

# fastText-compatible CLI
ftt predict model.bin input.txt 1 0.0 8
ftt predict-prob model.bin input.txt 5
ftt predict-prob model.bin - 1 0.0 8 < input.txt

ftt bench model.bin input.txt -threads 8 -repeat 3
ftt info model.ftz
ftt features model.bin "hello world"
```

`ftt predict` and `ftt predict-prob` accept fastText's positional order
`<model> <input> [k] [threshold] [threads]`, plus named flags `-k`,
`-threshold`, `-threads`, `-batch`, `--no-eos`, and `--prob`.

### Library

```rust
use fasttext_turbo::{Model, PredictOptions};

fn main() -> fasttext_turbo::Result<()> {
    let model = Model::load("model.bin")?;
    let opts = PredictOptions::k(3).with_threads(8);

    // One line.
    for p in model.predict("Hello, how are you?", &opts)? {
        println!("{}\t{:.6}", p.label, p.probability);
    }

    // Many lines: batched and parallel, results in input order.
    let lines = ["bonjour", "привет", "你好"];
    let all = model.predict_batch(&lines, &opts)?;
    for row in all {
        for p in row {
            print!("{} {:.6}  ", p.label, p.probability);
        }
        println!();
    }
    Ok(())
}
```

Byte input (fastText hashes raw bytes, so this also accepts data that is not
valid UTF-8):

```rust
let preds = model.predict_bytes(b"hello \xff world", &PredictOptions::k(1))?;
```

For high-throughput callers, `predict_batch_bytes_into` writes all results
into one reusable flat vector plus one length per line — no `Vec` per
document. This is what the CLI uses:

```rust
let mut flat = Vec::new();
let mut lengths = Vec::new();
model.predict_batch_bytes_into(&lines_as_bytes, &opts, &mut flat, &mut lengths)?;
```

For a tight loop of single-line predictions, reuse a `Predictor`; it keeps the
feature, hidden-vector and score buffers alive:

```rust
let mut predictor = model.predictor()?;
for line in lines {
    let preds = predictor.predict(line, &PredictOptions::k(1))?;
}
```

`Prediction.label` borrows the model, so no string is allocated per result.
`Model` is `Sync`; wrap it in `Arc` to share it between threads.

## Compatibility and correctness

Supported model files:

| Feature | Status |
| --- | --- |
| File format v12 (and v11) | yes |
| Supervised models | yes |
| Softmax, one-vs-all, negative sampling | yes |
| Hierarchical softmax | yes |
| Character n-grams and word n-grams | yes |
| Quantized input (`input.ftz`, qnorm) | yes |
| Quantized output (`qout`) | yes |
| Pruned n-gram dictionary | yes |
| Unsupervised models | loaded, but prediction is rejected |
| Training / quantization | out of scope |

The test suite ships eleven tiny trained models plus a hand-written peaked
softmax model, covering softmax, hierarchical softmax, one-vs-all, negative
sampling, word n-grams, odd dimensions, UTF-8, OOV text, quantized input,
qnorm, `dsub=4`, quantized output (including hierarchical softmax), and
norm-bound skipping. Every recorded prediction is compared label-by-label and
probability-by-probability through the serial, batch, multi-threaded and
single-line paths.

On those fixtures: **0 label mismatches, maximum probability difference
1.6e-6**. On the real `lid.176.ftz` (quantized hierarchical softmax, 42,765
pruned n-grams, 176 labels): **0 label mismatches, max 4.2e-7**. A 500-line
bulk run over mixed multilingual text gives 0 label mismatches and at most
3.3e-6 for both `lid.176.ftz` and the 125 MB dense `lid.176.bin`.

The Python API appends a newline to every string, so an end-of-sentence token
is included by default. Use `append_eos: false` (or `--no-eos`) to predict the
raw line without it. A newline inside a line terminates processing, exactly as
in fastText.

## Benchmarks

20,000 lines; best of a few runs after a warm-up; Windows 11, AMD Ryzen 7 PRO
3700 (8 cores / 16 threads), 32 GB RAM. Reference is the stock `fasttext`
0.9.2 Python wheel, single-threaded (`multilinePredict`), on the same model and
machine. "1 thread" and "8 threads" are this crate, with the word cache on.
Machine load and page-cache state move these numbers by 20–30%, so treat the
ratios as indicative.

| Model | Size | Shape | fastText | ftt ×1 | Speedup | ftt ×8 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `lid.176.ftz` | 0.9 MB | 176 labels, dim 16, HS, PQ | 50,800 | 250,000 | **4.9×** | 1,300,000 |
| `lid.176.bin` | 125 MB | 176 labels, dim 16, HS | 44,900 | 290,000 | **6.5×** | 1,550,000 |
| synthetic | 217 MB | 2,102 labels, dim 256, softmax | 1,250 | 20,000 | **16×** | 95,000 |

The synthetic benchmark uses random in-vocabulary words, so the per-word cache
is fully effective; `bench/lines.txt` against that model is an OOV-heavy worst
case. The model has the label count and dimension of a GlotLID-v3-class
classifier and a 200,000-bucket (217 MB) n-gram table, but its weights are
random: it is a throughput benchmark, not an accuracy benchmark. GlotLID v3
itself has about 2M buckets and a 1.7 GB table, so its absolute lines/s are
lower. The 8-thread numbers depend on memory bandwidth: prediction is largely
a random-access streaming problem.

Cache effect on one core: `lid.176.ftz` 192k → 272k lines/s, `lid.176.bin`
202k → 309k, and the in-vocabulary synthetic input 14.1k → 21.7k. Flat output
buffers were 10–20% faster than a `Vec<Vec<Prediction>>` on 20,000 LID lines
(the difference is noisy at this throughput, but it removes 20,000
allocations). End-to-end CLI on 200,000 `lid.176.ftz` lines, including process
startup, model load and writing output: 276k lines/s on one thread and 922k on
eight (cache on).

Norm-bound skipping is workload-dependent. On the random synthetic model it
does not pay off, and the adaptive switch falls back to the dense batch kernel.
On a deliberately peaked 2,102-label softmax model
(`scripts/make_peaked_model.py`), it skips the vast majority of label blocks
per line:

| Peaked synthetic | lines/s, one core |
| --- | ---: |
| exhaustive batch kernel (`--no-norm-skip`) | 51,000 |
| fused online softmax + norm skip | 218,000 (**4.3×**) |

`ftt info` reports whether norm skipping is available and whether the adaptive
policy has turned it on or off.

Reproduce:

```bash
py -3.11 scripts/make_bench.py
py -3.11 scripts/make_synthetic_model.py
cargo build --release
./target/release/ftt bench bench/synthetic.bin bench/synthetic.txt -threads 1 -repeat 3
FT_MODEL=bench/synthetic.bin BENCH_LINES=bench/synthetic.txt \
  py -3.11 scripts/bench_reference.py
```

## How it works

```text
line ──► bytes ──► words + word n-grams
                └─► char n-grams (UTF-8 code-point boundaries)
                      └─► bucket hash ──► prune map ──► row ids
row ids ──► gather rows ──► average ──► hidden vector
hidden  ──► scores = hidden · Wᵀ ──► softmax / HS / sigmoid
                                             └─► top-k + threshold
```

- `src/dictionary.rs` — file dictionary, FNV-1a, UTF-8 n-grams, word n-grams,
  pruning, exact `hash % bucket` without division.
- `src/matrix.rs`, `src/quant.rs` — dense and product-quantized views over the
  mapping, including qnorm.
- `src/simd.rs` — AVX2/FMA, NEON and scalar gather/GEMM kernels, runtime
  dispatch, vectorized exp.
- `src/model.rs` — model loading, word cache, softmax/HS/sigmoid
  finalization, reusable top-k, batching and scoped-thread parallelism.
- `src/main.rs` — the `ftt` CLI, mmap/flat-buffer I/O and formatting.

The model mapping is read-only. As with any memory map, do not modify or
replace a model file while a `Model` is alive. The word cache is built during
`Model::load`; set `FASTTEXT_TURBO_WORD_CACHE_MB=0` to skip it (lowest load
latency and memory) or raise the budget for a large vocabulary. `ftt info`
prints how many words were cached.

## Building and testing

```bash
cargo test --release           # unit tests + reference fixtures
cargo check --target aarch64-unknown-linux-gnu   # cross-check NEON paths
```

Fixtures are committed under `tests/data/`; regenerate them (requires Python
3.9–3.12 with `fasttext`) with:

```bash
py -3.11 scripts/make_fixtures.py
```

The optional external-model test runs when `FASTTEXT_TURBO_MODEL` is set:

```bash
FASTTEXT_TURBO_MODEL=/path/to/lid.176.ftz cargo test --release optional_external_model
```

Dependencies: only [`memmap2`](https://crates.io/crates/memmap2). No Rayon, no
BLAS, no `serde`, no build scripts.

## Limitations

- Prediction only; training and quantization are deliberately out of scope.
- The convenient `&str` methods require UTF-8. `predict_bytes`,
  `predict_batch_bytes` and `feature_ids_bytes` (and the CLI) accept arbitrary
  bytes, matching fastText's byte-oriented tokenizer.
- Quantized-output models are dequantized per label on the fly; they are fast
  because the code rows are small, not because of SIMD.
- The per-word cache trades memory and load time for throughput. It is built
  for at most `FASTTEXT_TURBO_WORD_CACHE_MB` (default 32 MiB) and covers the
  most frequent words first.
- File input in the CLI is memory-mapped; stdin is streamed in 4,096-line
  chunks through a reusable byte arena. Input files are not copied.
- The softmax denominator uses a vectorized exp approximation with relative
  error around 1e-7, and skips terms below `exp(-20)`. Labels are unaffected.
- Norm skipping only applies to softmax models with at least 64 labels and
  `k < labels`, and only when the adaptive policy sees enough skippable
  blocks. Disable it explicitly with `--no-norm-skip` or
  `PredictOptions { norm_skip: false, .. }`.
- For softmax, ties among labels whose probability rounds to the `+1e-5`
  output bias can follow raw-score order instead of fastText's
  `log(p + 1e-5)` order. This affects only the identity of negligible tail
  labels when `k` is larger than the number of meaningful labels; top-1 and
  normal `k` are unaffected.

## License

MIT. fastText is MIT-licensed by Meta; this is an independent implementation
of its model format and prediction semantics. The speed-oriented design was
inspired by the [fastText-turbo](https://commonglot.github.io/blog/fasttext-turbo.html)
write-up.
