//! Model loading and prediction.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;

use memmap2::Mmap;

use crate::args::{Args, Loss, ModelKind};
use crate::dictionary::Dictionary;
use crate::io::Cursor;
use crate::matrix::MatrixView;
use crate::Error;
use crate::Result;

const MAGIC: i32 = 793_712_314;
const MAX_SUPPORTED_VERSION: i32 = 12;

/// Options for a prediction.
///
/// Construct it with [`PredictOptions::default`] or [`PredictOptions::k`] and
/// the `with_*` builder methods. The struct is `#[non_exhaustive]` so new
/// options can be added without breaking callers.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct PredictOptions {
    /// Number of labels to return. `0` returns no labels.
    pub k: usize,
    /// Minimum probability. Labels below it are omitted, as in fastText.
    pub threshold: f32,
    /// Worker threads for [`Model::predict_batch`]. `0` uses all available
    /// cores; values are clamped to the core count and to the amount of work
    /// (at least 128 lines per worker). A single-line prediction always runs
    /// on the calling thread.
    pub threads: usize,
    /// Lines per batch. `0` chooses a size that keeps the score matrix in
    /// cache (between 8 and 64 lines). Batches are independent, so this only
    /// affects speed and the last bits of the probabilities.
    pub batch_size: usize,
    /// Append the end-of-sentence token. The Python API appends a newline to
    /// every string, so this is `true` by default for compatibility. Set it
    /// to `false` to score the raw line; a literal newline in `line` still
    /// produces an EOS token, as in fastText.
    pub append_eos: bool,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            k: 1,
            threshold: 0.0,
            threads: 0,
            batch_size: 0,
            append_eos: true,
        }
    }
}

impl PredictOptions {
    pub fn k(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the number of lines per scoring batch. `0` selects automatically.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets whether an end-of-sentence token is appended to each line.
    pub fn with_append_eos(mut self, append_eos: bool) -> Self {
        self.append_eos = append_eos;
        self
    }
}

/// One label and its probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction<'a> {
    /// Full label string, including the model's label prefix.
    pub label: &'a str,
    /// Probability as fastText reports it. Like the reference implementation,
    /// this includes a small `+1e-5` bias added when converting its internal
    /// log-probability back to a probability.
    pub probability: f32,
}

/// State of the lazy per-word sum cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordCacheState {
    /// The model configuration or cache budget disables the cache.
    Disabled,
    /// Configured, but not built yet. The first prediction builds it.
    Pending,
    /// Ready to use.
    Built { words: usize, bytes: usize },
}

/// A loaded fastText model. All matrices are memory-mapped; loading a 2 GB
/// model does not copy it into the heap.
///
/// The mapping is read-only. Do not modify or replace the model file while a
/// `Model` is alive. `Model` is `Sync`, so an `Arc<Model>` can be shared by
/// many threads.
pub struct Model {
    mmap: Mmap,
    version: i32,
    args: Args,
    dict: Dictionary,
    input: MatrixView,
    output: MatrixView,
    quantized: bool,
    hs: Option<HsTree>,
    /// Built on the first prediction. `None` inside means the configuration
    /// disables the cache.
    word_cache: OnceLock<Option<WordCache>>,
}

/// Sum of the input rows contributed by each frequent vocabulary word.
///
/// Character n-grams make a word contribute dozens of rows, but the sum of
/// those rows never changes. Precomputing it turns a 650-row gather into one
/// vector add per word plus a handful of rows for OOV words and word n-grams.
/// The result is exact up to floating-point summation order.
struct WordCache {
    dim: usize,
    sums: Vec<f32>,
    counts: Vec<u32>,
    /// word id -> slot in `sums`/`counts`, or -1 when not cached.
    index: Vec<i32>,
}

impl WordCache {
    const DEFAULT_MB: usize = 32;

    fn budget_mb() -> usize {
        std::env::var("FASTTEXT_TURBO_WORD_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(Self::DEFAULT_MB)
    }

    fn configured(args: &Args, nwords: usize) -> bool {
        nwords > 0
            && args.model == ModelKind::Supervised
            && args.maxn > 0
            && args.minn <= args.maxn
            && args.bucket > 0
    }

    fn build(args: &Args, dict: &Dictionary, input: &MatrixView, data: &[u8]) -> Option<Self> {
        let nwords = dict.nwords();
        if !Self::configured(args, nwords) {
            return None;
        }
        let budget_mb = Self::budget_mb();
        if budget_mb == 0 {
            return None;
        }
        let dim = args.dim;
        let per_word = dim.checked_mul(4)?.checked_add(4)?;
        let max_slots = budget_mb.saturating_mul(1024 * 1024) / per_word;
        if max_slots == 0 {
            return None;
        }

        // For a large vocabulary, cache the most frequent words first: they
        // account for most tokens. The smallest models fit entirely.
        let selected: Vec<u32> = if nwords <= max_slots {
            (0..nwords as u32).collect()
        } else {
            let counts = dict.word_counts();
            let mut order: Vec<u32> = (0..nwords as u32).collect();
            order.sort_unstable_by_key(|&i| std::cmp::Reverse(counts[i as usize]));
            order.truncate(max_slots);
            order
        };

        let mut index = vec![-1i32; nwords];
        let mut sums = Vec::with_capacity(selected.len() * dim);
        let mut counts = Vec::with_capacity(selected.len());
        let mut ids = Vec::new();
        let mut scratch = Vec::new();
        let mut sum = vec![0.0f32; dim];
        for wid in selected {
            dict.word_feature_ids(wid as usize, &mut ids, &mut scratch);
            if ids.is_empty() {
                continue;
            }
            input.average_rows(data, &ids, &mut sum);
            let n = ids.len() as f32;
            for value in &mut sum {
                *value *= n;
            }
            let slot = counts.len() as i32;
            index[wid as usize] = slot;
            sums.extend_from_slice(&sum);
            counts.push(ids.len() as u32);
        }
        if counts.is_empty() {
            None
        } else {
            Some(Self {
                dim,
                sums,
                counts,
                index,
            })
        }
    }

    #[inline]
    fn sum(&self, slot: usize) -> &[f32] {
        let start = slot * self.dim;
        &self.sums[start..start + self.dim]
    }
}

/// Reusable buffers for repeated single-line predictions.
pub struct Predictor<'m> {
    model: &'m Model,
    ws: Workspace,
}

#[derive(Default)]
struct Workspace {
    feats: Vec<i32>,
    hashes: Vec<i32>,
    scratch: Vec<u8>,
    word_indices: Vec<u32>,
    hidden: Vec<f32>,
    scores: Vec<f32>,
    line_hidden: Vec<f32>,
    feat_avg: Vec<f32>,
    top: TopK,
    ranked: Vec<(u32, f32)>,
    stack: Vec<(usize, f32)>,
    offsets: Vec<usize>,
    valid: Vec<usize>,
}

impl Model {
    /// Loads a fastText `.bin` or `.ftz` model from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // The mapping is read-only. As with any mmap, modifying the file while
        // the model is alive is undefined behaviour.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> Result<Self> {
        let mut c = Cursor::new(&mmap);
        let magic = c.i32()?;
        if magic != MAGIC {
            return Err(Error::Format(format!(
                "bad magic {magic:#x} (expected {MAGIC:#x})"
            )));
        }
        let version = c.i32()?;
        if version > MAX_SUPPORTED_VERSION {
            return Err(Error::Unsupported(format!(
                "file format version {version} (this build supports up to {MAX_SUPPORTED_VERSION})"
            )));
        }

        let mut args = Args::load(&mut c)?;
        if version == 11 && args.model == ModelKind::Supervised {
            // Old supervised models did not use character n-grams.
            args.maxn = 0;
        }
        let dict = Dictionary::load(&mut c, &args)?;

        let quant_input = c.bool()?;
        let input = MatrixView::load(&mut c, quant_input, args.dim)?;

        if !quant_input && dict.prune_size() != -1 {
            return Err(Error::Format(
                "pruned dictionary in an unquantized model (see fastText issue #332)".into(),
            ));
        }

        let qout = c.bool()?;
        let output = MatrixView::load(&mut c, quant_input && qout, args.dim)?;

        if input.cols() != args.dim || output.cols() != args.dim {
            return Err(Error::Format("matrix dimension mismatch".into()));
        }
        if input.rows() < dict.nwords() {
            return Err(Error::Format(format!(
                "input matrix has {} rows but the vocabulary has {} words",
                input.rows(),
                dict.nwords()
            )));
        }
        // Every feature row must be inside the input matrix. Without this a
        // crafted model could make the scoring kernels read out of bounds.
        let uses_hashes = args.maxn > 0 || args.word_ngrams > 1;
        if uses_hashes {
            let hashed_rows = if dict.prune_size() >= 0 {
                dict.prune_size() as usize
            } else {
                args.bucket
            };
            if hashed_rows > 0 {
                let needed = dict
                    .nwords()
                    .checked_add(hashed_rows)
                    .ok_or_else(|| Error::Format("vocabulary size overflow".into()))?;
                if needed > i32::MAX as usize {
                    return Err(Error::Format(
                        "feature row ids do not fit in a signed 32-bit integer".into(),
                    ));
                }
                if input.rows() < needed {
                    return Err(Error::Format(format!(
                        "input matrix has {} rows but features can address {}",
                        input.rows(),
                        needed
                    )));
                }
            }
        }
        if args.model == ModelKind::Supervised && output.rows() != dict.nlabels() {
            return Err(Error::Format(format!(
                "output matrix has {} rows but the model has {} labels",
                output.rows(),
                dict.nlabels()
            )));
        }

        let hs = if args.model == ModelKind::Supervised && args.loss == Loss::HierarchicalSoftmax {
            Some(HsTree::build(dict.label_counts()))
        } else {
            None
        };
        Ok(Self {
            mmap,
            version,
            args,
            dict,
            input,
            output,
            quantized: quant_input,
            hs,
            word_cache: OnceLock::new(),
        })
    }

    /// `(cached words, bytes used by cached sums)` once the lazy cache has
    /// been built. Returns `None` before the first prediction or when the
    /// cache is disabled.
    pub fn word_cache_stats(&self) -> Option<(usize, usize)> {
        self.word_cache
            .get()?
            .as_ref()
            .map(|c| (c.counts.len(), c.sums.len() * 4))
    }

    /// Current state of the lazy per-word cache.
    pub fn word_cache_state(&self) -> WordCacheState {
        match self.word_cache.get() {
            Some(Some(cache)) => WordCacheState::Built {
                words: cache.counts.len(),
                bytes: cache.sums.len() * 4,
            },
            Some(None) => WordCacheState::Disabled,
            None if self.word_cache_wanted() => WordCacheState::Pending,
            None => WordCacheState::Disabled,
        }
    }

    fn word_cache_wanted(&self) -> bool {
        WordCache::configured(&self.args, self.dict.nwords()) && WordCache::budget_mb() > 0
    }

    #[inline]
    fn word_cache(&self) -> Option<&WordCache> {
        self.word_cache
            .get_or_init(|| WordCache::build(&self.args, &self.dict, &self.input, &self.mmap))
            .as_ref()
    }

    #[inline]
    pub fn args(&self) -> &Args {
        &self.args
    }

    #[inline]
    pub fn dictionary(&self) -> &Dictionary {
        &self.dict
    }

    #[inline]
    pub fn version(&self) -> i32 {
        self.version
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.args.dim
    }

    #[inline]
    pub fn nlabels(&self) -> usize {
        self.dict.nlabels()
    }

    #[inline]
    pub fn is_supervised(&self) -> bool {
        self.args.model == ModelKind::Supervised
    }

    #[inline]
    pub fn is_quantized(&self) -> bool {
        self.quantized || self.output.is_quantized()
    }

    /// Number of bytes in the model mapping.
    pub fn mapped_bytes(&self) -> usize {
        self.mmap.len()
    }

    /// Number of worker threads [`Model::predict_batch`] would use for a
    /// batch of `lines` input lines with the given requested count.
    pub fn worker_threads(&self, lines: usize, requested: usize) -> usize {
        resolve_threads(requested, lines)
    }

    /// Computes the input row ids for one line, exactly as fastText does.
    /// Mostly useful for tests and diagnostics.
    pub fn feature_ids(&self, line: &str, append_eos: bool) -> Vec<i32> {
        self.feature_ids_bytes(line.as_bytes(), append_eos)
    }

    /// Byte-oriented version of [`Model::feature_ids`]. fastText hashes raw
    /// UTF-8 bytes, so this also accepts input that is not valid UTF-8.
    pub fn feature_ids_bytes(&self, line: &[u8], append_eos: bool) -> Vec<i32> {
        let mut feats = Vec::new();
        let mut hashes = Vec::new();
        let mut words = Vec::new();
        let mut scratch = Vec::new();
        self.dict.features_fused(
            line,
            append_eos,
            None,
            &mut feats,
            &mut hashes,
            &mut words,
            &mut scratch,
        );
        feats
    }

    /// Creates a reusable predictor (holds scratch buffers).
    pub fn predictor(&self) -> Result<Predictor<'_>> {
        if !self.is_supervised() {
            return Err(Error::Unsupported(
                "prediction requires a supervised model".into(),
            ));
        }
        Ok(Predictor {
            model: self,
            ws: Workspace::default(),
        })
    }

    /// Predicts labels for one line.
    pub fn predict<'a>(&'a self, line: &str, opts: &PredictOptions) -> Result<Vec<Prediction<'a>>> {
        self.predict_bytes(line.as_bytes(), opts)
    }

    /// Predicts labels for one raw byte string. fastText hashes bytes, so this
    /// also accepts input that is not valid UTF-8.
    pub fn predict_bytes<'a>(
        &'a self,
        line: &[u8],
        opts: &PredictOptions,
    ) -> Result<Vec<Prediction<'a>>> {
        self.predictor()?.predict_bytes(line, opts)
    }

    /// Predicts labels for many lines, optionally on several threads. Results
    /// are returned in input order.
    pub fn predict_batch<'a>(
        &'a self,
        lines: &[&str],
        opts: &PredictOptions,
    ) -> Result<Vec<Vec<Prediction<'a>>>> {
        let lines: Vec<&[u8]> = lines.iter().map(|s| s.as_bytes()).collect();
        self.predict_batch_bytes(&lines, opts)
    }

    /// Byte-oriented version of [`Model::predict_batch`].
    pub fn predict_batch_bytes<'a>(
        &'a self,
        lines: &[&[u8]],
        opts: &PredictOptions,
    ) -> Result<Vec<Vec<Prediction<'a>>>> {
        if !self.is_supervised() {
            return Err(Error::Unsupported(
                "prediction requires a supervised model".into(),
            ));
        }
        let threads = resolve_threads(opts.threads, lines.len());
        if threads <= 1 {
            return Ok(self.predict_batch_seq(lines, opts));
        }

        let chunk = lines.len().div_ceil(threads);
        let parts: Vec<Vec<Vec<Prediction<'_>>>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for part in lines.chunks(chunk) {
                handles.push(scope.spawn(move || self.predict_batch_seq(part, opts)));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("prediction worker panicked"))
                .collect()
        });
        Ok(parts.into_iter().flatten().collect())
    }

    /// Predicts many byte lines into flat, reusable buffers.
    ///
    /// `out` receives the predictions of all lines back-to-back and `lengths`
    /// receives one count per line, so callers such as the CLI can avoid a
    /// `Vec` allocation per document. Both buffers are cleared first.
    pub fn predict_batch_bytes_into<'a>(
        &'a self,
        lines: &[&[u8]],
        opts: &PredictOptions,
        out: &mut Vec<Prediction<'a>>,
        lengths: &mut Vec<usize>,
    ) -> Result<()> {
        if !self.is_supervised() {
            return Err(Error::Unsupported(
                "prediction requires a supervised model".into(),
            ));
        }
        out.clear();
        lengths.clear();
        out.reserve(lines.len().saturating_mul(opts.k.min(16)));
        lengths.reserve(lines.len());
        let threads = resolve_threads(opts.threads, lines.len());
        if threads <= 1 {
            self.predict_batch_seq_flat(lines, opts, out, lengths);
            return Ok(());
        }
        let chunk = lines.len().div_ceil(threads);
        let parts: Vec<(Vec<Prediction<'_>>, Vec<usize>)> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for part in lines.chunks(chunk) {
                handles.push(scope.spawn(move || {
                    let mut flat = Vec::new();
                    let mut lens = Vec::new();
                    self.predict_batch_seq_flat(part, opts, &mut flat, &mut lens);
                    (flat, lens)
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("prediction worker panicked"))
                .collect()
        });
        for (flat, lens) in parts {
            out.extend(flat);
            lengths.extend(lens);
        }
        Ok(())
    }

    fn predict_batch_seq_flat<'a>(
        &'a self,
        lines: &[&[u8]],
        opts: &PredictOptions,
        out: &mut Vec<Prediction<'a>>,
        lengths: &mut Vec<usize>,
    ) {
        let mut ws = Workspace::default();
        let batch_size = resolve_batch_size(opts.batch_size, self.nlabels(), self.output.rows());
        let mut start = 0usize;
        while start < lines.len() {
            let end = start.saturating_add(batch_size).min(lines.len());
            self.predict_batch_chunk(&lines[start..end], opts, &mut ws, |_, ranked| {
                lengths.push(ranked.len());
                out.extend(ranked.iter().map(|&(idx, p)| Prediction {
                    label: self.dict.label(idx as usize),
                    probability: p,
                }));
            });
            start = end;
        }
    }

    /// Convenience wrapper for owned strings (for example lines read from a
    /// file). Avoids building a temporary `Vec<&str>` at the call site.
    pub fn predict_batch_owned<'a>(
        &'a self,
        lines: &[String],
        opts: &PredictOptions,
    ) -> Result<Vec<Vec<Prediction<'a>>>> {
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        self.predict_batch(&refs, opts)
    }

    fn predict_batch_seq<'a>(
        &'a self,
        lines: &[&[u8]],
        opts: &PredictOptions,
    ) -> Vec<Vec<Prediction<'a>>> {
        let mut ws = Workspace::default();
        let mut out: Vec<Vec<Prediction<'a>>> = Vec::with_capacity(lines.len());
        let batch_size = resolve_batch_size(opts.batch_size, self.nlabels(), self.output.rows());
        let mut start = 0usize;
        while start < lines.len() {
            let end = start.saturating_add(batch_size).min(lines.len());
            self.predict_batch_chunk(&lines[start..end], opts, &mut ws, |_, ranked| {
                out.push(
                    ranked
                        .iter()
                        .map(|&(idx, p)| Prediction {
                            label: self.dict.label(idx as usize),
                            probability: p,
                        })
                        .collect(),
                );
            });
            start = end;
        }
        out
    }

    fn predict_batch_chunk<F>(
        &self,
        lines: &[&[u8]],
        opts: &PredictOptions,
        ws: &mut Workspace,
        mut on_line: F,
    ) where
        F: FnMut(usize, &[(u32, f32)]),
    {
        let dim = self.dim();
        ws.offsets.clear();
        ws.valid.clear();
        ws.hidden.clear();
        for (i, line) in lines.iter().enumerate() {
            ws.offsets.push(ws.hidden.len());
            ws.hidden.resize(ws.hidden.len() + dim, 0.0);
            let at = ws.hidden.len() - dim;
            let total = self.hidden_from_line(
                line,
                opts.append_eos,
                &mut ws.feats,
                &mut ws.hashes,
                &mut ws.word_indices,
                &mut ws.scratch,
                &mut ws.feat_avg,
                &mut ws.hidden[at..],
            );
            if total == 0 {
                ws.hidden.truncate(at);
                continue;
            }
            ws.valid.push(i);
        }

        let labels = self.nlabels();
        if self.args.loss != Loss::HierarchicalSoftmax {
            let n = ws.valid.len();
            ws.scores.clear();
            ws.scores.resize(n.saturating_mul(labels), 0.0);
            if n > 0 {
                self.output
                    .score_batch(&self.mmap, &ws.hidden, n, &mut ws.scores);
            }
        }

        let mut taken = 0usize;
        for i in 0..lines.len() {
            if taken < ws.valid.len() && ws.valid[taken] == i {
                let at = ws.offsets[i];
                let hidden = &ws.hidden[at..at + dim];
                match (self.args.loss, &self.hs) {
                    (Loss::Softmax, _) => {
                        let s = &ws.scores[taken * labels..(taken + 1) * labels];
                        finalize_softmax_into(
                            s,
                            opts.k,
                            opts.threshold,
                            &mut ws.top,
                            &mut ws.ranked,
                        );
                    }
                    (Loss::HierarchicalSoftmax, Some(hs)) => {
                        hs.predict_into(
                            hidden,
                            &self.output,
                            &self.mmap,
                            opts.k,
                            opts.threshold,
                            &mut ws.top,
                            &mut ws.stack,
                            &mut ws.ranked,
                        );
                    }
                    (Loss::OneVsAll | Loss::NegativeSampling, _) => {
                        let s = &ws.scores[taken * labels..(taken + 1) * labels];
                        finalize_sigmoid_into(
                            s,
                            opts.k,
                            opts.threshold,
                            &mut ws.top,
                            &mut ws.ranked,
                        );
                    }
                    (Loss::HierarchicalSoftmax, None) => ws.ranked.clear(),
                }
                on_line(i, &ws.ranked);
                taken += 1;
            } else {
                on_line(i, &[]);
            }
        }
    }
}

impl<'m> Predictor<'m> {
    /// Predicts labels for one line using this predictor's scratch buffers.
    pub fn predict(&mut self, line: &str, opts: &PredictOptions) -> Result<Vec<Prediction<'m>>> {
        self.predict_bytes(line.as_bytes(), opts)
    }

    /// Byte-oriented version of [`Predictor::predict`].
    pub fn predict_bytes(
        &mut self,
        line: &[u8],
        opts: &PredictOptions,
    ) -> Result<Vec<Prediction<'m>>> {
        let model = self.model;
        let dim = model.dim();
        self.ws.line_hidden.resize(dim, 0.0);
        let total = model.hidden_from_line(
            line,
            opts.append_eos,
            &mut self.ws.feats,
            &mut self.ws.hashes,
            &mut self.ws.word_indices,
            &mut self.ws.scratch,
            &mut self.ws.feat_avg,
            &mut self.ws.line_hidden,
        );
        if total == 0 {
            return Ok(Vec::new());
        }
        let ws = &mut self.ws;
        model.predict_hidden_into(
            &ws.line_hidden,
            opts,
            &mut ws.scores,
            &mut ws.top,
            &mut ws.ranked,
            &mut ws.stack,
        );
        Ok(ws
            .ranked
            .iter()
            .map(|&(idx, p)| Prediction {
                label: model.dict.label(idx as usize),
                probability: p,
            })
            .collect())
    }

    /// Predicts many lines, in order. Uses the same threading policy as
    /// [`Model::predict_batch`] when `opts.threads` is greater than one.
    pub fn predict_batch(
        &mut self,
        lines: &[&str],
        opts: &PredictOptions,
    ) -> Result<Vec<Vec<Prediction<'m>>>> {
        let lines: Vec<&[u8]> = lines.iter().map(|s| s.as_bytes()).collect();
        self.predict_batch_bytes(&lines, opts)
    }

    /// Byte-oriented version of [`Predictor::predict_batch`].
    pub fn predict_batch_bytes(
        &mut self,
        lines: &[&[u8]],
        opts: &PredictOptions,
    ) -> Result<Vec<Vec<Prediction<'m>>>> {
        if resolve_threads(opts.threads, lines.len()) <= 1 {
            let mut out = Vec::with_capacity(lines.len());
            for line in lines {
                out.push(self.predict_bytes(line, opts)?);
            }
            Ok(out)
        } else {
            self.model.predict_batch_bytes(lines, opts)
        }
    }
}

impl Model {
    /// Builds the hidden vector for a line, adding cached word sums directly
    /// while the line is tokenized. Returns the number of input rows that
    /// contributed; zero means there were no features.
    #[allow(clippy::too_many_arguments)]
    fn hidden_from_line(
        &self,
        line: &[u8],
        append_eos: bool,
        feats: &mut Vec<i32>,
        hashes: &mut Vec<i32>,
        words: &mut Vec<u32>,
        scratch: &mut Vec<u8>,
        feat_avg: &mut Vec<f32>,
        out: &mut [f32],
    ) -> usize {
        let dim = self.dim();
        let cache = self.word_cache();
        self.dict.features_fused(
            line,
            append_eos,
            cache.map(|c| c.index.as_slice()),
            feats,
            hashes,
            words,
            scratch,
        );
        if words.is_empty() {
            // No cached word was used: let the optimized gather kernel sum
            // the rows directly (the common all-OOV / cache-disabled path).
            if feats.is_empty() {
                out.fill(0.0);
                return 0;
            }
            self.input.average_rows(&self.mmap, feats, out);
            return feats.len();
        }
        out.fill(0.0);
        let mut total = 0usize;
        if let Some(cache) = cache {
            for &slot in words.iter() {
                let slot = slot as usize;
                let sum = cache.sum(slot);
                for j in 0..dim {
                    out[j] += sum[j];
                }
                total += cache.counts[slot] as usize;
            }
        }
        if !feats.is_empty() {
            // OOV subwords, cache misses and word n-grams are individual rows.
            feat_avg.resize(dim, 0.0);
            self.input.average_rows(&self.mmap, feats, feat_avg);
            let n = feats.len() as f32;
            for j in 0..dim {
                out[j] += feat_avg[j] * n;
            }
            total += feats.len();
        }
        if total == 0 {
            return 0;
        }
        let inv = 1.0 / total as f32;
        for value in out {
            *value *= inv;
        }
        total
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_hidden_into(
        &self,
        hidden: &[f32],
        opts: &PredictOptions,
        scores: &mut Vec<f32>,
        top: &mut TopK,
        ranked: &mut Vec<(u32, f32)>,
        stack: &mut Vec<(usize, f32)>,
    ) {
        match (self.args.loss, &self.hs) {
            (Loss::Softmax, _) => {
                scores.clear();
                scores.resize(self.nlabels(), 0.0);
                self.output.score_line(&self.mmap, hidden, scores);
                finalize_softmax_into(scores, opts.k, opts.threshold, top, ranked);
            }
            (Loss::HierarchicalSoftmax, Some(hs)) => {
                hs.predict_into(
                    hidden,
                    &self.output,
                    &self.mmap,
                    opts.k,
                    opts.threshold,
                    top,
                    stack,
                    ranked,
                );
            }
            (Loss::OneVsAll | Loss::NegativeSampling, _) => {
                scores.clear();
                scores.resize(self.nlabels(), 0.0);
                self.output.score_line(&self.mmap, hidden, scores);
                finalize_sigmoid_into(scores, opts.k, opts.threshold, top, ranked);
            }
            (Loss::HierarchicalSoftmax, None) => ranked.clear(),
        }
    }
}

/// A worker needs at least this many lines to amortize spawn and join costs.
/// Below this, staying on one thread is substantially faster.
const MIN_LINES_PER_THREAD: usize = 128;

fn resolve_threads(requested: usize, lines: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    resolve_threads_with(requested, lines, available)
}

fn resolve_threads_with(requested: usize, lines: usize, available: usize) -> usize {
    let requested = if requested == 0 {
        available
    } else {
        requested.min(available)
    };
    let by_work = (lines / MIN_LINES_PER_THREAD).max(1);
    requested.max(1).min(by_work).min(lines.max(1))
}

fn resolve_batch_size(requested: usize, labels: usize, out_rows: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    // Keep the batch score matrix around 2 MiB, which fits in a typical L2.
    let rows = out_rows.max(labels).max(1);
    let by_cache = (2 * 1024 * 1024) / (rows * 4).max(1);
    by_cache.clamp(8, 64)
}

// ---------------------------------------------------------------------------
// Top-k and probability finalization
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Candidate {
    score: f32,
    index: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score is better. fastText's std::heap ordering leaves tied
        // labels in reverse label-index order; matching that keeps the output
        // label sequence identical for quantized sigmoid ties.
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.index.cmp(&other.index))
    }
}

/// Reusable top-k accumulator. Small k (the common case) uses a sorted Vec
/// with no allocation after the first line; large k falls back to a heap.
struct TopK {
    k: usize,
    small: bool,
    items: Vec<Candidate>,
    heap: BinaryHeap<std::cmp::Reverse<Candidate>>,
    sorted: Vec<Candidate>,
}

impl Default for TopK {
    fn default() -> Self {
        Self {
            k: 0,
            small: true,
            items: Vec::new(),
            heap: BinaryHeap::new(),
            sorted: Vec::new(),
        }
    }
}

impl TopK {
    fn reset(&mut self, k: usize, labels: usize) {
        self.k = k.clamp(1, labels.max(1));
        self.small = self.k <= 16;
        self.items.clear();
        self.heap.clear();
        self.sorted.clear();
        if !self.small {
            self.heap
                .reserve(self.k.saturating_sub(self.heap.capacity()));
        }
    }

    #[inline]
    fn push(&mut self, score: f32, index: usize) {
        let cand = Candidate {
            score,
            index: index as u32,
        };
        if self.small {
            if self.items.len() < self.k {
                let pos = self.items.partition_point(|x| *x <= cand);
                self.items.insert(pos, cand);
            } else if cand > self.items[0] {
                self.items.remove(0);
                let pos = self.items.partition_point(|x| *x <= cand);
                self.items.insert(pos, cand);
            }
        } else if self.heap.len() < self.k {
            self.heap.push(std::cmp::Reverse(cand));
        } else if let Some(worst) = self.heap.peek() {
            if cand > worst.0 {
                self.heap.pop();
                self.heap.push(std::cmp::Reverse(cand));
            }
        }
    }

    #[inline]
    fn len(&self) -> usize {
        if self.small {
            self.items.len()
        } else {
            self.heap.len()
        }
    }

    #[inline]
    fn worst(&self) -> Option<f32> {
        if self.small {
            self.items.first().map(|c| c.score)
        } else {
            self.heap.peek().map(|r| (r.0).score)
        }
    }

    /// Ascending by score (worst first); iterate in reverse for best-first.
    fn sorted(&mut self) -> &[Candidate] {
        if self.small {
            &self.items
        } else {
            self.sorted.clear();
            while let Some(std::cmp::Reverse(c)) = self.heap.pop() {
                self.sorted.push(c);
            }
            self.sorted.sort_unstable();
            &self.sorted
        }
    }
}

#[inline]
fn std_log(x: f32) -> f32 {
    (x + 1e-5).ln()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// fastText's quantized sigmoid lookup table, used by the one-vs-all and
/// negative-sampling prediction paths. Reproducing the table (rather than
/// calling `exp`) is what makes their probabilities match the reference
/// implementation, including the occasional exactly-equal scores.
fn sigmoid_table(x: f32) -> f32 {
    const SIZE: usize = 512;
    const MAX: f32 = 8.0;
    use std::sync::OnceLock;
    static TABLE: OnceLock<[f32; SIZE + 1]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0.0f32; SIZE + 1];
        for (i, slot) in t.iter_mut().enumerate() {
            let v = (i as f32 * 2.0 * MAX) / SIZE as f32 - MAX;
            *slot = 1.0 / (1.0 + (-v).exp());
        }
        t
    });
    if x < -MAX {
        0.0
    } else if x > MAX {
        1.0
    } else {
        let i = ((x + MAX) * SIZE as f32 / MAX / 2.0) as usize;
        table[i.min(SIZE)]
    }
}

/// Softmax scores + top-k. Labels whose exponential is negligible are skipped
/// when summing the denominator, which is where large label sets spend their
/// time; the effect on any probability is far below single precision.
fn finalize_softmax_into(
    scores: &[f32],
    k: usize,
    threshold: f32,
    top: &mut TopK,
    out: &mut Vec<(u32, f32)>,
) {
    out.clear();
    if scores.is_empty() || k == 0 {
        return;
    }
    let mut max = scores[0];
    for &s in &scores[1..] {
        if s > max {
            max = s;
        }
    }

    top.reset(k, scores.len());
    for (i, &s) in scores.iter().enumerate() {
        top.push(s, i);
    }
    let z = crate::simd::exp_sum(scores, max);
    if !z.is_finite() || z <= 0.0 {
        return;
    }
    let inv = 1.0 / z;
    for c in top.sorted().iter().rev() {
        let p = ((c.score - max).exp() * inv).min(1.0);
        if p < threshold {
            continue;
        }
        // fastText stores log(p + 1e-5) in its heap and exponentiates it in
        // predictLine, so returned probabilities carry the same small bias.
        out.push((c.index, p + 1e-5));
    }
}

fn finalize_sigmoid_into(
    scores: &[f32],
    k: usize,
    threshold: f32,
    top: &mut TopK,
    out: &mut Vec<(u32, f32)>,
) {
    out.clear();
    if scores.is_empty() || k == 0 {
        return;
    }
    top.reset(k, scores.len());
    for (i, &s) in scores.iter().enumerate() {
        let p = sigmoid_table(s);
        if p < threshold {
            continue;
        }
        top.push(p, i);
    }
    for c in top.sorted().iter().rev() {
        out.push((c.index, c.score + 1e-5));
    }
}

#[cfg(test)]
fn finalize_softmax(scores: &[f32], k: usize, threshold: f32) -> Vec<(u32, f32)> {
    let mut top = TopK::default();
    let mut out = Vec::new();
    finalize_softmax_into(scores, k, threshold, &mut top, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Hierarchical softmax
// ---------------------------------------------------------------------------

struct HsNode {
    parent: i32,
    left: i32,
    right: i32,
    count: i64,
    binary: bool,
}

struct HsTree {
    osz: usize,
    nodes: Vec<HsNode>,
}

impl HsTree {
    fn build(counts: Vec<i64>) -> Self {
        let osz = counts.len();
        if osz == 0 {
            return Self {
                osz,
                nodes: Vec::new(),
            };
        }
        let mut nodes = Vec::with_capacity(2 * osz - 1);
        for _ in 0..2 * osz - 1 {
            nodes.push(HsNode {
                parent: -1,
                left: -1,
                right: -1,
                count: 1_000_000_000_000_000,
                binary: false,
            });
        }
        for (i, &count) in counts.iter().enumerate() {
            nodes[i].count = count;
        }

        let mut leaf = osz as i64 - 1;
        let mut node = osz as i64;
        for i in osz..2 * osz - 1 {
            let mut mini = [0i32; 2];
            for slot in mini.iter_mut() {
                if leaf >= 0 && nodes[leaf as usize].count < nodes[node as usize].count {
                    *slot = leaf as i32;
                    leaf -= 1;
                } else {
                    *slot = node as i32;
                    node += 1;
                }
            }
            nodes[i].left = mini[0];
            nodes[i].right = mini[1];
            nodes[i].count = nodes[mini[0] as usize]
                .count
                .saturating_add(nodes[mini[1] as usize].count);
            nodes[mini[0] as usize].parent = i as i32;
            nodes[mini[1] as usize].parent = i as i32;
            nodes[mini[1] as usize].binary = true;
        }
        Self { osz, nodes }
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_into(
        &self,
        hidden: &[f32],
        output: &MatrixView,
        data: &[u8],
        k: usize,
        threshold: f32,
        top: &mut TopK,
        stack: &mut Vec<(usize, f32)>,
        out: &mut Vec<(u32, f32)>,
    ) {
        out.clear();
        if self.osz == 0 || k == 0 {
            return;
        }
        top.reset(k, self.osz);
        let min_score = std_log(threshold);
        let root = 2 * self.osz - 2;
        // Iterative DFS. Huffman trees can be very deep for adversarial label
        // counts, and a recursive walk could overflow the thread stack.
        stack.clear();
        stack.push((root, 0.0f32));
        while let Some((node, score)) = stack.pop() {
            if score < min_score {
                continue;
            }
            if top.len() == top.k {
                if let Some(worst) = top.worst() {
                    if score < worst {
                        continue;
                    }
                }
            }
            let n = &self.nodes[node];
            if n.left == -1 && n.right == -1 {
                top.push(score, node);
                continue;
            }
            let row = node - self.osz;
            let f = sigmoid(output.dot_row(data, row, hidden));
            let right = score + std_log(f);
            let left = score + std_log(1.0 - f);
            // Visit the more probable child first. Because the stack is
            // LIFO, push the less probable child first. Exact ties keep
            // fastText's left-first DFS order.
            if f > 0.5 {
                stack.push((n.left as usize, left));
                stack.push((n.right as usize, right));
            } else {
                stack.push((n.right as usize, right));
                stack.push((n.left as usize, left));
            }
        }
        for c in top.sorted().iter().rev() {
            out.push((c.index, c.score.exp()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_orders_by_score_then_index() {
        let mut top = TopK::default();
        top.reset(3, 5);
        for (i, s) in [0.5f32, 0.9, 0.5, 0.1, 0.9].iter().enumerate() {
            top.push(*s, i);
        }
        let got: Vec<(f32, u32)> = top
            .sorted()
            .iter()
            .rev()
            .map(|c| (c.score, c.index))
            .collect();
        assert_eq!(got, vec![(0.9, 4), (0.9, 1), (0.5, 2)]);
    }

    #[test]
    fn softmax_probabilities_are_ordered_and_sum_to_one() {
        let v = [1.0, 2.0, 0.5, 2.0];
        let out = finalize_softmax(&v, 4, 0.0);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].0, 3);
        let sum: f32 = out.iter().map(|x| x.1 - 1e-5).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hs_tree_is_a_full_binary_tree() {
        let tree = HsTree::build(vec![5, 3, 2, 1]);
        for i in 0..tree.osz {
            assert_eq!(tree.nodes[i].left, -1);
            assert_eq!(tree.nodes[i].right, -1);
        }
        let root = 2 * tree.osz - 2;
        assert_eq!(tree.nodes[root].parent, -1);
        for i in tree.osz..2 * tree.osz - 1 {
            assert!(tree.nodes[i].left >= 0);
            assert!(tree.nodes[i].right >= 0);
        }
    }

    #[test]
    fn thread_count_scales_with_available_work() {
        assert_eq!(resolve_threads_with(0, 0, 16), 1);
        assert_eq!(resolve_threads_with(0, 16, 16), 1);
        assert_eq!(resolve_threads_with(8, 16, 16), 1);
        assert_eq!(resolve_threads_with(0, 63, 16), 1);
        assert_eq!(resolve_threads_with(0, 127, 16), 1);
        assert_eq!(resolve_threads_with(0, 128, 16), 1);
        assert_eq!(resolve_threads_with(0, 255, 16), 1);
        assert_eq!(resolve_threads_with(0, 256, 16), 2);
        assert_eq!(resolve_threads_with(0, 512, 16), 4);
        assert_eq!(resolve_threads_with(0, 4096, 16), 16);
        assert_eq!(resolve_threads_with(8, 20_000, 16), 8);
        assert_eq!(resolve_threads_with(64, 20_000, 16), 16);
    }

    #[test]
    fn topk_small_and_heap_modes_match_a_full_sort() {
        for k in [1usize, 3, 16, 17, 32] {
            let mut top = TopK::default();
            top.reset(k, 64);
            let values: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 * 0.5).collect();
            for (i, &v) in values.iter().enumerate() {
                top.push(v, i);
            }
            let mut all: Vec<Candidate> = values
                .iter()
                .enumerate()
                .map(|(i, &score)| Candidate {
                    score,
                    index: i as u32,
                })
                .collect();
            all.sort_unstable();
            let expected = &all[all.len() - k..];
            let got = top.sorted();
            assert_eq!(got.len(), k, "k={k}");
            for (a, b) in got.iter().zip(expected) {
                assert_eq!((a.score, a.index), (b.score, b.index), "k={k}");
            }
        }
    }
}

#[cfg(test)]
mod fused_tests {
    use super::*;

    #[test]
    fn cached_words_do_not_emit_feature_rows() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/models/dense_softmax.bin");
        let model = Model::load(&path).unwrap();
        assert!(matches!(model.word_cache_state(), WordCacheState::Pending));
        let dim = model.dim();
        let mut ws = Workspace::default();
        let mut out = vec![0.0f32; dim];
        let total = model.hidden_from_line(
            b"good great excellent movie",
            true,
            &mut ws.feats,
            &mut ws.hashes,
            &mut ws.word_indices,
            &mut ws.scratch,
            &mut ws.feat_avg,
            &mut out,
        );
        assert!(total > 0);
        assert!(matches!(
            model.word_cache_state(),
            WordCacheState::Built { .. }
        ));
        assert!(ws.feats.is_empty(), "unexpected rows: {:?}", ws.feats);
    }
}
