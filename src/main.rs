use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use fasttext_turbo::{Model, PredictOptions, Prediction, WordCacheState};
use memmap2::Mmap;

/// Lines handed to one parallel batch. A chunk is large enough to amortize
/// thread spawn/join over many internal batches.
const CHUNK_LINES: usize = 65_536;

/// Large `k` produces many predictions per line; keep those chunks small to
/// bound the flat output buffer.
fn chunk_lines(opts: &PredictOptions) -> usize {
    if opts.k <= 16 {
        CHUNK_LINES
    } else {
        4_096
    }
}

enum InputData {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl InputData {
    fn bytes(&self) -> &[u8] {
        match self {
            InputData::Mapped(m) => m,
            InputData::Owned(v) => v,
        }
    }
}

fn load_input(path: Option<&str>) -> Result<InputData, String> {
    match path {
        Some(p) => {
            let file = File::open(p).map_err(|e| format!("cannot open {p}: {e}"))?;
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("cannot map {p}: {e}"))?;
            Ok(InputData::Mapped(mmap))
        }
        None => {
            let mut buf = Vec::new();
            io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| e.to_string())?;
            Ok(InputData::Owned(buf))
        }
    }
}

fn strip_cr(line: &[u8]) -> &[u8] {
    if line.last() == Some(&b'\r') {
        &line[..line.len() - 1]
    } else {
        line
    }
}

/// Splits a byte buffer into lines without copying. A trailing newline is a
/// terminator, not an extra empty line; an empty file has no lines.
fn split_lines(data: &[u8]) -> Vec<&[u8]> {
    if data.is_empty() {
        return Vec::new();
    }
    let trailing = data.last() == Some(&b'\n');
    let mut lines = Vec::new();
    let mut parts = data.split(|&b| b == b'\n').peekable();
    while let Some(line) = parts.next() {
        if trailing && parts.peek().is_none() {
            break;
        }
        lines.push(strip_cr(line));
    }
    lines
}

const HELP: &str = "\
fasttext-turbo — fast inference for fastText models

USAGE:
    ftt predict       <model> [input] [k] [threshold] [threads] [--prob]
    ftt predict-prob  <model> [input] [k] [threshold] [threads]
    ftt bench         <model> [input] [-k N] [-threads N] [-repeat R]
    ftt features      <model> <text>
    ftt info          <model>
    ftt --version

INPUT:
    A file path, or - for standard input (the default). One line per document,
    as in fastText. Lines are whitespace-tokenized exactly like fastText.

OPTIONS:
    -k N            labels per line (also accepted as a positional after input)
    -threshold P    minimum probability (default 0)
    -threads N      worker threads (default: all; small batches stay serial)
    -batch N        lines per scoring batch (default: automatic)
    --no-eos        do not append the end-of-sentence token
    --prob          print probabilities (predict-prob always does)
    -repeat R       bench repetitions (default 3)

The Python API appends a newline to every string, so the end-of-sentence token
is included by default. `--no-eos` matches raw single-line input without it.
";

const VALUE_FLAGS: &[&str] = &[
    "-k",
    "--k",
    "-threshold",
    "--threshold",
    "-threads",
    "--threads",
    "-batch",
    "--batch",
    "-repeat",
    "--repeat",
];
const BOOL_FLAGS: &[&str] = &["-h", "--help", "--version", "-V", "--no-eos", "--prob"];

struct Cli {
    args: Vec<String>,
}

impl Cli {
    fn new() -> Self {
        Self {
            args: std::env::args().skip(1).collect(),
        }
    }

    fn flag_value(&self, names: &[&str]) -> Option<&str> {
        for (i, a) in self.args.iter().enumerate() {
            if names.contains(&a.as_str()) {
                return self.args.get(i + 1).map(|s| s.as_str());
            }
        }
        None
    }

    fn has(&self, names: &[&str]) -> bool {
        self.args.iter().any(|a| names.contains(&a.as_str()))
    }

    fn validate_flags(&self) -> Result<(), String> {
        let mut i = 1; // args[0] is the command
        while i < self.args.len() {
            let a = self.args[i].as_str();
            if a == "--" {
                return Ok(());
            }
            let positional_number = a.parse::<f64>().is_ok();
            if a == "-" || !a.starts_with('-') || positional_number {
                i += 1;
                continue;
            }
            if VALUE_FLAGS.contains(&a) {
                if i + 1 >= self.args.len() {
                    return Err(format!("option {a} needs a value"));
                }
                i += 2;
            } else if BOOL_FLAGS.contains(&a) {
                i += 1;
            } else {
                return Err(format!("unknown option {a:?}; see --help"));
            }
        }
        Ok(())
    }

    fn positional(&self) -> Vec<&str> {
        let mut out = Vec::new();
        let mut i = 0;
        let mut after_dashdash = false;
        while i < self.args.len() {
            let a = self.args[i].as_str();
            if after_dashdash {
                out.push(a);
                i += 1;
                continue;
            }
            if a == "--" {
                after_dashdash = true;
                i += 1;
                continue;
            }
            if a == "-" {
                out.push(a);
                i += 1;
                continue;
            }
            if a.starts_with('-') && a.parse::<f64>().is_err() {
                i += if VALUE_FLAGS.contains(&a) { 2 } else { 1 };
                continue;
            }
            out.push(a);
            i += 1;
        }
        out
    }
}

fn parse_flag<T: std::str::FromStr>(
    cli: &Cli,
    names: &[&str],
    what: &str,
) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    cli.flag_value(names)
        .map(|s| {
            s.parse::<T>()
                .map_err(|e| format!("invalid {what} {s:?}: {e}"))
        })
        .transpose()
}

fn parse_positional<T: std::str::FromStr>(
    rest: &[&str],
    at: &mut usize,
    what: &str,
) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    let Some(s) = rest.get(*at) else {
        return Ok(None);
    };
    *at += 1;
    s.parse::<T>()
        .map(Some)
        .map_err(|e| format!("invalid {what} {s:?}: {e}"))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ftt: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::new();
    let command = cli.args.first().map(String::as_str).unwrap_or("");
    if cli.has(&["--version", "-V"]) {
        println!("ftt {}", fasttext_turbo::VERSION);
        return Ok(());
    }
    if cli.has(&["-h", "--help"]) {
        print!("{HELP}");
        return Ok(());
    }
    cli.validate_flags()?;
    match command {
        "" => {
            print!("{HELP}");
            Ok(())
        }
        "predict" | "predict-prob" | "predict_prob" => {
            let positional = cli.positional();
            let model_path = positional
                .get(1)
                .ok_or_else(|| "missing model path\n\n".to_string() + HELP)?;
            let input = positional.get(2).copied().filter(|p| *p != "-");
            // fastText's CLI order is: model input [k] [threshold] [threads].
            // Flags take precedence and consume no positional slot, so the two
            // forms can be mixed without shifting the remaining arguments.
            let rest = positional.get(3..).unwrap_or(&[]);
            let mut at = 0usize;
            let k = match parse_flag::<usize>(&cli, &["-k", "--k"], "k")? {
                Some(v) => v,
                None => parse_positional::<usize>(rest, &mut at, "k")?.unwrap_or(1),
            };
            if k == 0 {
                return Err("k needs to be 1 or higher".into());
            }
            let threshold =
                match parse_flag::<f32>(&cli, &["-threshold", "--threshold"], "threshold")? {
                    Some(v) => v,
                    None => parse_positional::<f32>(rest, &mut at, "threshold")?.unwrap_or(0.0),
                };
            let threads = match parse_flag::<usize>(&cli, &["-threads", "--threads"], "threads")? {
                Some(v) => v,
                None => parse_positional::<usize>(rest, &mut at, "threads")?.unwrap_or(0),
            };
            let batch_size =
                parse_flag::<usize>(&cli, &["-batch", "--batch"], "batch size")?.unwrap_or(0);
            if let Some(extra) = rest.get(at) {
                return Err(format!("unexpected argument {extra:?}"));
            }
            let opts = PredictOptions::k(k)
                .with_threshold(threshold)
                .with_threads(threads)
                .with_batch_size(batch_size)
                .with_append_eos(!cli.has(&["--no-eos"]));
            let show_prob = command != "predict" || cli.has(&["--prob"]);

            let model = Model::load(model_path).map_err(|e| e.to_string())?;
            predict_stream(&model, input, &opts, show_prob)
        }
        "bench" => {
            let positional = cli.positional();
            let model_path = positional
                .get(1)
                .ok_or_else(|| "missing model path\n\n".to_string() + HELP)?;
            let input = positional.get(2).copied().filter(|p| *p != "-");
            let k = parse_flag::<usize>(&cli, &["-k", "--k"], "k")?.unwrap_or(1);
            if k == 0 {
                return Err("k needs to be 1 or higher".into());
            }
            let threads =
                parse_flag::<usize>(&cli, &["-threads", "--threads"], "threads")?.unwrap_or(0);
            let repeat = parse_flag::<usize>(&cli, &["-repeat", "--repeat"], "repeat")?
                .unwrap_or(3)
                .max(1);
            let batch_size =
                parse_flag::<usize>(&cli, &["-batch", "--batch"], "batch size")?.unwrap_or(0);
            let opts = PredictOptions::k(k)
                .with_threads(threads)
                .with_batch_size(batch_size)
                .with_append_eos(!cli.has(&["--no-eos"]));
            bench(model_path, input, &opts, repeat)
        }
        "features" => {
            let positional = cli.positional();
            let model_path = positional
                .get(1)
                .ok_or_else(|| "missing model path".to_string())?;
            let text = positional.get(2).copied().unwrap_or("");
            let model = Model::load(model_path).map_err(|e| e.to_string())?;
            println!("{:?}", model.feature_ids(text, !cli.has(&["--no-eos"])));
            Ok(())
        }
        "info" => {
            let positional = cli.positional();
            let model_path = positional
                .get(1)
                .ok_or_else(|| "missing model path".to_string())?;
            print_info(model_path)
        }
        other => Err(format!("unknown command {other:?}\n\n{HELP}")),
    }
}

fn write_flat(
    out: &mut dyn Write,
    flat: &[Prediction<'_>],
    lengths: &[usize],
    show_prob: bool,
) -> Result<(), String> {
    let mut pos = 0usize;
    for &len in lengths {
        let end = pos + len;
        write_predictions(out, &flat[pos..end], show_prob)?;
        pos = end;
    }
    debug_assert_eq!(pos, flat.len());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flush_stdin_chunk<'a>(
    model: &'a Model,
    arena: &[u8],
    spans: &[(usize, usize)],
    opts: &PredictOptions,
    show_prob: bool,
    out: &mut dyn Write,
    flat: &mut Vec<Prediction<'a>>,
    lengths: &mut Vec<usize>,
) -> Result<(), String> {
    if spans.is_empty() {
        return Ok(());
    }
    let refs: Vec<&[u8]> = spans
        .iter()
        .map(|&(start, len)| &arena[start..start + len])
        .collect();
    model
        .predict_batch_bytes_into(&refs, opts, flat, lengths)
        .map_err(|e| e.to_string())?;
    write_flat(out, flat, lengths, show_prob)
}

fn predict_stream(
    model: &Model,
    input: Option<&str>,
    opts: &PredictOptions,
    show_prob: bool,
) -> Result<(), String> {
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 16, stdout.lock());
    let mut flat: Vec<Prediction<'_>> = Vec::new();
    let mut lengths: Vec<usize> = Vec::new();
    let chunk_lines = chunk_lines(opts);

    if let Some(path) = input {
        let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("cannot map {path}: {e}"))?;
        for chunk in split_lines(&mmap).chunks(chunk_lines) {
            model
                .predict_batch_bytes_into(chunk, opts, &mut flat, &mut lengths)
                .map_err(|e| e.to_string())?;
            write_flat(&mut out, &flat, &lengths, show_prob)?;
        }
    } else {
        // Stdin cannot be mapped; keep lines in a reusable arena so no line is
        // allocated individually. Flush in large chunks to amortize threads.
        let mut reader = BufReader::new(io::stdin());
        let mut line = Vec::new();
        let mut arena = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        loop {
            line.clear();
            let n = reader
                .read_until(b'\n', &mut line)
                .map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            if line.last() == Some(&b'\n') {
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
            }
            let start = arena.len();
            arena.extend_from_slice(&line);
            spans.push((start, line.len()));
            if spans.len() >= chunk_lines {
                flush_stdin_chunk(
                    model,
                    &arena,
                    &spans,
                    opts,
                    show_prob,
                    &mut out,
                    &mut flat,
                    &mut lengths,
                )?;
                arena.clear();
                spans.clear();
            }
        }
        flush_stdin_chunk(
            model,
            &arena,
            &spans,
            opts,
            show_prob,
            &mut out,
            &mut flat,
            &mut lengths,
        )?;
    }
    out.flush().map_err(|e| e.to_string())
}

/// Writes a probability like C++'s default `ostream << float`: six significant
/// digits, trailing zeros removed. Uses a stack buffer, not a String.
fn write_probability(out: &mut dyn Write, p: f32) -> io::Result<()> {
    if p == 0.0 {
        return out.write_all(b"0");
    }
    let exponent = p.abs().log10().floor() as i32;
    let mut buf = [0u8; 64];
    let len = {
        let mut slice = &mut buf[..];
        if !(-4..6).contains(&exponent) {
            write!(&mut slice, "{p:.5e}")?;
        } else {
            let decimals = (5 - exponent).max(0) as usize;
            write!(&mut slice, "{p:.decimals$}")?;
        }
        64 - slice.len()
    };
    let text = std::str::from_utf8(&buf[..len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if text.contains('.') {
        let end = text.find(['e', 'E']).unwrap_or(len);
        let mut trimmed = end;
        while trimmed > 0 && buf[trimmed - 1] == b'0' {
            trimmed -= 1;
        }
        if trimmed > 0 && buf[trimmed - 1] == b'.' {
            trimmed -= 1;
        }
        out.write_all(&buf[..trimmed])?;
        out.write_all(&buf[end..len])?;
    } else {
        out.write_all(&buf[..len])?;
    }
    Ok(())
}

#[cfg(test)]
fn format_probability(p: f32) -> String {
    let mut buf = Vec::new();
    write_probability(&mut buf, p).unwrap();
    String::from_utf8(buf).unwrap()
}

fn write_predictions(
    out: &mut dyn Write,
    preds: &[Prediction<'_>],
    show_prob: bool,
) -> Result<(), String> {
    if show_prob {
        for (i, p) in preds.iter().enumerate() {
            if i > 0 {
                write!(out, " ").map_err(|e| e.to_string())?;
            }
            write!(out, "{} ", p.label).map_err(|e| e.to_string())?;
            write_probability(out, p.probability).map_err(|e| e.to_string())?;
        }
    } else {
        for (i, p) in preds.iter().enumerate() {
            if i > 0 {
                write!(out, " ").map_err(|e| e.to_string())?;
            }
            write!(out, "{}", p.label).map_err(|e| e.to_string())?;
        }
    }
    writeln!(out).map_err(|e| e.to_string())
}

fn bench(
    model_path: &str,
    input: Option<&str>,
    opts: &PredictOptions,
    repeat: usize,
) -> Result<(), String> {
    let load_start = Instant::now();
    let model = Model::load(model_path).map_err(|e| e.to_string())?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1e3;

    let input_data = load_input(input)?;
    let lines = split_lines(input_data.bytes());
    if lines.is_empty() {
        return Err("no input lines".into());
    }
    let threads = model.worker_threads(lines.len(), opts.threads);

    eprintln!(
        "model: {} ({:.1} MiB, {:.2} ms to load)\n\
         {} labels, dim {}, {} words, model {}, loss {}, quantized {}\n\
         input: {} lines; k {}, threads {}, batch {}\n\
         word cache: {}",
        Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(model_path),
        model.mapped_bytes() as f64 / (1024.0 * 1024.0),
        load_ms,
        model.nlabels(),
        model.dim(),
        model.dictionary().nwords(),
        model.args().model.name(),
        model.args().loss.name(),
        model.is_quantized(),
        lines.len(),
        opts.k,
        threads,
        if opts.batch_size == 0 {
            "auto".to_string()
        } else {
            opts.batch_size.to_string()
        },
        match model.word_cache_state() {
            WordCacheState::Built { words, bytes } => {
                format!("{words} words, {:.1} MiB", bytes as f64 / 1048576.0)
            }
            WordCacheState::Pending => "lazy (builds on first prediction)".to_string(),
            WordCacheState::Disabled => "disabled".to_string(),
        }
    );

    let mut flat: Vec<Prediction<'_>> = Vec::new();
    let mut lengths: Vec<usize> = Vec::new();

    // Warm-up touches every page and primes the caches.
    let warm = Instant::now();
    model
        .predict_batch_bytes_into(&lines, opts, &mut flat, &mut lengths)
        .map_err(|e| e.to_string())?;
    std::hint::black_box(&flat);
    let warm_s = warm.elapsed().as_secs_f64();

    if let WordCacheState::Built { words, bytes } = model.word_cache_state() {
        eprintln!(
            "word cache built: {words} words ({:.1} MiB)",
            bytes as f64 / 1048576.0
        );
    }

    let mut best = f64::INFINITY;
    let mut total = 0.0;
    for run in 0..repeat {
        let start = Instant::now();
        model
            .predict_batch_bytes_into(&lines, opts, &mut flat, &mut lengths)
            .map_err(|e| e.to_string())?;
        std::hint::black_box(&flat);
        let s = start.elapsed().as_secs_f64();
        total += s;
        best = best.min(s);
        eprintln!(
            "run {}/{}: {:.3} s, {:.0} lines/s",
            run + 1,
            repeat,
            s,
            lines.len() as f64 / s
        );
    }
    eprintln!(
        "warm-up {:.3} s; best {:.0} lines/s; mean {:.0} lines/s",
        warm_s,
        lines.len() as f64 / best,
        lines.len() as f64 * repeat as f64 / total
    );
    Ok(())
}

fn print_info(model_path: &str) -> Result<(), String> {
    let model = Model::load(model_path).map_err(|e| e.to_string())?;
    let a = model.args();
    let d = model.dictionary();
    println!("file          {model_path}");
    println!("bytes         {}", model.mapped_bytes());
    println!("format        v{}", model.version());
    println!("model         {}", a.model.name());
    println!("loss          {}", a.loss.name());
    println!("dim           {}", a.dim);
    println!("words         {}", d.nwords());
    println!("labels        {}", d.nlabels());
    println!("tokens        {}", d.ntokens());
    println!("bucket        {}", a.bucket);
    println!("minn/maxn     {}/{}", a.minn, a.maxn);
    println!("wordNgrams    {}", a.word_ngrams);
    println!("quantized     {}", model.is_quantized());
    println!("pruned ngrams {}", d.prune_size());
    match model.word_cache_state() {
        WordCacheState::Built { words, bytes } => println!(
            "word cache    {words} words ({:.1} MiB)",
            bytes as f64 / 1048576.0
        ),
        WordCacheState::Pending => {
            println!("word cache    lazy (built on first prediction)")
        }
        WordCacheState::Disabled => println!("word cache    disabled"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_size_adapts_to_k() {
        assert_eq!(chunk_lines(&PredictOptions::k(1)), CHUNK_LINES);
        assert_eq!(chunk_lines(&PredictOptions::k(16)), CHUNK_LINES);
        assert_eq!(chunk_lines(&PredictOptions::k(17)), 4_096);
        assert_eq!(chunk_lines(&PredictOptions::k(100)), 4_096);
    }

    #[test]
    fn probability_format_matches_default_ostream_precision() {
        assert_eq!(format_probability(0.0), "0");
        assert_eq!(format_probability(0.974_427_16), "0.974427");
        assert_eq!(format_probability(0.000_722_140_2), "0.00072214");
        assert_eq!(format_probability(1.000_01), "1.00001");
        assert_eq!(format_probability(7.221_4e-6), "7.2214e-6");
    }

    #[test]
    fn positional_arguments_follow_fasttext_order() {
        let cli = Cli {
            args: "predict model.bin input.txt 5 0.25 8"
                .split(' ')
                .map(String::from)
                .collect(),
        };
        let p = cli.positional();
        assert_eq!(
            p,
            vec!["predict", "model.bin", "input.txt", "5", "0.25", "8"]
        );
    }
}
