//! End-to-end CLI tests. These also cover raw (non-UTF-8) byte input, which
//! fastText accepts because its tokenizer is byte-oriented.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn ftt() -> &'static str {
    env!("CARGO_BIN_EXE_ftt")
}

fn model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/models/dense_softmax.bin")
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir();
    dir.join(format!("ftt-cli-{}-{name}", std::process::id()))
}

#[test]
fn raw_bytes_are_tokenized_like_fasttext() {
    let input = temp("raw.txt");
    fs::write(&input, b"good great\nbad \xff\xfe\n").unwrap();
    let out = Command::new(ftt())
        .args([
            "predict-prob",
            model().to_str().unwrap(),
            input.to_str().unwrap(),
            "1",
            "0",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 2, "got {stdout:?}");
    assert!(stdout.lines().all(|l| l.starts_with("__label__")));
    let _ = fs::remove_file(input);
}

#[test]
fn positional_negative_threshold_and_single_dash_stdin() {
    let out = Command::new(ftt())
        .args([
            "predict-prob",
            model().to_str().unwrap(),
            "-",
            "3",
            "-1",
            "1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(b"good great\ntext\n")?;
            child.wait_with_output()
        })
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let rows: Vec<&str> = stdout.lines().collect();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row.split_whitespace().count(), 6, "row {row:?}");
    }
}

#[test]
fn invalid_options_and_k_zero_fail() {
    for args in [
        vec!["predict", model().to_str().unwrap(), "-", "0"],
        vec!["predict", model().to_str().unwrap(), "-", "abc"],
        vec!["predict", model().to_str().unwrap(), "-", "--bogus"],
        vec!["predict", model().to_str().unwrap(), "-", "-threads", "-1"],
    ] {
        let out = Command::new(ftt())
            .args(&args)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(
            !out.status.success(),
            "command {args:?} unexpectedly succeeded"
        );
        assert!(!out.stderr.is_empty());
    }
}

#[test]
fn empty_input_produces_no_lines() {
    let input = temp("empty.txt");
    fs::write(&input, b"").unwrap();
    let out = Command::new(ftt())
        .args([
            "predict",
            model().to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"");
    let _ = fs::remove_file(input);
}

#[test]
fn file_and_stdin_line_splitting_agree() {
    let cases: &[(&[u8], usize)] = &[
        (b"good\nbad\n", 2),
        (b"good\nbad", 2),
        (b"good\n\n", 2),
        (b"good\n\n\n", 3),
        (b"\n", 1),
        (b"good\r\nbad\r\n", 2),
        (b"good", 1),
        (b"", 0),
    ];
    let model = model();
    let path = temp("split.txt");
    for &(data, expected_lines) in cases {
        fs::write(&path, data).unwrap();
        let file_out = Command::new(ftt())
            .args(["predict", model.to_str().unwrap(), path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(file_out.status.success());
        assert_eq!(
            file_out.stdout.iter().filter(|&&b| b == b'\n').count(),
            expected_lines,
            "file case {data:?}"
        );

        let mut child = Command::new(ftt())
            .args(["predict", model.to_str().unwrap(), "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.as_mut().unwrap().write_all(data).unwrap();
        let stdin_out = child.wait_with_output().unwrap();
        assert!(stdin_out.status.success());
        assert_eq!(stdin_out.stdout, file_out.stdout, "stdin case {data:?}");
    }
    let _ = fs::remove_file(path);
}
