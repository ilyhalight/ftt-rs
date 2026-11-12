//! Minimal prediction example.
//!
//! cargo run --release --example predict -- model.bin

use std::env;

use fasttext_turbo::{Model, PredictOptions};

fn main() -> fasttext_turbo::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: predict <model.bin>");
        std::process::exit(2);
    });
    let model = Model::load(path)?;
    println!(
        "loaded {} labels, dim {}, {} words ({:.1} MiB mapped)",
        model.nlabels(),
        model.dim(),
        model.dictionary().nwords(),
        model.mapped_bytes() as f64 / (1024.0 * 1024.0)
    );

    let lines = [
        "Hello, how are you today?",
        "Bonjour tout le monde",
        "Привет, как дела?",
        "你好，今天过得怎么样？",
        "",
    ];
    let opts = PredictOptions::k(3).with_threads(0);
    for (line, preds) in lines.iter().zip(model.predict_batch(&lines, &opts)?) {
        print!("{line:?} ->");
        for p in preds {
            print!(" {} {:.5}", p.label, p.probability);
        }
        println!();
    }
    Ok(())
}
