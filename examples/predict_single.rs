use std::env;

use ftt_rs::{Model, PredictOptions};

fn main() -> ftt_rs::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: predict <model.bin>");
        std::process::exit(2);
    });
    let model = Model::load(path)?;
    let opts = PredictOptions::default();
    let predictions = model.predict("Hello, how are you today?", &opts)?;
    println!("{:?}", predictions);
    Ok(())
}
