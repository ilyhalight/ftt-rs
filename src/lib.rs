//! # ftt-rs
//!
//! A fast, memory-mapped inference engine for fastText models insipred by fasttext-turbo,
//! written from scratch in Rust.
//!
//! ```no_run
//! # fn main() -> ftt_rs::Result<()> {
//! use ftt_rs::{Model, PredictOptions};
//!
//! let model = Model::load("model.bin")?;
//! let predictions = model.predict("hello world", &PredictOptions::k(3))?;
//! for p in predictions {
//!     println!("{}\t{:.5}", p.label, p.probability);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Design
//!
//! * The model file is memory-mapped and never copied.
//! * Character n-grams are computed on demand with the exact fastText hash
//!   (signed-byte FNV-1a). For frequent words the sum of their embedding rows
//!   is precomputed lazily on first use and reused; OOV words and cache misses
//!   take the exact n-gram path.
//! * Label scoring uses AVX2+FMA (x86-64) or NEON (aarch64) with runtime
//!   dispatch, plus a portable scalar fallback.
//! * Batched prediction uses a cache-blocked matrix-matrix kernel: each output
//!   row is read once per batch rather than once per line.
//! * The softmax denominator skips labels whose contribution is below single
//!   precision, and batches can run on all cores.

mod args;
mod dictionary;
mod error;
mod io;
mod matrix;
mod model;
mod quant;
mod simd;

pub use args::{Args, Loss, ModelKind};
pub use dictionary::Dictionary;
pub use error::{Error, Result};
pub use model::{Model, PredictOptions, Prediction, Predictor, WordCacheState};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
