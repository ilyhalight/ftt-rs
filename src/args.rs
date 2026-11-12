//! The training arguments embedded in a fastText model file.
//!
//! Only the first 13 fields are stored by fastText's `Args::save`; the rest of
//! the file is the dictionary and the two matrices. Some fields (labels prefix,
//! quantization flags) are stored elsewhere or not at all.

use crate::io::Cursor;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    HierarchicalSoftmax,
    NegativeSampling,
    Softmax,
    OneVsAll,
}

impl Loss {
    fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            1 => Loss::HierarchicalSoftmax,
            2 => Loss::NegativeSampling,
            3 => Loss::Softmax,
            4 => Loss::OneVsAll,
            other => return Err(Error::Format(format!("unknown loss id {other}"))),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Loss::HierarchicalSoftmax => "hs",
            Loss::NegativeSampling => "ns",
            Loss::Softmax => "softmax",
            Loss::OneVsAll => "ova",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Cbow,
    SkipGram,
    Supervised,
}

impl ModelKind {
    fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            1 => ModelKind::Cbow,
            2 => ModelKind::SkipGram,
            3 => ModelKind::Supervised,
            other => return Err(Error::Format(format!("unknown model id {other}"))),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            ModelKind::Cbow => "cbow",
            ModelKind::SkipGram => "sg",
            ModelKind::Supervised => "sup",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Args {
    pub dim: usize,
    pub ws: i32,
    pub epoch: i32,
    pub min_count: i32,
    pub neg: i32,
    pub word_ngrams: i32,
    pub loss: Loss,
    pub model: ModelKind,
    pub bucket: usize,
    pub minn: i32,
    pub maxn: i32,
    pub lr_update_rate: i32,
    pub t: f64,
}

impl Args {
    pub(crate) fn load(c: &mut Cursor<'_>) -> Result<Self> {
        let dim = c.i32()?;
        let ws = c.i32()?;
        let epoch = c.i32()?;
        let min_count = c.i32()?;
        let neg = c.i32()?;
        let word_ngrams = c.i32()?;
        let loss = Loss::from_i32(c.i32()?)?;
        let model = ModelKind::from_i32(c.i32()?)?;
        let bucket = c.i32()?;
        let minn = c.i32()?;
        let maxn = c.i32()?;
        let lr_update_rate = c.i32()?;
        let t = c.f64()?;

        if dim <= 0 {
            return Err(Error::Format(format!(
                "non-positive embedding dimension {dim}"
            )));
        }
        if bucket < 0 {
            return Err(Error::Format(format!("negative bucket count {bucket}")));
        }
        Ok(Self {
            dim: dim as usize,
            ws,
            epoch,
            min_count,
            neg,
            word_ngrams,
            loss,
            model,
            bucket: bucket as usize,
            minn,
            maxn,
            lr_update_rate,
            t,
        })
    }
}
