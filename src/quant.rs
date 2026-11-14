//! Product-quantized matrices (fastText `.ftz` models).
//!
//! The on-disk layout and the arithmetic order follow fastText's
//! `ProductQuantizer` exactly, so quantized predictions agree with the
//! reference implementation.

use crate::io::{f32_at, Cursor};
use crate::{Error, Result};

const KSUB: usize = 256;

#[derive(Clone, Debug)]
pub(crate) struct ProductQuantizer {
    pub dim: usize,
    pub nsubq: usize,
    pub dsub: usize,
    pub lastdsub: usize,
    /// Byte offset of the centroid table in the model mapping.
    pub centroids_offset: usize,
}

impl ProductQuantizer {
    pub(crate) fn load(c: &mut Cursor<'_>) -> Result<Self> {
        let dim = c.i32()?;
        let nsubq = c.i32()?;
        let dsub = c.i32()?;
        let lastdsub = c.i32()?;
        if dim <= 0 || nsubq <= 0 || dsub <= 0 || lastdsub <= 0 {
            return Err(Error::Format("invalid product-quantizer dimensions".into()));
        }
        let dim = dim as usize;
        let nsubq = nsubq as usize;
        let dsub = dsub as usize;
        let lastdsub = lastdsub as usize;
        if nsubq.checked_mul(dsub).is_none() || (nsubq - 1) * dsub + lastdsub != dim {
            return Err(Error::Format(format!(
                "product quantizer: {nsubq} x {dsub} + {lastdsub} != {dim}"
            )));
        }
        let centroids_offset = c.pos();
        c.skip(dim * KSUB * 4)?;
        Ok(Self {
            dim,
            nsubq,
            dsub,
            lastdsub,
            centroids_offset,
        })
    }

    /// Byte offset of centroid `code` in subquantizer `m`.
    #[inline]
    fn centroid_off(&self, m: usize, code: u8) -> usize {
        let idx = if m + 1 == self.nsubq {
            m * KSUB * self.dsub + code as usize * self.lastdsub
        } else {
            (m * KSUB + code as usize) * self.dsub
        };
        self.centroids_offset + idx * 4
    }

    #[inline]
    pub(crate) fn centroid1(&self, data: &[u8], code: u8) -> f32 {
        // Used for qnorm: a one-dimensional PQ with nsubq = 1, dsub = 1.
        unsafe { f32_at(data, self.centroid_off(0, code)) }
    }

    /// `dot(x, decoded_row)` in the same order as fastText's `mulcode`.
    pub(crate) fn dot_code(&self, data: &[u8], x: &[f32], codes_off: usize, alpha: f32) -> f32 {
        let full = self.nsubq - 1;
        let mut res = 0.0f32;
        let mut m = 0usize;
        // dsub == 2 is what fastText writes for most models; it gets an
        // unrolled body instead of the inner loop.
        if self.dsub == 2 {
            while m < full {
                let c = self.centroid_off(m, data[codes_off + m]);
                res += x[2 * m] * unsafe { f32_at(data, c) };
                res += x[2 * m + 1] * unsafe { f32_at(data, c + 4) };
                m += 1;
            }
        } else {
            while m < full {
                let c = self.centroid_off(m, data[codes_off + m]);
                for n in 0..self.dsub {
                    res += x[m * self.dsub + n] * unsafe { f32_at(data, c + 4 * n) };
                }
                m += 1;
            }
        }
        let c = self.centroid_off(full, data[codes_off + full]);
        for n in 0..self.lastdsub {
            res += x[full * self.dsub + n] * unsafe { f32_at(data, c + 4 * n) };
        }
        res * alpha
    }

    /// `x += alpha * decoded_row`, the same additions as fastText's `addcode`.
    pub(crate) fn add_code(&self, data: &[u8], x: &mut [f32], codes_off: usize, alpha: f32) {
        let full = self.nsubq - 1;
        let mut m = 0usize;
        if self.dsub == 2 {
            while m < full {
                let c = self.centroid_off(m, data[codes_off + m]);
                x[2 * m] += alpha * unsafe { f32_at(data, c) };
                x[2 * m + 1] += alpha * unsafe { f32_at(data, c + 4) };
                m += 1;
            }
        } else {
            while m < full {
                let c = self.centroid_off(m, data[codes_off + m]);
                for n in 0..self.dsub {
                    x[m * self.dsub + n] += alpha * unsafe { f32_at(data, c + 4 * n) };
                }
                m += 1;
            }
        }
        let c = self.centroid_off(full, data[codes_off + full]);
        for n in 0..self.lastdsub {
            x[full * self.dsub + n] += alpha * unsafe { f32_at(data, c + 4 * n) };
        }
    }
}
