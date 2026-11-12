//! Dense and product-quantized matrix views over the mapped model file.

use crate::io::{f32_at, Cursor};
use crate::quant::ProductQuantizer;
use crate::{simd, Error, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseView {
    pub rows: usize,
    pub cols: usize,
    pub data_offset: usize,
}

#[derive(Debug)]
pub(crate) struct QuantView {
    pub rows: usize,
    pub cols: usize,
    pub qnorm: bool,
    pub codes_offset: usize,
    pub pq: ProductQuantizer,
    pub norm_codes_offset: usize,
    pub npq: Option<ProductQuantizer>,
}

#[derive(Debug)]
pub(crate) enum MatrixView {
    Dense(DenseView),
    Quant(QuantView),
}

impl MatrixView {
    /// Loads one matrix. `quantized` comes from the bool that fastText writes
    /// immediately before each matrix.
    pub(crate) fn load(c: &mut Cursor<'_>, quantized: bool, dim: usize) -> Result<Self> {
        if quantized {
            let qnorm = c.bool()?;
            let rows = checked_dim(c.i64()?, "matrix rows")?;
            let cols = checked_dim(c.i64()?, "matrix columns")?;
            // QuantMatrix stores codesize_ as int32_t.
            let codes_len = checked_dim(c.i32()? as i64, "quantized code length")?;
            if cols != dim {
                return Err(Error::Format(format!(
                    "matrix has {cols} columns, model dimension is {dim}"
                )));
            }
            let codes_offset = c.pos();
            c.skip(codes_len)?;
            let pq = ProductQuantizer::load(c)?;
            if pq.dim != cols || codes_len < rows.saturating_mul(pq.nsubq) {
                return Err(Error::Format(format!(
                    "quantized matrix shape mismatch (rows {rows}, codes {codes_len}, \
                     nsubq {})",
                    pq.nsubq
                )));
            }
            let mut norm_codes_offset = 0usize;
            let mut npq = None;
            if qnorm {
                norm_codes_offset = c.pos();
                c.skip(rows)?;
                let quantizer = ProductQuantizer::load(c)?;
                if quantizer.dim != 1 || quantizer.nsubq != 1 || quantizer.lastdsub != 1 {
                    return Err(Error::Format(
                        "qnorm quantizer must be one-dimensional".into(),
                    ));
                }
                npq = Some(quantizer);
            }
            Ok(MatrixView::Quant(QuantView {
                rows,
                cols,
                qnorm,
                codes_offset,
                pq,
                norm_codes_offset,
                npq,
            }))
        } else {
            let rows = checked_dim(c.i64()?, "matrix rows")?;
            let cols = checked_dim(c.i64()?, "matrix columns")?;
            if cols != dim {
                return Err(Error::Format(format!(
                    "matrix has {cols} columns, model dimension is {dim}"
                )));
            }
            let data_offset = c.pos();
            let bytes = rows
                .checked_mul(cols)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| Error::Format("matrix size overflow".into()))?;
            c.skip(bytes)?;
            Ok(MatrixView::Dense(DenseView {
                rows,
                cols,
                data_offset,
            }))
        }
    }

    #[inline]
    pub(crate) fn rows(&self) -> usize {
        match self {
            MatrixView::Dense(m) => m.rows,
            MatrixView::Quant(m) => m.rows,
        }
    }

    #[inline]
    pub(crate) fn cols(&self) -> usize {
        match self {
            MatrixView::Dense(m) => m.cols,
            MatrixView::Quant(m) => m.cols,
        }
    }

    #[inline]
    pub(crate) fn is_quantized(&self) -> bool {
        matches!(self, MatrixView::Quant(_))
    }

    pub(crate) fn average_rows(&self, data: &[u8], rows: &[i32], out: &mut [f32]) {
        if rows.is_empty() {
            out.fill(0.0);
            return;
        }
        match self {
            MatrixView::Dense(m) => {
                let base = m.data_offset;
                // A sub-slice keeps the dense kernel's offsets relative to the
                // matrix start.
                simd::average_rows(rows, m.cols, &data[base..], out);
            }
            MatrixView::Quant(m) => {
                out.fill(0.0);
                for &r in rows {
                    let alpha = if m.qnorm {
                        let npq = m.npq.as_ref().expect("qnorm without norm quantizer");
                        let code = data[m.norm_codes_offset + r as usize];
                        npq.centroid1(data, code)
                    } else {
                        1.0
                    };
                    let off = m.codes_offset + r as usize * m.pq.nsubq;
                    m.pq.add_code(data, out, off, alpha);
                }
                let inv = 1.0 / rows.len() as f32;
                for v in out {
                    *v *= inv;
                }
            }
        }
    }

    pub(crate) fn score_line(&self, data: &[u8], hidden: &[f32], out: &mut [f32]) {
        match self {
            MatrixView::Dense(m) => {
                let base = m.data_offset;
                simd::score_line(hidden, &data[base..], m.rows, m.cols, out);
            }
            MatrixView::Quant(m) => {
                for (i, o) in out.iter_mut().enumerate().take(m.rows) {
                    let alpha = self.quant_alpha(m, data, i);
                    let off = m.codes_offset + i * m.pq.nsubq;
                    *o = m.pq.dot_code(data, hidden, off, alpha);
                }
            }
        }
    }

    /// Scores `rows` labels starting at `start` against one hidden vector.
    /// Used by the norm-bound softmax path, which visits label blocks in a
    /// norm-sorted order and skips blocks that cannot matter.
    pub(crate) fn score_block(
        &self,
        data: &[u8],
        hidden: &[f32],
        start: usize,
        rows: usize,
        out: &mut [f32],
    ) {
        assert_eq!(out.len(), rows);
        match self {
            MatrixView::Dense(m) => {
                let base = m.data_offset + start * m.cols * 4;
                simd::score_line(hidden, &data[base..], rows, m.cols, out);
            }
            MatrixView::Quant(m) => {
                for (r, o) in out.iter_mut().enumerate() {
                    let row = start + r;
                    let alpha = self.quant_alpha(m, data, row);
                    let off = m.codes_offset + row * m.pq.nsubq;
                    *o = m.pq.dot_code(data, hidden, off, alpha);
                }
            }
        }
    }

    /// L2 norm of every row, used to bound dot products during prediction.
    pub(crate) fn row_norms(&self, data: &[u8], dim: usize) -> Vec<f32> {
        match self {
            MatrixView::Dense(m) => {
                let mut norms = Vec::with_capacity(m.rows);
                for row in 0..m.rows {
                    let off = m.data_offset + row * m.cols * 4;
                    let mut sum = 0.0f32;
                    for j in 0..dim {
                        let v = unsafe { f32_at(data, off + j * 4) };
                        sum += v * v;
                    }
                    norms.push(sum.sqrt());
                }
                norms
            }
            MatrixView::Quant(m) => {
                let mut row = vec![0.0f32; dim];
                let mut norms = Vec::with_capacity(m.rows);
                for i in 0..m.rows {
                    row.fill(0.0);
                    let alpha = self.quant_alpha(m, data, i);
                    let off = m.codes_offset + i * m.pq.nsubq;
                    m.pq.add_code(data, &mut row, off, alpha);
                    norms.push(row.iter().map(|v| v * v).sum::<f32>().sqrt());
                }
                norms
            }
        }
    }

    pub(crate) fn score_batch(&self, data: &[u8], hidden: &[f32], batch: usize, out: &mut [f32]) {
        match self {
            MatrixView::Dense(m) => {
                let base = m.data_offset;
                simd::score_batch(hidden, batch, &data[base..], m.rows, m.cols, out);
            }
            MatrixView::Quant(m) => {
                for b in 0..batch {
                    let h = &hidden[b * m.cols..(b + 1) * m.cols];
                    let line = &mut out[b * m.rows..(b + 1) * m.rows];
                    for (i, o) in line.iter_mut().enumerate() {
                        let alpha = self.quant_alpha(m, data, i);
                        let off = m.codes_offset + i * m.pq.nsubq;
                        *o = m.pq.dot_code(data, h, off, alpha);
                    }
                }
            }
        }
    }

    #[inline]
    fn quant_alpha(&self, m: &QuantView, data: &[u8], row: usize) -> f32 {
        if m.qnorm {
            let code = data[m.norm_codes_offset + row];
            m.npq
                .as_ref()
                .expect("qnorm without norm quantizer")
                .centroid1(data, code)
        } else {
            1.0
        }
    }

    /// Dot product of one row, used by tests and word-vector helpers.
    pub(crate) fn dot_row(&self, data: &[u8], row: usize, x: &[f32]) -> f32 {
        match self {
            MatrixView::Dense(m) => {
                let off = m.data_offset + row * m.cols * 4;
                simd::dot_scalar(data, off, x)
            }
            MatrixView::Quant(m) => {
                let alpha = self.quant_alpha(m, data, row);
                let off = m.codes_offset + row * m.pq.nsubq;
                m.pq.dot_code(data, x, off, alpha)
            }
        }
    }
}

fn checked_dim(v: i64, what: &str) -> Result<usize> {
    if v < 0 {
        return Err(Error::Format(format!("negative {what}: {v}")));
    }
    usize::try_from(v).map_err(|_| Error::Format(format!("{what} does not fit in usize")))
}
