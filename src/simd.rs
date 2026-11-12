//! Dense matrix kernels.
//!
//! All entry points are safe and use runtime CPU dispatch. The hot x86 path
//! uses AVX2 + FMA: eight independent accumulators for eight label rows, and
//! eight lines at a time in the batch kernel so that each output row is
//! streamed from memory once per batch instead of once per line.

use crate::io::f32_at;

/// Average the selected rows of a dense matrix into `out`.
pub(crate) fn average_rows(rows: &[i32], dim: usize, data: &[u8], out: &mut [f32]) {
    assert_eq!(out.len(), dim);
    debug_assert!(rows
        .iter()
        .all(|&r| r >= 0 && (r as usize) < data.len() / (dim * 4).max(1)));
    if rows.is_empty() {
        out.fill(0.0);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                average_rows_avx2(rows, dim, data, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                average_rows_neon(rows, dim, data, out);
            }
            return;
        }
    }
    average_rows_scalar(rows, dim, data, out);
}

/// Scores all rows of a dense matrix against one hidden vector.
pub(crate) fn score_line(hidden: &[f32], data: &[u8], rows: usize, dim: usize, out: &mut [f32]) {
    assert_eq!(hidden.len(), dim);
    assert_eq!(out.len(), rows);
    debug_assert!(rows * dim * 4 <= data.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                score_line_avx2(hidden, data, rows, dim, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                score_line_neon(hidden, data, rows, dim, out);
            }
            return;
        }
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot_scalar(data, i * dim * 4, hidden);
    }
}

/// Scores `batch` hidden vectors against all rows, writing `out` row-major
/// (`out[line * rows + label]`).
pub(crate) fn score_batch(
    hidden: &[f32],
    batch: usize,
    data: &[u8],
    rows: usize,
    dim: usize,
    out: &mut [f32],
) {
    assert_eq!(hidden.len(), batch * dim);
    assert_eq!(out.len(), batch * rows);
    debug_assert!(rows * dim * 4 <= data.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                score_batch_avx2(hidden, batch, data, rows, dim, out);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                score_batch_neon(hidden, batch, data, rows, dim, out);
            }
            return;
        }
    }
    for b in 0..batch {
        let h = &hidden[b * dim..(b + 1) * dim];
        for i in 0..rows {
            out[b * rows + i] = dot_scalar(data, i * dim * 4, h);
        }
    }
}

#[inline]
pub(crate) fn dot_scalar(data: &[u8], off: usize, x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (j, &xj) in x.iter().enumerate() {
        sum += unsafe { f32_at(data, off + j * 4) } * xj;
    }
    sum
}

#[inline]
fn average_rows_scalar(rows: &[i32], dim: usize, data: &[u8], out: &mut [f32]) {
    out.fill(0.0);
    for &r in rows {
        let off = r as usize * dim * 4;
        for (j, o) in out.iter_mut().enumerate() {
            *o += unsafe { f32_at(data, off + j * 4) };
        }
    }
    let inv = 1.0 / rows.len() as f32;
    for o in out {
        *o *= inv;
    }
}

// ---------------------------------------------------------------------------
// x86 AVX2 + FMA
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let mut lanes = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
        lanes.iter().sum()
    }

    #[inline]
    unsafe fn dot_avx2(data: &[u8], off: usize, x: &[f32]) -> f32 {
        let dim = x.len();
        let nv = dim & !7;
        let mut acc = unsafe { _mm256_setzero_ps() };
        let mut j = 0usize;
        while j < nv {
            let w = unsafe { _mm256_loadu_ps(data.as_ptr().add(off + j * 4) as *const f32) };
            let h = unsafe { _mm256_loadu_ps(x.as_ptr().add(j)) };
            acc = unsafe { _mm256_fmadd_ps(w, h, acc) };
            j += 8;
        }
        let mut sum = unsafe { hsum256(acc) };
        while j < dim {
            sum += unsafe { f32_at(data, off + j * 4) } * x[j];
            j += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn average_rows_avx2(rows: &[i32], dim: usize, data: &[u8], out: &mut [f32]) {
        out.fill(0.0);
        let nv = dim & !7;
        let out_ptr = out.as_mut_ptr();
        let base_ptr = data.as_ptr();
        for (ri, &row) in rows.iter().enumerate() {
            // The rows are scattered over the embedding table, so hardware
            // prefetchers cannot follow them. Prefetch several rows ahead.
            if let Some(&next) = rows.get(ri + 32) {
                let p = base_ptr.add(next as usize * dim * 4);
                _mm_prefetch::<_MM_HINT_T0>(p as *const i8);
            }
            let off = row as usize * dim * 4;
            let mut j = 0usize;
            while j < nv {
                let v = unsafe { _mm256_loadu_ps(base_ptr.add(off + j * 4) as *const f32) };
                let a = unsafe { _mm256_loadu_ps(out_ptr.add(j)) };
                unsafe { _mm256_storeu_ps(out_ptr.add(j), _mm256_add_ps(a, v)) };
                j += 8;
            }
            while j < dim {
                unsafe { *out_ptr.add(j) += f32_at(data, off + j * 4) };
                j += 1;
            }
        }
        let inv = 1.0 / rows.len() as f32;
        let invv = _mm256_set1_ps(inv);
        let mut j = 0usize;
        while j < nv {
            let a = unsafe { _mm256_loadu_ps(out_ptr.add(j)) };
            unsafe { _mm256_storeu_ps(out_ptr.add(j), _mm256_mul_ps(a, invv)) };
            j += 8;
        }
        while j < dim {
            unsafe { *out_ptr.add(j) *= inv };
            j += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn score_line_avx2(
        hidden: &[f32],
        data: &[u8],
        rows: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let nv = dim & !7;
        let mut i = 0usize;
        while i + 8 <= rows {
            let mut acc = [_mm256_setzero_ps(); 8];
            let mut j = 0usize;
            while j < nv {
                let h = unsafe { _mm256_loadu_ps(hidden.as_ptr().add(j)) };
                for (r, a) in acc.iter_mut().enumerate() {
                    let off = (i + r) * dim * 4 + j * 4;
                    let w = unsafe { _mm256_loadu_ps(data.as_ptr().add(off) as *const f32) };
                    *a = _mm256_fmadd_ps(w, h, *a);
                }
                j += 8;
            }
            for (r, a) in acc.iter().enumerate() {
                let mut sum = unsafe { hsum256(*a) };
                let off = (i + r) * dim * 4;
                let mut jj = nv;
                while jj < dim {
                    sum += unsafe { f32_at(data, off + jj * 4) } * hidden[jj];
                    jj += 1;
                }
                unsafe { *out.get_unchecked_mut(i + r) = sum };
            }
            i += 8;
        }
        while i < rows {
            let v = unsafe { dot_avx2(data, i * dim * 4, hidden) };
            unsafe { *out.get_unchecked_mut(i) = v };
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn score_batch_avx2(
        hidden: &[f32],
        batch: usize,
        data: &[u8],
        rows: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let nv = dim & !7;
        let full = batch & !7;
        let base = data.as_ptr();
        for label in 0..rows {
            let row_off = label * dim * 4;
            let mut b = 0usize;
            while b < full {
                let mut acc = [_mm256_setzero_ps(); 8];
                let mut j = 0usize;
                while j < nv {
                    let w = unsafe { _mm256_loadu_ps(base.add(row_off + j * 4) as *const f32) };
                    for (r, a) in acc.iter_mut().enumerate() {
                        let h = unsafe { _mm256_loadu_ps(hidden.as_ptr().add((b + r) * dim + j)) };
                        *a = _mm256_fmadd_ps(w, h, *a);
                    }
                    j += 8;
                }
                for (r, a) in acc.iter().enumerate() {
                    let mut sum = unsafe { hsum256(*a) };
                    let mut jj = nv;
                    while jj < dim {
                        sum +=
                            unsafe { f32_at(data, row_off + jj * 4) } * hidden[(b + r) * dim + jj];
                        jj += 1;
                    }
                    unsafe { *out.get_unchecked_mut((b + r) * rows + label) = sum };
                }
                b += 8;
            }
        }
        // Fewer than eight lines left: score them one at a time.
        for b in full..batch {
            let h = &hidden[b * dim..(b + 1) * dim];
            unsafe {
                score_line_avx2(h, data, rows, dim, &mut out[b * rows..(b + 1) * rows]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
use x86::*;

// ---------------------------------------------------------------------------
// aarch64 NEON
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use arm::*;

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn hsum128(v: float32x4_t) -> f32 {
        unsafe { vaddvq_f32(v) }
    }

    #[inline]
    unsafe fn dot_neon(data: &[u8], off: usize, x: &[f32]) -> f32 {
        let dim = x.len();
        let nv = dim & !3;
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut j = 0usize;
        while j < nv {
            let w = unsafe { vld1q_f32(data.as_ptr().add(off + j * 4) as *const f32) };
            let h = unsafe { vld1q_f32(x.as_ptr().add(j)) };
            acc = unsafe { vfmaq_f32(acc, w, h) };
            j += 4;
        }
        let mut sum = unsafe { hsum128(acc) };
        while j < dim {
            sum += unsafe { f32_at(data, off + j * 4) } * x[j];
            j += 1;
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn average_rows_neon(rows: &[i32], dim: usize, data: &[u8], out: &mut [f32]) {
        out.fill(0.0);
        let nv = dim & !3;
        let out_ptr = out.as_mut_ptr();
        let base = data.as_ptr();
        for &row in rows {
            let off = row as usize * dim * 4;
            let mut j = 0usize;
            while j < nv {
                let v = unsafe { vld1q_f32(base.add(off + j * 4) as *const f32) };
                let a = unsafe { vld1q_f32(out_ptr.add(j)) };
                unsafe { vst1q_f32(out_ptr.add(j), vaddq_f32(a, v)) };
                j += 4;
            }
            while j < dim {
                unsafe { *out_ptr.add(j) += f32_at(data, off + j * 4) };
                j += 1;
            }
        }
        let inv = 1.0 / rows.len() as f32;
        let invv = vdupq_n_f32(inv);
        let mut j = 0usize;
        while j < nv {
            let a = unsafe { vld1q_f32(out_ptr.add(j)) };
            unsafe { vst1q_f32(out_ptr.add(j), vmulq_f32(a, invv)) };
            j += 4;
        }
        while j < dim {
            unsafe { *out_ptr.add(j) *= inv };
            j += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn score_line_neon(
        hidden: &[f32],
        data: &[u8],
        rows: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let nv = dim & !3;
        let mut i = 0usize;
        while i + 8 <= rows {
            let mut acc = [vdupq_n_f32(0.0); 8];
            let mut j = 0usize;
            while j < nv {
                let h = unsafe { vld1q_f32(hidden.as_ptr().add(j)) };
                for (r, a) in acc.iter_mut().enumerate() {
                    let off = (i + r) * dim * 4 + j * 4;
                    let w = unsafe { vld1q_f32(data.as_ptr().add(off) as *const f32) };
                    *a = vfmaq_f32(*a, w, h);
                }
                j += 4;
            }
            for (r, a) in acc.iter().enumerate() {
                let mut sum = unsafe { hsum128(*a) };
                let off = (i + r) * dim * 4;
                let mut jj = nv;
                while jj < dim {
                    sum += unsafe { f32_at(data, off + jj * 4) } * hidden[jj];
                    jj += 1;
                }
                unsafe { *out.get_unchecked_mut(i + r) = sum };
            }
            i += 8;
        }
        while i < rows {
            let v = unsafe { dot_neon(data, i * dim * 4, hidden) };
            unsafe { *out.get_unchecked_mut(i) = v };
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn score_batch_neon(
        hidden: &[f32],
        batch: usize,
        data: &[u8],
        rows: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let nv = dim & !3;
        let full = batch & !7;
        let base = data.as_ptr();
        for label in 0..rows {
            let row_off = label * dim * 4;
            let mut b = 0usize;
            while b < full {
                let mut acc = [vdupq_n_f32(0.0); 8];
                let mut j = 0usize;
                while j < nv {
                    let w = unsafe { vld1q_f32(base.add(row_off + j * 4) as *const f32) };
                    for (r, a) in acc.iter_mut().enumerate() {
                        let h = unsafe { vld1q_f32(hidden.as_ptr().add((b + r) * dim + j)) };
                        *a = vfmaq_f32(*a, w, h);
                    }
                    j += 4;
                }
                for (r, a) in acc.iter().enumerate() {
                    let mut sum = unsafe { hsum128(*a) };
                    let mut jj = nv;
                    while jj < dim {
                        sum +=
                            unsafe { f32_at(data, row_off + jj * 4) } * hidden[(b + r) * dim + jj];
                        jj += 1;
                    }
                    unsafe { *out.get_unchecked_mut((b + r) * rows + label) = sum };
                }
                b += 8;
            }
        }
        for b in full..batch {
            let h = &hidden[b * dim..(b + 1) * dim];
            unsafe {
                score_line_neon(h, data, rows, dim, &mut out[b * rows..(b + 1) * rows]);
            }
        }
    }
}

/// Sum of `exp(x - max())` over all entries, skipping terms below `-20`
/// (their relative contribution is under single precision) and using a
/// vectorized exp approximation. Only the softmax denominator uses this;
/// selected labels are re-scored with the standard library exp.
pub(crate) fn exp_sum(scores: &[f32], max: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // The polynomial uses FMA, so AVX2 alone is not enough. Calling the
        // function without FMA would be undefined behaviour.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { exp_sum_avx2(scores, max) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { exp_sum_neon(scores, max) };
        }
    }
    let mut sum = 0.0f32;
    for &s in scores {
        let d = s - max;
        if d > -20.0 {
            sum += d.exp();
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
mod exp_x86 {
    use std::arch::x86_64::*;

    /// exp(x) for x in roughly [-20, 0], relative error about 1e-7.
    #[inline]
    unsafe fn exp256(x: __m256) -> __m256 {
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        // k = round(x / ln 2)
        let k = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT);
        // r = x - k * ln 2. The f32 LN_2 residual is below 2e-9 per unit of
        // k, so a split high/low constant is unnecessary here.
        let r = _mm256_sub_ps(x, _mm256_mul_ps(k, ln2));
        // exp(r) by Horner's method, degree 7 on |r| <= ln(2)/2.
        let mut p = _mm256_set1_ps(1.0 / 5040.0);
        for c in [
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            0.5,
            1.0,
            1.0,
        ] {
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(c));
        }
        // 2^k
        let ki = _mm256_cvtps_epi32(k);
        let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(ki, _mm256_set1_epi32(127)),
            23,
        ));
        _mm256_mul_ps(p, pow2)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_sum_avx2(scores: &[f32], max: f32) -> f32 {
        let n = scores.len();
        let nv = n & !7;
        let vmax = _mm256_set1_ps(max);
        let cutoff = _mm256_set1_ps(-20.0);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i < nv {
            let s = _mm256_loadu_ps(scores.as_ptr().add(i));
            let d = _mm256_sub_ps(s, vmax);
            let e = exp256(d);
            let mask = _mm256_cmp_ps(d, cutoff, _CMP_GT_OQ);
            acc = _mm256_add_ps(acc, _mm256_and_ps(e, mask));
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum: f32 = lanes.iter().sum();
        while i < n {
            let d = scores[i] - max;
            if d > -20.0 {
                sum += d.exp();
            }
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
use exp_x86::*;

#[cfg(target_arch = "aarch64")]
mod exp_arm {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn exp128(x: float32x4_t) -> float32x4_t {
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let k = vrndnq_f32(vmulq_f32(x, log2e));
        let r = vsubq_f32(x, vmulq_f32(k, ln2));
        let mut p = vdupq_n_f32(1.0 / 5040.0);
        for c in [
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            0.5,
            1.0,
            1.0,
        ] {
            p = vfmaq_f32(vdupq_n_f32(c), p, r);
        }
        let ki = vcvtq_s32_f32(k);
        let pow2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ki, vdupq_n_s32(127)), 23));
        vmulq_f32(p, pow2)
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn exp_sum_neon(scores: &[f32], max: f32) -> f32 {
        let n = scores.len();
        let nv = n & !3;
        let vmax = vdupq_n_f32(max);
        let cutoff = vdupq_n_f32(-20.0);
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i < nv {
            let s = vld1q_f32(scores.as_ptr().add(i));
            let d = vsubq_f32(s, vmax);
            let e = exp128(d);
            let mask = vcgtq_f32(d, cutoff);
            acc = vaddq_f32(
                acc,
                vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(e), mask)),
            );
            i += 4;
        }
        let mut sum = vaddvq_f32(acc);
        while i < n {
            let d = scores[i] - max;
            if d > -20.0 {
                sum += d.exp();
            }
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
use exp_arm::*;

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn bytes_of(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    #[test]
    fn dense_kernels_match_scalar_reference() {
        for &dim in &[1usize, 3, 7, 8, 15, 16, 31, 96, 100, 256, 512] {
            for &rows in &[1usize, 3, 7, 8, 9, 17, 64] {
                let mut rng = Lcg(0x1234_5678_9abc_def0 ^ (dim as u64) << 8 ^ rows as u64);
                let data: Vec<f32> = (0..rows * dim).map(|_| rng.next()).collect();
                let bytes = bytes_of(&data);
                let hidden: Vec<f32> = (0..dim).map(|_| rng.next()).collect();

                let mut got = vec![0.0f32; rows];
                score_line(&hidden, &bytes, rows, dim, &mut got);
                for i in 0..rows {
                    let mut want = 0.0f32;
                    for j in 0..dim {
                        want += data[i * dim + j] * hidden[j];
                    }
                    let tol = 2e-5 * want.abs().max(1.0);
                    assert!(
                        (got[i] - want).abs() <= tol,
                        "score_line dim={dim} rows={rows} row={i}: {} vs {want}",
                        got[i]
                    );
                }

                let nrows = 9.min(rows);
                let ids: Vec<i32> = (0..nrows)
                    .map(|_| ((rng.next() + 1.0) * 0.5 * rows as f32) as i32 % rows as i32)
                    .collect();
                let mut avg = vec![0.0f32; dim];
                average_rows(&ids, dim, &bytes, &mut avg);
                let mut want = vec![0.0f32; dim];
                for &id in &ids {
                    for j in 0..dim {
                        want[j] += data[id as usize * dim + j];
                    }
                }
                let inv = 1.0 / ids.len() as f32;
                for j in 0..dim {
                    want[j] *= inv;
                    assert!(
                        (avg[j] - want[j]).abs() <= 2e-5,
                        "average dim={dim} rows={rows} j={j}: {} vs {}",
                        avg[j],
                        want[j]
                    );
                }

                let batch = 13usize;
                let hidden_multi: Vec<f32> = (0..batch * dim).map(|_| rng.next()).collect();
                let mut scores = vec![0.0f32; batch * rows];
                score_batch(&hidden_multi, batch, &bytes, rows, dim, &mut scores);
                for b in 0..batch {
                    let mut line = vec![0.0f32; rows];
                    score_line(
                        &hidden_multi[b * dim..(b + 1) * dim],
                        &bytes,
                        rows,
                        dim,
                        &mut line,
                    );
                    for i in 0..rows {
                        let tol = 2e-5 * line[i].abs().max(1.0);
                        assert!(
                            (scores[b * rows + i] - line[i]).abs() <= tol,
                            "batch dim={dim} rows={rows} b={b} i={i}"
                        );
                    }
                }
            }
        }
    }
}
