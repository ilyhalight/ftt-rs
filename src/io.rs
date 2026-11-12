//! Little-endian cursor over a memory-mapped model file.

use crate::{Error, Result};

pub(crate) struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    #[inline]
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    #[inline]
    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(Error::Format(format!(
                "unexpected end of file at byte {} (need {n} more bytes)",
                self.pos
            )));
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    #[inline]
    pub(crate) fn skip(&mut self, n: usize) -> Result<()> {
        self.need(n)?;
        self.pos += n;
        Ok(())
    }

    #[inline]
    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    #[inline]
    pub(crate) fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    #[inline]
    pub(crate) fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    #[inline]
    pub(crate) fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    pub(crate) fn i64(&mut self) -> Result<i64> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    #[inline]
    pub(crate) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.i64()? as u64))
    }

    /// Reads a NUL-terminated string (fastText's `saveString`).
    pub(crate) fn cstr(&mut self) -> Result<&'a str> {
        let start = self.pos;
        let relative = self
            .buf
            .get(start..)
            .and_then(|rest| rest.iter().position(|&b| b == 0));
        let Some(len) = relative else {
            return Err(Error::Format(format!(
                "unterminated string at byte {start}"
            )));
        };
        let bytes = &self.buf[start..start + len];
        self.pos = start + len + 1;
        std::str::from_utf8(bytes)
            .map_err(|_| Error::Format(format!("invalid UTF-8 string at byte {start}")))
    }
}

/// Reads an unaligned little-endian `f32` from a model buffer.
///
/// # Safety
///
/// `off + 4` must be within `data`. Callers in hot loops ensure this with the
/// matrix bounds established at load time.
#[inline]
pub(crate) unsafe fn f32_at(data: &[u8], off: usize) -> f32 {
    debug_assert!(off + 4 <= data.len());
    unsafe { (data.as_ptr().add(off) as *const f32).read_unaligned() }
}
