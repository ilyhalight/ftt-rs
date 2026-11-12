//! fastText dictionary loading and feature extraction.
//!
//! Feature extraction intentionally mirrors `Dictionary::getLine` from
//! fastText, including byte-level tokenization, the signed-byte FNV-1a hash,
//! character n-grams (UTF-8 aware), word n-grams, and pruning. This is what
//! makes predictions label-identical to the reference implementation.

use crate::io::Cursor;
use crate::{Args, Error, Result};

const FNV_OFFSET: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;
const WORD_NGRAM_PRIME: u64 = 116_049_371;
const EOS: &[u8] = b"</s>";
const BOW: u8 = b'<';
const EOW: u8 = b'>';
const LABEL_PREFIX: &[u8] = b"__label__";

pub(crate) const WORD: u8 = 0;
pub(crate) const LABEL: u8 = 1;

/// Exact `hash % d` for all 32-bit hashes without a division.
///
/// Lemire, Kaser and Kurz, "Faster remainder by direct computation" (2019),
/// with `M = floor((2^64 - 1) / d) + 1`.
#[derive(Clone, Copy)]
struct FastMod {
    d: u32,
    m: u64,
}

impl FastMod {
    #[inline]
    fn new(d: u32) -> Self {
        Self {
            d,
            m: if d == 0 { 0 } else { u64::MAX / d as u64 + 1 },
        }
    }

    #[inline]
    fn rem(self, h: u32) -> u32 {
        debug_assert!(self.d != 0);
        // (M * h mod 2^64) * d >> 64
        let low = self.m.wrapping_mul(h as u64) as u128;
        ((low * self.d as u128) >> 64) as u32
    }
}

/// Small open-addressing map from a pruned hash id to its new row id.
struct PruneMap {
    keys: Vec<u32>,
    vals: Vec<u32>,
    mask: usize,
}

impl PruneMap {
    fn build(pairs: &[(i32, i32)]) -> Self {
        let mut cap = 8usize;
        while cap < pairs.len() * 2 {
            cap <<= 1;
        }
        let mut map = Self {
            keys: vec![u32::MAX; cap],
            vals: vec![0; cap],
            mask: cap - 1,
        };
        for &(k, v) in pairs {
            let mut slot = Self::hash(k as u32) & map.mask;
            while map.keys[slot] != u32::MAX {
                if map.keys[slot] == k as u32 {
                    break;
                }
                slot = (slot + 1) & map.mask;
            }
            map.keys[slot] = k as u32;
            map.vals[slot] = v as u32;
        }
        map
    }

    #[inline]
    fn hash(k: u32) -> usize {
        (k.wrapping_mul(2_654_435_761) >> 8) as usize
    }

    #[inline]
    fn get(&self, key: u32) -> Option<u32> {
        let mut slot = Self::hash(key) & self.mask;
        loop {
            let k = self.keys[slot];
            if k == key {
                return Some(self.vals[slot]);
            }
            if k == u32::MAX {
                return None;
            }
            slot = (slot + 1) & self.mask;
        }
    }
}

pub(crate) struct DictEntry {
    pub word: Box<str>,
    pub count: i64,
    pub kind: u8,
}

/// The vocabulary and n-gram hashing rules of a model.
pub struct Dictionary {
    entries: Vec<DictEntry>,
    /// fastText-compatible open-addressing word lookup table.
    table: Vec<i32>,
    nwords: usize,
    nlabels: usize,
    ntokens: i64,
    prune_size: i64,
    prune: Option<PruneMap>,

    // Cached from Args, used on every token.
    bucket: usize,
    fast_mod: FastMod,
    minn: i32,
    maxn: i32,
    word_ngrams: i32,

    pub(crate) eos_id: i32,
}

impl Dictionary {
    pub(crate) fn load(c: &mut Cursor<'_>, args: &Args) -> Result<Self> {
        let size = c.i32()?;
        let nwords = c.i32()?;
        let nlabels = c.i32()?;
        let ntokens = c.i64()?;
        let prune_size = c.i64()?;

        if size < 0 || nwords < 0 || nlabels < 0 {
            return Err(Error::Format("negative dictionary sizes".into()));
        }
        if nwords as i64 + nlabels as i64 != size as i64 {
            return Err(Error::Format(format!(
                "dictionary has {nwords} words + {nlabels} labels but size {size}"
            )));
        }
        if prune_size < -1 {
            return Err(Error::Format(format!(
                "implausible pruned-ngram count {prune_size}"
            )));
        }
        // Every entry needs at least its NUL terminator, an i64 count and an
        // i8 type. Reject impossible counts before allocating for them, so a
        // tiny corrupt file cannot request a terabyte-sized Vec.
        let min_entry_bytes = 1 + 8 + 1;
        if size as usize > c.remaining() / min_entry_bytes {
            return Err(Error::Format(format!(
                "dictionary claims {size} entries but only {} bytes remain",
                c.remaining()
            )));
        }
        if prune_size > 0 && prune_size as u64 > (c.remaining() / 8) as u64 {
            return Err(Error::Format(format!(
                "pruned-ngram table claims {prune_size} entries but only {} bytes remain",
                c.remaining()
            )));
        }

        let mut entries = Vec::with_capacity(size as usize);
        for _ in 0..size {
            let word = c.cstr()?;
            let count = c.i64()?;
            let kind = c.i8()? as u8;
            if kind != WORD && kind != LABEL {
                return Err(Error::Format(format!(
                    "invalid dictionary entry type {kind}"
                )));
            }
            entries.push(DictEntry {
                word: word.into(),
                count,
                kind,
            });
        }

        let prune = if prune_size > 0 {
            // Re-check against the space actually left after the entries, so
            // a corrupt count cannot allocate more than the file can hold.
            if prune_size as u64 > (c.remaining() / 8) as u64 {
                return Err(Error::Format(format!(
                    "pruned-ngram table claims {prune_size} entries but only {} bytes remain",
                    c.remaining()
                )));
            }
            let mut pairs = Vec::with_capacity(prune_size as usize);
            for _ in 0..prune_size {
                pairs.push((c.i32()?, c.i32()?));
            }
            for &(key, value) in &pairs {
                if key < 0 || value < 0 || value as i64 >= prune_size {
                    return Err(Error::Format(format!(
                        "invalid pruning entry {key} -> {value}"
                    )));
                }
            }
            Some(PruneMap::build(&pairs))
        } else {
            None
        };

        // Build the same open-addressing table fastText uses (size / 0.7).
        let table_size = ((size.max(1) as f64) / 0.7).ceil() as usize;
        let mut table = vec![-1i32; table_size.max(1)];
        for (id, entry) in entries.iter().enumerate() {
            let h = fnv1a(entry.word.as_bytes());
            let mut slot = (h as usize) % table.len();
            while table[slot] >= 0 {
                slot = (slot + 1) % table.len();
            }
            table[slot] = id as i32;
        }

        let eos_id = entries
            .iter()
            .position(|e| e.word.as_bytes() == EOS)
            .map(|id| id as i32)
            .unwrap_or(-1);

        Ok(Self {
            entries,
            table,
            nwords: nwords as usize,
            nlabels: nlabels as usize,
            ntokens,
            prune_size,
            prune,
            bucket: args.bucket,
            fast_mod: FastMod::new(args.bucket as u32),
            minn: args.minn,
            maxn: args.maxn,
            word_ngrams: args.word_ngrams,
            eos_id,
        })
    }

    #[inline]
    pub fn nwords(&self) -> usize {
        self.nwords
    }

    #[inline]
    pub fn nlabels(&self) -> usize {
        self.nlabels
    }

    #[inline]
    pub fn ntokens(&self) -> i64 {
        self.ntokens
    }

    /// `-1` when the model has no pruned n-grams, `0` when every hashed n-gram
    /// was pruned, and the size of the remapping table otherwise.
    #[inline]
    pub fn prune_size(&self) -> i64 {
        self.prune_size
    }

    #[inline]
    pub fn bucket(&self) -> usize {
        self.bucket
    }

    /// The raw feature row ids contributed by one in-vocabulary word: the word
    /// row itself plus its character n-grams (already pruned/remapped).
    pub(crate) fn word_feature_ids(&self, wid: usize, out: &mut Vec<i32>, scratch: &mut Vec<u8>) {
        out.clear();
        let token = self.entries[wid].word.as_bytes();
        self.add_subwords(token, wid as i32, out, scratch);
    }

    /// Vocabulary id of `</s>`, or `-1` when it is not in the vocabulary.
    #[inline]
    pub fn eos_id(&self) -> i32 {
        self.eos_id
    }

    #[inline]
    pub fn word(&self, id: usize) -> &str {
        &self.entries[id].word
    }

    #[inline]
    pub fn label(&self, id: usize) -> &str {
        &self.entries[self.nwords + id].word
    }

    #[inline]
    pub fn is_label(&self, id: usize) -> bool {
        self.entries[id].kind == LABEL
    }

    pub fn label_counts(&self) -> Vec<i64> {
        (0..self.nlabels)
            .map(|i| self.entries[self.nwords + i].count)
            .collect()
    }

    pub fn word_counts(&self) -> Vec<i64> {
        (0..self.nwords).map(|i| self.entries[i].count).collect()
    }

    /// Looks a token up in the vocabulary, the same way fastText does.
    #[inline]
    fn lookup(&self, token: &[u8], h: u32) -> i32 {
        let mut slot = (h as usize) % self.table.len();
        loop {
            let id = self.table[slot];
            if id < 0 {
                return -1;
            }
            if self.entries[id as usize].word.as_bytes() == token {
                return id;
            }
            slot = (slot + 1) % self.table.len();
        }
    }

    /// Applies the pruning map and appends the final input-matrix row id.
    #[inline]
    fn push_id(&self, id: u32, out: &mut Vec<i32>) {
        if self.prune_size == 0 {
            return;
        }
        let row = if self.prune_size < 0 {
            id
        } else {
            match self.prune.as_ref().and_then(|m| m.get(id)) {
                Some(mapped) => mapped,
                None => return,
            }
        };
        out.push(self.nwords as i32 + row as i32);
    }

    #[inline]
    fn add_subwords(&self, token: &[u8], wid: i32, out: &mut Vec<i32>, scratch: &mut Vec<u8>) {
        if wid >= 0 {
            out.push(wid);
        }
        if token == EOS || self.maxn <= 0 || self.bucket == 0 {
            return;
        }
        // fastText hashes "<word>" (with the boundary markers) for every
        // character n-gram starting at a UTF-8 code point boundary.
        scratch.clear();
        scratch.reserve(token.len() + 2);
        scratch.push(BOW);
        scratch.extend_from_slice(token);
        scratch.push(EOW);

        let word = scratch.as_slice();
        let len = word.len();
        let maxn = self.maxn as usize;
        let minn = self.minn.max(1) as usize;
        if minn > maxn {
            return;
        }

        for i in 0..len {
            if word[i] & 0xC0 == 0x80 {
                continue;
            }
            let mut h = FNV_OFFSET;
            let mut j = i;
            let mut n = 0usize;
            while j < len && n < maxn {
                loop {
                    h = (h ^ (word[j] as i8 as i32 as u32)).wrapping_mul(FNV_PRIME);
                    j += 1;
                    if j >= len || word[j] & 0xC0 != 0x80 {
                        break;
                    }
                }
                n += 1;
                // A one-code-point n-gram touching a boundary is just the
                // marker itself and is not a feature.
                if n >= minn && !(n == 1 && (i == 0 || j == len)) {
                    self.push_id(self.fast_mod.rem(h), out);
                }
            }
        }
    }

    #[inline]
    fn process_token(
        &self,
        token: &[u8],
        feats: &mut Vec<i32>,
        hashes: &mut Vec<i32>,
        scratch: &mut Vec<u8>,
        cache: Option<&[i32]>,
        words: &mut Vec<u32>,
    ) {
        let h = fnv1a(token);
        let wid = self.lookup(token, h);
        let kind = if wid >= 0 {
            self.entries[wid as usize].kind
        } else if token.starts_with(LABEL_PREFIX) {
            LABEL
        } else {
            WORD
        };
        if kind == WORD {
            if let Some(cache) = cache {
                if wid >= 0 {
                    let slot = cache[wid as usize];
                    if slot >= 0 {
                        words.push(slot as u32);
                        if self.word_ngrams > 1 {
                            hashes.push(h as i32);
                        }
                        return;
                    }
                }
            }
            self.add_subwords(token, wid, feats, scratch);
            if self.word_ngrams > 1 {
                hashes.push(h as i32);
            }
        }
    }

    #[inline]
    fn add_word_ngrams(&self, feats: &mut Vec<i32>, hashes: &[i32]) {
        let n = self.word_ngrams;
        if n <= 1 || self.bucket == 0 || hashes.len() < 2 {
            return;
        }
        let n = n as usize;
        for i in 0..hashes.len() {
            // fastText stores token hashes in a vector<int32_t> and widens
            // them to uint64_t when chaining word n-grams, so hashes with the
            // high bit set are sign-extended here too.
            let mut h = hashes[i] as i64 as u64;
            let end = (i + n).min(hashes.len());
            for &next in &hashes[i + 1..end] {
                h = h
                    .wrapping_mul(WORD_NGRAM_PRIME)
                    .wrapping_add(next as i64 as u64);
                let id = (h % self.bucket as u64) as u32;
                self.push_id(id, feats);
            }
        }
    }

    /// Computes the input-matrix row ids for one line.
    ///
    /// If `append_eos` is true a sentence-end token is appended, matching the
    /// Python API which appends a newline to every input string. A newline
    /// inside `line` terminates processing at that point, as in fastText.
    /// Tokenizes a line in one pass. Cached words are reported as slots in
    /// `words`; OOV subwords, cache misses and word n-grams go to `feats`.
    ///
    /// If `append_eos` is true a sentence-end token is appended, matching the
    /// Python API which appends a newline to every input string. A newline
    /// inside `line` terminates processing at that point, as in fastText.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn features_fused(
        &self,
        line: &[u8],
        append_eos: bool,
        cache: Option<&[i32]>,
        feats: &mut Vec<i32>,
        hashes: &mut Vec<i32>,
        words: &mut Vec<u32>,
        scratch: &mut Vec<u8>,
    ) {
        feats.clear();
        hashes.clear();
        words.clear();

        let bytes = line;
        let mut i = 0usize;
        let mut saw_eos = false;
        while i < bytes.len() {
            let b = bytes[i];
            if is_delimiter(b) {
                if b == b'\n' {
                    self.process_token(EOS, feats, hashes, scratch, cache, words);
                    saw_eos = true;
                    break;
                }
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && !is_delimiter(bytes[i]) {
                i += 1;
            }
            let token = &bytes[start..i];
            self.process_token(token, feats, hashes, scratch, cache, words);
            if token == EOS {
                saw_eos = true;
                break;
            }
        }
        if append_eos && !saw_eos {
            self.process_token(EOS, feats, hashes, scratch, cache, words);
        }
        self.add_word_ngrams(feats, hashes);
    }
}

#[inline]
fn is_delimiter(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t' | 0x0b | 0x0c | 0)
}

/// The signed-byte FNV-1a hash fastText has used since its first release.
#[inline]
pub(crate) fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h = (h ^ (b as i8 as i32 as u32)).wrapping_mul(FNV_PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_mod_matches_division() {
        let mut h = 0x9e37_79b9u32;
        for &d in &[1u32, 2, 3, 7, 777, 2_000_000, 4_294_967_295] {
            let f = FastMod::new(d);
            for _ in 0..10_000 {
                h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                assert_eq!(f.rem(h), h % d, "d={d} h={h}");
            }
        }
    }

    #[test]
    fn tokenizer_delimiters_and_eos() {
        assert!(is_delimiter(b' '));
        assert!(is_delimiter(b'\n'));
        assert!(is_delimiter(b'\r'));
        assert!(is_delimiter(b'\t'));
        assert!(is_delimiter(0));
        assert!(!is_delimiter(b'a'));
        // FNV-1a is signed-byte FNV, not the unsigned variant.
        assert_eq!(fnv1a(b""), FNV_OFFSET);
        assert_eq!(
            fnv1a(b"\xff"),
            (FNV_OFFSET ^ (-1i32 as u32)).wrapping_mul(FNV_PRIME)
        );
    }
}
