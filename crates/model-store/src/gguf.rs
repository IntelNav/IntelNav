//! Minimal, zero-copy GGUF v3 parser.
//!
//! Scope: parse the header, KV metadata, and tensor index of a GGUF
//! file without loading any tensor data. We only ever *read* GGUF
//! files here; writing/editing is out of scope (libllama already
//! does that via its own tooling).
//!
//! Why hand-roll instead of pulling a crate:
//!
//! * The format is short (~300 lines of parsing code) and stable
//!   (v3 since 2024).
//! * We want byte-exact control over KV-range identification so we
//!   can split KV metadata into hparams / tokenizer / other groups
//!   without re-reading the file. Existing crates either load
//!   everything into a dynamic tree or expose only typed accessors.
//! * Phase 2's loader needs a parser of its own; one implementation
//!   shared across phases is simpler than two.
//!
//! Spec reference: `llama.cpp/ggml/include/gguf.h` header comment.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use thiserror::Error;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION_SUPPORTED: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// Sanity caps — a crafted GGUF could otherwise claim billions of
// tensors/KVs/dims and force a huge `Vec::with_capacity` before any
// data is read. These numbers are ~100x real-world worst cases
// (DeepSeek 33B has ~900 tensors; biggest KV counts top out around
// 30; tensors in published models use 1-4 dims).
pub const GGUF_MAX_TENSORS: u64 = 1 << 20;  // 1,048,576
pub const GGUF_MAX_KV:      u64 = 1 << 16;  // 65,536
pub const GGUF_MAX_DIMS:    u32 = 8;        // ggml itself caps at GGML_MAX_DIMS=8

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("not a GGUF file: bad magic {got:?}")]
    BadMagic { got: [u8; 4] },
    #[error("unsupported GGUF version {got} (parser knows {GGUF_VERSION_SUPPORTED})")]
    BadVersion { got: u32 },
    #[error("GGUF truncated or malformed at byte {offset}: {reason}")]
    Malformed { offset: usize, reason: String },
    #[error("GGUF KV type {got} is unknown")]
    UnknownKvType { got: u32 },
    #[error("GGUF ggml tensor type {got} is unknown")]
    UnknownGgmlType { got: u32 },
}

type Result<T> = std::result::Result<T, GgufError>;

// -------- KV value types --------
//
// Mirror of `enum gguf_type` in ggml/include/gguf.h. Kept as u32
// so round-tripping an unknown-to-us value doesn't lose it.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl KvType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8, 1 => Self::I8, 2 => Self::U16, 3 => Self::I16,
            4 => Self::U32, 5 => Self::I32, 6 => Self::F32, 7 => Self::Bool,
            8 => Self::String, 9 => Self::Array,
            10 => Self::U64, 11 => Self::I64, 12 => Self::F64,
            _ => return Err(GgufError::UnknownKvType { got: v }),
        })
    }

    /// Byte size of a scalar value of this type. Returns None for
    /// variable-length kinds (string, array) where the size lives
    /// in the payload itself.
    fn scalar_size(self) -> Option<usize> {
        Some(match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String | Self::Array => return None,
        })
    }
}

// -------- ggml tensor dtype --------
//
// Mirror of `enum ggml_type` in ggml/include/ggml.h. We don't
// interpret the bytes, so the enum's main use here is surfacing a
// string name on the manifest side. If ggml adds a new type we
// record the raw number; chunker still works.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub fn name(self) -> &'static str {
        // Values from ggml/include/ggml.h as of libllama @ 0d0764d.
        // If a new quant lands upstream and we haven't synced, we
        // fall back to "GGML_TYPE_<n>" so the chunker still produces
        // a valid manifest (the load side will surface the mismatch).
        match self.0 {
            0  => "F32",    1  => "F16",   2  => "Q4_0", 3  => "Q4_1",
            6  => "Q5_0",   7  => "Q5_1",  8  => "Q8_0", 9  => "Q8_1",
            10 => "Q2_K",   11 => "Q3_K",  12 => "Q4_K", 13 => "Q5_K",
            14 => "Q6_K",   15 => "Q8_K",
            16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS",
            19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S",
            23 => "IQ4_XS", 24 => "I8",   25 => "I16", 26 => "I32",
            27 => "I64",   28 => "F64",   29 => "IQ1_M",
            30 => "BF16",  34 => "TQ1_0", 35 => "TQ2_0",
            _ => "GGML_UNKNOWN",
        }
    }
}

// -------- parsed structures --------

/// A KV entry as found in the GGUF header. The value is carried as
/// a byte range into the mmap so we can re-emit it verbatim later
/// (critical for the chunker — rewriting KV metadata would require
/// re-computing every downstream offset).
#[derive(Clone, Debug)]
pub struct KvEntry<'m> {
    pub key:   &'m str,
    pub ty:    KvType,
    /// For Array: the element type; ignored for scalars.
    pub array_element_ty: Option<KvType>,
    /// Full byte range covering the *value* portion of this KV
    /// entry in the mmap (not including the leading key/type tag
    /// bytes). Re-emit by copying this slice.
    pub value_range: std::ops::Range<usize>,
    /// Byte range covering this entry's entire record (key, type,
    /// value). Useful for slicing groups of KVs into a separate
    /// chunk — e.g. the tokenizer block becomes one CID chunk.
    pub entry_range: std::ops::Range<usize>,
}

/// A tensor index entry. Offsets are RELATIVE to the tensor data
/// blob (starts at `GgufHeader::tensor_data_offset`). Name and
/// shape are for the manifest; the byte range is what the chunker
/// writes as a single CID blob.
#[derive(Clone, Debug)]
pub struct TensorIndexEntry<'m> {
    pub name:  &'m str,
    pub shape: Vec<i64>,
    pub dtype: GgmlType,
    /// Offset of this tensor's bytes, RELATIVE to
    /// `GgufHeader::tensor_data_offset`.
    pub data_offset_rel: u64,
    /// Size in bytes.
    pub n_bytes: u64,
}

/// The parsed GGUF header. `mmap` is held here so every returned
/// slice stays valid for the life of this struct.
pub struct Gguf {
    pub path:               PathBuf,
    mmap:                   Mmap,
    pub version:            u32,
    pub alignment:          u64,
    /// Absolute file offset where the tensor data blob starts.
    /// Every tensor's absolute offset = `tensor_data_offset +
    /// entry.data_offset_rel`.
    pub tensor_data_offset: u64,
    pub n_kv:               u64,
    pub n_tensors:          u64,
    /// End of the KV metadata block, start of the tensor index.
    pub kv_end_offset:      usize,
    /// End of the tensor index, start of the tensor data padding.
    pub index_end_offset:   usize,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| GgufError::Io { path: path.clone(), source: e })?;
        // Safety: we only read through the mmap; the file is
        // immutable for the life of this object.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| GgufError::Io { path: path.clone(), source: e })?;

        let mut p = Cursor::new(&mmap[..]);
        let magic = p.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            let mut got = [0u8; 4];
            got.copy_from_slice(magic);
            return Err(GgufError::BadMagic { got });
        }
        let version = p.read_u32()?;
        if version != GGUF_VERSION_SUPPORTED {
            return Err(GgufError::BadVersion { got: version });
        }
        let n_tensors = p.read_u64()?;
        let n_kv      = p.read_u64()?;
        if n_tensors > GGUF_MAX_TENSORS {
            return Err(GgufError::Malformed {
                offset: 16,
                reason: format!("n_tensors = {n_tensors} exceeds cap {GGUF_MAX_TENSORS}"),
            });
        }
        if n_kv > GGUF_MAX_KV {
            return Err(GgufError::Malformed {
                offset: 24,
                reason: format!("n_kv = {n_kv} exceeds cap {GGUF_MAX_KV}"),
            });
        }

        // KV metadata: walk once to locate each entry's byte range.
        // We don't interpret values — just record where they are.
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        for _ in 0..n_kv {
            let entry_start = p.pos;
            let key = p.read_string()?;
            let ty  = KvType::from_u32(p.read_u32()?)?;
            let value_start = p.pos;
            // Read the key "general.alignment" on the fly — it changes
            // tensor-data-offset padding and we need it before the
            // tensor index loop finishes.
            skip_kv_value(&mut p, ty)?;
            let value_end = p.pos;
            if key == "general.alignment" && ty == KvType::U32 {
                // Reread: safe because skip_kv_value is deterministic.
                let raw = &mmap[value_start..value_start + 4];
                alignment = u32::from_le_bytes(raw.try_into().unwrap()) as u64;
            }
            let _ = (entry_start, value_end); // kept for future KV-range extraction
        }
        let kv_end = p.pos;

        // Tensor index: one entry per tensor.
        for _ in 0..n_tensors {
            let _name = p.read_string()?;
            let n_dims = p.read_u32()?;
            if n_dims > GGUF_MAX_DIMS {
                return Err(GgufError::Malformed {
                    offset: p.pos,
                    reason: format!("tensor n_dims = {n_dims} exceeds cap {GGUF_MAX_DIMS}"),
                });
            }
            for _ in 0..n_dims {
                let _d = p.read_u64()?; // shape element
            }
            let _dtype = p.read_u32()?;
            let _offset_rel = p.read_u64()?;
        }
        let index_end = p.pos;

        // Tensor data starts at the next aligned offset.
        let tensor_data_offset = align_up(index_end as u64, alignment);

        Ok(Self {
            path,
            mmap,
            version,
            alignment,
            tensor_data_offset,
            n_kv,
            n_tensors,
            kv_end_offset: kv_end,
            index_end_offset: index_end,
        })
    }

    /// Slice containing the whole mmapped file.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Walk the KV metadata again, producing borrowed entries that
    /// the chunker can group by prefix (e.g. "tokenizer.*",
    /// "qwen2.*", etc.) and emit as separate CID chunks.
    pub fn kv_entries(&self) -> Result<Vec<KvEntry<'_>>> {
        let mut out = Vec::with_capacity(self.n_kv as usize);
        let mut p = Cursor::new(&self.mmap[..]);
        // Re-seek past fixed header: 4 magic + 4 version + 8 n_tensors + 8 n_kv = 24.
        p.pos = 24;
        for _ in 0..self.n_kv {
            let entry_start = p.pos;
            let key = p.read_str_borrowed(self.as_bytes())?;
            let ty  = KvType::from_u32(p.read_u32()?)?;
            let value_start = p.pos;
            let array_elem_ty = if ty == KvType::Array {
                Some(KvType::from_u32(peek_u32(&mut p)?)?)
            } else {
                None
            };
            skip_kv_value(&mut p, ty)?;
            let entry_end = p.pos;
            out.push(KvEntry {
                key,
                ty,
                array_element_ty: array_elem_ty,
                value_range: value_start..entry_end,
                entry_range: entry_start..entry_end,
            });
        }
        Ok(out)
    }

    /// Walk the tensor index, producing borrowed entries. Each
    /// entry's `data_offset_rel` is relative to
    /// [`Self::tensor_data_offset`]; adding the two gives the
    /// absolute byte position of that tensor's data.
    pub fn tensors(&self) -> Result<Vec<TensorIndexEntry<'_>>> {
        let mut out = Vec::with_capacity(self.n_tensors as usize);
        let mut p = Cursor::new(&self.mmap[..]);
        p.pos = self.kv_end_offset;
        for _ in 0..self.n_tensors {
            let name = p.read_str_borrowed(self.as_bytes())?;
            let n_dims = p.read_u32()?;
            if n_dims > GGUF_MAX_DIMS {
                return Err(GgufError::Malformed {
                    offset: p.pos,
                    reason: format!("tensor `{name}` n_dims = {n_dims} exceeds cap {GGUF_MAX_DIMS}"),
                });
            }
            let n_dims = n_dims as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(p.read_u64()? as i64);
            }
            let dtype = GgmlType(p.read_u32()?);
            let data_offset_rel = p.read_u64()?;
            out.push(TensorIndexEntry {
                name, shape, dtype, data_offset_rel,
                n_bytes: 0, // filled in below once we have the full list
            });
        }
        // Derive n_bytes: for each tensor, its data runs from its
        // offset to the next tensor's offset (or to the end of the
        // file for the last one). GGUF guarantees each tensor is
        // aligned, so we use aligned offsets to bound size.
        let mut sorted_idxs: Vec<usize> = (0..out.len()).collect();
        sorted_idxs.sort_by_key(|&i| out[i].data_offset_rel);
        let file_end_rel = self.mmap.len() as u64 - self.tensor_data_offset;
        for w in 0..sorted_idxs.len() {
            let this = sorted_idxs[w];
            let next_offset_rel = if w + 1 < sorted_idxs.len() {
                out[sorted_idxs[w + 1]].data_offset_rel
            } else {
                file_end_rel
            };
            out[this].n_bytes = next_offset_rel - out[this].data_offset_rel;
        }
        Ok(out)
    }

    /// Raw bytes of a tensor's data — a borrow into the mmap.
    pub fn tensor_bytes<'a>(&'a self, entry: &TensorIndexEntry<'a>) -> &'a [u8] {
        let start = (self.tensor_data_offset + entry.data_offset_rel) as usize;
        let end = start + entry.n_bytes as usize;
        &self.mmap[start..end]
    }
}

// -------- parsing primitives --------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(GgufError::Malformed {
                offset: self.pos,
                reason: format!("short read of {n} bytes (have {})", self.buf.len() - self.pos),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    /// A GGUF string: u64 length + utf8 bytes (no null terminator).
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|_| GgufError::Malformed {
                offset: self.pos - len,
                reason: "string is not valid UTF-8".into(),
            })
    }

    /// Borrowed form — for the caller-visible parsing passes.
    fn read_str_borrowed(&mut self, full: &'a [u8]) -> Result<&'a str> {
        let len = self.read_u64()? as usize;
        let start = self.pos;
        let _ = self.read_bytes(len)?;
        std::str::from_utf8(&full[start..start + len])
            .map_err(|_| GgufError::Malformed {
                offset: start,
                reason: "string is not valid UTF-8".into(),
            })
    }
}

fn peek_u32(p: &mut Cursor<'_>) -> Result<u32> {
    if p.pos + 4 > p.buf.len() {
        return Err(GgufError::Malformed {
            offset: p.pos,
            reason: "peek_u32 past EOF".into(),
        });
    }
    Ok(u32::from_le_bytes(p.buf[p.pos..p.pos + 4].try_into().unwrap()))
}

fn skip_kv_value(p: &mut Cursor<'_>, ty: KvType) -> Result<()> {
    match ty {
        KvType::String => {
            let n = p.read_u64()? as usize;
            let _ = p.read_bytes(n)?;
        }
        KvType::Array => {
            let element_ty = KvType::from_u32(p.read_u32()?)?;
            let n_elems = p.read_u64()? as usize;
            match element_ty {
                KvType::String => {
                    for _ in 0..n_elems {
                        let n = p.read_u64()? as usize;
                        let _ = p.read_bytes(n)?;
                    }
                }
                KvType::Array => {
                    return Err(GgufError::Malformed {
                        offset: p.pos,
                        reason: "nested arrays are not supported in GGUF v3".into(),
                    });
                }
                _ => {
                    let sz = element_ty.scalar_size().unwrap();
                    let _ = p.read_bytes(sz * n_elems)?;
                }
            }
        }
        _ => {
            let sz = ty.scalar_size().unwrap();
            let _ = p.read_bytes(sz)?;
        }
    }
    Ok(())
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    let rem = offset % alignment;
    if rem == 0 { offset } else { offset + (alignment - rem) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = "/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    fn qwen() -> Option<Gguf> {
        if !Path::new(QWEN).exists() { return None; }
        Some(Gguf::open(QWEN).expect("parse qwen gguf"))
    }

    #[test]
    fn header_is_plausible() {
        let Some(g) = qwen() else { return; };
        assert_eq!(g.version, 3);
        assert_eq!(g.alignment, 32);
        assert_eq!(g.n_tensors, 291);
        assert_eq!(g.n_kv, 26);
        assert_eq!(g.tensor_data_offset, 5947744);
    }

    #[test]
    fn kv_entries_cover_n_kv() {
        let Some(g) = qwen() else { return; };
        let kvs = g.kv_entries().unwrap();
        assert_eq!(kvs.len(), g.n_kv as usize);
        assert!(kvs.iter().any(|e| e.key == "general.architecture"));
        assert!(kvs.iter().any(|e| e.key.starts_with("tokenizer.")));
    }

    #[test]
    fn tensor_index_covers_every_tensor() {
        let Some(g) = qwen() else { return; };
        let ts = g.tensors().unwrap();
        assert_eq!(ts.len(), g.n_tensors as usize);

        // Sum of n_bytes + tensor_data_offset ≈ file size
        // (rounded up to alignment for each tensor boundary).
        let total: u64 = ts.iter().map(|t| t.n_bytes).sum();
        let expected = g.as_bytes().len() as u64 - g.tensor_data_offset;
        // Slack = alignment padding at the tail.
        assert!(
            total <= expected && expected - total < 2 * g.alignment,
            "sum(n_bytes)={total}, expected≈{expected}"
        );

        // Biggest tensors should be output.weight and token_embd.weight.
        let output = ts.iter().find(|t| t.name == "output.weight").unwrap();
        let embd   = ts.iter().find(|t| t.name == "token_embd.weight").unwrap();
        assert_eq!(output.n_bytes, 144_643_072);
        assert_eq!(embd.n_bytes,    93_592_576);
    }

    #[test]
    fn tensor_bytes_slice_is_that_size() {
        let Some(g) = qwen() else { return; };
        let ts = g.tensors().unwrap();
        let t = ts.iter().find(|t| t.name == "output.weight").unwrap();
        let bytes = g.tensor_bytes(t);
        assert_eq!(bytes.len() as u64, t.n_bytes);
    }
}
