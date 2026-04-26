//! Minimal GGUF header reader.
//!
//! We only need two things from the header:
//!
//! * the model architecture (`general.architecture`, a string) — tells us
//!   which key holds the layer count;
//! * the layer count itself (`<arch>.block_count`, a u32).
//!
//! Everything else in the metadata KV table gets skipped. The format is
//! fully described at <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};

const MAGIC: u32 = 0x46554747; // "GGUF" little-endian

// GGUF value type codes (§2 of the spec).
#[derive(Debug)]
#[repr(u32)]
enum ValueType {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5,
    F32 = 6, Bool = 7, String = 8, Array = 9,
    U64 = 10, I64 = 11, F64 = 12,
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0  => Self::U8, 1  => Self::I8, 2  => Self::U16, 3  => Self::I16,
            4  => Self::U32, 5  => Self::I32, 6 => Self::F32, 7 => Self::Bool,
            8  => Self::String, 9 => Self::Array,
            10 => Self::U64, 11 => Self::I64, 12 => Self::F64,
            _  => return Err(anyhow!("unknown gguf value type {v}")),
        })
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16            => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String | Self::Array       => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GgufInfo {
    pub architecture: String,
    pub block_count:  u32,
}

// ----------------------------------------------------------------------

pub fn read_info(path: &Path) -> Result<GgufInfo> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::with_capacity(64 * 1024, f);

    let magic = read_u32(&mut r)?;
    if magic != MAGIC {
        return Err(anyhow!("not a gguf file (magic={magic:#x})"));
    }
    let version = read_u32(&mut r)?;
    if !(1..=3).contains(&version) {
        return Err(anyhow!("unsupported gguf version {version}"));
    }
    let _tensor_count = read_u64(&mut r)?;
    let kv_count      = read_u64(&mut r)?;

    let mut architecture: Option<String> = None;
    // We may see block_count before architecture; stash candidates keyed
    // by the "<arch>.block_count" key and resolve at the end.
    let mut block_counts: Vec<(String, u32)> = Vec::new();

    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let ty  = ValueType::from_u32(read_u32(&mut r)?)?;

        if key == "general.architecture" {
            architecture = Some(read_value_as_string(&mut r, &ty)?);
            continue;
        }
        if key.ends_with(".block_count") {
            let v = read_value_as_u32(&mut r, &ty)
                .with_context(|| format!("reading {key} as u32"))?;
            block_counts.push((key, v));
            continue;
        }
        skip_value(&mut r, &ty)?;
    }

    let arch = architecture.ok_or_else(||
        anyhow!("gguf is missing general.architecture"))?;
    let expected_key = format!("{arch}.block_count");
    let block_count = block_counts.iter()
        .find(|(k, _)| k == &expected_key)
        .or_else(|| block_counts.first())
        .map(|(_, v)| *v)
        .ok_or_else(|| anyhow!("gguf is missing block_count"))?;

    Ok(GgufInfo { architecture: arch, block_count })
}

pub fn sha256_hex(path: &Path) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

// ----------------------------------------------------------------------
//  primitive readers
// ----------------------------------------------------------------------

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    if len > 64 * 1024 {
        return Err(anyhow!("gguf string too long ({len})"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow!("gguf string utf-8: {e}"))
}

fn read_value_as_string<R: Read>(r: &mut R, ty: &ValueType) -> Result<String> {
    match ty {
        ValueType::String => read_string(r),
        _ => Err(anyhow!("expected string, got {ty:?}")),
    }
}

fn read_value_as_u32<R: Read>(r: &mut R, ty: &ValueType) -> Result<u32> {
    Ok(match ty {
        ValueType::U8  => { let mut b = [0u8;1]; r.read_exact(&mut b)?; b[0] as u32 }
        ValueType::U16 => { let mut b = [0u8;2]; r.read_exact(&mut b)?; u16::from_le_bytes(b) as u32 }
        ValueType::U32 => read_u32(r)?,
        ValueType::U64 => {
            let v = read_u64(r)?;
            if v > u32::MAX as u64 { return Err(anyhow!("block_count u64={v} overflows u32")); }
            v as u32
        }
        _ => return Err(anyhow!("expected unsigned int for block_count, got {ty:?}")),
    })
}

fn skip_value<R: Read + Seek>(r: &mut R, ty: &ValueType) -> Result<()> {
    if let Some(n) = ty.fixed_size() {
        r.seek(SeekFrom::Current(n as i64))?;
        return Ok(());
    }
    match ty {
        ValueType::String => {
            let len = read_u64(r)?;
            r.seek(SeekFrom::Current(len as i64))?;
        }
        ValueType::Array => {
            let inner = ValueType::from_u32(read_u32(r)?)?;
            let n = read_u64(r)? as usize;
            if let Some(sz) = inner.fixed_size() {
                r.seek(SeekFrom::Current((sz * n) as i64))?;
            } else {
                for _ in 0..n {
                    skip_value(r, &inner)?;
                }
            }
        }
        _ => unreachable!("fixed_size covers the rest"),
    }
    Ok(())
}
