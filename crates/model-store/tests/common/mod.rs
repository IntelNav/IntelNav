//! Test helpers shared across integration tests.
//!
//! The main export is [`synth_gguf`] — a tiny hand-built GGUF v3 file
//! with one "embed", two blocks, and one "head" — enough to exercise
//! the chunker/stitcher/fetcher without shipping or downloading a
//! real model. The tensors are zero-initialized F32 since libllama
//! never touches these files; only our own Rust parsers do.
//!
//! Do NOT load a synthetic GGUF with libllama — the arch tag is
//! `"intelnav_test"`, which libllama doesn't know about and would
//! reject. Use a real Qwen/Llama file for the libllama-side tests.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// GGUF spec constants mirrored here to keep this file standalone.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const KV_U32: u32 = 4;
const KV_STRING: u32 = 8;

/// ggml_type codes (same as `crate::gguf::GgmlType`'s numbering).
const GGML_TYPE_F32: u32 = 0;

/// Build a synthetic GGUF file at `path`. Returns the path for
/// chaining. Layout:
///
/// * `general.architecture` = `"intelnav_test"`
/// * `general.name`         = `"intelnav-synth"`
/// * `intelnav_test.block_count` = `2`
/// * Tensor index (all F32, zero-inited, 32-byte aligned):
///     * `token_embd.weight`        shape [8, 16]     — 512 B
///     * `blk.0.attn_norm.weight`   shape [8]         —  32 B
///     * `blk.0.ffn_norm.weight`    shape [8]         —  32 B
///     * `blk.1.attn_norm.weight`   shape [8]         —  32 B
///     * `blk.1.ffn_norm.weight`    shape [8]         —  32 B
///     * `output_norm.weight`       shape [8]         —  32 B
///     * `output.weight`            shape [8, 16]     — 512 B
pub fn synth_gguf(path: &Path) -> std::io::Result<PathBuf> {
    let alignment: u64 = 32;

    // -------- KV block --------
    let mut kv = Vec::new();
    write_kv_string(&mut kv, "general.architecture", "intelnav_test");
    write_kv_string(&mut kv, "general.name", "intelnav-synth");
    write_kv_u32   (&mut kv, "intelnav_test.block_count", 2);
    // Force a u32 `general.alignment` so our parser sees the non-
    // default-32 path at least once (here we set it back to 32, but
    // the KV-rewrite code in the stitcher treats this key specially).
    write_kv_u32(&mut kv, "general.alignment", 32);
    let n_kv: u64 = 4;

    // -------- Tensor list (shape × nbytes) --------
    //
    // ne are [dim0, dim1, ...]. 1D tensors store a single dim.
    let tensors: &[(&str, Vec<u64>)] = &[
        ("token_embd.weight",      vec![8, 16]),
        ("blk.0.attn_norm.weight", vec![8]),
        ("blk.0.ffn_norm.weight",  vec![8]),
        ("blk.1.attn_norm.weight", vec![8]),
        ("blk.1.ffn_norm.weight",  vec![8]),
        ("output_norm.weight",     vec![8]),
        ("output.weight",          vec![8, 16]),
    ];

    let byte_size = |shape: &[u64]| -> u64 {
        shape.iter().product::<u64>() * 4 /* sizeof(f32) */
    };

    // First pass: compute offsets so we can bake them into the index.
    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for (_, shape) in tensors {
        offsets.push(cursor);
        cursor = align_up(cursor + byte_size(shape), alignment);
    }
    let tensor_data_size = cursor;

    // -------- Tensor index block --------
    let mut index = Vec::new();
    for (i, (name, shape)) in tensors.iter().enumerate() {
        write_string(&mut index, name);
        write_u32(&mut index, shape.len() as u32);
        for &d in shape {
            write_u64(&mut index, d);
        }
        write_u32(&mut index, GGML_TYPE_F32);
        write_u64(&mut index, offsets[i]);
    }

    // -------- Assemble header --------
    let mut header = Vec::new();
    header.extend_from_slice(GGUF_MAGIC);
    write_u32(&mut header, GGUF_VERSION);
    write_u64(&mut header, tensors.len() as u64);
    write_u64(&mut header, n_kv);
    header.extend_from_slice(&kv);
    header.extend_from_slice(&index);

    let tensor_data_offset = align_up(header.len() as u64, alignment);
    let header_pad = tensor_data_offset - header.len() as u64;

    // -------- Write the file --------
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = File::create(path)?;
    f.write_all(&header)?;
    write_zeros(&mut f, header_pad as usize)?;

    // Tensor data: all zeros for F32 tensors, with alignment pad.
    for (i, (_, shape)) in tensors.iter().enumerate() {
        let sz = byte_size(shape) as usize;
        write_zeros(&mut f, sz)?;
        // Pad up to next aligned offset.
        let next_start = if i + 1 < offsets.len() { offsets[i + 1] } else { tensor_data_size };
        let cur_end = offsets[i] + sz as u64;
        let pad = next_start - cur_end;
        write_zeros(&mut f, pad as usize)?;
    }
    f.sync_all()?;
    Ok(path.to_path_buf())
}

// -------- byte-writer helpers --------
fn write_u32(b: &mut Vec<u8>, v: u32) { b.extend_from_slice(&v.to_le_bytes()); }
fn write_u64(b: &mut Vec<u8>, v: u64) { b.extend_from_slice(&v.to_le_bytes()); }
fn write_string(b: &mut Vec<u8>, s: &str) {
    write_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}
fn write_kv_u32(b: &mut Vec<u8>, key: &str, value: u32) {
    write_string(b, key);
    write_u32(b, KV_U32);
    write_u32(b, value);
}
fn write_kv_string(b: &mut Vec<u8>, key: &str, value: &str) {
    write_string(b, key);
    write_u32(b, KV_STRING);
    write_string(b, value);
}
fn write_zeros<W: Write>(w: &mut W, n: usize) -> std::io::Result<()> {
    const BUF: [u8; 4096] = [0u8; 4096];
    let mut left = n;
    while left > 0 {
        let take = left.min(BUF.len());
        w.write_all(&BUF[..take])?;
        left -= take;
    }
    Ok(())
}
fn align_up(offset: u64, alignment: u64) -> u64 {
    let rem = offset % alignment;
    if rem == 0 { offset } else { offset + (alignment - rem) }
}

#[allow(dead_code)] // used by specific test files, not all
pub fn unique_tmpdir(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "intelnav-test-{label}-{}-{seq}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}
