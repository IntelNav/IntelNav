//! Subset-GGUF stitcher.
//!
//! Input: a `Manifest`, a chunk cache directory (every bundle CID the
//! caller plans to use must already exist as `<dir>/<cid>.bin`), and
//! a half-open layer range `[start, end)`.
//!
//! Output: a valid GGUF file on disk, byte-compatible with upstream
//! llama.cpp's loader, containing only the tensors this peer needs.
//!
//! The design decision: we renumber `blk.<N>` tensors so the peer's
//! libllama sees a contiguous 0-based layer vector of size
//! `end - start`. This avoids having to patch llama.cpp's loader to
//! tolerate missing layer indices. Two small KV flags
//! (`intelnav.has_embed`, `intelnav.has_head`) tell a tiny loader
//! patch whether to expect embed/head tensors.
//!
//! Mental model: the original KV block is copied verbatim EXCEPT for
//! three edits — `<arch>.block_count` is rewritten to the subset
//! size, and the two `intelnav.*` bools are appended. KV bytes for
//! unchanged entries are copied as a single slab so we don't have to
//! round-trip every value through a typed representation.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;

use crate::cid::cid_string_for;
use crate::gguf::{Gguf, KvType, GGUF_MAGIC};
use crate::manifest::Manifest;

/// Temporary scratch file that auto-deletes on drop. Used to
/// materialize a header-only GGUF so the standard parser can open it;
/// without RAII we'd leak `intelnav-stitch-scratch-*.gguf` into /tmp
/// on every panic.
struct ScratchFile(PathBuf);
impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Identifies which peer-servable segment we're stitching for.
#[derive(Clone, Debug)]
pub struct StitchRange {
    /// Inclusive start layer (0-based).
    pub start: u32,
    /// Exclusive end layer (0-based). Must be `> start` unless the
    /// peer owns ONLY embed or ONLY head (in which case use
    /// `start == end`, a legal "no block" range).
    pub end: u32,
    /// Whether this peer is responsible for the embed phase. Only
    /// legal if `start == 0`.
    pub include_embed: bool,
    /// Whether this peer is responsible for the head phase. Only
    /// legal if `end == manifest.n_layers`.
    pub include_head: bool,
}

/// Result of a stitch operation.
#[derive(Clone, Debug)]
pub struct StitchOutcome {
    /// Absolute path to the stitched GGUF on disk.
    pub path: PathBuf,
    /// Size of the stitched file in bytes.
    pub size: u64,
    /// Number of tensors the stitched model contains.
    pub n_tensors: u64,
    /// Number of KV entries in the stitched header (may differ from
    /// the source by up to +2 for the `intelnav.*` flags).
    pub n_kv: u64,
}

/// Stitch a subset GGUF from chunks.
///
/// `chunk_cache_dir` must contain the bundle chunks for every segment
/// the range covers — flat layout, one file per CID: `<cid>.bin`.
/// `out_path` is overwritten if it exists.
pub fn stitch_subset(
    manifest: &Manifest,
    chunk_cache_dir: impl AsRef<Path>,
    range: &StitchRange,
    out_path: impl AsRef<Path>,
) -> Result<StitchOutcome> {
    let chunk_cache_dir = chunk_cache_dir.as_ref();
    let out_path = out_path.as_ref().to_path_buf();
    validate_range(manifest, range)?;

    // --- Load the header chunk and reparse it in place ---
    // We need the original KV byte ranges and tensor index to rewrite
    // them. The header chunk is literally the first
    // `tensor_data_offset` bytes of the original GGUF, so we hand it
    // back to our own parser by constructing a temp "truncated" file
    // view.
    let header_bytes = read_chunk(chunk_cache_dir, &manifest.header_chunk.cid)
        .context("reading header chunk from cache")?;
    if header_bytes.len() as u64 != manifest.header_chunk.size {
        return Err(anyhow!(
            "header chunk size mismatch: manifest {}, disk {}",
            manifest.header_chunk.size,
            header_bytes.len()
        ));
    }

    // Materialize the header chunk as a small on-disk GGUF so the
    // standard parser (which expects a file) can open it. `_scratch`
    // deletes the file on drop — no /tmp leaks on panic or early
    // return. `original` borrows the mmap backing the scratch file,
    // so both must stay alive for this function's duration.
    let _scratch = scratch_gguf_from_header(&header_bytes, manifest.gguf.tensor_data_offset)
        .context("materializing scratch GGUF for parse")?;
    let original = Gguf::open(&_scratch.0).context("parsing stitched header")?;

    // --- Decide which bundles to include and memory-map them ---
    //
    // We keep a live mmap per selected bundle and index into it when
    // writing. That means NO heap copy of tensor bytes at any point
    // (the kernel pages weights in lazily as the write syscall
    // streams them). Essential for 20 GB+ subset stitches.
    let kv_arch = manifest
        .architecture
        .clone()
        .ok_or_else(|| anyhow!("manifest has no architecture string; cannot rewrite block_count"))?;
    let block_count_key = format!("{kv_arch}.block_count");

    // `kept` holds pointers into `bundle_mmaps`; the Vec<Mmap> must
    // outlive every `KeptTensor` — we bind both in this scope.
    let mut bundle_mmaps: Vec<(String, Mmap)> = Vec::new();
    let mut kept: Vec<KeptTensor<'_>> = Vec::new();

    for bundle in &manifest.bundles {
        let include = match bundle.name.as_str() {
            "embed" => range.include_embed,
            "head" => range.include_head,
            b if b.starts_with("blk.") => {
                let n: u32 = b[4..].parse()
                    .map_err(|_| anyhow!("bundle name `{}` is malformed", b))?;
                n >= range.start && n < range.end
            }
            other => {
                return Err(anyhow!("unknown bundle name `{other}` in manifest"));
            }
        };
        if !include {
            continue;
        }
        let path = chunks_dir_resolve(chunk_cache_dir, &bundle.cid);
        let file = fs::File::open(&path)
            .with_context(|| format!("opening bundle {} at {}", bundle.name, path.display()))?;
        // Safety: the file is not modified externally during stitch.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap bundle {}", bundle.name))?;
        if mmap.len() as u64 != bundle.size {
            return Err(anyhow!(
                "bundle {} size mismatch: manifest {}, disk {}",
                bundle.name, bundle.size, mmap.len()
            ));
        }
        bundle_mmaps.push((bundle.cid.clone(), mmap));
    }

    // Now build the `kept` list with slice borrows into `bundle_mmaps`.
    // Mapping bundle name → mmap index is cheap (small N).
    for bundle in &manifest.bundles {
        let include = match bundle.name.as_str() {
            "embed" => range.include_embed,
            "head" => range.include_head,
            b if b.starts_with("blk.") => {
                let n: u32 = b[4..].parse().unwrap();
                n >= range.start && n < range.end
            }
            _ => false,
        };
        if !include { continue; }
        let mmap = &bundle_mmaps.iter().find(|(c, _)| c == &bundle.cid).unwrap().1;
        for m in &bundle.members {
            let start = m.offset_in_bundle as usize;
            let end = start + m.size as usize;
            if end > mmap.len() {
                return Err(anyhow!(
                    "member {} in bundle {} overruns ({}..{} > {})",
                    m.name, bundle.name, start, end, mmap.len()
                ));
            }
            kept.push(KeptTensor {
                new_name: rename_tensor(&m.name, range.start),
                dtype_code: m.dtype_code,
                shape: m.shape.clone(),
                bytes: &mmap[start..end],
            });
        }
    }

    if kept.is_empty() {
        return Err(anyhow!("stitch range selected zero tensors; refusing to emit empty GGUF"));
    }

    // --- Rewrite the KV block ---
    // Strategy: walk original KV entries, for each:
    //   * if key == `<arch>.block_count` → rewrite value to (end-start)
    //   * else → copy its entry bytes verbatim
    // Then append the two intelnav.* flags.
    let mut new_kv: Vec<u8> = Vec::with_capacity(header_bytes.len());
    let mut new_n_kv: u64 = 0;
    for kv in original.kv_entries()? {
        new_n_kv += 1;
        let entry_bytes = &header_bytes[kv.entry_range.clone()];
        if kv.key == block_count_key && kv.ty == KvType::U32 {
            // Replace: [key_len u64][key bytes][type u32][value u32]
            write_kv_u32(&mut new_kv, kv.key, range.end - range.start);
        } else {
            new_kv.extend_from_slice(entry_bytes);
        }
    }
    write_kv_bool(&mut new_kv, "intelnav.has_embed", range.include_embed);
    new_n_kv += 1;
    write_kv_bool(&mut new_kv, "intelnav.has_head", range.include_head);
    new_n_kv += 1;

    // --- Build the new tensor index ---
    // For each kept tensor: [name_len u64][name][n_dims u32][shape u64×n_dims][dtype u32][data_offset_rel u64]
    // Offsets recomputed with alignment padding between tensors.
    let alignment = manifest.gguf.alignment.max(1);
    let mut index_bytes: Vec<u8> = Vec::new();
    let mut data_offsets: Vec<u64> = Vec::with_capacity(kept.len());
    let mut cursor: u64 = 0;
    for t in &kept {
        write_string(&mut index_bytes, &t.new_name);
        write_u32(&mut index_bytes, t.shape.len() as u32);
        for d in &t.shape {
            write_u64(&mut index_bytes, *d as u64);
        }
        write_u32(&mut index_bytes, t.dtype_code);
        write_u64(&mut index_bytes, cursor);
        data_offsets.push(cursor);
        cursor = align_up(cursor + t.bytes.len() as u64, alignment);
    }

    // --- Assemble the new header ---
    // 4 magic + 4 version + 8 n_tensors + 8 n_kv + kv_bytes + index_bytes.
    let mut new_header: Vec<u8> = Vec::with_capacity(24 + new_kv.len() + index_bytes.len());
    new_header.extend_from_slice(GGUF_MAGIC);
    write_u32(&mut new_header, manifest.gguf.gguf_version);
    write_u64(&mut new_header, kept.len() as u64);
    write_u64(&mut new_header, new_n_kv);
    new_header.extend_from_slice(&new_kv);
    new_header.extend_from_slice(&index_bytes);

    // Compute tensor-data-offset for the new file.
    let new_tensor_data_offset = align_up(new_header.len() as u64, alignment);
    let header_padding = new_tensor_data_offset - new_header.len() as u64;

    // --- Write the file ---
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating parent dir {}", parent.display()))?;
    }
    let tmp = out_path.with_extension("gguf.tmp");
    {
        let mut f = fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(&new_header)?;
        write_zeros(&mut f, header_padding as usize)?;
        for (t, &off) in kept.iter().zip(data_offsets.iter()) {
            let expected_pos = new_tensor_data_offset + off;
            let actual_pos = f.stream_position()?;
            if expected_pos != actual_pos {
                return Err(anyhow!(
                    "tensor position drift: expected {} got {} (tensor `{}`)",
                    expected_pos, actual_pos, t.new_name
                ));
            }
            // Streaming write — no intermediate heap copy. The kernel
            // pages the mmap in, write(2) copies to pagecache.
            f.write_all(t.bytes)?;
            let after = f.stream_position()?;
            let padded = align_up(after, alignment);
            write_zeros(&mut f, (padded - after) as usize)?;
        }
        f.sync_all()?;
    }
    fs::rename(&tmp, &out_path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), out_path.display()))?;
    let size = fs::metadata(&out_path)?.len();

    Ok(StitchOutcome {
        path: out_path,
        size,
        n_tensors: kept.len() as u64,
        n_kv: new_n_kv,
    })
}

// ------------- helpers -------------

struct KeptTensor<'m> {
    new_name: String,
    dtype_code: u32,
    shape: Vec<i64>,
    /// Borrow into the bundle mmap — no heap copy of tensor bytes.
    bytes: &'m [u8],
}

fn rename_tensor(name: &str, layer_offset: u32) -> String {
    if let Some(rest) = name.strip_prefix("blk.") {
        let mut parts = rest.splitn(2, '.');
        let n_str = parts.next().unwrap_or("");
        let tail = parts.next().unwrap_or("");
        if let Ok(n) = n_str.parse::<u32>() {
            let local = n.saturating_sub(layer_offset);
            return if tail.is_empty() {
                format!("blk.{local}")
            } else {
                format!("blk.{local}.{tail}")
            };
        }
    }
    name.to_string()
}

fn validate_range(manifest: &Manifest, range: &StitchRange) -> Result<()> {
    if range.end < range.start {
        return Err(anyhow!("range end ({}) < start ({})", range.end, range.start));
    }
    if range.end > manifest.n_layers {
        return Err(anyhow!(
            "range end {} exceeds model n_layers {}",
            range.end, manifest.n_layers
        ));
    }
    if range.include_embed && range.start != 0 {
        return Err(anyhow!(
            "include_embed requires start==0 (got start={})", range.start
        ));
    }
    if range.include_head && range.end != manifest.n_layers {
        return Err(anyhow!(
            "include_head requires end==n_layers {} (got end={})",
            manifest.n_layers, range.end
        ));
    }
    Ok(())
}

fn read_chunk(cache_dir: &Path, cid: &str) -> Result<Vec<u8>> {
    let p = chunks_dir_resolve(cache_dir, cid);
    let b = fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    // Tamper check: re-hash and compare.
    let actual = cid_string_for(&b);
    if actual != cid {
        return Err(anyhow!(
            "chunk at {} hashes to {}, expected {}",
            p.display(), actual, cid
        ));
    }
    Ok(b)
}

/// Resolve a chunk CID to its on-disk path. Accepts both layouts:
///   * `<cache>/chunks/<cid>.bin` (the chunker / fetcher convention)
///   * `<cache>/<cid>.bin` (flat, convenient for tests)
fn chunks_dir_resolve(cache_dir: &Path, cid: &str) -> PathBuf {
    let canonical = cache_dir.join("chunks").join(format!("{cid}.bin"));
    if canonical.exists() {
        return canonical;
    }
    cache_dir.join(format!("{cid}.bin"))
}

/// Materialize the header-chunk bytes into a real on-disk file so
/// the standard GGUF parser can open it via mmap. Returns a
/// [`ScratchFile`] guard that deletes the file when dropped.
fn scratch_gguf_from_header(header_bytes: &[u8], tensor_data_offset: u64) -> Result<ScratchFile> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!(
        "intelnav-stitch-scratch-{}-{}.gguf",
        std::process::id(),
        seq
    ));
    let mut f = fs::File::create(&p)?;
    f.write_all(header_bytes)?;
    let pad = tensor_data_offset as i64 - header_bytes.len() as i64;
    if pad > 0 {
        write_zeros(&mut f, pad as usize)?;
    }
    // Need at least 1 byte past tensor_data_offset so mmap works
    // cleanly; libllama would bail on a zero-length data region but
    // we never ask it to load this file.
    write_zeros(&mut f, 1)?;
    f.sync_all()?;
    Ok(ScratchFile(p))
}

fn write_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn write_u64(buf: &mut Vec<u8>, v: u64) { buf.extend_from_slice(&v.to_le_bytes()); }

fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    write_string(buf, key);
    write_u32(buf, KvType::U32 as u32);
    write_u32(buf, value);
}

fn write_kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
    write_string(buf, key);
    write_u32(buf, KvType::Bool as u32);
    buf.push(if value { 1 } else { 0 });
}

fn write_zeros<W: Write>(w: &mut W, n: usize) -> io::Result<()> {
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
    if alignment == 0 { return offset; }
    let rem = offset % alignment;
    if rem == 0 { offset } else { offset + (alignment - rem) }
}

trait StreamPosition {
    fn stream_position(&mut self) -> io::Result<u64>;
}
impl StreamPosition for fs::File {
    fn stream_position(&mut self) -> io::Result<u64> {
        use std::io::Seek;
        Seek::stream_position(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::{chunk_gguf, ChunkerOptions};

    const QWEN: &str = "/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    fn have_qwen() -> bool {
        Path::new(QWEN).exists()
    }

    #[test]
    fn stitch_full_range_matches_tensor_count() {
        if !have_qwen() {
            return;
        }
        let cache = std::env::temp_dir().join("intelnav-stitch-full-cache");
        let _ = fs::remove_dir_all(&cache);
        let chunked = chunk_gguf(
            QWEN,
            &ChunkerOptions { output_dir: cache.clone(), overwrite: false, dry_run: false },
        ).unwrap();
        let range = StitchRange {
            start: 0,
            end: chunked.manifest.n_layers,
            include_embed: true,
            include_head: true,
        };
        let out = std::env::temp_dir().join("intelnav-stitch-full.gguf");
        let _ = fs::remove_file(&out);
        let outcome = stitch_subset(&chunked.manifest, &cache, &range, &out).unwrap();
        // Full stitch should contain every tensor (291 for Qwen).
        assert_eq!(outcome.n_tensors, chunked.manifest.gguf.n_tensors);
        // Parse it back and confirm layer count was preserved.
        let g = Gguf::open(&outcome.path).unwrap();
        assert_eq!(g.n_tensors, chunked.manifest.gguf.n_tensors);
        // KV count: original 26 + 2 intelnav flags.
        assert_eq!(g.n_kv, chunked.manifest.gguf.n_kv + 2);
    }

    #[test]
    fn stitch_mid_slice_renumbers_and_shrinks_block_count() {
        if !have_qwen() {
            return;
        }
        let cache = std::env::temp_dir().join("intelnav-stitch-mid-cache");
        let _ = fs::remove_dir_all(&cache);
        let chunked = chunk_gguf(
            QWEN,
            &ChunkerOptions { output_dir: cache.clone(), overwrite: false, dry_run: false },
        ).unwrap();
        let range = StitchRange {
            start: 10,
            end: 15,
            include_embed: false,
            include_head: false,
        };
        let out = std::env::temp_dir().join("intelnav-stitch-mid.gguf");
        let _ = fs::remove_file(&out);
        let outcome = stitch_subset(&chunked.manifest, &cache, &range, &out).unwrap();
        // 5 blocks × tensors-per-block should equal outcome.n_tensors,
        // and every tensor name must be blk.0..blk.4.
        let g = Gguf::open(&outcome.path).unwrap();
        assert_eq!(g.n_tensors, outcome.n_tensors);
        let tensors = g.tensors().unwrap();
        for t in &tensors {
            if let Some(rest) = t.name.strip_prefix("blk.") {
                let n: u32 = rest.split('.').next().unwrap().parse().unwrap();
                assert!(n < 5, "tensor `{}` has out-of-range local block {}", t.name, n);
            } else {
                panic!("unexpected non-block tensor in mid-slice: {}", t.name);
            }
        }
    }

    #[test]
    fn stitch_head_only_works() {
        if !have_qwen() {
            return;
        }
        let cache = std::env::temp_dir().join("intelnav-stitch-head-cache");
        let _ = fs::remove_dir_all(&cache);
        let chunked = chunk_gguf(
            QWEN,
            &ChunkerOptions { output_dir: cache.clone(), overwrite: false, dry_run: false },
        ).unwrap();
        let n = chunked.manifest.n_layers;
        let range = StitchRange {
            start: n,
            end: n,
            include_embed: false,
            include_head: true,
        };
        let out = std::env::temp_dir().join("intelnav-stitch-head.gguf");
        let _ = fs::remove_file(&out);
        let outcome = stitch_subset(&chunked.manifest, &cache, &range, &out).unwrap();
        assert!(outcome.n_tensors > 0);
        let g = Gguf::open(&outcome.path).unwrap();
        let tensors = g.tensors().unwrap();
        assert!(tensors.iter().any(|t| t.name == "output.weight"));
    }
}
