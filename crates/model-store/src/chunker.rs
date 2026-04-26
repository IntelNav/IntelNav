//! Orchestrates: parse a GGUF → route tensors into bundles → compute
//! CIDs for every bundle and every member → write chunk files +
//! manifest.
//!
//! The chunker is read-only with respect to its input (we mmap and
//! slice) and write-only with respect to its output (one directory,
//! no in-place edits). Re-running with the same input and the same
//! options produces byte-identical output.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use crate::bundle::{classify_tensor, BundleEntry, BundleKind, BundleMember};
use crate::cid::{cid_for, cid_string_for, cid_string_from_sha256};
use crate::gguf::{Gguf, KvType};
use crate::manifest::{Chunk, GgufInfo, Manifest, MANIFEST_VERSION};

/// Knobs for a single chunking run.
#[derive(Clone, Debug)]
pub struct ChunkerOptions {
    /// Output directory. Will be created if missing. Must be empty
    /// unless [`Self::overwrite`] is set.
    pub output_dir: PathBuf,
    /// Allow writing into a non-empty output directory. Existing
    /// files may be overwritten; unrelated files are left alone.
    pub overwrite: bool,
    /// Suppress writing the manifest file. Still returns the manifest
    /// struct in the outcome — useful for tests and dry-run CLI.
    pub dry_run: bool,
}

impl ChunkerOptions {
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            overwrite: false,
            dry_run: false,
        }
    }
}

/// Result of a successful chunking run.
#[derive(Clone, Debug)]
pub struct ChunkOutcome {
    /// The manifest as written. If `dry_run` is false, also persisted
    /// to `manifest.json` in the output directory.
    pub manifest: Manifest,
    /// CID of the manifest file's JSON bytes — the "model CID" that
    /// a peer would look up to fetch this model.
    pub manifest_cid: String,
    /// Bytes of the manifest as persisted. Useful for callers that
    /// want to forward it somewhere without re-reading the file.
    pub manifest_bytes: Vec<u8>,
    /// Count of bundles emitted (embed + N blocks + head, typically).
    pub n_bundles: usize,
    /// Count of tensors that went into those bundles.
    pub n_tensors: usize,
    /// Total bytes written to disk (chunk bodies + header + manifest).
    /// Zero in dry-run.
    pub bytes_written: u64,
}

/// Entry point. Parse the GGUF at `input`, chunk it, and materialize
/// the result into `opts.output_dir`.
pub fn chunk_gguf(input: impl AsRef<Path>, opts: &ChunkerOptions) -> Result<ChunkOutcome> {
    let input = input.as_ref();
    let gguf = Gguf::open(input).with_context(|| format!("parsing GGUF at {}", input.display()))?;
    info!(
        path = %input.display(),
        n_tensors = gguf.n_tensors,
        n_kv = gguf.n_kv,
        alignment = gguf.alignment,
        "parsed GGUF"
    );

    // --- Diagnostics from KV: name, architecture, layer count ---
    let mut name_kv: Option<String> = None;
    let mut arch_kv: Option<String> = None;
    let mut block_count_kv: Option<u32> = None;
    for kv in gguf.kv_entries()? {
        let val_bytes = &gguf.as_bytes()[kv.value_range.clone()];
        match (kv.ty, kv.key) {
            (KvType::String, "general.name") => {
                name_kv = read_gguf_string(val_bytes);
            }
            (KvType::String, "general.architecture") => {
                arch_kv = read_gguf_string(val_bytes);
            }
            // `<arch>.block_count` is the canonical layer-count KV.
            // We don't know the arch up front, so match by suffix.
            (KvType::U32, key) if key.ends_with(".block_count") => {
                if val_bytes.len() >= 4 {
                    block_count_kv =
                        Some(u32::from_le_bytes(val_bytes[..4].try_into().unwrap()));
                }
            }
            _ => {}
        }
    }

    // --- Route every tensor into a bundle ---
    let tensors = gguf.tensors()?;
    let mut groups: BTreeMap<BundleKey, Vec<usize>> = BTreeMap::new();
    let mut max_block: i64 = -1;
    for (i, t) in tensors.iter().enumerate() {
        let kind = classify_tensor(t.name);
        if let BundleKind::Block(n) = kind {
            if (n as i64) > max_block {
                max_block = n as i64;
            }
        }
        groups.entry(BundleKey::from(&kind)).or_default().push(i);
    }
    let n_layers = block_count_kv.unwrap_or_else(|| (max_block + 1).max(0) as u32);

    // --- Header chunk: bytes [0, tensor_data_offset) ---
    //
    // The header is typically a few MB; we copy it once to hash and
    // queue it for writing. Bundles below stream directly to disk.
    let header_end = gguf.tensor_data_offset as usize;
    let header_bytes = &gguf.as_bytes()[..header_end];
    let header_cid = cid_string_for(header_bytes);
    let header_chunk = Chunk {
        cid: header_cid.clone(),
        size: header_bytes.len() as u64,
    };
    debug!(cid = %header_cid, size = header_bytes.len(), "computed header chunk");

    // Prepare output dir up front so we can stream bundles into it.
    // In dry-run we never write, so skip.
    let chunks_dir = opts.output_dir.join("chunks");
    if !opts.dry_run {
        prepare_output_dir(&opts.output_dir, opts.overwrite)
            .with_context(|| format!("preparing output dir {}", opts.output_dir.display()))?;
        fs::create_dir_all(&chunks_dir)
            .with_context(|| format!("creating {}", chunks_dir.display()))?;
    }

    let mut bytes_written: u64 = 0;
    let mut written_cids: std::collections::HashSet<String> = std::collections::HashSet::new();

    if !opts.dry_run {
        write_chunk(&chunks_dir.join(format!("{header_cid}.bin")), header_bytes)
            .with_context(|| format!("writing header chunk {header_cid}"))?;
        bytes_written += header_bytes.len() as u64;
        written_cids.insert(header_cid.clone());
    }

    // --- Build each bundle, streaming to disk as we go ---
    //
    // For a 33B-class model the embed bundle alone is ~500 MB; holding
    // every bundle in RAM before writing blows past typical laptop
    // memory. Instead we stream each bundle into a tempfile while
    // feeding its bytes through a Sha256 hasher, then atomically
    // rename to `<cid>.bin`. Per-member CIDs are still computed by
    // hashing the mmap slice directly (zero copy).
    let mut bundle_entries: Vec<BundleEntry> = Vec::with_capacity(groups.len());

    for (key, indices) in groups.iter() {
        let kind = key.to_kind();
        let mut members: Vec<BundleMember> = Vec::with_capacity(indices.len());

        // Unique tempfile per bundle; renamed once we know the CID.
        let tmp_path = if opts.dry_run {
            PathBuf::new() // unused
        } else {
            chunks_dir.join(format!(".bundle-{}.tmp", kind.name()))
        };

        let mut tmp_file = if opts.dry_run { None } else {
            Some(File::create(&tmp_path)
                .with_context(|| format!("creating bundle tmp {}", tmp_path.display()))?)
        };
        let mut hasher = Sha256::new();
        let mut size: u64 = 0;

        for &ti in indices {
            let t = &tensors[ti];
            let bytes = gguf.tensor_bytes(t);
            let offset_in_bundle = size;
            hasher.update(bytes);
            if let Some(f) = tmp_file.as_mut() {
                f.write_all(bytes)
                    .with_context(|| format!("writing tensor `{}` into bundle tmp", t.name))?;
            }
            size += bytes.len() as u64;

            members.push(BundleMember {
                name: t.name.to_string(),
                shape: t.shape.clone(),
                dtype: t.dtype.name().to_string(),
                dtype_code: t.dtype.0,
                data_offset_rel: t.data_offset_rel,
                cid: cid_string_for(bytes),
                size: bytes.len() as u64,
                offset_in_bundle,
            });
        }

        let digest = hasher.finalize();
        let bundle_cid = cid_string_from_sha256(&digest);

        if let Some(f) = tmp_file {
            f.sync_all().context("fsync bundle tmp")?;
            drop(f);
            let final_path = chunks_dir.join(format!("{bundle_cid}.bin"));
            // Skip rename if we've already written this CID (two bundles
            // with identical bytes — rare but possible for zero-inited
            // tensors). Clean up the redundant tmp.
            if written_cids.insert(bundle_cid.clone()) {
                fs::rename(&tmp_path, &final_path)
                    .with_context(|| format!("renaming {} to {}",
                        tmp_path.display(), final_path.display()))?;
                bytes_written += size;
            } else {
                let _ = fs::remove_file(&tmp_path);
            }
        }

        bundle_entries.push(BundleEntry {
            name: kind.name(),
            layer_range: kind.layer_range(),
            cid: bundle_cid,
            size,
            members,
        });
    }

    let n_bundles = bundle_entries.len();
    let n_tensors = bundle_entries.iter().map(|b| b.members.len()).sum::<usize>();

    let manifest = Manifest {
        format: "intelnav-model".to_string(),
        version: MANIFEST_VERSION,
        name: name_kv,
        architecture: arch_kv,
        n_layers,
        gguf: GgufInfo {
            gguf_version: gguf.version,
            alignment: gguf.alignment,
            tensor_data_offset: gguf.tensor_data_offset,
            n_kv: gguf.n_kv,
            n_tensors: gguf.n_tensors,
        },
        header_chunk,
        bundles: bundle_entries,
    };
    let manifest_bytes = manifest.to_json_bytes().context("serializing manifest")?;
    let manifest_cid = cid_string_for(&manifest_bytes);

    if !opts.dry_run {
        let manifest_path = opts.output_dir.join("manifest.json");
        fs::write(&manifest_path, &manifest_bytes)
            .with_context(|| format!("writing {}", manifest_path.display()))?;
        bytes_written += manifest_bytes.len() as u64;
    }

    Ok(ChunkOutcome {
        manifest,
        manifest_cid,
        manifest_bytes,
        n_bundles,
        n_tensors,
        bytes_written,
    })
}

/// Verify a freshly-emitted chunk directory: re-hash every bundle
/// chunk on disk, confirm it matches the manifest, and cross-check
/// each member's per-tensor CID by slicing the bundle.
pub fn verify_chunks(output_dir: impl AsRef<Path>) -> Result<()> {
    let output_dir = output_dir.as_ref();
    let manifest_path = output_dir.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest = Manifest::from_json_bytes(&manifest_bytes)
        .with_context(|| format!("parsing {}", manifest_path.display()))?;

    let chunks_dir = output_dir.join("chunks");
    let read_chunk = |cid: &str| -> Result<Vec<u8>> {
        let path = chunks_dir.join(format!("{cid}.bin"));
        Ok(fs::read(&path).with_context(|| format!("reading {}", path.display()))?)
    };

    // Header chunk.
    let bytes = read_chunk(&manifest.header_chunk.cid)?;
    if bytes.len() as u64 != manifest.header_chunk.size {
        anyhow::bail!(
            "header size mismatch: manifest {} disk {}",
            manifest.header_chunk.size,
            bytes.len()
        );
    }
    let actual = cid_for(&bytes).to_string();
    if actual != manifest.header_chunk.cid {
        anyhow::bail!(
            "header hash mismatch: manifest {} actual {}",
            manifest.header_chunk.cid,
            actual
        );
    }

    // Every bundle — check the bundle CID, then slice each member out
    // and confirm its per-tensor CID too.
    for bundle in &manifest.bundles {
        let bundle_bytes = read_chunk(&bundle.cid)
            .with_context(|| format!("reading bundle {}", bundle.name))?;
        if bundle_bytes.len() as u64 != bundle.size {
            anyhow::bail!(
                "bundle {} size mismatch: manifest {} disk {}",
                bundle.name,
                bundle.size,
                bundle_bytes.len()
            );
        }
        let actual = cid_for(&bundle_bytes).to_string();
        if actual != bundle.cid {
            anyhow::bail!(
                "bundle {} hash mismatch: manifest {} actual {}",
                bundle.name,
                bundle.cid,
                actual
            );
        }
        for m in &bundle.members {
            let start = m.offset_in_bundle as usize;
            let end = start + m.size as usize;
            if end > bundle_bytes.len() {
                anyhow::bail!(
                    "member {} in bundle {} overruns bundle ({}..{} > {})",
                    m.name,
                    bundle.name,
                    start,
                    end,
                    bundle_bytes.len()
                );
            }
            let member_actual = cid_for(&bundle_bytes[start..end]).to_string();
            if member_actual != m.cid {
                anyhow::bail!(
                    "member {} in bundle {} hash mismatch: manifest {} actual {}",
                    m.name,
                    bundle.name,
                    m.cid,
                    member_actual
                );
            }
        }
    }
    Ok(())
}

fn read_gguf_string(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 8 {
        return None;
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if 8 + n > bytes.len() {
        return None;
    }
    std::str::from_utf8(&bytes[8..8 + n]).ok().map(str::to_owned)
}

fn prepare_output_dir(dir: &Path, overwrite: bool) -> io::Result<()> {
    if dir.exists() {
        if !overwrite && fs::read_dir(dir)?.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "output directory {} is not empty; pass --overwrite to reuse it",
                    dir.display()
                ),
            ));
        }
    } else {
        fs::create_dir_all(dir)?;
    }
    Ok(())
}

fn write_chunk(path: &Path, bytes: &[u8]) -> io::Result<()> {
    // Write-then-rename keeps partial writes from leaving a chunk
    // file with the right name but wrong contents — important when
    // the chunker is interrupted mid-run.
    let tmp = path.with_extension("bin.tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Sort key giving us `Embed < Block(0) < Block(1) < ... < Head`.
/// We can't derive `Ord` on `BundleKind` directly because it holds a
/// `u32`, so this wrapper encodes the desired ordering explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BundleKey(u8, u32);

impl From<&BundleKind> for BundleKey {
    fn from(k: &BundleKind) -> Self {
        match k {
            BundleKind::Embed => BundleKey(0, 0),
            BundleKind::Block(n) => BundleKey(1, *n),
            BundleKind::Head => BundleKey(2, 0),
        }
    }
}

impl BundleKey {
    fn to_kind(self) -> BundleKind {
        match self.0 {
            0 => BundleKind::Embed,
            1 => BundleKind::Block(self.1),
            _ => BundleKind::Head,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = "/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    fn have_qwen() -> bool {
        Path::new(QWEN).exists()
    }

    #[test]
    fn dry_run_produces_consistent_cids() {
        if !have_qwen() {
            return;
        }
        let opts = ChunkerOptions {
            output_dir: std::env::temp_dir().join("intelnav-chunk-dryrun"),
            overwrite: true,
            dry_run: true,
        };
        let a = chunk_gguf(QWEN, &opts).unwrap();
        let b = chunk_gguf(QWEN, &opts).unwrap();
        assert_eq!(a.manifest_cid, b.manifest_cid);
        assert_eq!(a.manifest.header_chunk.cid, b.manifest.header_chunk.cid);
        assert_eq!(a.n_tensors, 291);
        // Qwen2.5-0.5B has 24 transformer layers.
        assert_eq!(a.manifest.n_layers, 24);
        // embed + 24 blocks + head.
        assert_eq!(a.manifest.bundles.len(), 26);
        assert_eq!(a.n_bundles, 26);
    }

    #[test]
    fn bundle_order_is_stable() {
        if !have_qwen() {
            return;
        }
        let opts = ChunkerOptions {
            output_dir: std::env::temp_dir().join("intelnav-chunk-order"),
            overwrite: true,
            dry_run: true,
        };
        let out = chunk_gguf(QWEN, &opts).unwrap();
        let names: Vec<&str> = out.manifest.bundles.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names.first(), Some(&"embed"));
        assert_eq!(names.last(), Some(&"head"));
        for n in 0..24 {
            assert!(names.iter().any(|s| *s == format!("blk.{n}")));
        }
    }

    #[test]
    fn write_and_verify_roundtrip() {
        if !have_qwen() {
            return;
        }
        let dir = std::env::temp_dir().join("intelnav-chunk-verify-v2");
        let _ = fs::remove_dir_all(&dir);
        let opts = ChunkerOptions {
            output_dir: dir.clone(),
            overwrite: false,
            dry_run: false,
        };
        let out = chunk_gguf(QWEN, &opts).unwrap();
        assert!(out.bytes_written > 0);
        assert!(dir.join("manifest.json").exists());
        // Spot-check: the `embed` bundle's file exists.
        let embed = out
            .manifest
            .bundles
            .iter()
            .find(|b| b.name == "embed")
            .expect("embed bundle present");
        assert!(dir.join("chunks").join(format!("{}.bin", embed.cid)).exists());
        // Full verification covers bundles + members.
        verify_chunks(&dir).unwrap();
    }
}
