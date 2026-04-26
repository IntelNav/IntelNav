//! HTTP chunk fetching with CID verification and a disk cache.
//!
//! This is Path B phase 3: a peer that doesn't have a model locally
//! points at `manifest.json` on some HTTP endpoint, learns which
//! bundle CIDs it needs, and fetches each one. Every byte is SHA-256
//! verified against its CID before the file enters the cache — a
//! malicious server can at most make us refuse the download, not
//! slip us a wrong-weight chunk that silently computes garbage.
//!
//! The cache layout matches what `intelnav-chunk chunk` produces:
//!
//! ```text
//! <cache_root>/
//!   <manifest-cid>/
//!     manifest.json
//!     chunks/
//!       <cid>.bin
//! ```
//!
//! So a peer that fetched chunks over HTTP can be driven by the same
//! `stitch_subset(&manifest, <dir>, …)` path as a peer that ran the
//! chunker locally.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tracing::{debug, info, warn};
use url::Url;

use crate::cid::{cid_string_for, cid_string_from_sha256};
use crate::manifest::{Chunk, Manifest};

/// Options for an HTTP fetch session.
#[derive(Clone, Debug)]
pub struct FetchOptions {
    /// Root directory for the cache. A per-manifest subdirectory is
    /// created under it. `~/.cache/intelnav/models` is the convention.
    pub cache_root: PathBuf,
    /// Request timeout for any single chunk. Chunks can be ~500 MB
    /// on big models so the default is intentionally generous.
    pub request_timeout: Duration,
    /// Whether to use HTTP range requests to resume a `.tmp` file
    /// that was half-written by a previous run.
    pub resume: bool,
    /// Maximum concurrent chunk downloads. Set conservatively — most
    /// consumer home uplinks flatline at 2-4 parallel TCP streams.
    pub max_concurrent: usize,
    /// Hard cap on manifest JSON size, in bytes. A hostile server
    /// could otherwise claim any Content-Length and we'd buffer the
    /// whole response before even trying to parse. 64 MiB is ~1000x
    /// the largest real manifest we'd ever emit.
    pub max_manifest_bytes: u64,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            cache_root: default_cache_root(),
            request_timeout: Duration::from_secs(15 * 60),
            resume: true,
            max_concurrent: 4,
            max_manifest_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Default cache root, platform-aware:
///
/// * Unix (macOS, Linux): `$XDG_CACHE_HOME/intelnav/models` when set,
///   else `$HOME/.cache/intelnav/models`.
/// * Windows: `%LOCALAPPDATA%\intelnav\models` when set, else
///   `%USERPROFILE%\AppData\Local\intelnav\models`.
/// * Fallback on any platform: `<tempdir>/intelnav-cache`.
///
/// The final leg keeps tests and headless containers working when
/// none of the usual env vars are set.
pub fn default_cache_root() -> PathBuf {
    // Prefer the XDG spec if honored; then fall back per OS.
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("intelnav").join("models");
    }
    #[cfg(windows)]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local).join("intelnav").join("models");
        }
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            return PathBuf::from(profile)
                .join("AppData").join("Local")
                .join("intelnav").join("models");
        }
    }
    #[cfg(not(windows))]
    {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(".cache/intelnav/models");
        }
    }
    std::env::temp_dir().join("intelnav-cache")
}

/// Which chunks to fetch. `Subset` lets a peer pull only the bundles
/// it needs to serve its layer range.
#[derive(Clone, Debug)]
pub enum FetchPlan {
    /// Every bundle + header. Expensive; mostly useful for seeders.
    Full,
    /// The header (always needed by the stitcher) plus the named
    /// bundles (see `Manifest::bundles`).
    Bundles(Vec<String>),
}

impl FetchPlan {
    /// Select the bundle chunks needed for a peer serving
    /// `[start, end)` layers, mirroring `StitchRange`'s logic:
    ///   * `embed`   if `start == 0`
    ///   * `blk.<N>` for every N in `[start, end)`
    ///   * `head`    if `end == n_layers`
    pub fn for_range(manifest: &Manifest, start: u32, end: u32) -> Self {
        let mut names = Vec::new();
        if start == 0 {
            names.push("embed".to_string());
        }
        for n in start..end {
            names.push(format!("blk.{n}"));
        }
        if end == manifest.n_layers {
            names.push("head".to_string());
        }
        FetchPlan::Bundles(names)
    }
}

/// Where the manifest + its cache directory ended up on disk.
#[derive(Clone, Debug)]
pub struct FetchedManifest {
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub manifest_cid: String,
    pub manifest_url: String,
}

/// Fetch JUST the manifest JSON (no bundles). Callers that know what
/// layer range they'll serve should use this first so they can feed
/// `FetchPlan::for_range(&manifest, start, end)` into
/// [`fetch_chunks`] — otherwise a peer serving 2 of 80 layers ends up
/// downloading the whole model.
pub async fn fetch_manifest_only(
    manifest_url: &str,
    opts: &FetchOptions,
) -> Result<FetchedManifest> {
    let client = build_client(opts)?;
    let manifest_bytes = fetch_manifest_bytes(&client, manifest_url, opts.max_manifest_bytes).await?;
    let manifest = Manifest::from_json_bytes(&manifest_bytes)
        .context("parsing manifest JSON")?;
    let manifest_cid = cid_string_for(&manifest_bytes);

    fs::create_dir_all(&opts.cache_root).await
        .with_context(|| format!("creating cache root {}", opts.cache_root.display()))?;
    let dir = opts.cache_root.join(&manifest_cid);
    let chunks_dir = dir.join("chunks");
    fs::create_dir_all(&chunks_dir).await
        .with_context(|| format!("creating {}", chunks_dir.display()))?;
    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, &manifest_bytes).await
        .with_context(|| format!("writing {}", manifest_path.display()))?;
    info!(manifest_cid = %manifest_cid, dir = %dir.display(), "manifest cached");

    Ok(FetchedManifest {
        dir,
        manifest,
        manifest_cid,
        manifest_url: manifest_url.to_string(),
    })
}

/// Fetch the chunks selected by `plan`, streaming + verifying each
/// one into the cache directory opened by [`fetch_manifest_only`].
pub async fn fetch_chunks(
    fetched: &FetchedManifest,
    plan: &FetchPlan,
    opts: &FetchOptions,
) -> Result<FetchOutcome> {
    let client = build_client(opts)?;
    let base = derive_chunk_base(&fetched.manifest_url)?;
    let chunks_dir = fetched.dir.join("chunks");

    let mut wanted: Vec<Chunk> = Vec::new();
    wanted.push(fetched.manifest.header_chunk.clone());
    match plan {
        FetchPlan::Full => {
            for b in &fetched.manifest.bundles {
                wanted.push(Chunk { cid: b.cid.clone(), size: b.size });
            }
        }
        FetchPlan::Bundles(names) => {
            for n in names {
                let b = fetched.manifest.bundles.iter().find(|b| &b.name == n)
                    .ok_or_else(|| anyhow!("manifest has no bundle named {n}"))?;
                wanted.push(Chunk { cid: b.cid.clone(), size: b.size });
            }
        }
    }
    dedup_in_place(&mut wanted);

    let total_bytes: u64 = wanted.iter().map(|c| c.size).sum();
    info!(n = wanted.len(), bytes = total_bytes, "fetching chunks");

    let results = futures_util::stream::iter(wanted.into_iter().map(|chunk| {
        let client = client.clone();
        let base = base.clone();
        let chunks_dir = chunks_dir.clone();
        let resume = opts.resume;
        async move { fetch_one(&client, &base, &chunks_dir, &chunk, resume).await }
    }))
    .buffer_unordered(opts.max_concurrent)
    .collect::<Vec<_>>()
    .await;

    let mut bytes_downloaded: u64 = 0;
    let mut bytes_reused: u64 = 0;
    for r in results {
        let f = r?;
        if f.already_cached {
            bytes_reused += f.size;
        } else {
            bytes_downloaded += f.size;
        }
    }

    Ok(FetchOutcome {
        dir: fetched.dir.clone(),
        manifest: fetched.manifest.clone(),
        manifest_cid: fetched.manifest_cid.clone(),
        bytes_downloaded,
        bytes_reused,
    })
}

/// Convenience wrapper that does both steps in sequence. Use this
/// when the plan doesn't depend on manifest contents; otherwise call
/// [`fetch_manifest_only`] + [`fetch_chunks`] yourself to avoid
/// over-fetching.
pub async fn fetch_manifest_and_chunks(
    manifest_url: &str,
    plan: &FetchPlan,
    opts: &FetchOptions,
) -> Result<FetchOutcome> {
    let fetched = fetch_manifest_only(manifest_url, opts).await?;
    fetch_chunks(&fetched, plan, opts).await
}

fn build_client(opts: &FetchOptions) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(opts.request_timeout)
        .user_agent(concat!("intelnav-model-store/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("building HTTP client")
}

/// Stream the manifest body through a size cap. Rejects responses
/// that exceed `max_bytes` to avoid OOM from a hostile server.
async fn fetch_manifest_bytes(
    client: &reqwest::Client,
    url: &str,
    max_bytes: u64,
) -> Result<Vec<u8>> {
    let resp = client.get(url).send().await.context("GET manifest")?
        .error_for_status().context("manifest HTTP error")?;

    // If the server advertises Content-Length, short-circuit before
    // streaming a single byte.
    if let Some(len) = resp.content_length() {
        if len > max_bytes {
            return Err(anyhow!(
                "manifest Content-Length {len} exceeds cap {max_bytes}"
            ));
        }
    }

    let mut body = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = body.next().await {
        let b = chunk.context("streaming manifest body")?;
        if buf.len() as u64 + b.len() as u64 > max_bytes {
            return Err(anyhow!(
                "manifest body exceeds cap {max_bytes} bytes"
            ));
        }
        buf.extend_from_slice(&b);
    }
    Ok(buf)
}

#[derive(Clone, Debug)]
pub struct FetchOutcome {
    /// Directory containing `manifest.json` + `chunks/<cid>.bin`.
    /// Hand this straight to `stitch_subset`.
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub manifest_cid: String,
    pub bytes_downloaded: u64,
    pub bytes_reused: u64,
}

struct FetchOne {
    size: u64,
    already_cached: bool,
}

async fn fetch_one(
    client: &reqwest::Client,
    base: &Url,
    chunks_dir: &Path,
    chunk: &Chunk,
    resume: bool,
) -> Result<FetchOne> {
    let final_path = chunks_dir.join(format!("{}.bin", chunk.cid));

    // Already in cache? Verify size + hash; skip on match, redownload on drift.
    if fs::try_exists(&final_path).await.unwrap_or(false) {
        let bytes = fs::read(&final_path).await
            .with_context(|| format!("reading cached {}", final_path.display()))?;
        if bytes.len() as u64 == chunk.size && cid_string_for(&bytes) == chunk.cid {
            debug!(cid = %chunk.cid, "cache hit");
            return Ok(FetchOne { size: chunk.size, already_cached: true });
        }
        warn!(cid = %chunk.cid, "cached chunk is corrupt, redownloading");
        let _ = fs::remove_file(&final_path).await;
    }

    let tmp = chunks_dir.join(format!("{}.bin.tmp", chunk.cid));
    let url = base.join(&format!("chunks/{}.bin", chunk.cid))
        .with_context(|| format!("joining chunk URL for {}", chunk.cid))?;

    // Resume from a partial .tmp via Range request.
    let mut resume_from: u64 = 0;
    if resume {
        if let Ok(meta) = fs::metadata(&tmp).await {
            let existing = meta.len();
            if existing < chunk.size {
                resume_from = existing;
            } else {
                // Partial file is larger than expected — discard, redo.
                let _ = fs::remove_file(&tmp).await;
            }
        }
    }

    let mut req = client.get(url);
    if resume_from > 0 {
        req = req.header("Range", format!("bytes={resume_from}-"));
        debug!(cid = %chunk.cid, from = resume_from, "resuming chunk");
    }
    let resp = req.send().await
        .with_context(|| format!("GET chunk {}", chunk.cid))?
        .error_for_status()
        .with_context(|| format!("chunk {} HTTP error", chunk.cid))?;

    // If the server ignored our Range and sent the whole body (status
    // 200 instead of 206), start the hasher fresh and truncate the
    // temp file. Otherwise, fold the existing bytes into the hasher
    // so the final digest matches whole-file contents.
    let mut hasher = Sha256::new();
    let mut f = if resp.status() == reqwest::StatusCode::PARTIAL_CONTENT && resume_from > 0 {
        let mut existing = fs::File::options().read(true).write(true).open(&tmp).await?;
        let mut buf = vec![0u8; 64 * 1024];
        let mut read_so_far: u64 = 0;
        use tokio::io::AsyncReadExt;
        while read_so_far < resume_from {
            let n = existing.read(&mut buf).await?;
            if n == 0 { break; }
            hasher.update(&buf[..n]);
            read_so_far += n as u64;
        }
        existing.seek(tokio::io::SeekFrom::End(0)).await?;
        existing
    } else {
        fs::File::create(&tmp).await?
    };

    let mut body = resp.bytes_stream();
    let mut total: u64 = resume_from;
    while let Some(chunk_bytes) = body.next().await {
        let b = chunk_bytes.with_context(|| format!("streaming {}", chunk.cid))?;
        hasher.update(&b);
        f.write_all(&b).await?;
        total += b.len() as u64;
    }
    f.flush().await?;
    f.sync_all().await?;
    drop(f);

    if total != chunk.size {
        let _ = fs::remove_file(&tmp).await;
        return Err(anyhow!(
            "chunk {} size mismatch: manifest {}, got {}",
            chunk.cid, chunk.size, total
        ));
    }

    // Reconstruct the same CID string from the hasher and compare.
    let digest = hasher.finalize();
    let actual_cid = cid_string_from_sha256(&digest);
    if actual_cid != chunk.cid {
        let _ = fs::remove_file(&tmp).await;
        return Err(anyhow!(
            "chunk {} hash mismatch: manifest {}, got {}",
            chunk.cid, chunk.cid, actual_cid
        ));
    }

    fs::rename(&tmp, &final_path).await
        .with_context(|| format!("renaming {} to {}", tmp.display(), final_path.display()))?;
    info!(cid = %chunk.cid, size = total, "fetched");
    Ok(FetchOne { size: chunk.size, already_cached: false })
}

/// Turn a manifest URL into a base URL that chunks live under.
/// `https://host/path/manifest.json` → `https://host/path/`.
/// `https://host/manifest.json` → `https://host/`.
fn derive_chunk_base(manifest_url: &str) -> Result<Url> {
    let u = Url::parse(manifest_url).context("parsing manifest URL")?;
    let segments: Vec<String> = u
        .path_segments()
        .map(|it| it.map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let mut base = u.clone();
    if segments.last().map(|s| s == "manifest.json").unwrap_or(false) {
        // Strip the trailing filename. The leading '/' is implicit in
        // set_path, and the trailing '/' must be explicit so Url::join
        // treats the result as a directory rather than replacing its
        // last segment.
        let parent_path = if segments.len() <= 1 {
            "/".to_string()
        } else {
            format!("/{}/", segments[..segments.len() - 1].join("/"))
        };
        base.set_path(&parent_path);
    }
    Ok(base)
}

fn dedup_in_place(chunks: &mut Vec<Chunk>) {
    let mut seen = std::collections::HashSet::new();
    chunks.retain(|c| seen.insert(c.cid.clone()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_plan_for_range() {
        let m = Manifest {
            format: "intelnav-model".into(),
            version: 2,
            name: None,
            architecture: None,
            n_layers: 24,
            gguf: crate::manifest::GgufInfo {
                gguf_version: 3,
                alignment: 32,
                tensor_data_offset: 0,
                n_kv: 0,
                n_tensors: 0,
            },
            header_chunk: Chunk { cid: "h".into(), size: 0 },
            bundles: vec![],
        };
        // Start of model: includes embed + first few blocks, no head.
        if let FetchPlan::Bundles(v) = FetchPlan::for_range(&m, 0, 3) {
            assert_eq!(v, vec!["embed", "blk.0", "blk.1", "blk.2"]);
        } else { panic!("wrong variant"); }
        // End of model: includes last block range + head, no embed.
        if let FetchPlan::Bundles(v) = FetchPlan::for_range(&m, 22, 24) {
            assert_eq!(v, vec!["blk.22", "blk.23", "head"]);
        } else { panic!("wrong variant"); }
        // Middle: neither embed nor head.
        if let FetchPlan::Bundles(v) = FetchPlan::for_range(&m, 5, 8) {
            assert_eq!(v, vec!["blk.5", "blk.6", "blk.7"]);
        } else { panic!("wrong variant"); }
    }

    #[test]
    fn derive_base_strips_manifest_filename() {
        let b = derive_chunk_base("https://example.com/models/qwen/manifest.json").unwrap();
        assert_eq!(b.as_str(), "https://example.com/models/qwen/");
        let b = derive_chunk_base("http://host:8080/x/manifest.json").unwrap();
        assert_eq!(b.as_str(), "http://host:8080/x/");
        // Root-level manifest collapses to host root — the previous
        // bug doubled the leading slash here.
        let b = derive_chunk_base("http://127.0.0.1:8080/manifest.json").unwrap();
        assert_eq!(b.as_str(), "http://127.0.0.1:8080/");
        assert_eq!(b.join("chunks/x.bin").unwrap().as_str(),
                   "http://127.0.0.1:8080/chunks/x.bin");
    }

    #[test]
    fn cid_from_sha256_matches_full_path() {
        let bytes = b"intelnav phase 3 fetch";
        let reference = cid_string_for(bytes);
        let digest = Sha256::digest(bytes);
        let via_prehash = cid_string_from_sha256(&digest);
        assert_eq!(reference, via_prehash);
    }
}
