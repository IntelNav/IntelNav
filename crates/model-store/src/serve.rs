//! Tiny HTTP server that exposes a chunk directory.
//!
//! Serves two things under the given directory root:
//!
//! * `GET /manifest.json` — the manifest for this model.
//! * `GET /chunks/<cid>.bin` — the chunk bytes.
//!
//! This is deliberately minimal. It exists so integration tests can
//! stand up a real HTTP endpoint without external dependencies, and
//! so a dev running `intelnav-chunk serve ./my-model/` has a one-
//! command way to share a model on their LAN. Production peers would
//! point at nginx/S3/the libp2p swarm instead.
//!
//! Compiled only with the `serve` feature so the default build
//! doesn't drag in axum.

#![cfg(feature = "serve")]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{Path as AxPath, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

#[derive(Clone)]
struct AppState {
    root: PathBuf,
}

/// Bind on `addr` and serve `root` until the task is cancelled.
/// Returns the actual bound address (useful when caller passed :0).
pub async fn serve(root: impl AsRef<Path>, addr: SocketAddr) -> Result<()> {
    let (_bound, fut) = serve_bound(root, addr).await?;
    fut.await
}

/// Multi-shard chunk server.
///
/// Serves every manifest+chunks directory under `shards_root` at
/// `/{manifest_cid}/manifest.json` and `/{manifest_cid}/chunks/<cid>.bin`.
/// `manifest_cid` is derived by hashing the manifest.json bytes (or
/// taken from `pull_source.json` if present).
///
/// The map is rebuilt lazily on cache miss so a freshly-added shard
/// becomes serveable without restarting the server.
pub async fn serve_multi(
    shards_root: impl AsRef<Path>,
    addr: SocketAddr,
) -> Result<()> {
    let (_bound, fut) = serve_multi_bound(shards_root, addr).await?;
    fut.await
}

pub async fn serve_multi_bound(
    shards_root: impl AsRef<Path>,
    addr: SocketAddr,
) -> Result<(SocketAddr, impl std::future::Future<Output = Result<()>> + 'static)> {
    let shards_root = shards_root.as_ref().to_path_buf();
    if !shards_root.exists() {
        std::fs::create_dir_all(&shards_root)
            .with_context(|| format!("create {}", shards_root.display()))?;
    }
    let state = MultiState { shards_root };
    let app = Router::new()
        .route("/:manifest_cid/manifest.json", get(serve_multi_manifest))
        .route("/:manifest_cid/chunks/:chunk", get(serve_multi_chunk))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = TcpListener::bind(addr).await
        .with_context(|| format!("binding {addr}"))?;
    let bound = listener.local_addr()?;
    info!(%bound, "intelnav-chunk multi-shard serving");
    let fut = async move {
        axum::serve(listener, app).await
            .context("axum serve exited")?;
        Ok(())
    };
    Ok((bound, fut))
}

#[derive(Clone)]
struct MultiState {
    shards_root: PathBuf,
}

impl MultiState {
    /// Locate the on-disk shard root whose manifest_cid matches.
    /// Linear scan over `<shards_root>/*/`. Cheap (<100 shards typical).
    fn resolve_shard(&self, manifest_cid: &str) -> Option<PathBuf> {
        let rd = std::fs::read_dir(&self.shards_root).ok()?;
        for entry in rd.flatten() {
            let dir = entry.path();
            // Prefer pull_source.json when present (cheapest).
            if let Ok(bytes) = std::fs::read(dir.join("pull_source.json")) {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    if v.get("manifest_cid").and_then(|s| s.as_str()) == Some(manifest_cid) {
                        return Some(dir);
                    }
                }
            }
            // Otherwise hash the manifest bytes.
            let manifest_path = dir.join("manifest.json");
            let Ok(bytes) = std::fs::read(&manifest_path) else { continue; };
            let cid = crate::cid::cid_string_for(&bytes);
            if cid == manifest_cid { return Some(dir); }
        }
        None
    }
}

async fn serve_multi_manifest(
    State(state): State<MultiState>,
    AxPath(manifest_cid): AxPath<String>,
) -> Response {
    let Some(root) = state.resolve_shard(&manifest_cid) else {
        return (StatusCode::NOT_FOUND, "unknown manifest").into_response();
    };
    let path = root.join("manifest.json");
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            bytes,
        ).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, format!("read manifest: {e}")).into_response(),
    }
}

async fn serve_multi_chunk(
    State(state): State<MultiState>,
    AxPath((manifest_cid, chunk)): AxPath<(String, String)>,
) -> Response {
    if !is_valid_chunk_filename(&chunk) {
        return (StatusCode::BAD_REQUEST, "bad chunk id").into_response();
    }
    let Some(root) = state.resolve_shard(&manifest_cid) else {
        return (StatusCode::NOT_FOUND, "unknown manifest").into_response();
    };
    let path = root.join("chunks").join(&chunk);
    match tokio::fs::File::open(&path).await {
        Ok(file) => {
            let meta = match file.metadata().await {
                Ok(m) => m,
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("stat: {e}")).into_response(),
            };
            let len = meta.len();
            let stream = tokio_util::io::ReaderStream::new(file);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header(header::CONTENT_LENGTH, len)
                .header(header::ACCEPT_RANGES, "bytes")
                .body(Body::from_stream(stream))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(_) => (StatusCode::NOT_FOUND, "no such chunk").into_response(),
    }
}

/// Like [`serve`] but returns the bound address synchronously so
/// integration tests can read it before the server starts accepting.
/// The second tuple element is the serve future; `.await` it to run.
pub async fn serve_bound(
    root: impl AsRef<Path>,
    addr: SocketAddr,
) -> Result<(SocketAddr, impl std::future::Future<Output = Result<()>> + 'static)> {
    let root = root.as_ref().to_path_buf();
    if !root.exists() {
        anyhow::bail!("serve root does not exist: {}", root.display());
    }
    if !root.join("manifest.json").exists() {
        anyhow::bail!(
            "no manifest.json in {}; did you run `intelnav-chunk chunk` first?",
            root.display()
        );
    }

    let state = AppState { root };
    let app = Router::new()
        .route("/manifest.json", get(serve_manifest))
        .route("/chunks/:cid", get(serve_chunk))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = TcpListener::bind(addr).await
        .with_context(|| format!("binding {addr}"))?;
    let bound = listener.local_addr()?;
    info!(root = %bound, "intelnav-chunk serving");

    let fut = async move {
        axum::serve(listener, app).await
            .context("axum serve exited")?;
        Ok(())
    };
    Ok((bound, fut))
}

async fn serve_manifest(State(state): State<AppState>) -> Response {
    let path = state.root.join("manifest.json");
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            bytes,
        ).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, format!("read manifest: {e}")).into_response(),
    }
}

/// Tight filter for `<cid>.bin` filenames: lowercase base32
/// ([a-z2-7]) CID body, at least the minimum length a base32 CIDv1
/// reaches (~36 chars), plus the `.bin` suffix. CIDv1 cannot contain
/// uppercase, dots, slashes, or NULs under our codec/multihash
/// choice, so anything else is abuse.
fn is_valid_chunk_filename(s: &str) -> bool {
    let Some(stem) = s.strip_suffix(".bin") else { return false; };
    // Minimum base32 body for sha256+cidv1 is ~59 chars; 32 is a
    // conservative floor that still rejects obvious garbage without
    // being tightly coupled to the exact CID-string length.
    if stem.len() < 32 { return false; }
    stem.bytes().all(|b| matches!(b, b'a'..=b'z' | b'2'..=b'7'))
}

async fn serve_chunk(
    State(state): State<AppState>,
    AxPath(cid_bin): AxPath<String>,
) -> Response {
    // Strict allow-list: CIDv1-base32 filenames only. `bafkrei...`
    // followed by `.bin`. Anything else is rejected without opening
    // a file handle — a hostile peer can't path-traverse, can't hit
    // symlinks in the chunks dir (which shouldn't exist but we
    // assume nothing), and can't probe filesystem layout.
    if !is_valid_chunk_filename(&cid_bin) {
        return (StatusCode::BAD_REQUEST, "bad chunk id").into_response();
    }
    let path = state.root.join("chunks").join(&cid_bin);

    match tokio::fs::File::open(&path).await {
        Ok(file) => {
            let meta = match file.metadata().await {
                Ok(m) => m,
                Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("stat: {e}")).into_response(),
            };
            let len = meta.len();
            // tokio_util's ReaderStream turns an AsyncRead into a Stream<Bytes>.
            let stream = tokio_util::io::ReaderStream::new(file);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header(header::CONTENT_LENGTH, len)
                .header(header::ACCEPT_RANGES, "bytes") // fetcher's range-resume hint
                .body(Body::from_stream(stream))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(_) => (StatusCode::NOT_FOUND, "no such chunk").into_response(),
    }
}
