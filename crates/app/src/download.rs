//! Streaming file download with progress events.
//!
//! Emits `Event::Progress` at ~10 Hz plus `Done` / `Error` terminals
//! over an unbounded mpsc. Writes to `{dest}.part` and renames on
//! successful completion so an interrupted download never leaves a
//! half-written file pretending to be real.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum Event {
    /// Periodic update. `total` is None when the server didn't send
    /// a content-length (rare on HF).
    Progress { label: String, done: u64, total: Option<u64>, bps: f64 },
    /// Final file landed at this path.
    Done     { label: String, path: PathBuf },
    Error    { label: String, message: String },
}

/// Kick off a download in the background. Returns the receiver half
/// of the event stream. Kept for arbitrary-URL flows (operator
/// pasting a custom GGUF link); the curated-catalog path uses
/// [`download_catalog`] instead.
#[allow(dead_code)]
pub fn spawn(url: String, dest: PathBuf, label: String) -> mpsc::UnboundedReceiver<Event> {
    let (tx, rx) = mpsc::unbounded_channel();
    let label2 = label.clone();
    tokio::spawn(async move {
        match download(&url, &dest, &label, &tx).await {
            Ok(path) => { let _ = tx.send(Event::Done { label: label2, path }); }
            Err(e)   => { let _ = tx.send(Event::Error { label: label2, message: e.to_string() }); }
        }
    });
    rx
}

async fn download(
    url:   &str,
    dest:  &Path,
    label: &str,
    tx:    &mpsc::UnboundedSender<Event>,
) -> Result<PathBuf> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await
            .with_context(|| format!("mkdir {}", parent.display()))?;
    }

    let client = reqwest::Client::builder()
        .user_agent(concat!("intelnav-cli/", env!("CARGO_PKG_VERSION")))
        // HF redirects through a CDN; reqwest follows by default.
        .build()
        .context("http client")?;

    let resp = client.get(url).send().await
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("{} returned {}", url, resp.status()));
    }
    let total = resp.content_length();

    let part = dest.with_extension({
        let mut s = dest.extension().and_then(|e| e.to_str()).unwrap_or("").to_string();
        if !s.is_empty() { s.push('.'); }
        s.push_str("part");
        s
    });
    let mut file = tokio::fs::File::create(&part).await
        .with_context(|| format!("create {}", part.display()))?;

    let mut stream = resp.bytes_stream();
    let mut done: u64 = 0;
    let start     = Instant::now();
    let mut last_emit = Instant::now() - Duration::from_secs(1);
    let mut last_done = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("network read")?;
        file.write_all(&chunk).await.context("file write")?;
        done = done.saturating_add(chunk.len() as u64);

        if last_emit.elapsed() >= Duration::from_millis(120) {
            let dt = last_emit.elapsed().as_secs_f64().max(1e-3);
            let bps = (done - last_done) as f64 / dt;
            let _ = tx.send(Event::Progress {
                label: label.to_string(),
                done, total, bps,
            });
            last_emit = Instant::now();
            last_done = done;
        }
    }
    file.flush().await.ok();
    drop(file);

    tokio::fs::rename(&part, dest).await
        .with_context(|| format!("rename {} → {}", part.display(), dest.display()))?;

    // One final progress tick so the UI can show 100%.
    let dt = start.elapsed().as_secs_f64().max(1e-3);
    let _ = tx.send(Event::Progress {
        label: label.to_string(),
        done, total: Some(done), bps: done as f64 / dt,
    });

    Ok(dest.to_path_buf())
}

/// Download a catalog entry's two sidecar files, emitting progress
/// events from both serially (GGUF first — it's the big one).
pub fn install_entry(
    entry: &'static crate::catalog::CatalogEntry,
    models_dir: PathBuf,
) -> mpsc::UnboundedReceiver<Event> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let gguf_dest = models_dir.join(entry.gguf_file);
        let tok_dest  = models_dir.join(format!("{}.tokenizer.json",
            entry.gguf_file.trim_end_matches(".gguf")));

        // 1. Tokenizer first — it's tiny; failing early is cheaper.
        let tok_label = format!("{} · tokenizer", entry.display_name);
        if let Err(e) = download(&entry.tokenizer_url(), &tok_dest, &tok_label, &tx).await {
            let _ = tx.send(Event::Error { label: tok_label, message: e.to_string() });
            return;
        }

        // 2. GGUF.
        let gguf_label = format!("{} · weights", entry.display_name);
        let path = match download(&entry.gguf_url(), &gguf_dest, &gguf_label, &tx).await {
            Ok(p)  => p,
            Err(e) => {
                let _ = tx.send(Event::Error { label: gguf_label, message: e.to_string() });
                return;
            }
        };
        let _ = tx.send(Event::Done { label: entry.display_name.into(), path });
    });
    rx
}
