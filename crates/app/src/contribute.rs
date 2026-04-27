//! Hub → split → host: turn a downloaded GGUF into per-bundle chunks
//! and persist the (cid, kept-ranges) sidecar the announce loop will
//! later publish to the DHT.
//!
//! The flow runs as a single non-blocking tokio task because chunking
//! a multi-GB GGUF is CPU-bound — `spawn_blocking` keeps the runtime
//! responsive. Progress is reported on an unbounded channel so the
//! TUI can show stage transitions without polling.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use intelnav_model_store::{chunk_gguf, ChunkerOptions};

use crate::catalog::CatalogEntry;
use crate::local::LocalModel;

/// Progress events emitted while splitting a GGUF into shards.
#[derive(Debug)]
pub enum SplitEvent {
    Started   { gguf: PathBuf, output: PathBuf },
    Done      {
        manifest_cid: String,
        n_bundles:    usize,
        bytes:        u64,
        kept_ranges:  Vec<(u16, u16)>,
        shard_root:   PathBuf,
    },
    Error(String),
}

/// Sidecar written next to the manifest — the announce loop reads this
/// to know which (start, end) pairs to publish as provider records.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeptRanges {
    pub model_cid:   String,
    pub display_name: String,
    pub block_count: u16,
    pub gguf_path:   PathBuf,
    pub kept:        Vec<(u16, u16)>,
}

/// Default kept-range policy: when the catalog gives us standard
/// splits, the user offers to host *one* of them (the first by
/// default — TUI can prompt later). When no splits are declared,
/// the peer takes the whole model.
fn default_kept_ranges(entry: &CatalogEntry) -> Vec<(u16, u16)> {
    if entry.default_splits.is_empty() {
        vec![(0, entry.block_count)]
    } else {
        vec![entry.default_splits[0]]
    }
}

/// Where shards for `cid` live. One subdir per model under the
/// shards root keeps multi-model peers tidy.
pub fn shard_dir(models_dir: &std::path::Path, cid: &str) -> PathBuf {
    models_dir.join(".shards").join(cid)
}

/// Kick off the split. Returns a receiver of progress events.
///
/// `local` is the cached GGUF that's already been downloaded — the
/// caller must guarantee the file exists; this fn doesn't touch the
/// network.
pub fn start_split(
    entry: &'static CatalogEntry,
    local: LocalModel,
    models_dir: PathBuf,
) -> mpsc::UnboundedReceiver<SplitEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    let cid = entry.model_cid();
    let display = entry.display_name.to_string();
    let block_count = entry.block_count;
    let kept = default_kept_ranges(entry);
    let shard_root = shard_dir(&models_dir, &cid);

    tokio::task::spawn_blocking(move || {
        let _ = tx.send(SplitEvent::Started {
            gguf: local.path.clone(),
            output: shard_root.clone(),
        });
        if let Err(e) = std::fs::create_dir_all(&shard_root) {
            let _ = tx.send(SplitEvent::Error(format!("mkdir {}: {e}", shard_root.display())));
            return;
        }
        let mut opts = ChunkerOptions::new(shard_root.clone());
        opts.overwrite = true;
        let outcome = match chunk_gguf(&local.path, &opts) {
            Ok(o)  => o,
            Err(e) => {
                let _ = tx.send(SplitEvent::Error(format!("chunk_gguf: {e}")));
                return;
            }
        };

        // Write the kept-ranges sidecar. Future announce loop reads
        // this to decide which (start, end) keys to publish.
        let sidecar = KeptRanges {
            model_cid:    cid.clone(),
            display_name: display,
            block_count,
            gguf_path:    local.path.clone(),
            kept:         kept.clone(),
        };
        let path = shard_root.join("kept_ranges.json");
        if let Err(e) = std::fs::write(
            &path,
            serde_json::to_vec_pretty(&sidecar).unwrap_or_default(),
        ) {
            let _ = tx.send(SplitEvent::Error(format!("write {}: {e}", path.display())));
            return;
        }

        let _ = tx.send(SplitEvent::Done {
            manifest_cid: outcome.manifest_cid,
            n_bundles:    outcome.n_bundles,
            bytes:        outcome.bytes_written,
            kept_ranges:  kept,
            shard_root,
        });
    });
    rx
}
