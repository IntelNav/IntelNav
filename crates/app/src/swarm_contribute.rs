//! Swarm pre-split contribute: pull a single layer slice from a peer
//! that already serves it, persist it locally, and stage the
//! kept-ranges sidecar so the announce loop will publish ourselves
//! as another provider for that slice.
//!
//! This is the *light* on-ramp — a peer joining the network never
//! has to download the full GGUF, only the chunks for the slice
//! they commit to host.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use intelnav_model_store::{
    fetch_chunks, fetch_manifest_only, FetchOptions, FetchPlan,
};
use intelnav_net::ProviderRecord;

use crate::contribute::{shard_dir, KeptRanges};

#[derive(Debug)]
pub enum SwarmPullEvent {
    Started     { manifest_url: String, range: (u16, u16) },
    ManifestOk  { manifest_cid: String, n_layers: u32 },
    ChunksDone  { bytes: u64, n_chunks: usize },
    Done        { kept_ranges: Vec<(u16, u16)>, shard_root: PathBuf },
    Error(String),
}

/// Sidecar describing where this peer's slice came from. Written
/// alongside the manifest so a future operator can audit "I'm
/// hosting layers X..Y of model Z, fetched from peer P".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PullSource {
    pub source_peer_id: String,
    pub manifest_cid:   String,
    pub manifest_url:   String,
    pub range:          (u16, u16),
}

/// Pick the best provider from a SwarmModel's range coverage. Today
/// "best" = most recently minted record with a non-empty
/// `chunks_url` and `manifest_cid`.
pub fn pick_provider(providers: &[ProviderRecord]) -> Option<&ProviderRecord> {
    providers.iter()
        .filter(|p| p.chunks_url.is_some() && p.manifest_cid.is_some())
        .max_by_key(|p| p.minted_at)
}

/// Default range to ask for: the first range with at least one
/// reachable provider (chunks_url + manifest_cid present), so that
/// a single click on "contribute" succeeds without further input.
pub fn default_range(model_cid: &str, ranges: &[(u16, u16, Vec<ProviderRecord>)])
    -> Option<(u16, u16, ProviderRecord)>
{
    let _ = model_cid;
    ranges.iter().find_map(|(s, e, providers)| {
        pick_provider(providers).map(|p| (*s, *e, p.clone()))
    })
}

/// Kick off the slice pull. Returns a receiver of progress events.
///
/// The chunks_url + manifest_cid on the provider record give us the
/// canonical manifest URL `http://<chunks_url>/<manifest_cid>/manifest.json`,
/// which mirrors the chunk-server's on-disk layout.
pub fn start_pull(
    model_cid: String,
    range: (u16, u16),
    provider: ProviderRecord,
    models_dir: PathBuf,
) -> mpsc::UnboundedReceiver<SwarmPullEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    let shard_root = shard_dir(&models_dir, &model_cid);

    tokio::spawn(async move {
        let (chunks_host, manifest_cid) = match (&provider.chunks_url, &provider.manifest_cid) {
            (Some(c), Some(m)) => (c.clone(), m.clone()),
            _ => {
                let _ = tx.send(SwarmPullEvent::Error(
                    "provider record is missing chunks_url or manifest_cid".into(),
                ));
                return;
            }
        };
        // Reconstruct the manifest URL the way the chunk-server lays
        // it out: `<host:port>/<manifest_cid>/manifest.json`.
        let manifest_url = format!(
            "http://{chunks_host}/{manifest_cid}/manifest.json"
        );
        let _ = tx.send(SwarmPullEvent::Started {
            manifest_url: manifest_url.clone(),
            range,
        });

        if let Err(e) = std::fs::create_dir_all(&shard_root) {
            let _ = tx.send(SwarmPullEvent::Error(format!(
                "mkdir {}: {e}", shard_root.display()
            )));
            return;
        }
        let mut opts = FetchOptions::default();
        opts.cache_root = shard_root.clone();

        let fetched = match fetch_manifest_only(&manifest_url, &opts).await {
            Ok(f) => f,
            Err(e) => {
                let _ = tx.send(SwarmPullEvent::Error(format!(
                    "fetch_manifest_only: {e}"
                )));
                return;
            }
        };
        let _ = tx.send(SwarmPullEvent::ManifestOk {
            manifest_cid: fetched.manifest_cid.clone(),
            n_layers:     fetched.manifest.n_layers,
        });

        let plan = FetchPlan::for_range(&fetched.manifest, range.0 as u32, range.1 as u32);
        let outcome = match fetch_chunks(&fetched, &plan, &opts).await {
            Ok(o) => o,
            Err(e) => {
                let _ = tx.send(SwarmPullEvent::Error(format!("fetch_chunks: {e}")));
                return;
            }
        };
        let _ = tx.send(SwarmPullEvent::ChunksDone {
            bytes:     outcome.bytes_downloaded,
            n_chunks:  outcome.manifest.bundles.len(),
        });

        // Persist sidecars: kept_ranges (announce loop reads this)
        // and pull_source (audit / debugging).
        let kept = KeptRanges {
            model_cid:    model_cid.clone(),
            display_name: fetched.manifest.name.clone()
                .unwrap_or_else(|| model_cid.clone()),
            block_count:  fetched.manifest.n_layers as u16,
            gguf_path:    PathBuf::new(),  // no local GGUF — chunks only
            kept:         vec![range],
        };
        let kept_path = shard_root.join("kept_ranges.json");
        if let Err(e) = std::fs::write(
            &kept_path,
            serde_json::to_vec_pretty(&kept).unwrap_or_default(),
        ) {
            let _ = tx.send(SwarmPullEvent::Error(format!(
                "write {}: {e}", kept_path.display()
            )));
            return;
        }

        let source = PullSource {
            source_peer_id: provider.peer_id.clone(),
            manifest_cid:   fetched.manifest_cid.clone(),
            manifest_url,
            range,
        };
        let src_path = shard_root.join("pull_source.json");
        let _ = std::fs::write(
            &src_path,
            serde_json::to_vec_pretty(&source).unwrap_or_default(),
        );

        let _ = tx.send(SwarmPullEvent::Done {
            kept_ranges: vec![range],
            shard_root,
        });
    });
    rx
}
