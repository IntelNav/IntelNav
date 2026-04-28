//! N-peer pipeline driver for `RunMode::Network` when the user has
//! configured a `[peers]` list.
//!
//! Mirrors the `Delta` stream shape used by the local driver so the
//! TUI's consumer code doesn't care which backend is in play.
//!
//! One chain per turn: open → prefill → decode → close. A fresh
//! connection per turn costs ~1ms on LAN and gives us free resilience
//! to peer restarts. If any peer drops mid-turn, the error surfaces
//! through `Delta::Error` tagged with the offending peer index, and
//! the next turn tries to reopen the chain cleanly.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use tokio::sync::mpsc;

use intelnav_runtime::{
    build_chat_prompt, run_turn, run_turn_spec, Chain, ChainCfg, ChainError, ChatTurn,
    DevicePref, Dtype, ModelHandle, ModelKind, SamplingCfg, SpecCfg, Tok,
};

use crate::delta::{ChatMessage, Delta};
use crate::local::LocalModel;

/// Configured chain target with per-hop failover candidates.
///
/// `peers[i]` is the *currently active* socket for hop `i`;
/// `alternates[i]` holds backup providers ranked by latency (freshest
/// first). [`failover`] swaps `peers[i]` in and out of `alternates[i]`
/// so the driver can retry a connect without rebuilding the whole plan.
#[derive(Clone, Debug)]
pub struct ChainTarget {
    pub peers:  Vec<SocketAddr>,
    pub splits: Vec<u16>,
    pub alternates: Vec<Vec<SocketAddr>>,
}

/// Speculative-decoding draft config: path to a small GGUF (Qwen2
/// family) plus the per-round proposal count.
#[derive(Clone, Debug)]
pub struct DraftTarget {
    pub path: PathBuf,
    pub k:    usize,
}

impl DraftTarget {
    pub fn summary(&self) -> String {
        let name = self.path.file_name()
            .and_then(|s| s.to_str()).unwrap_or("<draft>");
        format!("{name} · k={}", self.k)
    }
}

impl ChainTarget {
    /// Parse `peers = ["a:7717", "b:7717"]` + `splits = [6, 12]` from
    /// the raw config shape into something `Chain::connect` accepts.
    /// Returns an informative error per field so the TUI can surface
    /// it in the status bar instead of silently falling back.
    pub fn from_config(peers: &[String], splits: &[u16]) -> Result<Self> {
        if peers.is_empty() {
            return Err(anyhow!("no peers configured"));
        }
        if peers.len() != splits.len() {
            return Err(anyhow!(
                "peers ({}) and splits ({}) length mismatch",
                peers.len(), splits.len()
            ));
        }
        let mut parsed = Vec::with_capacity(peers.len());
        for (i, p) in peers.iter().enumerate() {
            let addr: SocketAddr = p.parse()
                .with_context(|| format!("peer[{i}] = `{p}` is not a valid host:port"))?;
            parsed.push(addr);
        }
        let alternates = vec![Vec::new(); parsed.len()];
        Ok(Self { peers: parsed, splits: splits.to_vec(), alternates })
    }

    /// Build a chain target by ranking providers per range.
    ///
    /// For each range we keep up to [`MAX_CANDIDATES_PER_HOP`] providers,
    /// freshest-first, and put the top one in `peers` and the rest in
    /// `alternates`. This is what enables [`failover`] without a fresh
    /// DHT round trip.
    ///
    /// This is a *synchronous* fallback ranking by `minted_at`. For
    /// latency-aware ranking call [`ChainTarget::from_swarm_with_probe`]
    /// instead — it issues a parallel TCP probe and sorts by RTT.
    pub fn from_swarm(
        ranges: &[(u16, u16, Vec<intelnav_net::ProviderRecord>)],
    ) -> Result<Self> {
        if ranges.is_empty() {
            return Err(anyhow!("no ranges to assemble"));
        }
        let mut peers      = Vec::with_capacity(ranges.len());
        let mut splits     = Vec::with_capacity(ranges.len());
        let mut alternates = Vec::with_capacity(ranges.len());
        for (start, _end, providers) in ranges {
            // Rank providers freshest-first by minted_at, parse the
            // forward_url, dedupe so a peer republishing twice doesn't
            // waste a slot, cap at MAX_CANDIDATES_PER_HOP.
            let mut ranked: Vec<(u64, SocketAddr)> = providers.iter()
                .filter_map(|p| {
                    let u = p.forward_url.as_ref()?;
                    let addr: SocketAddr = u.parse().ok()?;
                    Some((p.minted_at, addr))
                })
                .collect();
            ranked.sort_by(|a, b| b.0.cmp(&a.0));
            let mut sockets: Vec<SocketAddr> = ranked.into_iter()
                .map(|(_, a)| a)
                .take(MAX_CANDIDATES_PER_HOP)
                .collect();
            sockets.dedup();
            if sockets.is_empty() {
                return Err(anyhow!(
                    "no provider with a forward_url for layers [{start}..{_end})"
                ));
            }
            let primary = sockets.remove(0);
            peers.push(primary);
            splits.push(*start);
            alternates.push(sockets);
        }
        Ok(Self { peers, splits, alternates })
    }

    /// Promote the next alternate for hop `i` into `peers[i]`. Returns
    /// `true` if a swap happened, `false` if no alternates remain.
    pub fn failover(&mut self, i: usize) -> bool {
        let Some(alts) = self.alternates.get_mut(i) else { return false; };
        if alts.is_empty() { return false; }
        let next = alts.remove(0);
        if let Some(slot) = self.peers.get_mut(i) {
            *slot = next;
            return true;
        }
        false
    }

    /// True when at least one hop still has a backup we haven't tried.
    pub fn has_alternates(&self) -> bool {
        self.alternates.iter().any(|a| !a.is_empty())
    }

    pub fn summary(&self) -> String {
        let peers: Vec<String> = self.peers.iter().map(|a| a.to_string()).collect();
        let alts: usize = self.alternates.iter().map(|a| a.len()).sum();
        if alts == 0 {
            format!("{} · splits={:?}", peers.join(","), self.splits)
        } else {
            format!("{} · splits={:?} · {alts} alt", peers.join(","), self.splits)
        }
    }
}

impl ChainTarget {
    /// Latency-aware version of [`from_swarm`]. Probes every candidate
    /// in parallel via a TCP handshake (cached for 60s), then ranks
    /// reachable peers by RTT and uses freshness as a tiebreaker.
    ///
    /// Picks "fastest reachable peer" as primary, with the rest of the
    /// reachable peers (RTT-ranked) as alternates. Unreachable peers
    /// drop off the list — there's no point keeping them.
    pub async fn from_swarm_with_probe(
        ranges: &[(u16, u16, Vec<intelnav_net::ProviderRecord>)],
    ) -> Result<Self> {
        if ranges.is_empty() {
            return Err(anyhow!("no ranges to assemble"));
        }
        let mut peers      = Vec::with_capacity(ranges.len());
        let mut splits     = Vec::with_capacity(ranges.len());
        let mut alternates = Vec::with_capacity(ranges.len());

        for (start, _end, providers) in ranges {
            let mut candidates: Vec<(u64, SocketAddr)> = providers.iter()
                .filter_map(|p| {
                    let u = p.forward_url.as_ref()?;
                    let addr: SocketAddr = u.parse().ok()?;
                    Some((p.minted_at, addr))
                })
                .collect();
            candidates.sort_by(|a, b| b.0.cmp(&a.0));
            let addrs: Vec<SocketAddr> = candidates.iter().map(|(_, a)| *a).collect();
            let probe_results = crate::probe_latency::probe_many(&addrs).await;
            // Pair each addr with its probe + freshness rank, then sort by
            // (reachable, lowest RTT, freshest).
            let mut ranked: Vec<(u64, SocketAddr)> = addrs.iter().zip(probe_results.iter())
                .zip(candidates.iter())
                .filter_map(|((addr, p), (mint, _))| {
                    p.rtt.map(|_rtt| (crate::probe_latency::score(p), *addr, *mint))
                })
                .map(|(score, addr, _mint)| (score, addr))
                .collect();
            ranked.sort_by_key(|(score, _)| *score);
            let mut sockets: Vec<SocketAddr> = ranked.into_iter()
                .map(|(_, a)| a)
                .take(MAX_CANDIDATES_PER_HOP)
                .collect();
            sockets.dedup();
            if sockets.is_empty() {
                return Err(anyhow!(
                    "no reachable provider for layers [{start}..{_end})"
                ));
            }
            let primary = sockets.remove(0);
            peers.push(primary);
            splits.push(*start);
            alternates.push(sockets);
        }
        Ok(Self { peers, splits, alternates })
    }
}

/// Cap on candidates kept per hop. 1 primary + 2 alternates is enough
/// in practice — if all three fail something bigger is wrong.
pub const MAX_CANDIDATES_PER_HOP: usize = 3;

/// Cap on how many failover swaps we try inside one connect attempt.
const MAX_FAILOVER_ATTEMPTS: usize = 4;

/// Cached-model driver — same shape as `LocalDriver` but its forward
/// pass funnels through a TCP peer chain.
#[derive(Clone)]
pub struct ChainDriver {
    inner:       Arc<Mutex<Option<Loaded>>>,
    device_pref: DevicePref,
    target:      Arc<Mutex<Option<ChainTarget>>>,
    draft:       Arc<Mutex<Option<DraftTarget>>>,
    draft_slot:  Arc<Mutex<Option<LoadedDraft>>>,
    wire_dtype:  Arc<Mutex<Dtype>>,
}

/// Parse a user-facing dtype name into the wire enum. Unknown values
/// fall back to Fp16 (the safe default). Returns `Ok(dtype, name_used)`
/// so the caller can surface "parsed as fp16" in the status bar.
pub fn parse_wire_dtype(s: &str) -> (Dtype, &'static str) {
    match s.trim().to_ascii_lowercase().as_str() {
        "int8" | "i8"  => (Dtype::Int8, "int8"),
        "fp16" | "f16" => (Dtype::Fp16, "fp16"),
        _              => (Dtype::Fp16, "fp16"),
    }
}

struct Loaded {
    path:   PathBuf,
    handle: ModelHandle,
    tok:    Tok,
    kind:   ModelKind,
}

struct LoadedDraft {
    path:   PathBuf,
    handle: ModelHandle,
}

impl ChainDriver {
    pub fn new(device_pref: DevicePref) -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
            device_pref,
            target: Arc::new(Mutex::new(None)),
            draft:  Arc::new(Mutex::new(None)),
            draft_slot: Arc::new(Mutex::new(None)),
            wire_dtype: Arc::new(Mutex::new(Dtype::Fp16)),
        }
    }

    pub fn set_wire_dtype(&self, dtype: Dtype) {
        *self.wire_dtype.lock().unwrap() = dtype;
    }

    pub fn wire_dtype(&self) -> Dtype {
        *self.wire_dtype.lock().unwrap()
    }

    pub fn set_target(&self, target: Option<ChainTarget>) {
        *self.target.lock().unwrap() = target;
    }

    pub fn target(&self) -> Option<ChainTarget> {
        self.target.lock().unwrap().clone()
    }

    pub fn set_draft(&self, draft: Option<DraftTarget>) {
        // Evict any cached draft handle when the config changes — a
        // fresh path needs a fresh load.
        *self.draft_slot.lock().unwrap() = None;
        *self.draft.lock().unwrap() = draft;
    }

    pub fn draft(&self) -> Option<DraftTarget> {
        self.draft.lock().unwrap().clone()
    }

    /// Stream a reply through the chain. Caller guarantees the model
    /// file is loadable locally (the driver owns the front slice +
    /// head, so it needs a full copy of the GGUF too — same as every
    /// peer in this M1 smoke).
    pub fn stream(
        &self,
        model:    LocalModel,
        messages: Vec<ChatMessage>,
        cfg:      SamplingCfg,
    ) -> mpsc::UnboundedReceiver<Delta> {
        let (tx, rx) = mpsc::unbounded_channel();
        let driver = self.clone();

        // The chain is async (tokio TCP) while forward passes are
        // sync blocking (ggml forward). Run the whole turn on a dedicated
        // single-thread runtime so we can freely await + compute in
        // sequence without blocking the main runtime's worker.
        std::thread::spawn(move || {
            // 2 worker threads so spec-dec can overlap the draft
            // forward (sync CPU on a blocking-pool thread) with the
            // commit chain step (async network I/O on a worker). The
            // former serial current_thread runtime forced them to
            // alternate.
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Delta::Error(format!("spawn rt: {e}")));
                    return;
                }
            };
            if let Err(e) = rt.block_on(driver.run(model, messages, cfg, tx.clone())) {
                let _ = tx.send(Delta::Error(e.to_string()));
            }
        });
        rx
    }

    async fn run(
        self,
        model:    LocalModel,
        messages: Vec<ChatMessage>,
        cfg:      SamplingCfg,
        tx:       mpsc::UnboundedSender<Delta>,
    ) -> Result<()> {
        let mut target = self.target.lock().unwrap().clone()
            .ok_or_else(|| anyhow!(
                "no peer chain configured — use /peers to set one"
            ))?;
        let draft_cfg = self.draft.lock().unwrap().clone();

        let kind = self.ensure(&model)?;
        let turns: Vec<ChatTurn<'_>> = messages.iter()
            .map(|m| ChatTurn { role: m.role.as_str(), content: m.content.as_str() })
            .collect();
        let prompt = build_chat_prompt(kind, &turns);

        // Load the draft model if spec-dec is enabled. Held in its
        // own slot so target and draft caches don't fight for one lock.
        if let Some(ref dc) = draft_cfg {
            self.ensure_draft(dc)?;
        }

        // Hold the model mutex for the full turn — concurrent turns
        // would corrupt the shared KV cache on the front slice anyway.
        let mut slot = self.inner.lock().unwrap();
        let loaded = slot.as_mut().ok_or_else(|| anyhow!("model unloaded mid-flight"))?;
        let n_blocks = loaded.handle.block_count() as u16;

        let mut chain = connect_with_failover(
            &mut target,
            &model,
            n_blocks,
            *self.wire_dtype.lock().unwrap(),
            tx.clone(),
        ).await?;
        // Persist any swap so subsequent turns reuse the working route.
        *self.target.lock().unwrap() = Some(target);

        let tx_cb = tx.clone();
        let run_result = if let Some(dc) = draft_cfg {
            // Spec-dec path. `run_turn_spec` takes the draft handle by
            // value (overlap task needs `'static` ownership) so we lift
            // the whole `LoadedDraft` out of its slot for the turn and
            // put it back after, whether the turn succeeded or errored.
            let spec_cfg = SamplingCfg {
                temperature:    0.0,
                top_p:          None,
                repeat_penalty: 1.0,
                ..cfg.clone()
            };
            let mut loaded_draft = {
                let mut slot = self.draft_slot.lock().unwrap();
                slot.take().ok_or_else(|| anyhow!("draft unloaded mid-flight"))?
            };
            let spec_result = run_turn_spec(
                &mut loaded.handle,
                &loaded.tok,
                loaded_draft.handle,
                &mut chain,
                &prompt,
                &spec_cfg,
                &SpecCfg { k: dc.k },
                move |txt| {
                    tx_cb.send(Delta::Token(txt.to_string()))
                        .map_err(|_| anyhow!("receiver dropped"))?;
                    Ok(())
                },
            ).await;
            match spec_result {
                Ok((n, draft_back)) => {
                    loaded_draft.handle = draft_back;
                    *self.draft_slot.lock().unwrap() = Some(loaded_draft);
                    Ok(n)
                }
                Err(e) => {
                    // Handle was consumed on error; evict the slot so
                    // the next turn reloads from disk rather than
                    // racing with a partially-torn-down KV cache.
                    Err(e)
                }
            }
        } else {
            run_turn(
                &mut loaded.handle,
                &loaded.tok,
                &mut chain,
                &prompt,
                &cfg,
                move |txt| {
                    tx_cb.send(Delta::Token(txt.to_string()))
                        .map_err(|_| anyhow!("receiver dropped"))?;
                    Ok(())
                },
            ).await
        };

        let close_reason = if run_result.is_ok() { "turn complete" } else { "driver error" };
        chain.close(close_reason).await;

        run_result?;
        let _ = tx.send(Delta::Done);
        Ok(())
    }

}

/// Try to open a chain, swapping in alternates for any hop whose connect
/// fails. Returns the live `Chain` on success, or the last error after
/// exhausting [`MAX_FAILOVER_ATTEMPTS`] swaps.
///
/// Surfaces each failover attempt as a `Delta::Status` so the TUI can
/// show "swapping peer 2 → backup" without a separate log channel.
async fn connect_with_failover(
    target: &mut ChainTarget,
    model:  &LocalModel,
    n_blocks: u16,
    wire_dtype: Dtype,
    tx: mpsc::UnboundedSender<Delta>,
) -> Result<Chain> {
    let mut attempts = 0;
    loop {
        let cfg = ChainCfg {
            peers:           target.peers.clone(),
            splits:          target.splits.clone(),
            proto_ver:       1,
            model_cid:       model.name.clone(),
            max_seq:         2048,
            step_timeout:    Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            wire_dtype,
        };
        match Chain::connect(cfg, n_blocks).await {
            Ok(chain) => return Ok(chain),
            Err(e) => {
                attempts += 1;
                let failed_index = match &e {
                    ChainError::Connect   { index, .. }
                    | ChainError::Timeout   { index, .. }
                    | ChainError::Handshake { index, .. }
                    | ChainError::Aborted   { index, .. }
                    | ChainError::Dropped   { index, .. } => Some(*index),
                    ChainError::Other(_) => None,
                };
                let Some(i) = failed_index else { return Err(anyhow!("{e}")); };
                if attempts > MAX_FAILOVER_ATTEMPTS || !target.failover(i) {
                    return Err(anyhow!("{e}"));
                }
                let _ = tx.send(Delta::Token(format!(
                    "[swarm] hop {i} unreachable, swapping to backup\n"
                )));
            }
        }
    }
}

impl ChainDriver {
    fn ensure_draft(&self, dc: &DraftTarget) -> Result<()> {
        let mut slot = self.draft_slot.lock().unwrap();
        if let Some(l) = slot.as_ref() {
            if l.path == dc.path { return Ok(()); }
        }
        let handle = ModelHandle::load(&dc.path, self.device_pref)
            .with_context(|| format!("loading draft {}", dc.path.display()))?;
        *slot = Some(LoadedDraft { path: dc.path.clone(), handle });
        Ok(())
    }

    fn ensure(&self, model: &LocalModel) -> Result<ModelKind> {
        let mut slot = self.inner.lock().unwrap();
        if let Some(l) = slot.as_ref() {
            if l.path == model.path { return Ok(l.kind); }
        }
        let tok_path = model.tokenizer.clone()
            .ok_or_else(|| anyhow!("no tokenizer.json next to {}", model.path.display()))?;
        let handle = ModelHandle::load(&model.path, self.device_pref)
            .with_context(|| format!("loading {}", model.path.display()))?;
        let tok = Tok::load(&tok_path)
            .with_context(|| format!("loading tokenizer {}", tok_path.display()))?;
        let kind = handle.kind();
        *slot = Some(Loaded { path: model.path.clone(), handle, tok, kind });
        Ok(kind)
    }
}
