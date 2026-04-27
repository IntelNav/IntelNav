//! Gateway chain driver.
//!
//! When the gateway is started with a local GGUF model and a list of
//! peer addresses, this module replaces the upstream-proxy chat path:
//! incoming `/v1/chat/completions` requests are tokenized locally,
//! the front slice runs in-process, the middle hops through the
//! configured peer chain, the head + sample run locally, and tokens
//! stream back to the browser as OpenAI-shape SSE deltas.
//!
//! Telemetry from each chain hop is fanned out through
//! [`intelnav_runtime::Telemetry`] so the SPA's `/v1/swarm/events`
//! stream lights up the topology graph in real time as hidden state
//! actually moves between peers.
//!
//! Threading model mirrors `cli/chain_driver.rs`: `ModelHandle` is
//! not `Send` (libllama context lives on one OS thread), so each
//! request spawns a dedicated thread that hosts its own tokio
//! runtime to drive the async chain interleaved with sync ggml
//! forward passes. Concurrent requests serialize on the model
//! mutex — multi-tenant continuous batching is M3 work.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use intelnav_core::Config;
use intelnav_runtime::{
    build_chat_prompt, run_turn, Chain, ChainCfg, ChatTurn, DevicePref,
    Dtype, ModelHandle, ModelKind, SamplingCfg, Telemetry, Tok,
};
use tokio::sync::mpsc;

/// One streamed delta from the chain driver to the SSE response.
#[derive(Debug)]
pub enum Delta {
    /// One textual chunk to relay to the browser.
    Token(String),
    /// Stream completed normally.
    Done,
    /// Chain failed mid-flight; the SSE response should surface the
    /// message to the user and close.
    Error(String),
}

/// Loaded model + tokenizer + chain config the gateway uses for every
/// `/v1/chat/completions` request when running in chain mode.
///
/// `Clone`-cheap: the heavy state (model, tokenizer) lives behind
/// `Arc`s so the driver handle can be cloned into request handlers
/// without re-loading.
#[derive(Clone)]
pub struct GatewayDriver {
    model:      Arc<Mutex<ModelHandle>>,
    tok:        Arc<Tok>,
    kind:       ModelKind,
    /// Wrapped in [`RwLock`] so the SPA can flip wire-dtype (and any
    /// future per-chain knob) live — the next turn picks the new
    /// config off the lock.
    chain_cfg:  Arc<RwLock<ChainCfg>>,
    telemetry:  Telemetry,
}

impl GatewayDriver {
    /// Load the GGUF + tokenizer and prepare the chain config from
    /// `config.peers` + `config.splits`. Fails fast on any of:
    ///   * the GGUF not present at `gguf`
    ///   * `config.peers` empty (chain mode requires peers)
    ///   * tokenizer not next to the GGUF
    ///   * invalid splits / peer-address parsing
    pub fn load(
        gguf:       &Path,
        config:     &Config,
        telemetry:  Telemetry,
    ) -> Result<Self> {
        if config.peers.is_empty() {
            return Err(anyhow!(
                "gateway chain mode requires config.peers (or INTELNAV_PEERS) to be non-empty"
            ));
        }

        // Resolve tokenizer next to the GGUF. Same heuristic LocalDriver
        // uses — `tokenizer.json` adjacent to the model file, with a
        // small set of stem-prefix fallbacks.
        let tok_path = Tok::locate_for(gguf)
            .ok_or_else(|| anyhow!(
                "no tokenizer.json found beside {} — drop one in or set --tokenizer",
                gguf.display()
            ))?;

        // Device pref: env > config > Auto. Same path the CLI takes,
        // ModelHandle::load reads INTELNAV_NGL itself.
        let pref: DevicePref = config.device.parse().unwrap_or(DevicePref::Auto);

        tracing::info!(model = %gguf.display(), peers = config.peers.len(),
                       "gateway: loading chain-mode driver");
        let model = ModelHandle::load(gguf, pref)
            .with_context(|| format!("loading {}", gguf.display()))?;
        let kind = model.kind();
        let n_blocks = model.block_count() as u16;
        let tok = Tok::load(&tok_path)
            .with_context(|| format!("loading tokenizer {}", tok_path.display()))?;

        // ChainCfg invariant: splits.len() == peers.len() (each split
        // is the *start* layer of that peer's range; gateway owns the
        // prefix [0..splits[0]); tail peer owns [splits[N-1]..n_blocks)).
        if config.splits.len() != config.peers.len() {
            return Err(anyhow!(
                "peers ({}) and splits ({}) length mismatch — \
                 set INTELNAV_SPLITS to one start-layer per peer (gateway \
                 owns the prefix [0..splits[0]) locally, model has {n_blocks} blocks)",
                config.peers.len(), config.splits.len()
            ));
        }
        if let Some(&last) = config.splits.last() {
            if last >= n_blocks {
                return Err(anyhow!(
                    "last split {last} must be < n_blocks {n_blocks} \
                     so the tail peer has at least one layer"
                ));
            }
        }

        // Parse peer addresses + build the chain config.
        let peer_addrs: Vec<std::net::SocketAddr> = config.peers.iter()
            .map(|s| s.parse().with_context(|| format!("parsing peer addr `{s}`")))
            .collect::<Result<_>>()?;
        let mut cfg = ChainCfg::many(peer_addrs, config.splits.clone());
        cfg.proto_ver = 1;
        cfg.model_cid = config.registry_model.clone()
            .unwrap_or_else(|| config.default_model.clone());
        cfg.wire_dtype = parse_wire_dtype(&config.wire_dtype);

        Ok(Self {
            model:     Arc::new(Mutex::new(model)),
            tok:       Arc::new(tok),
            kind,
            chain_cfg: Arc::new(RwLock::new(cfg)),
            telemetry,
        })
    }

    /// Live-mutate the wire dtype. Takes effect on the next chat turn
    /// — the current turn's `Chain` already has its dtype baked in.
    pub fn set_wire_dtype(&self, dtype: Dtype) {
        if let Ok(mut g) = self.chain_cfg.write() {
            g.wire_dtype = dtype;
        }
    }

    /// Read-only snapshot of current chain config (small, cheap to
    /// clone). Used by the gateway's `/v1/chain/config` endpoint.
    pub fn chain_config(&self) -> ChainCfg {
        self.chain_cfg.read()
            .map(|c| c.clone())
            .unwrap_or_else(|p| p.into_inner().clone())
    }

    /// Drive one chat turn through the chain. Returns an unbounded
    /// receiver of `Delta`s the SSE handler can pump out.
    ///
    /// Spawns a dedicated OS thread + tokio runtime per request so
    /// the non-`Send` `ModelHandle` stays pinned and the async chain
    /// I/O can interleave freely with sync ggml forward passes.
    pub fn stream(
        &self,
        messages: Vec<(String, String)>,    // (role, content) pairs
        cfg:      SamplingCfg,
    ) -> mpsc::UnboundedReceiver<Delta> {
        let (tx, rx) = mpsc::unbounded_channel();
        let driver = self.clone();

        std::thread::spawn(move || {
            // Single-thread runtime is enough for a chat turn; the
            // chain is sequential per step.
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Delta::Error(format!("spawn rt: {e}")));
                    return;
                }
            };
            if let Err(e) = rt.block_on(driver.run(messages, cfg, tx.clone())) {
                let _ = tx.send(Delta::Error(e.to_string()));
            } else {
                let _ = tx.send(Delta::Done);
            }
        });
        rx
    }

    /// Internal: build the prompt, open the chain, drive `run_turn`,
    /// close the chain. The `tx` is shared so token deltas + a final
    /// `Done`/`Error` go to the same SSE stream.
    async fn run(
        self,
        messages: Vec<(String, String)>,
        cfg:      SamplingCfg,
        tx:       mpsc::UnboundedSender<Delta>,
    ) -> Result<()> {
        // Render the conversation through the model's chat template.
        let turns: Vec<ChatTurn<'_>> = messages.iter()
            .map(|(role, content)| ChatTurn { role: role.as_str(), content: content.as_str() })
            .collect();
        let prompt = build_chat_prompt(self.kind, &turns);

        // Lock the model exclusively for the duration of the turn —
        // shared KV cache, can't have concurrent forwards.
        let model_arc = self.model.clone();
        let mut guard = model_arc.lock()
            .map_err(|_| anyhow!("model mutex poisoned"))?;

        let n_blocks = guard.block_count() as u16;
        // Snapshot the chain config under the lock — once we're past
        // here, mid-turn flips don't surprise this connection.
        let cfg_snapshot = self.chain_cfg.read()
            .map(|c| c.clone())
            .unwrap_or_else(|p| p.into_inner().clone());
        let mut chain = Chain::connect(cfg_snapshot, n_blocks).await
            .map_err(|e| anyhow!("chain connect: {e}"))?;
        chain.attach_telemetry(self.telemetry.clone());

        let result = run_turn(
            &mut guard, &self.tok, &mut chain, &prompt, &cfg,
            |chunk| {
                let _ = tx.send(Delta::Token(chunk.to_string()));
                Ok(())
            },
        ).await;

        // Close the chain even if the turn errored — peer-side log
        // lines should end with a clean reason, not a TCP RST.
        let close_reason = if result.is_ok() { "turn complete" } else { "driver error" };
        chain.close(close_reason).await;

        match result {
            Ok(_n) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

/// Resolve a string like `"int8"` / `"fp16"` into the wire enum.
/// Unknown values fall back to fp16 silently — same policy the
/// CLI's chain_driver uses.
pub fn parse_wire_dtype(s: &str) -> Dtype {
    match s.trim().to_ascii_lowercase().as_str() {
        "int8" | "i8"  => Dtype::Int8,
        _              => Dtype::Fp16,
    }
}

/// Inverse of [`parse_wire_dtype`] — used by `/v1/chain/config` so
/// the SPA knows which button to highlight.
pub fn wire_dtype_str(d: Dtype) -> &'static str {
    match d {
        Dtype::Int8 => "int8",
        _           => "fp16",
    }
}

/// Configuration knob the gateway honours when deciding whether to
/// load a chain driver. Reads `INTELNAV_GATEWAY_MODEL` (a path to a
/// GGUF). Returns `None` if unset/empty — gateway then falls back to
/// the upstream-proxy chat path.
pub fn gateway_model_path() -> Option<PathBuf> {
    std::env::var("INTELNAV_GATEWAY_MODEL").ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
}

/// Default per-request budget. Generous because cold-start on a real
/// 33B chain over LAN can take a few seconds for the first hop.
pub const DEFAULT_STEP_TIMEOUT: Duration = Duration::from_secs(45);
