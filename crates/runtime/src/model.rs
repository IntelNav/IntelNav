//! GGUF loading entry point.
//!
//! Every supported architecture goes through libllama via
//! [`GgmlBackend`]. The old candle-fork path (`Qwen2Weights` /
//! `LlamaWeights`) was removed once the ggml path was proven
//! bit-identical and end-to-end over TCP at GPU speed on AMD hardware.
//! `ModelHandle` is a thin newtype over `GgmlBackend`.

use std::path::Path;

use anyhow::{Context, Result};

use crate::device::DevicePref;
use crate::ggml_backend::GgmlBackend;
use crate::pipeline::{Forwarding, Hidden, Pipelined};

/// Source-compat shim. Every model goes through the same backend now,
/// but a handful of call sites still ask `kind()` to decide whether
/// they can pipeline. They all can, so this is a single-variant enum
/// retained only as a public type alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Ggml,
}

impl ModelKind {
    pub fn is_pipelined(self) -> bool { true }
}

/// Read just the GGUF header and return its architecture tag.
/// Kept for source-compat with callers that used to route per-arch;
/// the ggml backend handles every arch uniformly so this is now a
/// stub that returns [`ModelKind::Ggml`] whenever the file opens.
pub fn sniff_arch(_path: &Path) -> Result<ModelKind> {
    Ok(ModelKind::Ggml)
}

/// Loaded GGUF model + open inference context. Routes every
/// `Forwarding` / `Pipelined` call through [`GgmlBackend`].
pub struct ModelHandle {
    inner: GgmlBackend,
}

impl ModelHandle {
    pub fn kind(&self) -> ModelKind { ModelKind::Ggml }

    /// Load a GGUF model. `pref` is the operator's device hint
    /// (`Cpu` forces no GPU offload, anything else means "let ggml
    /// offload everything it can"). Env overrides:
    ///
    ///   * `INTELNAV_NGL=<int>` — wins over `pref` if set. `0` forces
    ///     CPU, `-1` offloads every layer onto whatever backend the
    ///     linked libllama was compiled with, positive N offloads
    ///     that many.
    ///   * `INTELNAV_NCTX` — context window (default 2048).
    ///   * `INTELNAV_NBATCH` — physical batch size (default 512).
    pub fn load(path: &Path, pref: DevicePref) -> Result<Self> {
        let n_ctx:    u32 = env_or("INTELNAV_NCTX",   2048);
        let n_batch:  u32 = env_or("INTELNAV_NBATCH",  512);
        let n_gpu_layers: i32 = env_or("INTELNAV_NGL", pref.n_gpu_layers());
        Self::load_ggml(path, n_ctx, n_batch, n_gpu_layers)
    }

    /// Explicit entry point — bypasses env-var dispatch. Useful for
    /// tests and benches that pin specific values.
    pub fn load_ggml(
        path:         &Path,
        n_ctx:        u32,
        n_batch:      u32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let inner = GgmlBackend::load(path, n_ctx, n_batch, n_gpu_layers)
            .with_context(|| format!("loading {} via GgmlBackend", path.display()))?;
        Ok(Self { inner })
    }

    // ---- forwarding delegation ----

    pub fn forwarding(&mut self) -> &mut dyn Forwarding {
        &mut self.inner
    }

    /// Returns the pipelined-capable handle. Always `Some` — every
    /// supported arch is pipelined.
    pub fn pipelined(&mut self) -> Option<&mut dyn Pipelined> {
        Some(&mut self.inner)
    }

    pub fn block_count(&self) -> usize {
        <GgmlBackend as Forwarding>::block_count(&self.inner)
    }

    pub fn forward(&mut self, input_ids: &[u32], index_pos: usize) -> Result<Hidden> {
        self.forwarding().forward(input_ids, index_pos)
    }

    pub fn reset_cache(&mut self) {
        self.forwarding().reset_cache()
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
