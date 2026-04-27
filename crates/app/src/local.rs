//! In-process inference driver.
//!
//! Loads a GGUF from [`Config::models_dir`] via the runtime crate and
//! streams tokens back into the TUI as `Delta` chunks.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use tokio::sync::mpsc;

use intelnav_runtime::{
    build_chat_prompt, generate, sniff_arch, ChatTurn, DevicePref,
    ModelHandle, ModelKind, SamplingCfg, Tok,
};

use crate::delta::{ChatMessage, Delta};

// ======================================================================
//  Model discovery
// ======================================================================

#[derive(Clone, Debug)]
pub struct LocalModel {
    /// Display name — the GGUF's stem (e.g. `qwen2.5-0.5b-instruct-q4_k_m`).
    pub name:       String,
    pub path:       PathBuf,
    pub tokenizer:  Option<PathBuf>,
    pub size_bytes: u64,
    pub arch:       Option<ModelKind>,
}

impl LocalModel {
    pub fn is_usable(&self) -> bool {
        self.tokenizer.is_some() && self.arch.is_some()
    }

    pub fn status_line(&self) -> String {
        let size = human_bytes(self.size_bytes);
        match (self.arch, &self.tokenizer) {
            (Some(a), Some(_)) => format!("{} · {size} · {:?}", self.name, a),
            (Some(a), None)    => format!("{} · {size} · {:?} · missing tokenizer.json", self.name, a),
            (None,    _)       => format!("{} · {size} · unsupported arch", self.name),
        }
    }
}

/// Scan `dir` for `*.gguf` files and return what's loadable.
/// Returns an empty vec if the dir doesn't exist.
pub fn list_models(dir: &Path) -> Vec<LocalModel> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") { continue; }
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let tokenizer = Tok::locate_for(&path);
        let arch = sniff_arch(&path).ok();
        out.push(LocalModel { name, path, tokenizer, size_bytes, arch });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Pick a model to use by default from a scan:
/// prefer usable ones, smallest first (safest fit).
pub fn pick_default(models: &[LocalModel]) -> Option<&LocalModel> {
    models
        .iter()
        .filter(|m| m.is_usable())
        .min_by_key(|m| m.size_bytes)
}

/// Resolve a user-provided model name against a scan. Accepts exact
/// stem match, the filename with `.gguf`, an absolute path, or a
/// substring (case-insensitive) for fuzzy convenience.
pub fn resolve(models: &[LocalModel], name: &str) -> Option<LocalModel> {
    // Absolute path passthrough.
    let p = PathBuf::from(name);
    if p.is_file() {
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or(name).to_string();
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        let tok  = Tok::locate_for(&p);
        let arch = sniff_arch(&p).ok();
        return Some(LocalModel { name: stem, path: p, tokenizer: tok, size_bytes: size, arch });
    }

    let needle = name.to_ascii_lowercase();
    if let Some(m) = models.iter().find(|m| m.name.eq_ignore_ascii_case(&needle)) {
        return Some(m.clone());
    }
    if let Some(m) = models.iter().find(|m| m.name.to_ascii_lowercase().contains(&needle)) {
        return Some(m.clone());
    }
    None
}

// ======================================================================
//  Driver
// ======================================================================

/// Shared-across-turns state.
struct Loaded {
    path:   PathBuf,
    handle: ModelHandle,
    tok:    Tok,
    kind:   ModelKind,
}

#[derive(Clone)]
pub struct LocalDriver {
    inner: Arc<Mutex<Option<Loaded>>>,
    device_pref: DevicePref,
}

impl LocalDriver {
    pub fn new(device_pref: DevicePref) -> Self {
        Self { inner: Arc::new(Mutex::new(None)), device_pref }
    }

    /// Ensure `model` is loaded. Returns the kind so the caller can
    /// pick a chat template.
    fn ensure(&self, model: &LocalModel) -> Result<ModelKind> {
        let mut slot = self.inner.lock().unwrap();
        if let Some(l) = slot.as_ref() {
            if l.path == model.path {
                return Ok(l.kind);
            }
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

    /// Stream a reply.
    pub fn stream(
        &self,
        model: LocalModel,
        messages: Vec<ChatMessage>,
        cfg: SamplingCfg,
    ) -> mpsc::UnboundedReceiver<Delta> {
        let (tx, rx) = mpsc::unbounded_channel();
        let driver = self.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = driver.run_blocking(&model, &messages, &cfg, &tx) {
                let _ = tx.send(Delta::Error(e.to_string()));
            }
        });
        rx
    }

    fn run_blocking(
        &self,
        model:    &LocalModel,
        messages: &[ChatMessage],
        cfg:      &SamplingCfg,
        tx:       &mpsc::UnboundedSender<Delta>,
    ) -> Result<()> {
        let kind = self.ensure(model)?;
        let turns: Vec<ChatTurn<'_>> = messages
            .iter()
            .map(|m| ChatTurn { role: m.role.as_str(), content: m.content.as_str() })
            .collect();
        let prompt = build_chat_prompt(kind, &turns);

        // Take the loaded model for the duration of the call. The
        // mutex is held for the whole generation — concurrent turns
        // would be incorrect anyway (shared KV cache).
        let mut slot = self.inner.lock().unwrap();
        let loaded = slot.as_mut().ok_or_else(|| anyhow!("model unloaded mid-flight"))?;
        let fw = loaded.handle.forwarding();
        generate(fw, &loaded.tok, &prompt, cfg, |chunk| {
            let _ = tx.send(Delta::Token(chunk.to_string()));
            Ok(())
        })?;
        let _ = tx.send(Delta::Done);
        Ok(())
    }
}

// ======================================================================
//  Small helpers
// ======================================================================

pub fn human_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    let mut s = String::new();
    if n >= GB { let _ = write!(s, "{:.1} GiB", n as f64 / GB as f64); }
    else if n >= MB { let _ = write!(s, "{:.0} MiB", n as f64 / MB as f64); }
    else if n >= KB { let _ = write!(s, "{:.0} KiB", n as f64 / KB as f64); }
    else            { let _ = write!(s, "{n} B"); }
    s
}
