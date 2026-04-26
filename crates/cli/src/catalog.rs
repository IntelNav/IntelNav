//! Curated download catalog.
//!
//! A small list of GGUFs we've verified load cleanly with our runtime
//! and whose tokenizers are available on public HuggingFace repos
//! (no gated models — one click should always work).
//!
//! Sizes are Q4_K_M-era; `ram_bytes` is a rough weights + KV + activation
//! ceiling we compare against the hardware probe's `available_bytes`.

use intelnav_runtime::{ModelKind, Probe};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    /// Weights + headroom comfortably fit.
    Fits,
    /// Fits but cuts it close — warn the user.
    Tight,
    /// Would almost certainly OOM — block unless forced.
    TooBig,
}

/// One downloadable model.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    /// Slug we surface to users and use as filename stem.
    pub id:              &'static str,
    pub display_name:    &'static str,
    pub family:          &'static str,
    pub arch:            ModelKind,
    pub params_b:        f32,          // in billions
    pub size_bytes:      u64,          // GGUF on disk (approx)
    pub ram_bytes_min:   u64,          // minimum free RAM to comfortably run
    pub hf_repo:         &'static str, // {repo}
    pub gguf_file:       &'static str, // file inside {repo}
    pub tokenizer_repo:  &'static str, // may differ from hf_repo
    pub tokenizer_file:  &'static str, // usually "tokenizer.json"
    pub note:            &'static str,
}

impl CatalogEntry {
    pub fn gguf_url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}?download=true",
                self.hf_repo, self.gguf_file)
    }
    pub fn tokenizer_url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}?download=true",
                self.tokenizer_repo, self.tokenizer_file)
    }
    pub fn fit(&self, probe: &Probe) -> Fit {
        let free = probe.memory.available_bytes;
        if free >= self.ram_bytes_min.saturating_mul(13) / 10 { Fit::Fits }
        else if free >= self.ram_bytes_min { Fit::Tight }
        else { Fit::TooBig }
    }
}

/// The curated list. Keep it small + reliable — quality over quantity.
pub fn catalog() -> &'static [CatalogEntry] {
    &CATALOG
}

pub fn find(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id.eq_ignore_ascii_case(id))
}

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

const CATALOG: [CatalogEntry; 6] = [
    CatalogEntry {
        id: "qwen2.5-0.5b-instruct-q4",
        display_name:   "Qwen 2.5 · 0.5B · Instruct",
        family:         "qwen",
        arch:           ModelKind::Ggml,
        params_b:       0.5,
        size_bytes:     398 * MB,
        ram_bytes_min:  700 * MB,
        hf_repo:        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        gguf_file:      "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer_file: "tokenizer.json",
        note:           "tiny, fits anywhere — good smoke test",
    },
    CatalogEntry {
        id: "qwen2.5-1.5b-instruct-q4",
        display_name:   "Qwen 2.5 · 1.5B · Instruct",
        family:         "qwen",
        arch:           ModelKind::Ggml,
        params_b:       1.5,
        size_bytes:     986 * MB,
        ram_bytes_min:  2 * GB,
        hf_repo:        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        gguf_file:      "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct",
        tokenizer_file: "tokenizer.json",
        note:           "snappy on laptops",
    },
    CatalogEntry {
        id: "qwen2.5-3b-instruct-q4",
        display_name:   "Qwen 2.5 · 3B · Instruct",
        family:         "qwen",
        arch:           ModelKind::Ggml,
        params_b:       3.0,
        size_bytes:     2 * GB,
        ram_bytes_min:  4 * GB,
        hf_repo:        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        gguf_file:      "qwen2.5-3b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-3B-Instruct",
        tokenizer_file: "tokenizer.json",
        note:           "sweet spot for most 16 GB laptops",
    },
    CatalogEntry {
        id: "qwen2.5-7b-instruct-q4",
        display_name:   "Qwen 2.5 · 7B · Instruct",
        family:         "qwen",
        arch:           ModelKind::Ggml,
        params_b:       7.0,
        size_bytes:     47 * GB / 10,     // ≈ 4.7 GiB
        ram_bytes_min:  9 * GB,
        hf_repo:        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        gguf_file:      "qwen2.5-7b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct",
        tokenizer_file: "tokenizer.json",
        note:           "workhorse — wants ~9 GiB free",
    },
    CatalogEntry {
        id: "qwen2.5-coder-1.5b-q4",
        display_name:   "Qwen 2.5 Coder · 1.5B",
        family:         "qwen-coder",
        arch:           ModelKind::Ggml,
        params_b:       1.5,
        size_bytes:     986 * MB,
        ram_bytes_min:  2 * GB,
        hf_repo:        "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        gguf_file:      "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tokenizer_file: "tokenizer.json",
        note:           "fast code completion",
    },
    CatalogEntry {
        id: "qwen2.5-coder-7b-q4",
        display_name:   "Qwen 2.5 Coder · 7B",
        family:         "qwen-coder",
        arch:           ModelKind::Ggml,
        params_b:       7.0,
        size_bytes:     47 * GB / 10,
        ram_bytes_min:  9 * GB,
        hf_repo:        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        gguf_file:      "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-Coder-7B-Instruct",
        tokenizer_file: "tokenizer.json",
        note:           "quality code — 16 GB+ machines",
    },
];
