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

/// One downloadable model. Schema-complete: several fields aren't
/// read by today's renderer but are part of the catalog contract
/// (used by future `--filter` / `intelnav doctor --suggest` paths
/// and the operator-facing manifest output).
#[allow(dead_code)]
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
    /// Total transformer blocks. Used to derive the canonical splits
    /// the swarm announces under and the SwarmIndex probes for.
    pub block_count:     u16,
    /// Standardized splits for this model on the swarm. The
    /// SwarmIndex queries `(start, end)` for each pair here when
    /// computing `RangeCoverage`. Empty = "we don't have a beta
    /// recommendation, treat as one big slice".
    pub default_splits:  &'static [(u16, u16)],
}

impl CatalogEntry {
    /// Deterministic CID for this catalog entry. Used as the DHT key
    /// for the model envelope and as the join key against
    /// `SwarmIndex` results. Stable across releases as long as the
    /// (hf_repo, gguf_file) pair doesn't change.
    pub fn model_cid(&self) -> String {
        let s = format!("hf|{}|{}", self.hf_repo, self.gguf_file);
        let h = blake3::hash(s.as_bytes());
        // bs58-encoded, truncated to 22 chars — short enough to fit
        // the picker without being collision-prone (88 bits of entropy).
        let full = bs58::encode(h.as_bytes()).into_string();
        full.chars().take(22).collect()
    }

    /// Splits to query against the SwarmIndex. Falls back to one
    /// (0..block_count) range when the entry doesn't declare splits.
    pub fn swarm_ranges(&self) -> Vec<(u16, u16)> {
        if self.default_splits.is_empty() {
            vec![(0, self.block_count)]
        } else {
            self.default_splits.to_vec()
        }
    }
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

/// Resolve a slug to its catalog entry. Used by the future
/// `intelnav models install <id>` flow; kept now so the catalog
/// has one canonical lookup entry point.
#[allow(dead_code)]
pub fn find(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id.eq_ignore_ascii_case(id))
}

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

// Standardized splits the swarm announces under for each architecture.
// Every contributor that hosts a slice of one of these models picks a
// (start, end) pair from this list so the SwarmIndex's coverage probe
// hits the same keys.
//
// 24-block models (Qwen 0.5B/1.5B): 4-way split, 6 blocks each.
// 36-block models (Qwen 3B):        4-way split, 9 blocks each.
// 28-block models (Qwen 7B / Coder): 4-way split, 7 blocks each.
const SPLITS_24_4: &[(u16, u16)] = &[(0, 6),  (6, 12),  (12, 18), (18, 24)];
const SPLITS_36_4: &[(u16, u16)] = &[(0, 9),  (9, 18),  (18, 27), (27, 36)];
const SPLITS_28_4: &[(u16, u16)] = &[(0, 7),  (7, 14),  (14, 21), (21, 28)];

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
        block_count:    24,
        default_splits: SPLITS_24_4,
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
        block_count:    28,
        default_splits: SPLITS_28_4,
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
        block_count:    36,
        default_splits: SPLITS_36_4,
    },
    CatalogEntry {
        id: "qwen2.5-7b-instruct-q4",
        display_name:   "Qwen 2.5 · 7B · Instruct",
        family:         "qwen",
        arch:           ModelKind::Ggml,
        params_b:       7.0,
        size_bytes:     47 * GB / 10,
        ram_bytes_min:  9 * GB,
        hf_repo:        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        gguf_file:      "qwen2.5-7b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct",
        tokenizer_file: "tokenizer.json",
        note:           "workhorse — wants ~9 GiB free",
        block_count:    28,
        default_splits: SPLITS_28_4,
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
        block_count:    28,
        default_splits: SPLITS_28_4,
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
        block_count:    28,
        default_splits: SPLITS_28_4,
    },
];
