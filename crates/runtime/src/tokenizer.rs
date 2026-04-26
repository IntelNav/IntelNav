//! Thin wrapper over `tokenizers::Tokenizer` with:
//!
//! * a known list of EOS / end-of-turn ids (Qwen: `<|im_end|>` and
//!   `<|endoftext|>`), so the generation loop can stop cleanly.
//! * an [`IncrementalDecoder`] that surfaces new text each time a
//!   token is appended — matches how we want to stream output from
//!   generate.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

pub struct Tok {
    inner:   Tokenizer,
    eos_ids: Vec<u32>,
}

impl Tok {
    pub fn load(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", path.display()))?;
        let mut eos_ids = Vec::new();
        for name in ["<|im_end|>", "<|endoftext|>", "</s>"] {
            if let Some(id) = inner.token_to_id(name) {
                eos_ids.push(id);
            }
        }
        Ok(Self { inner, eos_ids })
    }

    /// Locate a tokenizer for a given GGUF file. Tries, in order:
    ///
    /// 1. `<gguf>.tokenizer.json` exact match
    /// 2. `tokenizer.json` in the same dir
    /// 3. any `*.tokenizer.json` in the same dir (single-model dirs)
    /// 4. prefix match: `<stem-prefix>.tokenizer.json` for each
    ///    hyphen-truncated prefix of the GGUF stem
    pub fn locate_for(gguf: &Path) -> Option<PathBuf> {
        let dir  = gguf.parent()?;
        let stem = gguf.file_stem()?.to_str()?;

        let exact = dir.join(format!("{stem}.tokenizer.json"));
        if exact.is_file() { return Some(exact); }

        let plain = dir.join("tokenizer.json");
        if plain.is_file() { return Some(plain); }

        // Prefix-match: qwen2.5-0.5b-instruct-q4_k_m → try
        // `qwen2.5-0.5b-instruct-q4_k_m`, `...-instruct`, `qwen2.5-0.5b`, ...
        let mut parts: Vec<&str> = stem.split('-').collect();
        while parts.len() > 1 {
            parts.pop();
            let cand = dir.join(format!("{}.tokenizer.json", parts.join("-")));
            if cand.is_file() { return Some(cand); }
        }

        // Last-ditch: first `*.tokenizer.json` we find in the dir.
        std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).find_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_str()?;
            if name.ends_with(".tokenizer.json") { Some(p) } else { None }
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, true).map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }

    pub fn incremental(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder { tok: &self.inner, ids: Vec::new(), emitted: 0 }
    }
}

/// Buffers generated token IDs and yields newly-decoded text each call.
///
/// `tokenizers` is byte-level BPE, so a single new token often decodes
/// to empty text (mid-multibyte-char) and we need to wait for the next
/// one before flushing. Simplest-correct strategy: re-decode the full
/// buffer, emit the suffix that wasn't emitted before.
pub struct IncrementalDecoder<'a> {
    tok:     &'a Tokenizer,
    ids:     Vec<u32>,
    emitted: usize,
}

impl<'a> IncrementalDecoder<'a> {
    pub fn push(&mut self, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        let all = self.tok.decode(&self.ids, false).map_err(|e| anyhow!("decode: {e}"))?;
        if all.len() <= self.emitted {
            return Ok(None);
        }
        // Only emit if the suffix ends at a valid UTF-8 boundary — which
        // decode() guarantees since it returns a String.
        let delta = all[self.emitted..].to_string();
        self.emitted = all.len();
        Ok(Some(delta))
    }
}
