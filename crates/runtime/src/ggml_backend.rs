//! `Pipelined` adapter over the IntelNav-patched libllama.
//!
//! Since task #18 the adapter is a thin pass-through — the trait's
//! currency is `intelnav_ggml::Hidden { data: Vec<f32>, shape:
//! Vec<usize> }` which matches the ggml side's native in-memory
//! layout, so no `candle::Tensor` conversion happens. One memcpy per
//! call pulls data out of the ctx's embeddings pointer into a fresh
//! `Hidden`; one memcpy on the way in fills the batch. That's it.

use std::path::Path;

use anyhow::{anyhow, Context as _, Result};

use intelnav_ggml as gg;
use intelnav_ggml::Hidden;

use crate::pipeline::{Forwarding, Pipelined};

/// Scratch sequence id used by `head` / `head_all` / `embed`. They
/// all run zero layers on libllama's side, but the memory module
/// still claims a slot per position during its init_batch step.
/// Running those claims on seq 0 collides with the forward_range
/// calls building the real KV cache there. We route the scratch
/// work through this seq id and flush it after each call so seq 0
/// stays pristine.
const HEAD_SCRATCH_SEQ: i32 = 1;

/// Hard upper bound on a single Pipelined call's sequence length.
/// Stops a malicious peer (or a local bug) from asking for an
/// absurdly large batch. Matches the default `INTELNAV_NCTX=2048`
/// with headroom; callers wanting more must bump both.
const MAX_BATCH_TOKENS: usize = 4096;

/// ggml-backed `Pipelined` model. Only `batch == 1` is supported —
/// matches every IntelNav call path today; M3's continuous-batching
/// gateway (task #20) grows this.
pub struct GgmlBackend {
    session: gg::Session,
    n_embd:  i32,
    n_layer: i32,
    n_vocab: i32,
}

impl GgmlBackend {
    /// Load a GGUF model + open an inference context. `n_gpu_layers`:
    /// `0` forces CPU, `-1` offloads everything libllama can,
    /// positive N offloads that many.
    pub fn load(path: &Path, n_ctx: u32, n_batch: u32, n_gpu_layers: i32) -> Result<Self> {
        gg::backend_load_all().ok();

        // n_seq_max = 2: seq 0 for the layer loop's real KV, seq 1 as
        // HEAD_SCRATCH_SEQ. M3's gateway will parameterize this.
        let session = gg::Session::load(path, n_ctx, n_batch, n_batch, 2, n_gpu_layers)
            .with_context(|| format!("loading ggml model: {}", path.display()))?;

        let m = session.model();
        let n_embd = m.n_embd();
        let n_layer = m.n_layer();
        let n_vocab = m.vocab().n_tokens();

        Ok(Self { session, n_embd, n_layer, n_vocab })
    }

    pub fn n_embd(&self)  -> i32 { self.n_embd  }
    pub fn n_layer(&self) -> i32 { self.n_layer }
    pub fn n_vocab(&self) -> i32 { self.n_vocab }
}

// ---------- helpers ----------

fn ids_to_i32(ids: &[u32]) -> Result<Vec<i32>> {
    if ids.is_empty() || ids.len() > MAX_BATCH_TOKENS {
        return Err(anyhow!(
            "ggml adapter: token slice len {} out of bounds (0, {MAX_BATCH_TOKENS}]",
            ids.len()
        ));
    }
    Ok(ids.iter().map(|&v| v as i32).collect())
}

/// Validate an incoming Hidden as `[1, seq, n_embd]`. Returns `seq`.
fn check_hidden(hidden: &Hidden, expect_hidden: i32) -> Result<usize> {
    if hidden.shape.len() != 3 {
        return Err(anyhow!(
            "ggml adapter: expected rank-3 Hidden [1, seq, hidden], got shape {:?}",
            hidden.shape
        ));
    }
    let (b, seq, h) = (hidden.shape[0], hidden.shape[1], hidden.shape[2]);
    if b != 1 {
        return Err(anyhow!("ggml adapter: only batch=1 supported, got {b}"));
    }
    if seq == 0 || seq > MAX_BATCH_TOKENS {
        return Err(anyhow!(
            "ggml adapter: seq_len {seq} out of bounds (0, {MAX_BATCH_TOKENS}]"
        ));
    }
    if h as i32 != expect_hidden {
        return Err(anyhow!(
            "ggml adapter: hidden dim mismatch (input {h}, model {expect_hidden})"
        ));
    }
    Ok(seq)
}

fn pull_hidden(ctx: &mut gg::Context<'_>, seq: i32, n_embd: i32) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity((seq as usize) * (n_embd as usize));
    for i in 0..seq {
        let row = ctx.get_embeddings_ith(i, n_embd as usize)?;
        out.extend_from_slice(row);
    }
    Ok(out)
}

// ---------- Trait impls ----------

impl Forwarding for GgmlBackend {
    fn block_count(&self) -> usize {
        self.n_layer as usize
    }

    fn forward(&mut self, input_ids: &[u32], index_pos: usize) -> Result<Hidden> {
        let ids = ids_to_i32(input_ids)?;
        let seq = ids.len() as i32;
        let n_layer = self.n_layer;
        let n_vocab = self.n_vocab;

        let mut batch = gg::Batch::tokens(seq, 1);
        batch.fill_tokens(&ids, index_pos as i32, /*logits_last_only=*/ true);

        self.session.ctx().decode_layers(&batch, 0, n_layer, /*run_head=*/ true)?;

        let logits = self
            .session
            .ctx()
            .get_logits_ith(seq - 1, n_vocab as usize)?
            .to_vec();
        Hidden::new(logits, vec![1, n_vocab as usize])
    }

    fn reset_cache(&mut self) {
        // Fresh contexts error on seq_rm; harmless, cache is empty.
        let _ = self.session.ctx().kv_seq_rm(0, -1, -1);
    }
}

impl Pipelined for GgmlBackend {
    fn embed(&mut self, input_ids: &[u32]) -> Result<Hidden> {
        let ids = ids_to_i32(input_ids)?;
        let seq = ids.len() as i32;
        let n_embd = self.n_embd;

        let mut batch = gg::Batch::tokens(seq, 1);
        batch.fill_tokens_in(&ids, 0, HEAD_SCRATCH_SEQ, /*logits_last_only=*/ false);

        self.session.ctx().embed_only(&batch).context("embed_only")?;

        let flat = pull_hidden(self.session.ctx(), seq, n_embd)?;
        let _ = self.session.ctx().kv_seq_rm(HEAD_SCRATCH_SEQ, -1, -1);

        Hidden::new(flat, vec![1, seq as usize, n_embd as usize])
    }

    fn forward_range(
        &mut self,
        hidden:    &Hidden,
        index_pos: usize,
        start:     usize,
        end:       usize,
    ) -> Result<Hidden> {
        let seq = check_hidden(hidden, self.n_embd)? as i32;
        let n_embd = self.n_embd;

        let mut batch = gg::Batch::embeddings(seq, n_embd, 1);
        batch.fill_embeddings(&hidden.data, seq, index_pos as i32, /*logits_last_only=*/ false);

        self.session.ctx()
            .decode_layers(&batch, start as i32, end as i32, /*run_head=*/ false)
            .with_context(|| format!("decode_layers([{start}, {end}))"))?;

        let out = pull_hidden(self.session.ctx(), seq, n_embd)?;
        Hidden::new(out, vec![1, seq as usize, n_embd as usize])
    }

    fn head(&mut self, hidden: &Hidden) -> Result<Hidden> {
        let seq = check_hidden(hidden, self.n_embd)? as i32;
        let n_embd = self.n_embd;
        let n_vocab = self.n_vocab;

        // ROCm head_only correctness workaround.
        //
        // libllama's head_only path produces a ~0.2 max_abs_diff vs
        // stock decode on ROCm when the embd batch carries
        // n_tokens >= 9 (matches `MMVQ_MAX_BATCH_SIZE = 8` in
        // ggml-cuda's mmvq dispatch). The lm_head is position-wise,
        // so feeding *only* the last token's hidden row produces the
        // same logits as feeding the full sequence with
        // `logits_last_only=true`, but lets the graph stay below
        // the buggy threshold.
        //
        // Pre-sliced unconditionally (not just on ROCm) — saves
        // bytes on the upload, simplifies the kv_seq_rm scope, and
        // avoids backend-detection plumbing here. Mathematically
        // identical to the previous "fill seq, last_only=true"
        // shape on every backend.
        let last_off = ((seq - 1) as usize) * (n_embd as usize);
        let last_row = &hidden.data[last_off..last_off + (n_embd as usize)];

        let mut batch = gg::Batch::embeddings(1, n_embd, 1);
        batch.fill_embeddings_in(last_row, 1, 0, HEAD_SCRATCH_SEQ, true);

        self.session.ctx().head_only(&batch).context("head_only")?;

        let logits = self
            .session
            .ctx()
            .get_logits_ith(0, n_vocab as usize)?
            .to_vec();

        let _ = self.session.ctx().kv_seq_rm(HEAD_SCRATCH_SEQ, -1, -1);

        Hidden::new(logits, vec![1, n_vocab as usize])
    }

    fn head_all(&mut self, hidden: &Hidden) -> Result<Hidden> {
        let seq = check_hidden(hidden, self.n_embd)? as i32;
        let n_embd = self.n_embd;
        let n_vocab = self.n_vocab;

        let mut batch = gg::Batch::embeddings(seq, n_embd, 1);
        batch.fill_embeddings_in(&hidden.data, seq, 0, HEAD_SCRATCH_SEQ, false);

        self.session.ctx().head_only(&batch).context("head_only (all)")?;

        let mut all: Vec<f32> = Vec::with_capacity((seq as usize) * (n_vocab as usize));
        for i in 0..seq {
            let row = self.session.ctx().get_logits_ith(i, n_vocab as usize)?;
            all.extend_from_slice(row);
        }

        let _ = self.session.ctx().kv_seq_rm(HEAD_SCRATCH_SEQ, -1, -1);

        Hidden::new(all, vec![1, seq as usize, n_vocab as usize])
    }

    fn truncate_kv_to(&mut self, keep: usize) -> Result<()> {
        self.session.ctx()
            .kv_seq_rm(0, keep as i32, -1)
            .with_context(|| format!("truncate_kv_to({keep})"))
    }
}
