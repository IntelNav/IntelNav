//! Safe-ish Rust wrappers around the IntelNav-patched libllama.
//!
//! The three C functions our llama.cpp fork adds
//! (`llama_embed_only`, `llama_decode_layers`, `llama_head_only`) plus
//! enough scaffolding to exercise them — model loading, tokenization,
//! batch construction, output readback.
//!
//! Design notes:
//!
//! * libllama is **dlopened at runtime** via [`loader::Loader`]. No
//!   compile-time link dependency. Consumers don't need libllama on
//!   disk to build; they need it present at one of a set of
//!   well-known locations before they call into the ggml path.
//! * Handles (`Model`, `Context`, `Session`, `Batch`) are RAII — the
//!   C-side resource is freed on drop. They hold an `Arc<Loader>` so
//!   the underlying library is guaranteed to outlive them.
//! * `Model` is `Send + Sync`; `Context` is `Send` but not `Sync`
//!   (libllama contexts are single-threaded).
//! * All unsafe FFI lives under `sys` + `loader`. Nothing outside
//!   this crate should need to touch it.

pub mod hidden;
pub mod loader;
pub mod probe;
pub mod sys;

pub use hidden::{decode_hidden, encode_hidden, encode_hidden_with, Hidden, HiddenPayload};
pub use loader::{default_loader, find_libllama, Loader};
pub use probe::{BackendCheck, BackendStatus, DetectedGpu, GgmlProbe};

use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::Arc;

use anyhow::{anyhow, Context as _, Result};

/// Call once per process before loading any model. Loads every ggml
/// backend plugin from the directory libllama was dlopened from (via
/// `ggml_backend_load_all_from_path`) and falls back to ggml's default
/// discovery (dirname of `/proc/self/exe`) afterwards. This makes a
/// freshly-installed libllama work without any `LD_LIBRARY_PATH`,
/// `GGML_BACKEND_PATH`, or "symlink the .so next to your binary"
/// rituals — the exact UX friction the Path B laptop smoke exposed.
pub fn backend_load_all() -> Result<()> {
    let l = default_loader()?;
    if let Some(dir) = l.loaded_from.parent() {
        let c_dir = std::ffi::CString::new(dir.to_string_lossy().as_bytes())
            .map_err(|e| anyhow!("libllama dir path contains NUL: {e}"))?;
        // Safety: `dir_path` is a valid NUL-terminated C string that
        // outlives the call; ggml_backend_load_all_from_path doesn't
        // retain the pointer.
        unsafe { (l.ggml_backend_load_all_from_path)(c_dir.as_ptr()) };
    }
    // Belt-and-suspenders: ggml's default discovery also scans
    // `dirname(/proc/self/exe)` and any GGML_BACKEND_PATH entries.
    // Running both is idempotent on already-loaded backends.
    unsafe { (l.ggml_backend_load_all)() };
    Ok(())
}

/// Flip the shim's abort flag (global, lives inside libllama). When
/// `true`, any in-flight ggml compute will bail out via the normal
/// error path instead of calling `exit(1)` on an assertion. Used by
/// timeout / cancel paths; the Context wrappers already clear the
/// flag on every call's error path.
pub fn trip_abort(tripped: bool) -> Result<()> {
    let l = default_loader()?;
    unsafe { (l.intelnav_trip_abort)(tripped) };
    Ok(())
}

// ---------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------

/// A loaded GGUF model. Thread-safe to share by reference; freed on drop.
pub struct Model {
    raw: *mut sys::llama_model,
    loader: Arc<Loader>,
}

// Safety: llama.cpp models are immutable after load and are safe to
// share across threads for read-only use (tokenization, size queries,
// creating contexts). Contexts are the single-threaded resource.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    /// Load a GGUF model from disk using the default loader (dlopens
    /// libllama on first call per process). `n_gpu_layers`:
    /// * `0` forces CPU,
    /// * `-1` offloads as many layers as fit,
    /// * positive N offloads N layers.
    pub fn load_from_file(path: impl AsRef<Path>, n_gpu_layers: i32) -> Result<Self> {
        let loader = default_loader()?;
        Self::load_from_file_with(loader, path, n_gpu_layers)
    }

    /// Explicit-loader variant — for tests / benches that want to pin
    /// a particular libllama build (e.g. the ROCm vs CPU shootout).
    pub fn load_from_file_with(
        loader:       Arc<Loader>,
        path:         impl AsRef<Path>,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let path = path.as_ref();
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .with_context(|| format!("model path contains NUL byte: {}", path.display()))?;

        // Safety: path is valid NUL-terminated UTF-8; function pointer
        // was resolved at loader-construction time.
        let raw = unsafe { (loader.intelnav_load_model)(c_path.as_ptr(), n_gpu_layers) };
        if raw.is_null() {
            return Err(anyhow!("failed to load model: {}", path.display()));
        }
        Ok(Self { raw, loader })
    }

    /// Embedding dimension.
    pub fn n_embd(&self) -> i32 {
        unsafe { (self.loader.llama_model_n_embd)(self.raw) }
    }

    /// Number of transformer blocks.
    pub fn n_layer(&self) -> i32 {
        unsafe { (self.loader.llama_model_n_layer)(self.raw) }
    }

    /// Vocabulary accessor, scoped to this model's lifetime.
    pub fn vocab(&self) -> Vocab<'_> {
        let raw = unsafe { (self.loader.llama_model_get_vocab)(self.raw) };
        Vocab { raw, loader: self.loader.clone(), _m: std::marker::PhantomData }
    }

    pub(crate) fn raw(&self) -> *mut sys::llama_model {
        self.raw
    }

    /// Shared handle to the loader this model lives in — contexts and
    /// sessions reuse it so they avoid re-initializing function
    /// pointer tables.
    pub fn loader(&self) -> &Arc<Loader> {
        &self.loader
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.loader.llama_model_free)(self.raw) };
        }
    }
}

// ---------------------------------------------------------------------
// Vocab
// ---------------------------------------------------------------------

/// Borrowed vocabulary view. Lives as long as its `Model`.
pub struct Vocab<'m> {
    raw: *const sys::llama_vocab,
    loader: Arc<Loader>,
    _m: std::marker::PhantomData<&'m Model>,
}

impl<'m> Vocab<'m> {
    /// Number of tokens in the vocabulary (for sizing logits buffers).
    pub fn n_tokens(&self) -> i32 {
        unsafe { (self.loader.llama_vocab_n_tokens)(self.raw) }
    }

    /// Tokenize a UTF-8 string. Returns a `Vec<llama_token>`.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let probe_text = text.as_bytes();
        let text_ptr = probe_text.as_ptr() as *const i8;
        let text_len = probe_text.len() as i32;

        let needed = unsafe {
            (self.loader.llama_tokenize)(
                self.raw,
                text_ptr,
                text_len,
                ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };
        let needed = if needed < 0 { -needed } else { needed };
        if needed <= 0 {
            return Ok(Vec::new());
        }

        let mut out = vec![0_i32; needed as usize];
        let written = unsafe {
            (self.loader.llama_tokenize)(
                self.raw,
                text_ptr,
                text_len,
                out.as_mut_ptr(),
                out.len() as i32,
                add_special,
                parse_special,
            )
        };
        if written < 0 {
            return Err(anyhow!("llama_tokenize failed: {written}"));
        }
        out.truncate(written as usize);
        Ok(out)
    }
}

// ---------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------

/// An inference context pinned to a specific `Model`.
///
/// Not `Sync` — llama.cpp contexts hold mutable KV cache and output
/// buffers; only one caller at a time.
pub struct Context<'m> {
    raw: *mut sys::llama_context,
    loader: Arc<Loader>,
    _m:  std::marker::PhantomData<&'m Model>,
}

unsafe impl<'m> Send for Context<'m> {}

impl<'m> Context<'m> {
    /// Create a new context. `n_ctx` / `n_batch` / `n_ubatch` /
    /// `n_seq_max` map to the equivalent `llama_context_params`
    /// fields; everything else takes llama.cpp defaults plus
    /// `no_perf = true`. Installs the shim's abort callback so a
    /// crafted payload can't take the process down via `ggml_abort`.
    pub fn new(
        model:     &'m Model,
        n_ctx:     u32,
        n_batch:   u32,
        n_ubatch:  u32,
        n_seq_max: u32,
    ) -> Result<Self> {
        let raw = unsafe {
            (model.loader.intelnav_new_context)(model.raw(), n_ctx, n_batch, n_ubatch, n_seq_max)
        };
        if raw.is_null() {
            return Err(anyhow!("failed to create llama_context"));
        }
        Ok(Self { raw, loader: model.loader.clone(), _m: std::marker::PhantomData })
    }

    /// Stock full-forward. Zero-cost fallback to `llama_decode`.
    pub fn decode(&mut self, batch: &Batch) -> Result<()> {
        let rc = unsafe { (self.loader.llama_decode)(self.raw, batch.raw_view()) };
        check_ret("llama_decode", rc)
    }

    /// IntelNav: tokens → embedding lookup, no layers, no head.
    /// Per-position hidden state via [`Self::get_embeddings_ith`].
    pub fn embed_only(&mut self, batch: &Batch) -> Result<()> {
        let rc = unsafe { (self.loader.llama_embed_only)(self.raw, batch.raw_view()) };
        check_ret("llama_embed_only", rc)
    }

    /// IntelNav: run layers `[layer_start, layer_end)`.
    /// * `run_head=false` is the P2P middle-peer / tail-peer-without-head
    ///   case — skip `output_norm` and `lm_head`, expose hidden state
    ///   via [`Self::get_embeddings_ith`].
    /// * `run_head=true` requires `layer_end == n_layer`; applies
    ///   norm + head, exposes logits via [`Self::get_logits_ith`].
    pub fn decode_layers(
        &mut self,
        batch: &Batch,
        layer_start: i32,
        layer_end: i32,
        run_head: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.loader.llama_decode_layers)(self.raw, batch.raw_view(), layer_start, layer_end, run_head)
        };
        check_ret("llama_decode_layers", rc)
    }

    /// IntelNav: hidden-state → `output_norm` → `lm_head`. Skips the
    /// layer loop. `batch.embd` must carry the input hidden state.
    pub fn head_only(&mut self, batch: &Batch) -> Result<()> {
        let rc = unsafe { (self.loader.llama_head_only)(self.raw, batch.raw_view()) };
        check_ret("llama_head_only", rc)
    }

    /// Borrow the logits row for batch position `i` (as flagged by
    /// `batch.logits[i] != 0` during the last `decode_*` call).
    ///
    /// The returned slice is valid until the next forward call.
    pub fn get_logits_ith(&mut self, i: i32, n_vocab: usize) -> Result<&[f32]> {
        let p = unsafe { (self.loader.llama_get_logits_ith)(self.raw, i) };
        if p.is_null() {
            return Err(anyhow!("get_logits_ith({i}) returned null"));
        }
        Ok(unsafe { slice::from_raw_parts(p, n_vocab) })
    }

    /// Borrow the per-position hidden state at batch index `i`. Same
    /// lifetime rules as [`Self::get_logits_ith`].
    pub fn get_embeddings_ith(&mut self, i: i32, n_embd: usize) -> Result<&[f32]> {
        let p = unsafe { (self.loader.llama_get_embeddings_ith)(self.raw, i) };
        if p.is_null() {
            return Err(anyhow!("get_embeddings_ith({i}) returned null"));
        }
        Ok(unsafe { slice::from_raw_parts(p, n_embd) })
    }

    /// Reset the KV cache for sequence `seq_id` over positions `[p0, p1)`.
    /// Use `-1` as a sentinel for "all" on either bound.
    pub fn kv_seq_rm(&mut self, seq_id: i32, p0: i32, p1: i32) -> Result<()> {
        let mem = unsafe { (self.loader.llama_get_memory)(self.raw) };
        if mem.is_null() {
            return Err(anyhow!("llama_get_memory returned null (no KV cache on this ctx)"));
        }
        let ok = unsafe { (self.loader.llama_memory_seq_rm)(mem, seq_id, p0, p1) };
        if !ok {
            return Err(anyhow!("llama_memory_seq_rm rejected [{p0}, {p1})"));
        }
        Ok(())
    }

    /// Toggle `cparams.embeddings`.
    pub fn set_embeddings(&mut self, value: bool) {
        unsafe { (self.loader.llama_set_embeddings)(self.raw, value) }
    }
}

impl<'m> Drop for Context<'m> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.loader.llama_free)(self.raw) };
        }
    }
}

// ---------------------------------------------------------------------
// Session — owned Model + Context bundle
// ---------------------------------------------------------------------

/// Self-contained inference session: owns both a [`Model`] and a
/// [`Context`] borrowing from it. Frees them on drop in the order
/// llama.cpp requires (context first, then model).
pub struct Session {
    ctx:    Option<Context<'static>>,
    model:  Box<Model>,
}

unsafe impl Send for Session {}

impl Session {
    /// Load a GGUF model and open an inference context using the
    /// default [`Loader`]. `n_gpu_layers`: `0` for CPU, `-1` for
    /// "offload as many as fit", positive N offloads that many.
    pub fn load(
        path:         impl AsRef<Path>,
        n_ctx:        u32,
        n_batch:      u32,
        n_ubatch:     u32,
        n_seq_max:    u32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let loader = default_loader()?;
        Self::load_with(loader, path, n_ctx, n_batch, n_ubatch, n_seq_max, n_gpu_layers)
    }

    /// Explicit-loader variant.
    pub fn load_with(
        loader:       Arc<Loader>,
        path:         impl AsRef<Path>,
        n_ctx:        u32,
        n_batch:      u32,
        n_ubatch:     u32,
        n_seq_max:    u32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        let model = Box::new(Model::load_from_file_with(loader, path, n_gpu_layers)?);

        // Safety: we extend the borrow of `model` to `'static` so the
        // resulting Context can be stored alongside it. Sound because:
        //   * `model` is `Box`-allocated, so its address is stable;
        //   * `ctx` lives in an `Option` we drop BEFORE `model` in our
        //     explicit `Drop`;
        //   * `ctx` has no public accessor that escapes `self`.
        let model_ref: &'static Model = unsafe {
            std::mem::transmute::<&Model, &'static Model>(&*model)
        };
        let ctx = Context::new(model_ref, n_ctx, n_batch, n_ubatch, n_seq_max)?;

        Ok(Self { ctx: Some(ctx), model })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn ctx(&mut self) -> &mut Context<'static> {
        self.ctx.as_mut().expect("Session::ctx: context was already dropped")
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        drop(self.ctx.take());
    }
}

// ---------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------

pub struct Batch {
    raw:      sys::llama_batch,
    capacity: i32,
    embd_dim: i32,
    loader:   Arc<Loader>,
}

impl Batch {
    /// Allocate a token-input batch using the default loader.
    pub fn tokens(n_tokens: i32, n_seq_max: i32) -> Self {
        Self::tokens_with(default_loader().expect("no libllama loaded"), n_tokens, n_seq_max)
    }

    pub fn tokens_with(loader: Arc<Loader>, n_tokens: i32, n_seq_max: i32) -> Self {
        assert!(n_tokens > 0, "batch capacity must be > 0");
        let raw = unsafe { (loader.llama_batch_init)(n_tokens, 0, n_seq_max) };
        Self { raw, capacity: n_tokens, embd_dim: 0, loader }
    }

    pub fn embeddings(n_tokens: i32, n_embd: i32, n_seq_max: i32) -> Self {
        Self::embeddings_with(
            default_loader().expect("no libllama loaded"),
            n_tokens, n_embd, n_seq_max,
        )
    }

    pub fn embeddings_with(loader: Arc<Loader>, n_tokens: i32, n_embd: i32, n_seq_max: i32) -> Self {
        assert!(n_tokens > 0 && n_embd > 0, "batch capacity must be > 0");
        let raw = unsafe { (loader.llama_batch_init)(n_tokens, n_embd, n_seq_max) };
        Self { raw, capacity: n_tokens, embd_dim: n_embd, loader }
    }

    pub fn fill_tokens(&mut self, tokens: &[i32], pos_start: i32, logits_last_only: bool) {
        self.fill_tokens_in(tokens, pos_start, 0, logits_last_only)
    }

    pub fn fill_tokens_in(
        &mut self,
        tokens:           &[i32],
        pos_start:        i32,
        seq_id:           i32,
        logits_last_only: bool,
    ) {
        assert_eq!(self.embd_dim, 0, "fill_tokens called on an embedding batch");
        let n = tokens.len();
        assert!(n as i32 <= self.capacity, "n_tokens > batch capacity");
        self.raw.n_tokens = n as i32;
        unsafe {
            for i in 0..n {
                *self.raw.token.add(i) = tokens[i];
                *self.raw.pos.add(i) = pos_start + i as i32;
                *self.raw.n_seq_id.add(i) = 1;
                **self.raw.seq_id.add(i) = seq_id;
                *self.raw.logits.add(i) =
                    if logits_last_only { (i == n - 1) as i8 } else { 1 };
            }
        }
    }

    pub fn fill_embeddings(
        &mut self,
        embd:             &[f32],
        n_tokens:         i32,
        pos_start:        i32,
        logits_last_only: bool,
    ) {
        self.fill_embeddings_in(embd, n_tokens, pos_start, 0, logits_last_only)
    }

    pub fn fill_embeddings_in(
        &mut self,
        embd:             &[f32],
        n_tokens:         i32,
        pos_start:        i32,
        seq_id:           i32,
        logits_last_only: bool,
    ) {
        assert!(self.embd_dim > 0, "fill_embeddings called on a token batch");
        assert!(n_tokens <= self.capacity, "n_tokens > batch capacity");
        let expected = (n_tokens as usize) * (self.embd_dim as usize);
        assert_eq!(embd.len(), expected, "embd slice len mismatch");
        self.raw.n_tokens = n_tokens;
        unsafe {
            std::ptr::copy_nonoverlapping(embd.as_ptr(), self.raw.embd, expected);
            for i in 0..(n_tokens as usize) {
                *self.raw.pos.add(i) = pos_start + i as i32;
                *self.raw.n_seq_id.add(i) = 1;
                **self.raw.seq_id.add(i) = seq_id;
                *self.raw.logits.add(i) =
                    if logits_last_only { (i as i32 == n_tokens - 1) as i8 } else { 1 };
            }
        }
    }

    pub fn n_tokens(&self) -> i32 {
        self.raw.n_tokens
    }

    fn raw_view(&self) -> sys::llama_batch {
        sys::llama_batch {
            n_tokens: self.raw.n_tokens,
            token:    self.raw.token,
            embd:     self.raw.embd,
            pos:      self.raw.pos,
            n_seq_id: self.raw.n_seq_id,
            seq_id:   self.raw.seq_id,
            logits:   self.raw.logits,
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { (self.loader.llama_batch_free)(self.raw_view()) };
    }
}

// ---------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------

fn check_ret(func: &'static str, rc: i32) -> Result<()> {
    match rc {
        0 => Ok(()),
        1 => Err(anyhow!("{func}: no KV slot (rc=1)")),
        _ => Err(anyhow!("{func}: failed (rc={rc})")),
    }
}
