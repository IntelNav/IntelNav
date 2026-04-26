//! Raw FFI types + function-pointer signatures for the
//! IntelNav-patched libllama.
//!
//! Since the dlopen refactor (task #14 phase 2) this file contains
//! **no `extern "C"` blocks** — the library isn't linked at build
//! time. Function signatures are exposed as `type` aliases; the
//! [`crate::loader::Loader`] dlopens libllama and resolves a
//! function pointer per type alias.
//!
//! Safety: everything in this module is `unsafe` by contract. Use
//! the safe wrappers in `lib.rs` from code outside this crate.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_float};

pub type llama_token  = i32;
pub type llama_pos    = i32;
pub type llama_seq_id = i32;

pub enum llama_model {}
pub enum llama_context {}
pub enum llama_vocab {}
pub enum llama_memory_i {}

// Publicly declared in llama.h:235 — stable across versions, safe to
// mirror. The Rust side only ever constructs a `raw_view` of an
// existing allocation (via `llama_batch_init`) and never touches the
// inner pointers directly.
#[repr(C)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token:    *mut llama_token,
    pub embd:     *mut c_float,
    pub pos:      *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id:   *mut *mut llama_seq_id,
    pub logits:   *mut i8,
}

// --------------------------- function types ---------------------------
//
// One `type` alias per libllama symbol we call. The Loader resolves
// each from `dlsym(libllama, "<symbol_name>")` at startup; calls
// become indirect through the function-pointer field instead of a
// direct extern reference. Zero overhead at call time (one pointer
// deref vs direct symbol resolution).

pub type ggml_backend_load_all_fn = unsafe extern "C" fn();
pub type ggml_backend_load_all_from_path_fn = unsafe extern "C" fn(dir_path: *const c_char);

// model
pub type llama_model_free_fn       = unsafe extern "C" fn(*mut llama_model);
pub type llama_model_get_vocab_fn  = unsafe extern "C" fn(*const llama_model) -> *const llama_vocab;
pub type llama_model_n_embd_fn     = unsafe extern "C" fn(*const llama_model) -> i32;
pub type llama_model_n_layer_fn    = unsafe extern "C" fn(*const llama_model) -> i32;

// vocab / tokenizer
pub type llama_vocab_n_tokens_fn = unsafe extern "C" fn(*const llama_vocab) -> i32;
pub type llama_tokenize_fn       = unsafe extern "C" fn(
    *const llama_vocab,
    *const c_char,
    i32,
    *mut llama_token,
    i32,
    bool,
    bool,
) -> i32;

// context lifecycle
pub type llama_free_fn = unsafe extern "C" fn(*mut llama_context);

// batch
pub type llama_batch_init_fn = unsafe extern "C" fn(i32, i32, i32) -> llama_batch;
pub type llama_batch_free_fn = unsafe extern "C" fn(llama_batch);

// forward
pub type llama_decode_fn        = unsafe extern "C" fn(*mut llama_context, llama_batch) -> i32;
pub type llama_embed_only_fn    = unsafe extern "C" fn(*mut llama_context, llama_batch) -> i32;
pub type llama_decode_layers_fn = unsafe extern "C" fn(
    *mut llama_context,
    llama_batch,
    i32, // layer_start
    i32, // layer_end
    bool, // run_head
) -> i32;
pub type llama_head_only_fn     = unsafe extern "C" fn(*mut llama_context, llama_batch) -> i32;

// outputs
pub type llama_get_logits_ith_fn     = unsafe extern "C" fn(*mut llama_context, i32) -> *mut c_float;
pub type llama_get_embeddings_ith_fn = unsafe extern "C" fn(*mut llama_context, i32) -> *mut c_float;

// KV cache
pub type llama_get_memory_fn     = unsafe extern "C" fn(*const llama_context) -> *mut llama_memory_i;
pub type llama_memory_seq_rm_fn  = unsafe extern "C" fn(
    *mut llama_memory_i,
    llama_seq_id,
    llama_pos,
    llama_pos,
) -> bool;

// setters
pub type llama_set_embeddings_fn = unsafe extern "C" fn(*mut llama_context, bool);

// IntelNav shim (lives inside libllama since task #14.2)
pub type intelnav_load_model_fn  = unsafe extern "C" fn(*const c_char, i32) -> *mut llama_model;
pub type intelnav_new_context_fn = unsafe extern "C" fn(
    *mut llama_model,
    u32, // n_ctx
    u32, // n_batch
    u32, // n_ubatch
    u32, // n_seq_max
) -> *mut llama_context;
pub type intelnav_trip_abort_fn  = unsafe extern "C" fn(bool);
