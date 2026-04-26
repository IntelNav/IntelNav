// Integration test: exercise GgmlBackend through the Pipelined trait,
// simulating what chain.rs does on a real N-peer pipeline.
//
// Proves that the ggml adapter plugs into the runtime's existing trait
// machinery and produces the same logits as calling stock llama_decode
// directly — after the task #18 refactor to Hidden-based currency.
//
//     INTELNAV_TEST_MODEL=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//     cargo test -p intelnav-runtime --test ggml_pipelined -- --nocapture

use std::path::PathBuf;

use intelnav_runtime::{
    pipeline::{Forwarding, Pipelined},
    GgmlBackend,
};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INTELNAV_TEST_MODEL") {
        let p = PathBuf::from(p);
        if p.exists() { return Some(p); }
    }
    let p = PathBuf::from("/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if p.exists() { return Some(p); }
    None
}

const PROMPT_IDS: &[u32] = &[9707, 847, 829, 374]; // "Hello my name is" (qwen2.5 tokenizer)

#[test]
fn two_peer_pipeline_via_pipelined_trait() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INTELNAV_TEST_MODEL to a .gguf path to run");
        return;
    };

    // -------- reference: full forward via `Forwarding::forward` --------
    let logits_ref: Vec<f32> = {
        let mut m = GgmlBackend::load(&path, 512, 256, 0).unwrap();
        let logits = m.forward(PROMPT_IDS, 0).unwrap();
        logits.data
    };
    eprintln!(
        "reference: forward() -> {} logits, max = {:.4}",
        logits_ref.len(),
        logits_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    // -------- 2-peer pipeline via the Pipelined trait --------
    let logits_pipe: Vec<f32> = {
        let mut m = GgmlBackend::load(&path, 512, 256, 0).unwrap();
        let n_layer = m.block_count();
        let n_mid = n_layer / 2;

        // Step 1: tokens -> embedding.
        let h_embd = m.embed(PROMPT_IDS).unwrap();

        // Step 2 (peer 1): embedding -> forward_range(0, n_mid).
        m.reset_cache();
        let h_mid = m.forward_range(&h_embd, 0, 0, n_mid).unwrap();

        // Step 3 (peer 2): hidden_mid -> forward_range(n_mid, n_layer).
        // KV cache from step 2 lives at layers [0, n_mid) for
        // positions [0, seq). When we decode layers [n_mid, n_layer) on
        // the same seq positions, those layers' KV gets written — the
        // layers [0, n_mid) KV state is untouched.
        m.reset_cache();
        let h_tail = m.forward_range(&h_mid, 0, n_mid, n_layer).unwrap();

        // Step 4: hidden_tail -> head.
        m.reset_cache();
        let out = m.head(&h_tail).unwrap();
        out.data
    };
    eprintln!(
        "2-peer:    embed + forward_range×2 + head -> {} logits",
        logits_pipe.len(),
    );

    assert_eq!(logits_ref.len(), logits_pipe.len());
    let (mut max_abs, mut argmax_ref, mut argmax_new) = (0.0_f32, 0usize, 0usize);
    let (mut v_ref, mut v_new) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in 0..logits_ref.len() {
        let d = (logits_ref[i] - logits_pipe[i]).abs();
        if d > max_abs { max_abs = d; }
        if logits_ref[i] > v_ref { v_ref = logits_ref[i]; argmax_ref = i; }
        if logits_pipe[i] > v_new { v_new = logits_pipe[i]; argmax_new = i; }
    }
    eprintln!("max_abs_diff = {max_abs:e}, argmax ref={argmax_ref} new={argmax_new}");
    assert_eq!(argmax_ref, argmax_new, "argmax diverged");
    assert_eq!(
        max_abs, 0.0,
        "expected bit-identical logits via Pipelined trait, got max_abs_diff = {max_abs:e}"
    );
}

#[test]
fn truncate_kv_to_rolls_back() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INTELNAV_TEST_MODEL to a .gguf path to run");
        return;
    };

    let mut m = GgmlBackend::load(&path, 512, 256, 0).unwrap();
    let n_layer = m.block_count();

    let h_embd = m.embed(PROMPT_IDS).unwrap();
    m.reset_cache();
    let _ = m.forward_range(&h_embd, 0, 0, n_layer).unwrap();

    m.truncate_kv_to(0).expect("truncate_kv_to(0) must succeed");
}
