// Smoke test: load Qwen2.5-0.5B via the Rust FFI, run `llama_decode`,
// verify we get logits out. This is the "does the FFI actually work
// end-to-end" test — before the full 5-scenario bit-identical port.
//
// Gated on the model file existing so it doesn't break `cargo test`
// on CI / fresh clones. Run locally with:
//
//     INTELNAV_TEST_MODEL=/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//     cargo test -p intelnav-ggml --test smoke -- --nocapture

use std::path::PathBuf;

use intelnav_ggml::{backend_load_all, Batch, Context, Model};

fn test_ngl() -> i32 {
    std::env::var("INTELNAV_TEST_NGL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INTELNAV_TEST_MODEL") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // Fallback: the well-known dev checkout location.
    let p = PathBuf::from("/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if p.exists() {
        return Some(p);
    }
    None
}

#[test]
fn decode_smoke() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INTELNAV_TEST_MODEL to a .gguf path to run");
        return;
    };

    backend_load_all().unwrap();
    let model = Model::load_from_file(&path, test_ngl()).unwrap();
    assert!(model.n_embd() > 0);
    assert!(model.n_layer() > 0);
    eprintln!(
        "loaded {} (n_embd={}, n_layer={})",
        path.display(),
        model.n_embd(),
        model.n_layer()
    );

    let vocab = model.vocab();
    let n_vocab = vocab.n_tokens() as usize;
    let tokens = vocab.tokenize("Hello my name is", true, true).unwrap();
    assert!(!tokens.is_empty(), "tokenizer returned zero tokens");
    let n_prompt = tokens.len() as i32;
    eprintln!("tokenized to {n_prompt} tokens: {tokens:?}");

    let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
    let mut batch = Batch::tokens(n_prompt, 1);
    batch.fill_tokens(&tokens, 0, /*logits_last_only=*/ true);

    ctx.decode(&batch).unwrap();

    let logits = ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap();
    assert_eq!(logits.len(), n_vocab);

    // Reasonableness: the max logit should be a valid vocab index,
    // logits shouldn't be all zeros or NaN.
    let (mut best_i, mut best_v) = (0_usize, f32::NEG_INFINITY);
    let mut n_nan = 0_usize;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            n_nan += 1;
        } else if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    assert_eq!(n_nan, 0, "got {n_nan} NaN logits");
    eprintln!("argmax = token {best_i} (logit {best_v:.3})");
    assert!(best_v > 0.0, "max logit non-positive — something is off");
}

#[test]
fn decode_layers_matches_decode() {
    // Direct port of the C++ "test 0" — proves the FFI and the patched
    // library agree that decode_layers(tokens, 0, N, run_head=true)
    // produces bit-identical logits to llama_decode.
    let Some(path) = model_path() else {
        eprintln!("skipping: set INTELNAV_TEST_MODEL to a .gguf path to run");
        return;
    };

    backend_load_all().unwrap();
    let model = Model::load_from_file(&path, test_ngl()).unwrap();
    let vocab = model.vocab();
    let n_vocab = vocab.n_tokens() as usize;
    let n_layer = model.n_layer();
    let tokens = vocab.tokenize("Hello my name is", true, true).unwrap();
    let n_prompt = tokens.len() as i32;

    // Reference run: stock llama_decode.
    let logits_ref: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
        let mut batch = Batch::tokens(n_prompt, 1);
        batch.fill_tokens(&tokens, 0, /*logits_last_only=*/ true);
        ctx.decode(&batch).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // Patched run: decode_layers(0, n_layer, run_head=true).
    let logits_new: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
        let mut batch = Batch::tokens(n_prompt, 1);
        batch.fill_tokens(&tokens, 0, /*logits_last_only=*/ true);
        ctx.decode_layers(&batch, 0, n_layer, /*run_head=*/ true).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // Compare.
    let mut max_abs = 0.0_f32;
    let mut argmax_ref = 0usize;
    let mut argmax_new = 0usize;
    let (mut v_ref, mut v_new) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for i in 0..n_vocab {
        let d = (logits_ref[i] - logits_new[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        if logits_ref[i] > v_ref {
            v_ref = logits_ref[i];
            argmax_ref = i;
        }
        if logits_new[i] > v_new {
            v_new = logits_new[i];
            argmax_new = i;
        }
    }
    eprintln!("max_abs_diff = {max_abs:e}");
    eprintln!("argmax ref={argmax_ref}, new={argmax_new}");
    assert_eq!(argmax_ref, argmax_new, "argmax diverged");
    assert_eq!(max_abs, 0.0, "expected bit-identical logits, got max_abs_diff = {max_abs:e}");
}
