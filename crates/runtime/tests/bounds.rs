// Regression tests: every hostile-peer entry into the ggml adapter
// rejects malformed shapes with a clean Err *before* allocating a
// llama.cpp batch or asking the C side to compute. None of these
// should panic, abort, or call exit(1) — that's the whole point of
// the bounds-check layer in `crates/runtime/src/ggml_backend.rs`.
//
//     INTELNAV_TEST_MODEL=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//     cargo test -p intelnav-runtime --test bounds -- --nocapture

use std::path::PathBuf;

use intelnav_ggml::Hidden;
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

fn load_small() -> Option<GgmlBackend> {
    let path = model_path()?;
    Some(GgmlBackend::load(&path, /*n_ctx=*/ 128, /*n_batch=*/ 64, /*n_gpu_layers=*/ 0).unwrap())
}

#[test]
fn forward_range_rejects_oversized_seq() {
    let Some(mut m) = load_small() else {
        eprintln!("skipping: set INTELNAV_TEST_MODEL to a .gguf path to run");
        return;
    };
    let n_ctx = m.n_ctx() as usize;
    let n_embd = m.n_embd() as usize;

    // seq exactly one over n_ctx must Err, not panic.
    let bad_seq = n_ctx + 1;
    let bad = Hidden::new(vec![0.0f32; bad_seq * n_embd], vec![1, bad_seq, n_embd]).unwrap();
    let err = m.forward_range(&bad, 0, 0, 1).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("seq_len") && msg.contains("n_ctx"),
            "expected n_ctx-bound error, got: {msg}");
}

#[test]
fn forward_range_rejects_zero_seq() {
    let Some(mut m) = load_small() else { return; };
    let n_embd = m.n_embd() as usize;
    let bad = Hidden::new(Vec::<f32>::new(), vec![1, 0, n_embd]).unwrap();
    let err = m.forward_range(&bad, 0, 0, 1).unwrap_err();
    assert!(format!("{err:#}").contains("seq_len 0"));
}

#[test]
fn forward_range_rejects_wrong_n_embd() {
    let Some(mut m) = load_small() else { return; };
    let n_embd = m.n_embd() as usize;
    let wrong = n_embd + 7;
    let bad = Hidden::new(vec![0.0f32; 4 * wrong], vec![1, 4, wrong]).unwrap();
    let err = m.forward_range(&bad, 0, 0, 1).unwrap_err();
    assert!(format!("{err:#}").contains("hidden dim mismatch"));
}

#[test]
fn forward_range_rejects_wrong_rank() {
    let Some(mut m) = load_small() else { return; };
    let n_embd = m.n_embd() as usize;
    let bad = Hidden::new(vec![0.0f32; 4 * n_embd], vec![4, n_embd]).unwrap();
    let err = m.forward_range(&bad, 0, 0, 1).unwrap_err();
    assert!(format!("{err:#}").contains("rank-3"));
}

#[test]
fn forward_range_rejects_wrong_batch_dim() {
    let Some(mut m) = load_small() else { return; };
    let n_embd = m.n_embd() as usize;
    let bad = Hidden::new(vec![0.0f32; 2 * 4 * n_embd], vec![2, 4, n_embd]).unwrap();
    let err = m.forward_range(&bad, 0, 0, 1).unwrap_err();
    assert!(format!("{err:#}").contains("batch=1"));
}

#[test]
fn forward_range_rejects_data_len_mismatch() {
    let Some(mut m) = load_small() else { return; };
    let n_embd = m.n_embd() as usize;
    // Hidden::new() itself rejects this so the constructor is bypassed —
    // simulates a path where a Hidden is built up in pieces (or arrives
    // from a stale wire path that escapes validation). The adapter must
    // catch it regardless.
    let bad = Hidden { data: vec![0.0f32; 4 * n_embd - 1], shape: vec![1, 4, n_embd] };
    let err = m.forward_range(&bad, 0, 0, 1).unwrap_err();
    assert!(format!("{err:#}").contains("data len"));
}

#[test]
fn embed_rejects_oversized_token_slice() {
    let Some(mut m) = load_small() else { return; };
    let n_ctx = m.n_ctx() as usize;
    let too_many: Vec<u32> = vec![0u32; n_ctx + 1];
    let err = m.embed(&too_many).unwrap_err();
    assert!(format!("{err:#}").contains("token slice len"));
}

#[test]
fn embed_rejects_empty_token_slice() {
    let Some(mut m) = load_small() else { return; };
    let err = m.embed(&[]).unwrap_err();
    assert!(format!("{err:#}").contains("token slice len"));
}

#[test]
fn forward_rejects_oversized_token_slice() {
    let Some(mut m) = load_small() else { return; };
    let n_ctx = m.n_ctx() as usize;
    let too_many: Vec<u32> = vec![0u32; n_ctx + 1];
    let err = m.forward(&too_many, 0).unwrap_err();
    assert!(format!("{err:#}").contains("token slice len"));
}

#[test]
fn head_rejects_oversized_seq() {
    let Some(mut m) = load_small() else { return; };
    let n_ctx = m.n_ctx() as usize;
    let n_embd = m.n_embd() as usize;
    let bad_seq = n_ctx + 1;
    let bad = Hidden::new(vec![0.0f32; bad_seq * n_embd], vec![1, bad_seq, n_embd]).unwrap();
    let err = m.head(&bad).unwrap_err();
    assert!(format!("{err:#}").contains("seq_len"));
}
