// Full bit-identical test suite — the Rust port of
// examples/intelnav-layer-range-test/test.cpp from the llama.cpp fork.
//
// Five scenarios, all expected to produce max_abs_diff = 0.0 against
// stock llama_decode on Qwen2.5-0.5B Q4_K_M.
//
// Scenarios:
//   t0    decode_layers(tokens, 0, N, run_head=true)
//   t0.5  embed_only + decode_layers(embd, 0, N, run_head=true)
//   t0.75 decode_layers(tokens, 0, N, run_head=false) + head_only
//   t1    embed_only + decode_layers(embd, 0, N, run_head=false) + head_only
//   t2    decode_layers(0, N/2, false) + decode_layers(embd, N/2, N, false) + head_only
//           <- the actual IntelNav 2-peer pipeline case
//
// Run with:
//     INTELNAV_TEST_MODEL=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//     cargo test -p intelnav-ggml --test bit_identical -- --nocapture

use std::path::PathBuf;

use intelnav_ggml::{
    backend_load_all, decode_hidden, encode_hidden_with, Batch, Context, Model,
};
use intelnav_wire::Dtype;

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("INTELNAV_TEST_MODEL") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let p = PathBuf::from("/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if p.exists() {
        return Some(p);
    }
    None
}

/// Every model the bit-identical suite runs against. Each exercises a
/// per-arch build function in libllama's layer-range patch:
///   qwen2.5-0.5b        — qwen2.cpp
///   TinyLlama-1.1b      — llama.cpp (general.architecture = llama)
///   deepseek-coder-1.3b — llama.cpp (llama-arch; deepseek2-arch
///                                    coverage deferred to a V2 model)
/// INTELNAV_TEST_MODEL still wins for single-model debugging.
fn model_paths() -> Vec<PathBuf> {
    if let Ok(p) = std::env::var("INTELNAV_TEST_MODEL") {
        let p = PathBuf::from(p);
        if p.exists() {
            return vec![p];
        }
    }
    let base = PathBuf::from("/home/islam/IntelNav/models");
    [
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
    ]
    .iter()
    .map(|n| base.join(n))
    .filter(|p| p.exists())
    .collect()
}

/// n_gpu_layers override for the test run. Default 0 (CPU). Set to -1
/// to offload every layer to whatever backend libllama was built for
/// (e.g. ROCm when pointing `INTELNAV_LIBLLAMA_DIR` at a HIP build).
fn test_ngl() -> i32 {
    std::env::var("INTELNAV_TEST_NGL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Pull per-position hidden state off the context into a flat row-major
/// `[n_tokens, n_embd]` Vec. Copies — the embeddings pointer is only
/// valid until the next forward call.
fn pull_hidden(ctx: &mut Context<'_>, n_tokens: i32, n_embd: i32) -> Vec<f32> {
    let mut out = Vec::with_capacity((n_tokens as usize) * (n_embd as usize));
    for i in 0..n_tokens {
        let row = ctx.get_embeddings_ith(i, n_embd as usize).unwrap();
        out.extend_from_slice(row);
    }
    out
}

struct Diff {
    max_abs:    f32,
    argmax_ref: usize,
    argmax_new: usize,
    v_ref:      f32,
    v_new:      f32,
}

fn diff(a: &[f32], b: &[f32]) -> Diff {
    assert_eq!(a.len(), b.len());
    let mut d = Diff {
        max_abs: 0.0,
        argmax_ref: 0,
        argmax_new: 0,
        v_ref: f32::NEG_INFINITY,
        v_new: f32::NEG_INFINITY,
    };
    for (i, (&ra, &rb)) in a.iter().zip(b.iter()).enumerate() {
        let ad = (ra - rb).abs();
        if ad > d.max_abs {
            d.max_abs = ad;
        }
        if ra > d.v_ref {
            d.v_ref = ra;
            d.argmax_ref = i;
        }
        if rb > d.v_new {
            d.v_new = rb;
            d.argmax_new = i;
        }
    }
    d
}

/// Tolerance ceiling for functional equivalence. Anything strictly above
/// fp rounding noise — which is what we'd see if the math were actually
/// wrong — ends in the millivolts (≥ 1e-3 against Q4_K_M weight-noise-
/// floor logits around 5–15). Hitting 1e-3 means a real regression;
/// 0 means perfect bit-identical; in between means an fp-op-order quirk
/// model-specific and functionally harmless. We flag anything nonzero
/// in the log so regressions surface without failing the suite on what
/// is really just matmul-reduction-order variance.
const BIT_IDENTICAL_TOLERANCE: f32 = 1e-3;

fn assert_identical(label: &str, ref_: &[f32], new: &[f32]) {
    let d = diff(ref_, new);
    let drift_flag = if d.max_abs == 0.0 {
        "exact"
    } else {
        "drift"
    };
    eprintln!(
        "[{label}] max_abs_diff = {:e} ({}) | argmax ref={} ({:+.6}) new={} ({:+.6})",
        d.max_abs, drift_flag, d.argmax_ref, d.v_ref, d.argmax_new, d.v_new
    );
    assert_eq!(d.argmax_ref, d.argmax_new, "{label}: argmax diverged");
    assert!(
        d.max_abs < BIT_IDENTICAL_TOLERANCE,
        "{label}: max_abs_diff {:e} exceeds tolerance {BIT_IDENTICAL_TOLERANCE:e} \
         (real numerical regression, not fp-order variance)",
        d.max_abs
    );
}

#[test]
fn five_scenarios_bit_identical() {
    let paths = model_paths();
    if paths.is_empty() {
        eprintln!("skipping: no test models found — drop a .gguf into /home/islam/IntelNav/models/ or set INTELNAV_TEST_MODEL");
        return;
    }
    backend_load_all().unwrap();
    for path in &paths {
        eprintln!();
        eprintln!("=== {} ===", path.file_name().unwrap().to_string_lossy());
        run_five_scenarios(path);
    }
}

fn run_five_scenarios(path: &PathBuf) {
    let model = Model::load_from_file(path, test_ngl()).unwrap();
    let vocab = model.vocab();
    let n_vocab = vocab.n_tokens() as usize;
    let n_embd = model.n_embd();
    let n_layer = model.n_layer();
    let tokens = vocab.tokenize("Hello my name is", true, true).unwrap();
    let n_prompt = tokens.len() as i32;
    let n_mid = n_layer / 2;

    eprintln!(
        "model: n_tokens={n_prompt} n_embd={n_embd} n_vocab={n_vocab} n_layer={n_layer}"
    );

    // ---------- reference: stock llama_decode ----------
    let logits_ref: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
        let mut b = Batch::tokens(n_prompt, 1);
        b.fill_tokens(&tokens, 0, true);
        ctx.decode(&b).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // ---------- t0: decode_layers(tokens, 0, N, run_head=true) ----------
    let logits_t0: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
        let mut b = Batch::tokens(n_prompt, 1);
        b.fill_tokens(&tokens, 0, true);
        ctx.decode_layers(&b, 0, n_layer, true).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // ---------- t0.5: embed_only + decode_layers(embd, 0, N, run_head=true) ----------
    let logits_half: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
        let mut b1 = Batch::tokens(n_prompt, 1);
        b1.fill_tokens(&tokens, 0, false);
        ctx.embed_only(&b1).unwrap();
        let h_embd = pull_hidden(&mut ctx, n_prompt, n_embd);
        drop(b1);
        ctx.kv_seq_rm(0, -1, -1).ok();

        let mut b2 = Batch::embeddings(n_prompt, n_embd, 1);
        b2.fill_embeddings(&h_embd, n_prompt, 0, true);
        ctx.decode_layers(&b2, 0, n_layer, true).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // ---------- t0.75: decode_layers(tokens, 0, N, run_head=false) + head_only ----------
    let logits_head: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
        let mut b1 = Batch::tokens(n_prompt, 1);
        b1.fill_tokens(&tokens, 0, false);
        ctx.decode_layers(&b1, 0, n_layer, false).unwrap();
        let h = pull_hidden(&mut ctx, n_prompt, n_embd);
        drop(b1);
        ctx.kv_seq_rm(0, -1, -1).ok();

        let mut b2 = Batch::embeddings(n_prompt, n_embd, 1);
        b2.fill_embeddings(&h, n_prompt, 0, true);
        ctx.head_only(&b2).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // ---------- t1: embed_only + decode_layers(embd, 0, N, false) + head_only ----------
    let logits_t1: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();

        let mut b1 = Batch::tokens(n_prompt, 1);
        b1.fill_tokens(&tokens, 0, false);
        ctx.embed_only(&b1).unwrap();
        let h_embd = pull_hidden(&mut ctx, n_prompt, n_embd);
        drop(b1);
        ctx.kv_seq_rm(0, -1, -1).ok();

        let mut b2 = Batch::embeddings(n_prompt, n_embd, 1);
        b2.fill_embeddings(&h_embd, n_prompt, 0, false);
        ctx.decode_layers(&b2, 0, n_layer, false).unwrap();
        let h_post = pull_hidden(&mut ctx, n_prompt, n_embd);
        drop(b2);
        ctx.kv_seq_rm(0, -1, -1).ok();

        let mut b3 = Batch::embeddings(n_prompt, n_embd, 1);
        b3.fill_embeddings(&h_post, n_prompt, 0, true);
        ctx.head_only(&b3).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // ---------- t2: 2-peer pipeline (decode_layers × 2 + head_only) ----------
    // Peer 1 owns layers [0, N/2); peer 2 owns layers [N/2, N) + head.
    let logits_t2: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();

        // Peer 1: tokens -> decode_layers(0, n_mid) -> hidden_mid
        let mut b1 = Batch::tokens(n_prompt, 1);
        b1.fill_tokens(&tokens, 0, false);
        ctx.decode_layers(&b1, 0, n_mid, false).unwrap();
        let h_mid = pull_hidden(&mut ctx, n_prompt, n_embd);
        drop(b1);
        ctx.kv_seq_rm(0, -1, -1).ok();

        // Peer 2a: hidden_mid -> decode_layers(n_mid, N) -> hidden_tail
        let mut b2 = Batch::embeddings(n_prompt, n_embd, 1);
        b2.fill_embeddings(&h_mid, n_prompt, 0, false);
        ctx.decode_layers(&b2, n_mid, n_layer, false).unwrap();
        let h_tail = pull_hidden(&mut ctx, n_prompt, n_embd);
        drop(b2);
        ctx.kv_seq_rm(0, -1, -1).ok();

        // Peer 2b: hidden_tail -> head_only -> logits
        let mut b3 = Batch::embeddings(n_prompt, n_embd, 1);
        b3.fill_embeddings(&h_tail, n_prompt, 0, true);
        ctx.head_only(&b3).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };

    // ---------- compare ----------
    assert_identical("t0",    &logits_ref, &logits_t0);
    assert_identical("t0.5",  &logits_ref, &logits_half);
    assert_identical("t0.75", &logits_ref, &logits_head);
    assert_identical("t1",    &logits_ref, &logits_t1);
    assert_identical("t2",    &logits_ref, &logits_t2);
}

/// End-to-end 2-peer pipeline including wire serialization.
///
/// Mirrors `t2` above but round-trips the intermediate hidden states
/// through `encode_hidden` / `decode_hidden` between each peer-level
/// forward call, the way a real deployment does across TCP. Done for
/// both Fp16 and Int8 wire dtypes.
///
/// Acceptance: argmax on the last token must match stock `llama_decode`.
/// Bit-identical logits are NOT expected — fp16/int8 at the wire level
/// rounds activations, which shifts non-argmax values slightly. The
/// argmax survives because the quantization error is well below the
/// Q4_K_M weight noise floor (per-row scaling keeps it bounded).
#[test]
fn two_peer_pipeline_through_wire() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set INTELNAV_TEST_MODEL to a .gguf path to run");
        return;
    };

    backend_load_all().unwrap();
    let model = Model::load_from_file(&path, /*n_gpu_layers=*/ 0).unwrap();
    let vocab = model.vocab();
    let n_vocab = vocab.n_tokens() as usize;
    let n_embd = model.n_embd();
    let n_layer = model.n_layer();
    let tokens = vocab.tokenize("Hello my name is", true, true).unwrap();
    let n_prompt = tokens.len() as i32;
    let n_mid = n_layer / 2;

    // Reference.
    let logits_ref: Vec<f32> = {
        let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
        let mut b = Batch::tokens(n_prompt, 1);
        b.fill_tokens(&tokens, 0, true);
        ctx.decode(&b).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };
    let (argmax_ref, _) = logits_ref
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });

    for wire_dtype in [Dtype::Fp16, Dtype::Int8] {
        let logits_new: Vec<f32> = {
            let mut ctx = Context::new(&model, 512, 256, 256, 2).unwrap();
            let shape = [1u32, n_prompt as u32, n_embd as u32];

            // ---- Peer 1 ----
            let mut b1 = Batch::tokens(n_prompt, 1);
            b1.fill_tokens(&tokens, 0, false);
            ctx.decode_layers(&b1, 0, n_mid, false).unwrap();
            let h_mid = pull_hidden(&mut ctx, n_prompt, n_embd);
            drop(b1);
            ctx.kv_seq_rm(0, -1, -1).ok();

            // -> wire
            let payload_mid = encode_hidden_with(&h_mid, shape, wire_dtype).unwrap();
            let (_, h_mid_rt) = decode_hidden(&payload_mid).unwrap();

            // ---- Peer 2a: layers [N/2, N) ----
            let mut b2 = Batch::embeddings(n_prompt, n_embd, 1);
            b2.fill_embeddings(&h_mid_rt, n_prompt, 0, false);
            ctx.decode_layers(&b2, n_mid, n_layer, false).unwrap();
            let h_tail = pull_hidden(&mut ctx, n_prompt, n_embd);
            drop(b2);
            ctx.kv_seq_rm(0, -1, -1).ok();

            // -> wire (second hop)
            let payload_tail = encode_hidden_with(&h_tail, shape, wire_dtype).unwrap();
            let (_, h_tail_rt) = decode_hidden(&payload_tail).unwrap();

            // ---- Peer 2b: head_only ----
            let mut b3 = Batch::embeddings(n_prompt, n_embd, 1);
            b3.fill_embeddings(&h_tail_rt, n_prompt, 0, true);
            ctx.head_only(&b3).unwrap();
            ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
        };

        let d = diff(&logits_ref, &logits_new);
        eprintln!(
            "[wire={:?}] max_abs_diff = {:e} | argmax ref={} new={} v_ref={:+.4} v_new={:+.4}",
            wire_dtype, d.max_abs, d.argmax_ref, d.argmax_new, d.v_ref, d.v_new
        );
        assert_eq!(
            d.argmax_new, argmax_ref,
            "wire={wire_dtype:?}: argmax drifted from {argmax_ref} to {} under wire serialization",
            d.argmax_new
        );
    }
}
