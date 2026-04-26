//! End-to-end Phase 2 test: chunk → stitch → load via patched libllama.
//!
//! Covers:
//!
//! * Full-range stitch round-trips through libllama: chunk Qwen,
//!   stitch with every layer + embed + head, load the stitched file,
//!   confirm `n_layer` / `n_embd` match the source.
//! * Mid-slice stitch loads without error: chunk Qwen, stitch only
//!   layers [10..15) with `intelnav.has_embed=false` and
//!   `intelnav.has_head=false`, confirm libllama accepts the model
//!   (no "missing tensor 'token_embd.weight'" throw).
//!
//! Run with:
//!     INTELNAV_LIBLLAMA_DIR=/path/to/libllama/build/bin \
//!     cargo test -p intelnav-ggml --test stitched_load -- --nocapture

use std::path::{Path, PathBuf};

use intelnav_ggml::{backend_load_all, Batch, Context, Model};
use intelnav_model_store::{
    chunk_gguf, stitch_subset, ChunkerOptions, StitchRange,
};

fn qwen_path() -> Option<PathBuf> {
    let p = PathBuf::from("/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if p.exists() { Some(p) } else { None }
}

/// All test models. Each exercises a distinct libllama arch code path:
///   qwen2.5-0.5b        — `qwen2.cpp`
///   tinyllama-1.1b      — `llama.cpp` arch (covers llama/mistral/tinyllama)
///   deepseek-coder-1.3b — same `llama.cpp` arch (deepseek1 variant)
/// The centralized partial-model override in `llama-context.cpp`
/// should make all three work without per-arch patches.
fn all_model_paths() -> Vec<PathBuf> {
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

fn init_once() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .with_target(false)
        .try_init();
    let _ = backend_load_all();
}

fn chunk_cache(cache_dir: &Path, gguf: &Path) -> intelnav_model_store::Manifest {
    if cache_dir.join("manifest.json").exists() {
        let bytes = std::fs::read(cache_dir.join("manifest.json")).unwrap();
        return intelnav_model_store::Manifest::from_json_bytes(&bytes).unwrap();
    }
    let opts = ChunkerOptions {
        output_dir: cache_dir.to_path_buf(),
        overwrite: true,
        dry_run: false,
    };
    chunk_gguf(gguf, &opts).unwrap().manifest
}

#[test]
fn stitch_full_range_loads_in_libllama() {
    init_once();
    let Some(gguf) = qwen_path() else { return; };
    let cache = std::env::temp_dir().join("intelnav-stitched-load-full-cache");
    let manifest = chunk_cache(&cache, &gguf);

    let range = StitchRange {
        start: 0,
        end: manifest.n_layers,
        include_embed: true,
        include_head: true,
    };
    let out = std::env::temp_dir().join("intelnav-stitched-full.gguf");
    let _ = std::fs::remove_file(&out);
    let outcome = stitch_subset(&manifest, &cache, &range, &out).unwrap();
    eprintln!(
        "stitched full: path={} size={} n_tensors={} n_kv={}",
        outcome.path.display(), outcome.size, outcome.n_tensors, outcome.n_kv
    );

    // Load both and compare layer / embed counts. Bit-identical
    // forward is covered by the separate `bit_identical` suite; here
    // we only confirm the stitched file is a valid on-disk image that
    // libllama accepts.
    let orig = Model::load_from_file(&gguf, 0).expect("original qwen loads");
    let stitched = Model::load_from_file(&outcome.path, 0).expect("stitched full loads");
    assert_eq!(orig.n_layer(), stitched.n_layer());
    assert_eq!(orig.n_embd(), stitched.n_embd());

    // Forward on both models and compare logits — full-range stitch
    // should be bit-identical to the source (same tensor bytes,
    // same names, same hparams, just with two extra KV flags).
    let prompt = "Hello my name is";
    let tokens = orig.vocab().tokenize(prompt, true, true).unwrap();
    let n_prompt = tokens.len() as i32;
    let n_vocab = orig.vocab().n_tokens() as usize;

    let fwd = |m: &Model| -> Vec<f32> {
        let mut ctx = Context::new(m, 512, 256, 256, 2).unwrap();
        let mut b = Batch::tokens(n_prompt, 1);
        b.fill_tokens(&tokens, 0, true);
        ctx.decode(&b).unwrap();
        ctx.get_logits_ith(n_prompt - 1, n_vocab).unwrap().to_vec()
    };
    let logits_ref = fwd(&orig);
    let logits_stitched = fwd(&stitched);

    let max_abs = logits_ref.iter().zip(logits_stitched.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("full-range stitch: max_abs_diff vs original = {:e}", max_abs);
    assert!(
        max_abs < 1e-3,
        "full-range stitch diverges from source: max_abs_diff = {max_abs:e}"
    );

    // Argmax must match (this is the real correctness gate — fp
    // reduction order can produce tiny drift, but argmax divergence
    // means actual math error).
    let argmax = |v: &[f32]| v.iter().enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| if x > acc.1 { (i, x) } else { acc }).0;
    assert_eq!(argmax(&logits_ref), argmax(&logits_stitched));
}

#[test]
fn stitch_mid_slice_loads_with_partial_flags() {
    init_once();
    let Some(gguf) = qwen_path() else { return; };
    let cache = std::env::temp_dir().join("intelnav-stitched-load-mid-cache");
    let manifest = chunk_cache(&cache, &gguf);

    let range = StitchRange {
        start: 10,
        end: 15,
        include_embed: false,
        include_head: false,
    };
    let out = std::env::temp_dir().join("intelnav-stitched-mid.gguf");
    let _ = std::fs::remove_file(&out);
    let outcome = stitch_subset(&manifest, &cache, &range, &out).unwrap();
    eprintln!(
        "stitched mid [10..15): path={} size={} n_tensors={} n_kv={}",
        outcome.path.display(), outcome.size, outcome.n_tensors, outcome.n_kv
    );

    // The loader patch should see intelnav.has_embed=false /
    // intelnav.has_head=false and accept the absence of token_embd /
    // output tensors. The model should report n_layer = 5.
    let stitched = Model::load_from_file(&outcome.path, 0)
        .expect("mid-slice stitched model loads with partial flags honored");
    assert_eq!(stitched.n_layer(), 5, "stitched mid-slice should report n_layer=5");
}

/// The correctness gate for pipe_peer's Path B mode:
///
/// * Embed a prompt on the ORIGINAL Qwen, then run
///   `decode_layers(embd, 10, 15, run_head=false)` — this produces the
///   reference hidden state a mid-slice peer would emit in real life.
/// * Take the SAME embedded hidden state, push it into a mid-slice
///   stitched model, and run `decode_layers(embd, 0, 5, run_head=false)`
///   — the renumbered-local equivalent.
///
/// Both paths must produce bit-identical hidden output. If this
/// diverges, a pipe_peer running from chunks would silently compute
/// the wrong thing.
#[test]
fn mid_slice_decode_matches_original_range() {
    use intelnav_ggml::{encode_hidden_with, decode_hidden, Hidden};
    use intelnav_wire::Dtype;

    init_once();
    let Some(gguf) = qwen_path() else { return; };
    let cache = std::env::temp_dir().join("intelnav-stitched-match-cache");
    let manifest = chunk_cache(&cache, &gguf);

    // Stitch the same mid-slice we intend to serve.
    let range = StitchRange {
        start: 10,
        end: 15,
        include_embed: false,
        include_head: false,
    };
    let stitched_path = std::env::temp_dir().join("intelnav-stitched-match.gguf");
    let _ = std::fs::remove_file(&stitched_path);
    let _ = stitch_subset(&manifest, &cache, &range, &stitched_path).unwrap();

    let orig = Model::load_from_file(&gguf, 0).unwrap();
    let mid  = Model::load_from_file(&stitched_path, 0).unwrap();
    let n_embd = orig.n_embd();
    assert_eq!(mid.n_embd(), n_embd);

    let tokens = orig.vocab().tokenize("Hello my name is", true, true).unwrap();
    let n_prompt = tokens.len() as i32;

    // Step 1: embed on the original to get hidden state.
    let embd: Vec<f32> = {
        let mut ctx = Context::new(&orig, 512, 256, 256, 2).unwrap();
        let mut b = Batch::tokens(n_prompt, 1);
        b.fill_tokens(&tokens, 0, true);
        ctx.embed_only(&b).unwrap();
        let mut v = Vec::with_capacity((n_prompt as usize) * (n_embd as usize));
        for i in 0..n_prompt {
            v.extend_from_slice(ctx.get_embeddings_ith(i, n_embd as usize).unwrap());
        }
        v
    };

    // Step 2: reference — decode layers [10..15) on the original.
    let ref_hidden: Vec<f32> = {
        let mut ctx = Context::new(&orig, 512, 256, 256, 2).unwrap();
        let mut b = Batch::embeddings(n_prompt, n_embd, 1);
        b.fill_embeddings(&embd, n_prompt, 0, false);
        ctx.decode_layers(&b, 10, 15, false).unwrap();
        let mut v = Vec::with_capacity((n_prompt as usize) * (n_embd as usize));
        for i in 0..n_prompt {
            v.extend_from_slice(ctx.get_embeddings_ith(i, n_embd as usize).unwrap());
        }
        v
    };

    // Step 3: same input, stitched model, local range [0..5).
    let mid_hidden: Vec<f32> = {
        let mut ctx = Context::new(&mid, 512, 256, 256, 2).unwrap();
        let mut b = Batch::embeddings(n_prompt, n_embd, 1);
        b.fill_embeddings(&embd, n_prompt, 0, false);
        ctx.decode_layers(&b, 0, 5, false).unwrap();
        let mut v = Vec::with_capacity((n_prompt as usize) * (n_embd as usize));
        for i in 0..n_prompt {
            v.extend_from_slice(ctx.get_embeddings_ith(i, n_embd as usize).unwrap());
        }
        v
    };

    // Also round-trip through the wire encoder/decoder (fp16 — what
    // pipe_peer actually uses on the wire) so we exercise the same
    // path a live peer exercises.
    let shape = [1u32, n_prompt as u32, n_embd as u32];
    let rt = encode_hidden_with(&mid_hidden, shape, Dtype::Fp16).unwrap();
    let (_shape, rt_data) = decode_hidden(&rt).unwrap();
    assert_eq!(rt_data.len(), mid_hidden.len());

    let max_abs = ref_hidden.iter().zip(mid_hidden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("mid-slice decode match: max_abs_diff vs original = {:e}", max_abs);
    assert!(
        max_abs < 1e-3,
        "mid-slice stitched decode diverges from original: max_abs_diff = {max_abs:e}"
    );

    // Unused hidden var — silences warning without changing shape.
    let _ = Hidden::new(mid_hidden, vec![1, n_prompt as usize, n_embd as usize]).unwrap();
}

/// Exercise the full chunk → stitch → load → mid-slice decode path on
/// every available test model. Proves the centralized partial-model
/// override (llama-context.cpp: `if (model.output == nullptr …)`
/// forcing `run_head=false`) works arch-agnostic. If this ever fails
/// on a new arch, the arch-specific graph builder needs inspection —
/// NOT a per-file patch, just a look at whether its default-graph
/// path unconditionally touches `model.output`/`output_norm`.
#[test]
fn mid_slice_decode_matches_across_archs() {
    init_once();
    let paths = all_model_paths();
    if paths.is_empty() {
        eprintln!("skipping: no test models available under /home/islam/IntelNav/models");
        return;
    }

    for gguf in &paths {
        let name = gguf.file_name().unwrap().to_string_lossy().to_string();
        eprintln!();
        eprintln!("=== {name} ===");

        // One cache dir per model — avoid collisions if tests run in parallel.
        let cache = std::env::temp_dir().join(format!("intelnav-stitched-arch-{name}"));
        let manifest = chunk_cache(&cache, gguf);
        let n_layers = manifest.n_layers;
        eprintln!("n_layers={n_layers}");

        // Pick a mid-slice inside the model. [n/3, n/3+2) — 2 layers
        // is enough to exercise the forward without taking forever.
        let start = n_layers / 3;
        let end = (start + 2).min(n_layers);
        if end <= start {
            eprintln!("skipping {name}: too few layers ({n_layers}) for a mid-slice");
            continue;
        }

        let range = StitchRange {
            start,
            end,
            include_embed: false,
            include_head: false,
        };
        let stitched_path = std::env::temp_dir().join(format!("intelnav-stitched-arch-{name}.gguf"));
        let _ = std::fs::remove_file(&stitched_path);
        stitch_subset(&manifest, &cache, &range, &stitched_path).unwrap();

        let orig = Model::load_from_file(gguf, 0)
            .unwrap_or_else(|e| panic!("{name}: original load failed: {e:#}"));
        let mid = Model::load_from_file(&stitched_path, 0)
            .unwrap_or_else(|e| panic!("{name}: stitched load failed: {e:#}"));

        let n_embd = orig.n_embd();
        assert_eq!(mid.n_embd(), n_embd, "{name}: embd size mismatch");
        assert_eq!(mid.n_layer(), (end - start) as i32,
            "{name}: stitched should report {} layers, got {}", end - start, mid.n_layer());

        let tokens = orig.vocab().tokenize("Hello", true, true).unwrap();
        let n_prompt = tokens.len() as i32;

        // Embed on the original to get a hidden state to feed both forwards.
        let embd: Vec<f32> = {
            let mut ctx = Context::new(&orig, 512, 256, 256, 2).unwrap();
            let mut b = Batch::tokens(n_prompt, 1);
            b.fill_tokens(&tokens, 0, true);
            ctx.embed_only(&b).unwrap();
            let mut v = Vec::with_capacity((n_prompt as usize) * (n_embd as usize));
            for i in 0..n_prompt {
                v.extend_from_slice(ctx.get_embeddings_ith(i, n_embd as usize).unwrap());
            }
            v
        };

        let pull_after_decode = |m: &Model, s: i32, e: i32| -> Vec<f32> {
            let mut ctx = Context::new(m, 512, 256, 256, 2).unwrap();
            let mut b = Batch::embeddings(n_prompt, n_embd, 1);
            b.fill_embeddings(&embd, n_prompt, 0, false);
            ctx.decode_layers(&b, s, e, false).unwrap();
            let mut v = Vec::with_capacity((n_prompt as usize) * (n_embd as usize));
            for i in 0..n_prompt {
                v.extend_from_slice(ctx.get_embeddings_ith(i, n_embd as usize).unwrap());
            }
            v
        };

        let ref_hidden = pull_after_decode(&orig, start as i32, end as i32);
        let mid_hidden = pull_after_decode(&mid, 0, (end - start) as i32);

        let max_abs = ref_hidden.iter().zip(mid_hidden.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[{name}] range [{start}..{end}) → local [0..{}) max_abs_diff = {:e}",
                  end - start, max_abs);
        // Same tolerance as the bit_identical suite: 1e-3 is "real
        // math broke". Anything below is fp reduction-order noise.
        assert!(max_abs < 1e-3,
            "{name}: mid-slice diverges from original (max_abs_diff = {max_abs:e})");
    }
}
