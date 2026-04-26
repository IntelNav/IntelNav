//! Sanity check: load a GGUF, run the full forward, then split the
//! transformer at the midpoint (`embed → forward_range(0, N/2) →
//! forward_range(N/2, N) → head`) and assert the two paths produce
//! the same logits. This is the M1 pipeline-split correctness gate.
//!
//! Usage:
//!
//!     cargo run -p intelnav-runtime --example smoke_load -- \
//!         /home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use candle_core::Tensor;

use intelnav_runtime::{pick_device, ModelHandle};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: smoke_load <gguf-path>")
        .into();

    let device = pick_device()?;
    let t0 = Instant::now();
    let mut model = ModelHandle::load(&path, &device)?;
    println!("loaded {:?} in {:.2?}", model.kind(), t0.elapsed());

    let m = model.pipelined()
        .ok_or_else(|| anyhow!("model arch doesn't support layer-split yet"))?;

    let tokens = Tensor::new(&[1u32, 2, 3, 4, 5, 6, 7, 8], &device)?.unsqueeze(0)?;
    let n = m.block_count();
    let split = n / 2;

    // --- full forward ---
    m.reset_cache();
    let t = Instant::now();
    let logits_full = m.forward(&tokens, 0)?;
    println!("full forward: {n} layers in {:.2?}", t.elapsed());

    // --- split forward (same tokens, fresh caches) ---
    m.reset_cache();
    let t = Instant::now();
    let h0 = m.embed(&tokens)?;
    let h_mid = m.forward_range(&h0, 0, 0, split)?;
    let h_out = m.forward_range(&h_mid, 0, split, n)?;
    let logits_split = m.head(&h_out)?;
    println!(
        "split forward: embed + [0..{split}) + [{split}..{n}) + head in {:.2?}",
        t.elapsed()
    );

    // --- compare ---
    let a = logits_full.flatten_all()?.to_vec1::<f32>()?;
    let b = logits_split.flatten_all()?.to_vec1::<f32>()?;
    if a.len() != b.len() {
        return Err(anyhow!(
            "logits shape mismatch: full={} split={}",
            a.len(),
            b.len()
        ));
    }
    let (mut max_abs, mut max_rel) = (0f32, 0f32);
    for (x, y) in a.iter().zip(&b) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        let denom = x.abs().max(1e-6);
        max_rel = max_rel.max(d / denom);
    }
    println!("max_abs_diff={max_abs:e}  max_rel_diff={max_rel:e}  (logits_len={})", a.len());

    if max_abs > 1e-3 {
        return Err(anyhow!(
            "split != full (max_abs_diff={max_abs} > 1e-3) — pipeline would produce wrong tokens"
        ));
    }
    println!("OK — split forward matches full forward.");
    Ok(())
}
