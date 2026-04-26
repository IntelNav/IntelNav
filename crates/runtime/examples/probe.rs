//! Report what this machine can do, then optionally run a
//! mini-benchmark so operators can see realistic tok/s before
//! joining the network.
//!
//!     cargo run -p intelnav-runtime --example probe --release
//!     cargo run -p intelnav-runtime --example probe --release -- \
//!         --gguf /path/to/model.gguf --bench-tokens 32

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use candle_core::Tensor;
use clap::Parser;

use intelnav_runtime::{pick_device_with, DevicePref, ModelHandle, Probe, SamplingCfg, Tok};

#[derive(Parser, Debug)]
#[command(name = "probe", about = "Hardware report + optional model micro-bench")]
struct Args {
    #[arg(long, default_value = "auto")]
    device: DevicePref,

    /// If provided, load the GGUF and run a decode benchmark.
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// Tokens to decode in the benchmark.
    #[arg(long, default_value_t = 32)]
    bench_tokens: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse();
    let probe = Probe::collect();
    println!("--- hardware ---");
    print!("{probe}");

    let Some(gguf) = args.gguf else { return Ok(()); };

    let device  = pick_device_with(args.device)?;
    let tok_path = Tok::locate_for(&gguf)
        .ok_or_else(|| anyhow!("no tokenizer.json beside {}", gguf.display()))?;

    println!("\n--- load ---");
    let t = Instant::now();
    let mut model = ModelHandle::load(&gguf, &device)?;
    let tok = Tok::load(&tok_path)?;
    println!("loaded {:?} + tokenizer in {:.2?}", model.kind(), t.elapsed());

    let block_count = model.block_count();

    // Prefill: tokenize a short prompt, measure prompt-pass.
    let prompt = "The quick brown fox jumps over";
    let ids = tok.encode(prompt)?;
    model.reset_cache();
    let t = Instant::now();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let _ = model.forward(&input, 0)?;
    let prefill_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Decode: greedy, ignore logits processor — we just want timings.
    let mut tokens = ids.clone();
    let cfg = SamplingCfg { temperature: 0.0, top_p: None, ..Default::default() };
    let _ = cfg; // SamplingCfg unused in tight bench; greedy via argmax

    let t = Instant::now();
    for _ in 0..args.bench_tokens {
        let last = *tokens.last().unwrap();
        let idx = tokens.len() - 1;
        let input = Tensor::new(&[last], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, idx)?;
        let logits = logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
        let id = argmax_u32(&logits)?;
        tokens.push(id);
    }
    let decode_s = t.elapsed().as_secs_f64();
    let tok_per_s = args.bench_tokens as f64 / decode_s;

    println!("\n--- bench ---");
    println!("model:    {} layers", block_count);
    println!("prefill:  {} tokens in {prefill_ms:.1} ms", ids.len());
    println!("decode:   {} tokens in {decode_s:.2}s  →  {tok_per_s:.1} tok/s",
             args.bench_tokens);
    Ok(())
}

fn argmax_u32(logits: &Tensor) -> Result<u32> {
    let v = logits.to_vec1::<f32>()?;
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    Ok(best as u32)
}
