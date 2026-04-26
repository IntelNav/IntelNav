//! Report what this machine can do, then optionally run a
//! mini-benchmark so operators can see realistic tok/s before
//! joining the network.
//!
//! Mostly superseded by `intelnav doctor` (which covers the same
//! hardware probe plus libllama load + backend resolution). Kept
//! as an example because it exercises the runtime in isolation
//! — handy for bisecting runtime issues without the CLI stack.
//!
//!     cargo run -p intelnav-runtime --example probe --release
//!     cargo run -p intelnav-runtime --example probe --release -- \
//!         --gguf /path/to/model.gguf --bench-tokens 32

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Parser;

use intelnav_runtime::{DevicePref, ModelHandle, Probe, Tok};

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

    let tok_path = Tok::locate_for(&gguf)
        .ok_or_else(|| anyhow!("no tokenizer.json beside {}", gguf.display()))?;

    println!("\n--- load ---");
    let t = Instant::now();
    let mut model = ModelHandle::load(&gguf, args.device)?;
    let tok = Tok::load(&tok_path)?;
    println!("loaded {:?} + tokenizer in {:.2?}", model.kind(), t.elapsed());

    let block_count = model.block_count();

    // Prefill: tokenize a short prompt, measure prompt-pass.
    let prompt = "The quick brown fox jumps over";
    let ids = tok.encode(prompt)?;
    model.reset_cache();
    let t = Instant::now();
    let _ = model.forward(&ids, 0)?;
    let prefill_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Decode: greedy, argmax the head output directly.
    let mut tokens = ids.clone();

    let t = Instant::now();
    for _ in 0..args.bench_tokens {
        let last = *tokens.last().unwrap();
        let idx = tokens.len() - 1;
        let logits = model.forward(&[last], idx)?;
        let id = logits.argmax_last()?;
        tokens.push(id);
    }
    let decode_s = t.elapsed().as_secs_f64();
    let tok_per_s = args.bench_tokens as f64 / decode_s;

    println!("\n--- bench ---");
    println!("model:    {} layers", block_count);
    println!("prefill:  {} tokens in {prefill_ms:.1} ms", ids.len());
    println!("decode:   {} tokens in {decode_s:.2}s  ->  {tok_per_s:.1} tok/s",
             args.bench_tokens);
    Ok(())
}
