//! Minimal single-peer bench for `GgmlBackend` — measures decode tok/s
//! and prefill ms on whatever backend libllama was linked against.
//!
//! Run:
//!
//! ```
//! # CPU baseline
//! cargo run --release --example bench_ggml -p intelnav-runtime -- \
//!     --gguf /path/to/qwen2.5-0.5b.gguf --ngl 0
//!
//! # ROCm on the RX 6600 (requires HSA_OVERRIDE_GFX_VERSION=10.3.0)
//! INTELNAV_LIBLLAMA_DIR=/path/to/llama.cpp/build-rocm/bin \
//! HSA_OVERRIDE_GFX_VERSION=10.3.0 \
//! cargo run --release --example bench_ggml -p intelnav-runtime -- \
//!     --gguf /path/to/qwen2.5-0.5b.gguf --ngl -1
//! ```
//!
//! Outputs:
//!   * prefill_ms  — time to consume the prompt (single `forward` call).
//!   * decode tok/s — decode-only throughput (excludes prefill) over
//!                    `--max-new-tokens` greedy steps.
//!
//! This is the ggml-path mirror of `bench_chain.rs`. Multi-peer bench
//! over TCP comes in task #12 (once chain.rs runs through GgmlBackend).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use clap::Parser;

use intelnav_ggml::Hidden;
use intelnav_runtime::{pipeline::Forwarding, GgmlBackend};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    /// Number of layers to offload to the backend libllama was built for.
    /// `0` forces CPU, `-1` offloads all; positive N offloads that many.
    #[arg(long, default_value_t = 0)]
    ngl: i32,

    #[arg(long, default_value_t = 512)]
    n_ctx: u32,

    #[arg(long, default_value_t = 256)]
    n_batch: u32,

    /// Prompt token IDs (already tokenized). Defaults to a fixed
    /// 4-token Qwen2.5 sequence so the bench doesn't need a tokenizer
    /// file next to the GGUF.
    #[arg(long, value_delimiter = ',')]
    prompt: Option<Vec<u32>>,

    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    /// Discard the first N decode steps from the tok/s average so JIT /
    /// kernel-launch warmup doesn't skew the number.
    #[arg(long, default_value_t = 2)]
    warmup_steps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;

    // 4-token "Hello my name is" on Qwen2.5's tokenizer — the same
    // prompt the bit-identical tests use.
    let prompt_ids: Vec<u32> = args
        .prompt
        .unwrap_or_else(|| vec![9707, 847, 829, 374]);

    eprintln!(
        "bench_ggml: loading {} (ngl={}, n_ctx={}, n_batch={})",
        args.gguf.display(),
        args.ngl,
        args.n_ctx,
        args.n_batch
    );
    let t_load = Instant::now();
    let mut model = GgmlBackend::load(&args.gguf, args.n_ctx, args.n_batch, args.ngl)
        .context("loading GgmlBackend")?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    let n_layer = model.block_count();
    eprintln!(
        "bench_ggml: loaded in {:.0} ms — n_layer={}, prompt={} tok",
        load_ms,
        n_layer,
        prompt_ids.len()
    );

    // Prefill.
    let t_pf = Instant::now();
    let _ = &device; // candle Device kept only for signature parity
    let logits = model.forward(&prompt_ids, 0)?;
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
    let mut index_pos = prompt_ids.len();
    let mut next = greedy(&logits)?;

    // Decode.
    let mut step_us: Vec<u64> = Vec::with_capacity(args.max_new_tokens);
    let mut n_gen = 0usize;
    let t_dec = Instant::now();
    while n_gen < args.max_new_tokens {
        let t_step = Instant::now();
        let logits = model.forward(&[next], index_pos)?;
        index_pos += 1;
        next = greedy(&logits)?;
        step_us.push(t_step.elapsed().as_micros() as u64);
        n_gen += 1;
    }
    let decode_s = t_dec.elapsed().as_secs_f64();

    let warmup = args.warmup_steps.min(n_gen.saturating_sub(1));
    let effective_s: f64 =
        step_us[warmup..].iter().copied().sum::<u64>() as f64 / 1_000_000.0;
    let effective_toks = (n_gen - warmup) as f64;
    let decode_tps = if effective_s > 0.0 { effective_toks / effective_s } else { 0.0 };
    let decode_all_tps = if decode_s > 0.0 { n_gen as f64 / decode_s } else { 0.0 };

    let sorted = {
        let mut s = step_us[warmup..].to_vec();
        s.sort_unstable();
        s
    };
    let p50 = ms(sorted[sorted.len() / 2]);
    let p95_idx = ((0.95 * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    let p95 = ms(sorted[p95_idx]);
    let min = ms(*sorted.first().unwrap());
    let max = ms(*sorted.last().unwrap());

    eprintln!();
    eprintln!("--- bench_ggml ---");
    eprintln!("prefill:  {} tok  ->  {:.1} ms", prompt_ids.len(), prefill_ms);
    eprintln!(
        "decode:   {} tok  ->  {:.2} tok/s  (warmup-excluded),  {:.2} tok/s  (incl warmup)",
        n_gen, decode_tps, decode_all_tps,
    );
    eprintln!("warmup:   {} step(s) discarded", warmup);
    eprintln!("step latency (ms)       min     p50     p95     max");
    eprintln!("                      {:>7.2} {:>7.2} {:>7.2} {:>7.2}", min, p50, p95, max);

    Ok(())
}

fn greedy(logits: &Hidden) -> Result<u32> {
    logits.argmax_last()
}

fn ms(u: u64) -> f64 {
    (u as f64) / 1000.0
}

// Unused import guard when only `forward` is used — keeps clippy quiet.
#[allow(dead_code)]
fn _imports_used(_: &dyn Forwarding) {
    // refer to the trait to silence unused-import
    let _ = std::any::type_name::<dyn Forwarding>();
    let _ = anyhow!("silencer");
}
