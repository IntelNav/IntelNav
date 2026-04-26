//! Standalone end-to-end generation using the layer-split runtime.
//!
//!     cargo run --release -p intelnav-runtime --example generate -- \
//!         --gguf /path/to/model.gguf \
//!         --prompt "Write a haiku about TCP."
//!
//! A tokenizer.json is auto-discovered next to the GGUF.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Parser;

use intelnav_runtime::{
    generate, pick_device_with, qwen_chat_prompt, DevicePref, ModelHandle, SamplingCfg, Tok,
};

#[derive(Parser, Debug)]
#[command(name = "generate", about = "IntelNav runtime standalone generation")]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    /// Path to tokenizer.json (auto-detected next to the GGUF if omitted).
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    system: Option<String>,

    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Skip the Qwen chat template; send the prompt verbatim.
    #[arg(long)]
    raw: bool,

    /// Backend: `auto`, `cpu`, `cuda[:N]`, or `metal[:N]`.
    #[arg(long, default_value = "auto")]
    device: DevicePref,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,intelnav_runtime=info")),
        )
        .init();

    let args = Args::parse();
    let device = pick_device_with(args.device)?;

    let tok_path = args.tokenizer
        .clone()
        .or_else(|| Tok::locate_for(&args.gguf))
        .ok_or_else(|| anyhow!(
            "no tokenizer.json found next to {}. Pass --tokenizer or drop tokenizer.json beside the GGUF.",
            args.gguf.display()
        ))?;

    let t0 = Instant::now();
    let mut model = ModelHandle::load(&args.gguf, &device)?;
    let tok = Tok::load(&tok_path)?;
    eprintln!("loaded {:?} + tokenizer in {:.2?}", model.kind(), t0.elapsed());

    let prompt = if args.raw {
        args.prompt.clone()
    } else {
        qwen_chat_prompt(&args.prompt, args.system.as_deref())
    };

    let cfg = SamplingCfg {
        temperature:    args.temperature,
        top_p:          args.top_p,
        repeat_penalty: args.repeat_penalty,
        repeat_ctx:     64,
        seed:           args.seed,
        max_new_tokens: args.max_new_tokens,
    };

    let t = Instant::now();
    let n = generate(model.forwarding(), &tok, &device, &prompt, &cfg, |chunk| {
        print!("{chunk}");
        std::io::stdout().flush().ok();
        Ok(())
    })?;
    println!();

    let elapsed = t.elapsed().as_secs_f64();
    eprintln!(
        "generated {n} tokens in {elapsed:.2}s ({:.1} tok/s)",
        n as f64 / elapsed
    );
    Ok(())
}
