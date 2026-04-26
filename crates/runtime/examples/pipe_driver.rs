//! `pipe_driver` — head peer + sampler, drives a chain of `pipe_peer`s.
//!
//! **Single peer:**
//!
//! ```bash
//!   cargo run --release -p intelnav-runtime --example pipe_peer -- \
//!     --gguf qwen2.5-0.5b.gguf --start 12 --end 24 --bind 127.0.0.1:7717
//!
//!   cargo run --release -p intelnav-runtime --example pipe_driver -- \
//!     --gguf qwen2.5-0.5b.gguf --peers 127.0.0.1:7717 --splits 12 \
//!     --prompt "Write a haiku about TCP."
//! ```
//!
//! **Three peers:** `--peers a:7717,b:7717,c:7717 --splits 6,12,18`
//! means driver owns `[0..6)`, peer A owns `[6..12)`, B `[12..18)`,
//! C `[18..N)`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;

use intelnav_runtime::{
    pick_device_with, qwen_chat_prompt, run_turn, run_turn_spec, Chain, ChainCfg, DevicePref,
    ModelHandle, SamplingCfg, SpecCfg, Tok,
};

#[derive(Parser, Debug)]
#[command(name = "pipe_driver", about = "IntelNav pipeline driver — 1..N peers")]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', required = true)]
    peers: Vec<SocketAddr>,

    #[arg(long, value_delimiter = ',', required = true)]
    splits: Vec<u16>,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long)]
    raw: bool,

    #[arg(long, default_value = "cpu")]
    device: DevicePref,

    #[arg(long, default_value_t = 30)]
    step_timeout_secs: u64,

    /// Draft GGUF for speculative decoding (Qwen2 family — must share
    /// tokenizer with target). Absent = plain chained decode.
    #[arg(long)]
    draft: Option<PathBuf>,

    /// Proposals per spec-dec round. Ignored unless `--draft` is set.
    #[arg(long, default_value_t = 4)]
    spec_k: usize,

    /// Activation dtype on the chain wire. `fp16` is the baseline;
    /// `int8` per-row quant cuts bytes/step ~2× on LAN.
    #[arg(long, default_value = "fp16")]
    wire_dtype: String,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(
                "warn,intelnav_runtime=info,pipe_driver=info")))
        .init();

    let args = Args::parse();
    if args.peers.len() != args.splits.len() {
        return Err(anyhow!(
            "--peers has {} entries but --splits has {}",
            args.peers.len(), args.splits.len()
        ));
    }

    let device = pick_device_with(args.device)?;

    let t0 = Instant::now();
    let mut model = ModelHandle::load(&args.gguf, &device)?;
    let n_blocks = model.block_count() as u16;
    if model.pipelined().is_none() {
        return Err(anyhow!(
            "model arch {:?} is not pipelined — the smoke needs qwen2", model.kind()
        ));
    }
    eprintln!(
        "pipe_driver: loaded {:?} ({} blocks) in {:.2?}; owning [0..{})",
        model.kind(), n_blocks, t0.elapsed(), args.splits[0]
    );

    let mut cfg = ChainCfg::many(args.peers.clone(), args.splits.clone());
    cfg.step_timeout = Duration::from_secs(args.step_timeout_secs);
    cfg.wire_dtype = match args.wire_dtype.trim().to_ascii_lowercase().as_str() {
        "int8" | "i8"  => intelnav_wire::Dtype::Int8,
        "fp16" | "f16" => intelnav_wire::Dtype::Fp16,
        other => return Err(anyhow!(
            "--wire-dtype must be fp16 or int8, got `{other}`"
        )),
    };
    let mut chain = Chain::connect(cfg, n_blocks).await
        .with_context(|| "opening peer chain")?;
    eprintln!("pipe_driver: chain of {} peer(s) ready", chain.peer_count());

    let tok_path = args.tokenizer
        .clone()
        .or_else(|| Tok::locate_for(&args.gguf))
        .ok_or_else(|| anyhow!(
            "no tokenizer.json found next to {}. Pass --tokenizer.", args.gguf.display()
        ))?;
    let tok = Tok::load(&tok_path)?;

    let prompt = if args.raw { args.prompt.clone() } else { qwen_chat_prompt(&args.prompt, None) };

    let sampling = SamplingCfg {
        temperature:    args.temperature,
        top_p:          args.top_p,
        repeat_penalty: args.repeat_penalty,
        repeat_ctx:     64,
        seed:           args.seed,
        max_new_tokens: args.max_new_tokens,
    };

    // If --draft is set, spec-dec v1 is greedy-only: override sampling
    // knobs so the comparison against plain decode is apples-to-apples.
    let draft_model = if let Some(ref p) = args.draft {
        let t_draft = Instant::now();
        let mut h = ModelHandle::load(p, &device)
            .with_context(|| format!("loading draft {}", p.display()))?;
        if h.pipelined().is_none() {
            return Err(anyhow!("draft arch {:?} is not pipelined", h.kind()));
        }
        eprintln!(
            "pipe_driver: loaded draft {:?} in {:.2?}",
            h.kind(), t_draft.elapsed()
        );
        Some(h)
    } else {
        None
    };

    let print_token = |txt: &str| -> Result<()> {
        use std::io::Write;
        print!("{txt}");
        std::io::stdout().flush().ok();
        Ok(())
    };

    let t_gen_start = Instant::now();
    let n_gen = if let Some(draft) = draft_model {
        if args.spec_k < 2 {
            return Err(anyhow!("--spec-k must be >= 2 when --draft is set"));
        }
        let greedy = SamplingCfg {
            temperature:    0.0,
            top_p:          None,
            repeat_penalty: 1.0,
            ..sampling.clone()
        };
        let (n, _draft_back) = run_turn_spec(
            &mut model, &tok, draft, &mut chain, &prompt,
            &greedy, &SpecCfg { k: args.spec_k }, print_token,
        ).await?;
        n
    } else {
        run_turn(
            &mut model, &tok, &mut chain, &prompt, &sampling,
            print_token,
        ).await?
    };
    println!();
    let elapsed = t_gen_start.elapsed().as_secs_f64();
    eprintln!(
        "pipe_driver: generated {n_gen} tokens in {elapsed:.2}s ({:.1} tok/s) across {} peer(s)",
        n_gen as f64 / elapsed, chain.peer_count(),
    );

    chain.close("driver finished").await;
    Ok(())
}
