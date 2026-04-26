//! `bench_chain` — measure per-step round-trip latency and end-to-end
//! tok/s across an N-peer pipeline.
//!
//! Mirrors `pipe_driver` but instruments each forward step:
//!
//! * `prefill_ms` — driver front + chain round-trip + head, prompt-pass.
//! * Per-decode-step latency, broken down into:
//!   * `front_ms` — local `front_forward` on the driver.
//!   * `chain_ms` — full round-trip through every peer.
//!   * `head_ms` — local `head_forward` + sampling.
//! * `tok/s` — decode-only throughput (excludes prefill).
//!
//! Reports min / p50 / p95 / mean / max for each segment, plus the
//! aggregate. Greedy-only by design so runs are deterministic and
//! diffable against single-proc `generate`.
//!
//! ```bash
//!   cargo run --release -p intelnav-runtime --example pipe_peer -- \
//!     --gguf qwen2.5-0.5b.gguf --start 12 --end 24 --bind 127.0.0.1:7717
//!
//!   cargo run --release -p intelnav-runtime --example bench_chain -- \
//!     --gguf qwen2.5-0.5b.gguf --peers 127.0.0.1:7717 --splits 12 \
//!     --prompt "Count from 1 to 32." --max-new-tokens 32
//! ```
//!
//! With `--json`, prints a single-line JSON summary to stdout suitable
//! for diff tracking across runs.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use intelnav_ggml::Hidden;
use clap::Parser;
use intelnav_wire::Phase;

use intelnav_runtime::{
    front_forward, head_forward, pick_device_with, qwen_chat_prompt, Chain, ChainCfg, DevicePref,
    ModelHandle, Tok,
};

#[derive(Parser, Debug)]
#[command(name = "bench_chain", about = "IntelNav N-peer pipeline bench harness")]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', required = true)]
    peers: Vec<SocketAddr>,

    #[arg(long, value_delimiter = ',', required = true)]
    splits: Vec<u16>,

    #[arg(long, default_value = "Count from 1 to 32.")]
    prompt: String,

    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    #[arg(long)]
    raw: bool,

    #[arg(long, default_value = "cpu")]
    device: DevicePref,

    #[arg(long, default_value_t = 30)]
    step_timeout_secs: u64,

    #[arg(long, default_value = "fp16")]
    wire_dtype: String,

    /// Discard the first N decode steps from the per-step stats so JIT /
    /// page-fault warmup doesn't skew the percentiles.
    #[arg(long, default_value_t = 2)]
    warmup_steps: usize,

    /// Emit a single-line JSON summary on stdout in addition to the
    /// human-readable report on stderr.
    #[arg(long)]
    json: bool,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(
                "warn,intelnav_runtime=warn,bench_chain=info")))
        .init();

    let args = Args::parse();
    if args.peers.len() != args.splits.len() {
        return Err(anyhow!(
            "--peers has {} entries but --splits has {}",
            args.peers.len(), args.splits.len()
        ));
    }

    let device = pick_device_with(args.device)?;

    let t_load = Instant::now();
    let mut model = ModelHandle::load(&args.gguf, &device)?;
    let n_blocks = model.block_count() as u16;
    if model.pipelined().is_none() {
        return Err(anyhow!(
            "model arch {:?} is not pipelined", model.kind()
        ));
    }
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let mut cfg = ChainCfg::many(args.peers.clone(), args.splits.clone());
    cfg.step_timeout = Duration::from_secs(args.step_timeout_secs);
    cfg.wire_dtype = match args.wire_dtype.trim().to_ascii_lowercase().as_str() {
        "int8" | "i8"  => intelnav_wire::Dtype::Int8,
        "fp16" | "f16" => intelnav_wire::Dtype::Fp16,
        other => return Err(anyhow!(
            "--wire-dtype must be fp16 or int8, got `{other}`"
        )),
    };
    let wire_label = match cfg.wire_dtype {
        intelnav_wire::Dtype::Int8 => "int8",
        _                          => "fp16",
    };

    let mut chain = Chain::connect(cfg, n_blocks).await
        .with_context(|| "opening peer chain")?;
    let n_peers = chain.peer_count();
    let front_end = chain.front_range().1;

    let tok_path = args.tokenizer
        .clone()
        .or_else(|| Tok::locate_for(&args.gguf))
        .ok_or_else(|| anyhow!(
            "no tokenizer.json found next to {}. Pass --tokenizer.", args.gguf.display()
        ))?;
    let tok = Tok::load(&tok_path)?;

    let prompt = if args.raw { args.prompt.clone() } else { qwen_chat_prompt(&args.prompt, None) };
    let prompt_ids = tok.encode(&prompt)?;
    if prompt_ids.is_empty() {
        return Err(anyhow!("prompt tokenized to zero tokens"));
    }
    let prompt_len = prompt_ids.len();

    eprintln!(
        "bench_chain: model {:?} ({} blocks), {} peer(s), front=[0..{}), wire={}, load {:.0} ms",
        model.kind(), n_blocks, n_peers, front_end, wire_label, load_ms
    );
    eprintln!("bench_chain: prompt {} tok, decoding up to {} new", prompt_len, args.max_new_tokens);

    model.reset_cache();
    let mut tokens = prompt_ids.clone();
    let mut index_pos: usize = 0;

    // Prefill — single round-trip, multi-token.
    let t_pf = Instant::now();
    let front = front_forward(&mut model, &tokens, index_pos, front_end)?;
    let tail = chain.step(front, Phase::Prefill).await
        .map_err(|e| anyhow!("{e}"))?;
    let logits = head_forward(&mut model, &tail)?;
    index_pos += tokens.len();
    let mut next = greedy(&logits)?;
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;

    // Decode loop — record per-step segments.
    let mut front_us:  Vec<u64> = Vec::with_capacity(args.max_new_tokens);
    let mut chain_us:  Vec<u64> = Vec::with_capacity(args.max_new_tokens);
    let mut head_us:   Vec<u64> = Vec::with_capacity(args.max_new_tokens);
    let mut step_us:   Vec<u64> = Vec::with_capacity(args.max_new_tokens);
    let mut out_ids:   Vec<u32> = Vec::with_capacity(args.max_new_tokens);
    let mut decoder   = tok.incremental();
    let mut out_text  = String::new();

    let t_decode = Instant::now();
    let mut n_gen = 0usize;
    loop {
        if tok.is_eos(next) || n_gen >= args.max_new_tokens { break; }
        tokens.push(next);
        out_ids.push(next);
        if let Some(chunk) = decoder.push(next)? {
            out_text.push_str(&chunk);
        }
        n_gen += 1;

        let t_step = Instant::now();

        let t = Instant::now();
        let front = front_forward(&mut model, &[next], index_pos, front_end)?;
        let f_us = t.elapsed().as_micros() as u64;

        let t = Instant::now();
        let tail = chain.step(front, Phase::Decode).await
            .map_err(|e| anyhow!("{e}"))?;
        let c_us = t.elapsed().as_micros() as u64;

        let t = Instant::now();
        let logits = head_forward(&mut model, &tail)?;
        index_pos += 1;
        next = greedy(&logits)?;
        let h_us = t.elapsed().as_micros() as u64;

        front_us.push(f_us);
        chain_us.push(c_us);
        head_us.push(h_us);
        step_us.push(t_step.elapsed().as_micros() as u64);
    }
    let decode_s = t_decode.elapsed().as_secs_f64();
    chain.close("bench done").await;

    let warmup = args.warmup_steps.min(n_gen.saturating_sub(1));
    let f_stats = stats(&front_us[warmup..]);
    let c_stats = stats(&chain_us[warmup..]);
    let h_stats = stats(&head_us[warmup..]);
    let s_stats = stats(&step_us[warmup..]);
    let tok_per_s = if decode_s > 0.0 { n_gen as f64 / decode_s } else { 0.0 };

    eprintln!();
    eprintln!("--- bench ---");
    eprintln!("prefill:  {prompt_len} tok in {prefill_ms:.1} ms");
    eprintln!("decode:   {n_gen} tok in {decode_s:.2} s  ->  {tok_per_s:.2} tok/s  ({n_peers} peer(s), {wire_label})");
    eprintln!("warmup discarded: {warmup} step(s)");
    eprintln!("per-step latency (ms)        min     p50     p95    mean     max");
    eprintln!("  front (driver compute) {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}",
              f_stats.min_ms, f_stats.p50_ms, f_stats.p95_ms, f_stats.mean_ms, f_stats.max_ms);
    eprintln!("  chain (peer round-trip){:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}",
              c_stats.min_ms, c_stats.p50_ms, c_stats.p95_ms, c_stats.mean_ms, c_stats.max_ms);
    eprintln!("  head  (driver + sample){:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}",
              h_stats.min_ms, h_stats.p50_ms, h_stats.p95_ms, h_stats.mean_ms, h_stats.max_ms);
    eprintln!("  step  (total wall)     {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}",
              s_stats.min_ms, s_stats.p50_ms, s_stats.p95_ms, s_stats.mean_ms, s_stats.max_ms);
    eprintln!("output preview: {}", preview(&out_text, 72));

    if args.json {
        let tokens_json = ids_to_json(&out_ids);
        let text_json = escape_json(&out_text);
        let json = format!(
            "{{\"n_peers\":{n_peers},\"wire\":\"{wire_label}\",\"prompt_tok\":{prompt_len},\
             \"prefill_ms\":{:.3},\"n_gen\":{n_gen},\"decode_s\":{:.4},\"tok_per_s\":{:.3},\
             \"warmup\":{warmup},\
             \"front_ms\":{},\"chain_ms\":{},\"head_ms\":{},\"step_ms\":{},\
             \"tokens\":{tokens_json},\"text\":\"{text_json}\"}}",
            prefill_ms, decode_s, tok_per_s,
            f_stats.to_json(), c_stats.to_json(), h_stats.to_json(), s_stats.to_json(),
        );
        println!("{json}");
    }

    Ok(())
}

fn ids_to_json(ids: &[u32]) -> String {
    let mut s = String::with_capacity(ids.len() * 6 + 2);
    s.push('[');
    for (i, id) in ids.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&id.to_string());
    }
    s.push(']');
    s
}

fn escape_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Collapse newlines and clip to `max_chars` (char count, not bytes) for
/// a single-line terminal preview. Appends `…` if truncated.
fn preview(text: &str, max_chars: usize) -> String {
    let flat: String = text.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    let count = flat.chars().count();
    if count <= max_chars { flat }
    else {
        let clipped: String = flat.chars().take(max_chars).collect();
        format!("{clipped}…")
    }
}

fn greedy(logits: &Hidden) -> Result<u32> {
    logits.argmax_last()
}

#[derive(Debug, Clone, Copy)]
struct Stats {
    min_ms:  f64,
    p50_ms:  f64,
    p95_ms:  f64,
    mean_ms: f64,
    max_ms:  f64,
    n:       usize,
}

impl Stats {
    fn to_json(&self) -> String {
        format!(
            "{{\"n\":{},\"min\":{:.3},\"p50\":{:.3},\"p95\":{:.3},\"mean\":{:.3},\"max\":{:.3}}}",
            self.n, self.min_ms, self.p50_ms, self.p95_ms, self.mean_ms, self.max_ms,
        )
    }
}

fn stats(samples_us: &[u64]) -> Stats {
    if samples_us.is_empty() {
        return Stats { min_ms: 0.0, p50_ms: 0.0, p95_ms: 0.0, mean_ms: 0.0, max_ms: 0.0, n: 0 };
    }
    let mut s = samples_us.to_vec();
    s.sort_unstable();
    let to_ms = |u: u64| (u as f64) / 1000.0;
    let min  = to_ms(*s.first().unwrap());
    let max  = to_ms(*s.last().unwrap());
    let mean = to_ms(s.iter().copied().sum::<u64>() / s.len() as u64);
    let p50  = to_ms(s[s.len() / 2]);
    // Inclusive percentile index: ceil(p/100 * n) - 1
    let p95_idx = ((95.0 / 100.0) * (s.len() as f64)).ceil() as usize;
    let p95_idx = p95_idx.saturating_sub(1).min(s.len() - 1);
    let p95  = to_ms(s[p95_idx]);
    Stats { min_ms: min, p50_ms: p50, p95_ms: p95, mean_ms: mean, max_ms: max, n: s.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_basic() {
        let s = stats(&[1_000, 2_000, 3_000, 4_000, 5_000]);
        assert_eq!(s.n, 5);
        assert!((s.min_ms  - 1.0).abs() < 1e-9);
        assert!((s.max_ms  - 5.0).abs() < 1e-9);
        assert!((s.p50_ms  - 3.0).abs() < 1e-9);
        assert!((s.p95_ms  - 5.0).abs() < 1e-9);
        assert!((s.mean_ms - 3.0).abs() < 1e-9);
    }

    #[test]
    fn stats_empty_returns_zero() {
        let s = stats(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.min_ms, 0.0);
        assert_eq!(s.max_ms, 0.0);
    }

    #[test]
    fn stats_p95_picks_high_tail() {
        let mut xs: Vec<u64> = (1..=100).map(|i| i * 1_000).collect();
        // Shuffle a bit — stats() must sort internally.
        xs.swap(0, 99);
        xs.swap(10, 50);
        let s = stats(&xs);
        assert_eq!(s.n, 100);
        // p95 of 1..=100 (ms) should be 95.
        assert!((s.p95_ms - 95.0).abs() < 1e-9, "p95 was {}", s.p95_ms);
    }

    #[test]
    fn ids_to_json_roundtrip_shape() {
        assert_eq!(ids_to_json(&[]), "[]");
        assert_eq!(ids_to_json(&[42]), "[42]");
        assert_eq!(ids_to_json(&[1, 2, 3]), "[1,2,3]");
    }

    #[test]
    fn escape_json_handles_specials() {
        assert_eq!(escape_json("plain"), "plain");
        assert_eq!(escape_json("a\"b"), "a\\\"b");
        assert_eq!(escape_json("a\\b"), "a\\\\b");
        assert_eq!(escape_json("a\nb\tc"), "a\\nb\\tc");
        // Control byte (\x07, bell) gets \u escaped.
        assert_eq!(escape_json("x\x07y"), "x\\u0007y");
        // Non-ASCII passes through — JSON allows UTF-8 in string bodies.
        assert_eq!(escape_json("café"), "café");
    }

    #[test]
    fn preview_clips_and_flattens_newlines() {
        assert_eq!(preview("hello", 10), "hello");
        assert_eq!(preview("a\nb\nc", 10), "a b c");
        assert_eq!(preview("abcdefghij", 5), "abcde…");
        // Multi-byte char boundary: each grapheme counts as one char.
        assert_eq!(preview("αβγδε", 3), "αβγ…");
    }
}
