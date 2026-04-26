# intelnav-runtime

Layer-range inference backend — the piece that lets a peer run just
*some* of a model's layers on a hidden state and pass the result along.

Backend: [`candle`](https://github.com/huggingface/candle) with a
local fork of `quantized_qwen2` that adds `embed`, `forward_range(start, end)`,
`head`, `head_all` (all-position logits, needed for spec-dec verify),
and `truncate_kv_to` (per-layer KV rollback). Bit-identical to the
upstream full forward on q4_k_m.

### Layout

| Module          | Purpose                                                |
| --------------- | ------------------------------------------------------ |
| `device`        | `DevicePref::{Auto, Cpu, Cuda, Metal}`, feature-gated. |
| `model`         | `ModelHandle`, `ModelKind::from_arch` (sniffs GGUF).   |
| `pipeline`      | `Forwarding` / `Pipelined` traits — architecture plug point. |
| `qwen2`         | Qwen2 fork with layer-range forward + KV truncate.     |
| `llama`         | Llama / mistral / deepseek / mixtral via candle's `quantized_llama::ModelWeights` (full forward only; pipeline fork pending). |
| `tokenizer`     | Loader + Qwen chat template.                           |
| `generate`      | Greedy / top-p sampler with repeat penalty.            |
| `hidden`        | `Tensor ↔ ForwardHidden` payload bridge (fp16 default, int8 optional). |
| `chain`         | N-peer pipeline client: `Chain`, `ChainCfg`, `run_turn`. |
| `spec`          | Speculative decoding v1 (greedy draft-verify with compute/transfer overlap). |
| `probe`         | Host probe (backend + CPU/RAM + micro-bench).          |

### Examples

- `generate` — single-process end-to-end generation.
- `probe` — one-shot host characterization.
- `pipe_peer` — tail-half pipeline peer (localhost TCP smoke).
- `pipe_driver` — driver with 1..N peers; also supports `--draft / --spec-k`.
- `bench_chain` — per-segment percentiles + end-to-end tok/s.
- `smoke_load` — GGUF load smoke test.

Run any of them with `cargo run --release -p intelnav-runtime --example <name> -- …`.

`#![forbid(unsafe_code)]`.
