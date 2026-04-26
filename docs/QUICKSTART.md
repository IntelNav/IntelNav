# Quickstart

The actually-works-today path, top to bottom. Every command below is
verified on the dev box (Linux, Ryzen 5 5600X, no GPU needed for the
smoke runs).

---

## 0. Prerequisites

**Fast path:** `bash scripts/provision.sh` handles everything below on
Debian/Ubuntu, Fedora, Arch, and macOS. Manual list follows if you're
on something else.

- **Rust** — stable, 1.88+ (MSRV; see [`rust-toolchain.toml`](../rust-toolchain.toml)).
  The repo pins `channel = "stable"`; rustup users get the matching
  toolchain automatically on first build.
- **System deps** — C toolchain + pkg-config + OpenSSL headers:
  - Debian/Ubuntu: `build-essential pkg-config libssl-dev`
  - Fedora: `@development-tools pkgconf openssl-devel`
  - Arch: `base-devel pkgconf openssl`
  - macOS: `brew install pkg-config openssl`
- **A GGUF model file.** For smoke tests, use a small Qwen2 like
  `qwen2.5-0.5b-instruct-q4_k_m.gguf` (~470 MB) with its
  `tokenizer.json`.
- Optional: [`just`](https://github.com/casey/just) for the recipes
  in [`justfile`](../justfile).

---

## 1. Build

```bash
# one-shot check
cargo check --workspace --all-targets

# release binary
cargo build --release -p intelnav-cli
# → target/release/intelnav
```

---

## 2. Single-node generation

The fastest way to confirm everything works. No network, no TUI, just
the runtime:

```bash
cargo run --release -p intelnav-runtime --example generate -- \
    --gguf /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "Write a haiku about TCP."
```

`tokenizer.json` is auto-located next to the GGUF if it's named
sensibly. Expected: coherent output at ~8 tok/s greedy on CPU.

---

## 3. Layer-split probe

Confirm the arch sniffer + layer-split forward on your host:

```bash
cargo run --release -p intelnav-runtime --example probe -- \
    --gguf /path/to/qwen2.5-0.5b.gguf
```

Prints backend (CUDA/Metal/CPU), CPU, RAM, then a micro-bench
(prefill ms, decode tok/s).

---

## 4. Two-peer localhost pipeline

Proof of cross-process inference. One peer owns layers `[12..24)`,
the driver owns embedding + `[0..12)` + head.

```bash
# terminal 1 — the peer
cargo run --release -p intelnav-runtime --example pipe_peer -- \
    --gguf /path/to/qwen2.5-0.5b.gguf \
    --start 12 --end 24 --bind 127.0.0.1:7717

# terminal 2 — the driver
cargo run --release -p intelnav-runtime --example pipe_driver -- \
    --gguf /path/to/qwen2.5-0.5b.gguf \
    --peers 127.0.0.1:7717 \
    --splits 12 \
    --prompt "Count from 1 to 10."
```

Greedy output is bit-identical to single-process `generate`.
Throughput ~97% of single-process on LAN localhost.

### Three peers

```bash
--peers host-a:7717,host-b:7717,host-c:7717 --splits 6,12,18
```

Driver owns `[0..6)`, peer A owns `[6..12)`, B `[12..18)`, C `[18..N)`.

---

## 5. Bench harness

Apples-to-apples chain timing. Outputs per-segment percentiles
(`front_ms`, `chain_ms`, `head_ms`) and end-to-end tok/s.

```bash
cargo run --release -p intelnav-runtime --example bench_chain -- \
    --gguf /path/to/qwen2.5-0.5b.gguf \
    --peers 127.0.0.1:7717 \
    --splits 12 \
    --prompt "Write a haiku." \
    --warmup-steps 4 \
    --json > bench.json
```

Use `--warmup-steps` to discard JIT / page-fault outliers. Diff two
`--json` runs to track regressions.

---

## 6. Int8 wire + speculative decoding

```bash
# int8 activations on the chain (fp16 is the default)
--wire-dtype int8

# speculative decoding: 1.5B target + 0.5B draft, k=4
--draft /path/to/qwen2.5-0.5b.gguf --spec-k 4
```

Int8 halves the bytes/step on the chain; bit-identical to fp16 on
structured prompts, diverges on long free-form generations (first-token
argmax shifts under quant noise). Speculative decoding is correct on
CPU but currently slower there (batched-k verify is linear in k on
quantized CPU matmul); the win is on GPU.

---

## 7. TUI (chat REPL)

```bash
export INTELNAV_MODELS_DIR=/path/to/models_dir
./target/release/intelnav chat
```

`INTELNAV_MODE=auto` (the default) pings the gateway at 127.0.0.1:8787
with a 350 ms timeout; if no gateway is running, the TUI falls back to
the in-process runtime.

In the REPL:

- `/models` — browse local GGUFs + network + HF catalog.
- `/peers host:port,... splits …` — route this session through a chain.
- `/draft <path> [k]` — enable spec-dec.
- `/wire fp16|int8` — switch wire dtype.

Logs go to `$XDG_STATE_HOME/intelnav/intelnav.log` (the TUI redirects
both tracing and raw FD 2, so native deps can't paint over the
canvas). `-vv` raises verbosity.

---

## 8. Gateway + registry + Python shard

Runs the full M3 bootstrap stack: HTTP registry, one contributor
shard, OpenAI-compatible gateway.

```bash
# 1. build a manifest from a GGUF
cargo run --release -p intelnav-registry -- init /path/to/model.gguf \
    -o /tmp/manifest.toml

# 2. serve the registry
cargo run --release -p intelnav-registry -- serve \
    --manifest /tmp/manifest.toml \
    --bind 127.0.0.1:7810

# 3. run a contributor shard (Python)
cd python/intelnav_shard
pip install -e .
python -m intelnav_shard.shard_server \
    --registry http://127.0.0.1:7810 \
    --role volunteer

# 4. run the gateway
export INTELNAV_REGISTRY_URL=http://127.0.0.1:7810
export INTELNAV_REGISTRY_MODEL=<model cid from manifest>
./target/release/intelnav gateway
```

OpenAI-compatible endpoints on `127.0.0.1:8787`:

- `POST /v1/chat/completions` — streaming + non-streaming.
- `GET  /v1/models`
- `GET  /v1/network/peers`
- `GET  /v1/network/health`

---

## 9. Operator commands

```bash
intelnav init            # write default config + generate peer identity
intelnav doctor          # preflight: gateway reachable, identity valid, mDNS
intelnav health          # snapshot of gateway + upstream + network health
intelnav models          # all models reachable (local + network)
intelnav peers           # every known peer across directories
intelnav ask -m <model> "prompt"  # non-interactive one-shot
```

---

## 10. Test suites

```bash
# everything
cargo test --workspace --no-fail-fast

# end-to-end registry smoke
python python/intelnav_shard/smoke_registry.py

# single-shard session smoke
python python/intelnav_shard/smoke_client.py
```
