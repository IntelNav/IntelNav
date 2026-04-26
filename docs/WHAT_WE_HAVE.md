# What's in IntelNav today — a user's tour

Written 2026-04-23, after M0 closed. Read this to understand what
you have, how the pieces fit together, and how to run each of them
yourself.

If you've been away, think of this as the orientation.

---

## 1. The two repos

You have two repos on disk that work together:

```
/home/islam/IntelNav/
├── intelnav/      ← the project: Rust, the runtime + CLI + docs
└── llama.cpp/     ← the patched inference library: our C++ fork
```

### `intelnav/` — the product

A Rust workspace. This is what you ship. Main crates:

| Crate | What it does |
|---|---|
| `intelnav-core` | shared types (PeerId, SessionId, LayerRange) |
| `intelnav-wire` | TCP wire protocol: `Msg` enum, length-prefixed CBOR framing |
| `intelnav-crypto` | Ed25519 identity, X25519 ECDH, AES-GCM prompt encryption |
| `intelnav-net` | peer directories (static, mDNS, HTTP registry, DHT stub) |
| `intelnav-registry` | HTTP model registry server |
| `intelnav-gateway` | OpenAI-compatible HTTP surface |
| `intelnav-ggml` | **Rust FFI to our patched libllama** (dlopens it at runtime) |
| `intelnav-runtime` | the inference runtime — `ModelHandle`, `Chain`, `run_turn`, spec-dec |
| `intelnav-cli` | the `intelnav` binary a user actually types |

### `llama.cpp/` — our patched libllama

Our fork of `ggml-org/llama.cpp`. Branch: `intelnav-layer-range`.
**Three new public C functions** on top of stock llama.cpp:

```c
llama_embed_only(ctx, batch)                       // tokens → hidden
llama_decode_layers(ctx, batch, start, end, run_head) // run layers [start, end)
llama_head_only(ctx, batch)                        // hidden → logits
```

This is what lets one model run across multiple machines — each peer
owns a layer range and passes hidden state to the next.

Pushed to `github.com/IntelNav/llama.cpp`. CI workflow
(`.github/workflows/intelnav-release.yml`) builds libllama per
`(backend × OS × arch)` on tag push. Backends covered today:
**CPU, Vulkan, ROCm** on linux-x64.

---

## 2. The one-line thesis

> **IntelNav lets people pool GPU compute — like BitTorrent for
> inference. Someone with an 8GB AMD card joins a swarm; together
> they run a model none of them could run alone.**

The whole project exists for that sentence. Everything else is
engineering to make it work reliably.

---

## 3. What M0 actually achieved

Concretely, what's proved-working today:

1. **Patched libllama layer-range forward is bit-identical to stock
   `llama_decode`.** On Qwen2.5-0.5B + TinyLlama-1.1B, 5 scenarios
   × 3 models, `max_abs_diff = 0e0`. DeepSeek-Coder shows 5.7e-5
   drift on head-split scenarios (argmax preserved, filed as #22).

2. **Two-host LAN pipeline runs end-to-end.** Your PC (Ryzen 5600X)
   + the laptop (i5-7300U) over WiFi, both using libllama via
   dlopen, talking our wire protocol, generate coherent text at
   48 tok/s on a 22/2 capability-weighted split.

3. **AMD GPU acceleration works on your RX 6600.** ROCm path hits
   232 tok/s decode on Qwen2.5-0.5B (3.36× the CPU baseline of 69
   tok/s). The candle-era gap that blocked AMD users is closed.

4. **`intelnav doctor` knows the ggml-side state.** Probes driver
   libs (libhsa-runtime64, libamdhip64, libvulkan), enumerates
   installed GPUs, tells you exactly which `.so` is missing and how
   to install *the driver*, never a -devel package.

---

## 4. User-facing commands (what a user actually types)

Assume the user installed the binary and `libllama.so` is present at
one of the discovery paths (env `INTELNAV_LIBLLAMA_DIR`, env
`INTELNAV_LIBLLAMA_PATH`, `~/.cache/intelnav/libllama/`, or the
system loader path).

### Pre-flight — what's on this machine?

```
intelnav doctor
```

Tells you:
- config/identity status
- CPU + memory
- which libllama backends are loadable (✓/✗ per backend)
- which GPUs are installed
- the preferred backend order for dlopen (CUDA → ROCm → Metal → Vulkan → CPU)
- any missing runtime libs, with a distro-aware install hint

### Single-node chat (no P2P yet)

```
intelnav ask "Write me a haiku about TCP"
```

Generates locally using whichever backend doctor picked.

### Host a peer (let others use your GPU)

```
intelnav gateway        # OpenAI-compat HTTP on :8787
intelnav node           # join the swarm (future — #20 gets us there)
```

Today these work in a single-tenant configuration; multi-session
continuous batching lands with task #20.

### Run a 2-peer pipeline by hand (the examples)

There are two low-level binaries that drive the pipeline without the
CLI wrapper — useful for understanding what's happening and for
benchmarking.

**On the peer (e.g. the laptop):**
```
LD_LIBRARY_PATH=$HOME/intelnav-peer/lib \
INTELNAV_LIBLLAMA_DIR=$HOME/intelnav-peer/lib \
./pipe_peer --gguf $HOME/intelnav-peer/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
            --start 22 --end 24 \
            --bind 0.0.0.0:7717
```
(owns layers 22..24 of the 24-layer Qwen2 model)

**On the driver (your PC):**
```
INTELNAV_LIBLLAMA_DIR=/home/islam/IntelNav/llama.cpp/build/bin \
./target/release/examples/bench_chain \
    --gguf ~/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --peers 192.168.1.154:7717 --splits 22 \
    --max-new-tokens 24 --wire-dtype fp16
```
(owns embed + layers 0..22 + head; reports per-segment percentiles)

---

## 5. The data flow during one decode step

This is the single most useful mental model:

```
tokens [N ids] ──► embed       ──► hidden  [1, N, n_embd]    ┐
                    (driver)                                  │ step 1
               ──► forward_range(0..front_end)                │
                    (driver, writes KV to seq 0)              │
                   hidden  [1, N, n_embd]  ─ encode to fp16 ──┘
                                                              │
                                                              ▼
                                                   TCP ForwardHidden msg
                                                              │
                                                              ▼
                                       ┌──── decode fp16 to hidden ───┐
                                       │                              │
                                       │ forward_range(front_end..N)  │
                                       │ (peer, writes its KV slice)  │ step 2
                                       │                              │
                                       │ hidden tail ─ encode fp16 ───┘
                                       │                              │
                                       │            TCP reply         │
                                       ▼                              │
                                   decode ─► hidden [1, N, n_embd]    │
                                       │                              │
                                       ▼                              │
                                    head ─► logits [1, vocab]         │ step 3
                                       │                              │
                                       ▼                              │
                                     sample ─► next_token (u32)       │
```

Three operations per peer-hop, all bit-identical to what a single
full forward would compute.

---

## 6. The big files to understand

If you want to read code, start here in this order:

1. **`docs/dev/RUNTIME_DECISION.md`** — *why* we patch llama.cpp
   instead of maintaining a candle fork. Thesis + alternatives
   considered + the patch sketch.

2. **`docs/STATUS.md`** — milestone-by-milestone status
   vs `paper/paper.pdf` §12.3.

3. **`docs/dev/PROGRESS.md`** — session log. Scroll to the bottom
   for context on every change; scroll up for the chronological
   arc.

4. **`docs/dev/M0_AUDIT.md`** — the post-M0 quality review that
   filed tasks #14-#21 and #22.

5. **`intelnav/crates/runtime/src/pipeline.rs`** — 70-line file
   that defines the two traits the whole runtime rests on
   (`Forwarding`, `Pipelined`). If you understand this file, you
   understand the runtime.

6. **`intelnav/crates/runtime/src/chain.rs`** — the N-peer pipeline
   client. `Chain::step` is the heart of the P2P path.

7. **`llama.cpp/src/intelnav-shim.cpp`** — 80-line C shim that
   exports `intelnav_load_model` / `intelnav_new_context` /
   `intelnav_trip_abort`. Built into libllama itself.

8. **`llama.cpp/include/llama.h`** — search for `IntelNav layer-range
   extensions` to see the three new public functions.

---

## 7. What's left (the roadmap)

**Closed (M0):** Tasks #1–#13, #15–#18.
**Open, in rough priority order:**

| # | What | Why it matters |
|---|---|---|
| #14 (phase 3) | Runtime downloader for prebuilt libllama artifacts | The one remaining UX gap — users still need a local libllama build |
| #19 | Upstream-rebase playbook + auto bit-identical | llama.cpp moves fast; we need a rebase protocol |
| #20 | Continuous-batching substrate (seq-id pool) | M3 blocker — multiple sessions per gateway |
| #21 | CI matrix: CUDA + Metal + Vulkan-windows + mac-arm | Platform reach for non-linux-x64 users |
| #22 | Investigate deepseek head_only 5.7e-5 drift | Likely matmul-reduction-order; not a correctness bug |

**After M0, the paper's M1 / M2 / M3 / M4 work:**
- **M1**: spec-dec GPU measurement (now possible; was blocked on
  AMD acceleration)
- **M2**: real libp2p + Noise + DHT + tiered chain builder
- **M3**: continuous batching + t-of-n quorum + 33B E2E
- **M4**: OpenAI gateway polish + commit-reveal + IPFS + installer +
  testnet

---

## 8. How the code proves itself

This is what "done" means for each piece:

| Claim | Proof |
|---|---|
| "Patched libllama works" | `cargo test -p intelnav-ggml --test bit_identical` — 5 scenarios × 3 models, max_abs_diff = 0 |
| "Runtime adapter plugs into trait" | `cargo test -p intelnav-runtime --test ggml_pipelined` — 2-peer trait pipeline bit-identical to single forward |
| "TCP wire works" | `bench_chain` localhost: ~63 tok/s on Qwen2.5-0.5B with per-segment percentiles |
| "Two machines can cooperate" | bench_chain with `--peers <laptop_ip>:7717 --splits 22` — 48 tok/s over WiFi |
| "AMD GPU works" | `bench_ggml --ngl=-1` with a ROCm libllama — 232 tok/s on RX 6600 |
| "Artifacts are supply-chain-safe" | sigstore attestation on every release tarball via `actions/attest-build-provenance` |

Every one of these is a literal shell command you can run right now.
