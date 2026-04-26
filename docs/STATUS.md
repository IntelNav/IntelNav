# Status

Distilled from [`docs/dev/PROGRESS.md`](dev/PROGRESS.md). This file is
the *current picture*; PROGRESS is the session log.

Milestone reference: [`paper/paper.pdf`](../../paper/paper.pdf) §12.3.

> **2026-04-22 — runtime pivot.** The layer-split path is moving off
> candle and onto a patched llama.cpp so every consumer GPU (AMD ROCm,
> Intel Arc SYCL, Vulkan, CPU) gets GPU-speed swarm participation, not
> just NVIDIA/Apple. This adds an implicit **M0 — single-engine
> cross-device runtime** gate. Full rationale:
> [`dev/RUNTIME_DECISION.md`](dev/RUNTIME_DECISION.md).

---

## Milestone summary

| Milestone | Paper target | Actual | Notes |
| --------- | ------------ | ------ | ----- |
| **M0** — single-engine cross-device runtime (ggml-backed `Pipelined`, bit-identical chain test green, ROCm bench on RX 6600) | — | **not started** | Added 2026-04-22. Gates meaningful progress on GPU-dependent M1/M3 items. |
| **M1** — 7B two-LAN-peer TCP pipeline, wire format, speculative decoding ≥60% | week 4  | **partial** | All subsystems in place; GPU spec-dec measurement blocked on M0 (candle has no ROCm/Vulkan/SYCL backend). |
| **M2** — libp2p + Noise + DHT, tiered chain builder, WAN pipeline              | week 8  | **not started** | `DhtDirectory` is an in-memory stub. |
| **M3** — continuous batching, t-of-n quorum, HTTP registry, 33B E2E            | week 12 | **partial** | Registry landed early; batching/quorum pending; 33B E2E gated on M0. |
| **M4** — OpenAI gateway, commit-reveal, IPFS, installer, testnet               | week 16 | **partial** | Gateway skeleton done; rest pending. |

---

## What works today

**Infrastructure**
- Wire protocol (paper §A) — `Msg` enum, 16 MiB length-prefixed CBOR framing.
- Crypto — Ed25519 identity, ephemeral/static X25519, AES-256-GCM prompt
  encryption; byte-identical across Rust and Python sides.
- Peer directories — static, mDNS, HTTP registry. DHT is a stub.
- Config loader — XDG TOML + `INTELNAV_*` env overlay.

**Runtime**
- Qwen2 layer-split (candle-backed fork) — `embed / forward_range / head / head_all / truncate_kv_to`, bit-identical to full forward on q4_k_m.
- Arch sniffer — recognizes llama / mistral / deepseek / deepseek2 / mixtral / qwen / baichuan. Qwen2 is fully pipelined; the others have full-forward only (pipeline fork pending).
- Device picker — `auto / cpu / cuda[:N] / metal[:N]`, feature-gated.
- Generate loop — tokenizer auto-discovery, chat template, greedy/top_p with repeat penalty.
- Chain driver — N-peer TCP pipeline, per-step timeout, structured errors (`peer_index (addr): reason`).
- Speculative decoding v1 — greedy draft-verify with compute/transfer overlap, `kv_truncate_to` rollback.
- Int8 wire quantization — per-token symmetric int8, ~2× smaller bytes/step.

**Services**
- HTTP registry — signed-envelope RPCs (Ed25519 over CBOR), 120s replay window, assign rate limits, hysteresis, heartbeat eviction, cloud-seeder standby. 9/9 smoke checks pass.
- Python shard — auto-downloads `llama-server` per host backend (CUDA/ROCm/Metal/Vulkan/SYCL/CPU). 4s first-run, 755ms warm.
- Gateway — OpenAI-compat HTTP surface, upstream proxy, peer listing, health.
- `intelnav-registry init <gguf>` — one-command manifest generation from a GGUF.

**Frontend**
- Ratatui TUI — streaming chat, `/models` browser fusing local + network + HF catalog, `/peers` chain selector, `/draft`, `/wire`, download progress, log redirection that doesn't paint over the canvas.
- Operator CLI — `init`, `doctor`, `health`, `models`, `peers`, `ask`, `gateway`, `node`.

**Benches**
- `probe` — backend + CPU/RAM + micro-bench.
- `bench_chain` — per-segment percentiles + end-to-end tok/s with warmup + JSON output.

---

## What's not built yet

**M0 — single-engine cross-device runtime** *(added 2026-04-22)*
- [x] Fork llama.cpp, add `llama_embed_only / llama_decode_layers / llama_head_only` public functions. `IntelNav/llama.cpp@8097ebe`.
- [~] CI matrix: prebuilt `libllama-<backend>-<os>-<arch>` artifacts + sigstore attestation. `IntelNav/llama.cpp` now has `.github/workflows/intelnav-release.yml` — builds CPU + Vulkan + ROCm on linux-x64 on `intelnav-v*` tag push or workflow_dispatch; packages each as `libllama-<backend>-<sha>.tar.gz`, generates a keyless-sigstore build-provenance attestation per artifact, uploads both to the matching GitHub Release. First cut; CUDA / Metal / SYCL / Windows / macos-arm64 follow the same pattern (task #21). Downloader with attestation verification → task #14 phase 3.
- [x] `crates/ggml` — FFI bindings via `cc`-compiled shim; builds against the local llama.cpp checkout. Runtime `dlopen` + artifact caching deferred until CI artifacts exist.
- [x] Port `hidden.rs` bridge to ggml (`crates/ggml/src/hidden.rs`). Fp16 + per-row Int8 wire dtypes; byte-identical algorithm to the candle-side encoder so mixed peers interop. Wire round-trip pipeline test argmax-preserved on both dtypes.
- [x] ggml-backed `Pipelined` trait impl (`crates/runtime/src/ggml_backend.rs`). Adapter wraps `intelnav_ggml::Session`; `forward_range` + `forward` + `truncate_kv_to` go through the trait; a trait-level 2-peer pipeline test is bit-identical (`max_abs_diff = 0e0`) to the single-call `forward`. The candle fork still ships — task #12 drops it once the runtime consumers (chain driver, spec-dec, examples, CLI) actually route through the adapter.
- [x] Bit-identical chain test green on ggml (`max_abs_diff=0`). 5/5 scenarios from both C++ (`examples/intelnav-layer-range-test/`) and Rust (`crates/ggml/tests/bit_identical.rs`).
- [x] ROCm bench on RX 6600 — Qwen2.5-0.5B Q4_K_M at **232.81 tok/s** on gfx1032 via gfx1030 bytecode + `HSA_OVERRIDE_GFX_VERSION=10.3.0`, vs **69.30 tok/s** on the 5600X CPU. **3.36× speedup**; all 5 bit-identical scenarios still green on GPU. The candle/AMD gap that blocked M1 spec-dec is closed.
- [x] **Two-host LAN smoke** — PC driver + laptop peer (i5-7300U, WiFi), both via dlopen ggml path. 12/12 symmetric Fp16 hits **37.37 tok/s** (chain 14.95 ms/step). Capability-weighted **22/2 hits 47.98 tok/s** (+28% by letting the slower laptop own fewer layers — replicates the candle-era split-sweep finding). 2-peer inference across real machines through the patched libllama, bit-identical wire.
- [x] Deleted the candle fork. `crates/runtime/src/qwen2.rs` + `llama.rs` gone (~500 LOC); `ModelHandle` is now a single-variant enum over `GgmlBackend`; end-to-end `pipe_driver` + `pipe_peer` over TCP generates coherent text on the ggml path (47 tok/s CPU / 80 tok/s ROCm on the dev box). Candle *types* (`Tensor`, `Device`) remain as the currency between consumers; task #15 (future) swaps them for `Vec<f32>` + shape.
- Rationale: [`dev/RUNTIME_DECISION.md`](dev/RUNTIME_DECISION.md).

**M1 leftovers**
- [ ] Baseline latency/throughput measurements. Harness exists (`bench_chain`); cross-host numbers not yet captured.
- [ ] Speculative decoding speedup measurement on GPU. Algorithm verified on CPU; **blocked on M0** — candle has no ROCm/Vulkan/SYCL backend so the developer's RX 6600 can't run the pipeline at GPU speed.

**M2 — P2P substrate**
- [ ] libp2p + Noise XX.
- [ ] Kademlia DHT + `GetProviders` keyed on model CID.
- [ ] Circuit-v2 relay + DCUtR hole punching.
- [ ] RTT probing + GeoIP tier classification.
- [ ] Tiered chain builder (LAN → Continent → WAN), `NoViableRoute`, 4-hop cap.
- [ ] Gossipsub `/intelnav/v1/health` topic.

**M3 — batching & quorum**
- [ ] Continuous batching scheduler (target ≥3× aggregate at 5 concurrent sessions).
- [ ] t-of-n quorum over disjoint chains (default n=3, t=2).
- [ ] Peer reputation (EWMA α=0.02 + ε-exploration).
- [ ] DeepSeek-Coder 33B end-to-end across three 8 GB peers (gated on M2).

**M4 — the public network**
- [ ] Commit-reveal v2 (off by default; slot needed in quorum path).
- [ ] IPFS-backed model distribution.
- [ ] One-line installer (`curl … | sh`).
- [ ] Public testnet — bootstrap peers, signed release manifest, pinned CIDs.

**TUI polish** (see [`docs/dev/PROGRESS_TUI.md`](dev/PROGRESS_TUI.md))
- [ ] Transcript overlap under long sessions on small terminals.
- [ ] Wrap-aware scroll bookkeeping.
- [ ] Paste detection (bracketed paste).
- [ ] `/doctor`, `/resume`, exit-flow dialog, global search.

---

## Deviations from the paper (summarized)

Full list + rationale in [`docs/dev/PROGRESS.md`](dev/PROGRESS.md).

1. **Cloud-seeder as a first-class role** — `Role = Volunteer | Cloud` on `CapabilityV1`; registry hysteresis parks cloud peers when volunteer count is sufficient.
2. **Signed-envelope replay window on registry RPCs** — 120 s + nonce.
3. **llama.cpp binary auto-provisioning** in the Python shard instead of `pip install llama-cpp-python` from source.
4. **Multi-scheme model resolver** — `file://` and `https://` with sha256 verification; IPFS CIDs remain the M4 target.
5. **`Standby` ≠ `Draining`** in the registry lifecycle — lets cloud peers be parked atomically without losing their claim.
6. **candle-transformers instead of llama.cpp** for the layer-split runtime (`crates/runtime`). llama.cpp's public API doesn't expose layer-range forward; candle's Qwen2 fork does.
7. **Runtime is arch-agnostic via a `Pipelined` trait** — Qwen2 is the reference; llama/mistral/deepseek/mixtral hook in via `Forwarding`. Operators can point the runtime at any GGUF they already have.
