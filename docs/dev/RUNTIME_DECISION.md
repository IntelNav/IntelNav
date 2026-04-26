# Runtime Decision — Cross-Device Swarm Inference

Status: **decided 2026-04-22**, implementation pending (see tasks).
Supersedes: the candle-only pipeline path described in `paper/paper.pdf` §7.

---

## The thesis

IntelNav only makes sense for users whose hardware is **good enough to
contribute, not good enough to run alone**. A 4090 owner doesn't need
us — llama.cpp runs their 33B locally. A Mac Studio owner doesn't need
us — MLX runs 70B locally. Our user is the person with an RX 6600, an
Arc A770, a Steam Deck, a Ryzen iGPU, a five-year-old laptop: someone
who can host 6–8 GB of weights at GPU speed but not 20 GB.

> **Cross-device GPU acceleration is not an optimization. It is the
> product.** If the swarm path only runs fast on NVIDIA and Apple, the
> project is best for the users who need it least.

Everything else in the roadmap — libp2p, quorum, gateway, installer —
is downstream of this being true.

---

## Where we are today (2026-04-22)

Two inference engines live side by side:

| Engine | Path | Backends supported |
|---|---|---|
| **A. Python shard** (`python/intelnav_shard/`) | Single-node, full model | CUDA, **ROCm**, Metal, Vulkan, SYCL, CPU — via auto-downloaded `llama-server` binary |
| **B. Rust candle runtime** (`crates/runtime/`) | Layer-split pipeline (the P2P special sauce) | CUDA, Metal, CPU only — no ROCm, no Vulkan, no SYCL |

Concretely: the user with an RX 6600 (developer's own machine, and a
huge fraction of the AMD install base) can run a model solo at GPU
speed, but the moment they **join the swarm to host a layer range**,
they fall back to CPU. Same story for Intel Arc, iGPUs, older GPUs.

Engine B is the IntelNav-specific contribution. Engine A is just
llama.cpp. The thing that makes us interesting only works on half the
hardware.

---

## The paths we considered

### Path 1 — Turn on candle's experimental HIP backend

candle upstream has HIP work in progress. Add a `rocm` feature to
`crates/runtime/Cargo.toml`, a `DevicePref::Rocm` variant in
`device.rs`, try it.

- **Pros:** ~100 LOC on our side, one evening if it works.
- **Cons:** candle's HIP is experimental; quantized matmul (q4_k_m,
  our hot path) may not be implemented or may be wrong. Doesn't fix
  Intel Arc, Vulkan, SYCL.
- **Verdict:** rejected as the primary path. A spike may still be
  worthwhile as a parallel exploration, but we are not betting the
  roadmap on it.

### Path 2 — Write HIP kernels for candle ourselves

Fill in missing ops in candle's HIP backend: quantized matmul,
RMSNorm, RoPE, flash-attn equivalent.

- **Pros:** stays in Rust land.
- **Cons:** weeks of HIP kernel work, ongoing maintenance of GPU
  kernels across candle versions, **still doesn't solve Intel/Vulkan/
  SYCL**, reinvents ggml.
- **Verdict:** rejected. Writing and maintaining GPU kernels is not
  where a P2P-inference project spends its engineering.

### Path 3 — Layer-range forward inside llama.cpp (chosen)

Patch llama.cpp to expose three new public functions:

```c
llama_embed_only(ctx, tokens)              -> hidden_state
llama_decode_layers(ctx, hidden_in, start_layer, end_layer) -> hidden_out
llama_head_only(ctx, hidden_in)            -> logits
```

Each peer calls `llama_decode_layers` for its owned layer range. The
hot path (matmul, attention, RoPE, RMSNorm) runs on whatever backend
llama.cpp was compiled for — **CUDA, ROCm, Metal, Vulkan, SYCL, CPU**,
all already upstream, already tuned.

- **Pros:**
  - Universal GPU support in one move: every consumer GPU shipped in
    the last 8 years is covered.
  - **Collapses two engines into one.** Python shard and Rust runtime
    share a backend. One place to patch bugs, one compatibility matrix.
  - **Model coverage explodes.** Candle fork only pipelines Qwen2
    today (llama/mistral/deepseek/mixtral are full-forward only). With
    ggml, every model llama.cpp supports — DeepSeek Coder 33B, Llama 3,
    Mistral, Mixtral, Qwen2.5, future Llama 4 — works in swarm mode on
    day one.
  - Faster single-peer throughput across the board (ggml's kernels are
    better tuned than candle's, especially on CPU).
  - Delete ~3000 lines of maintained candle fork.
- **Cons:**
  - FFI boundary: C++ lifetimes vs Rust lifetimes. Expect to hunt a
    segfault or two during bring-up.
  - Runs on our fork of llama.cpp until the patch is upstreamed.
  - Bit-identical property has to be re-earned against ggml (was
    previously proved against candle).
  - Build pipeline gets a C++ dependency — mitigated by prebuilt
    binaries (see below).
- **Verdict: chosen.** See "Why" below.

---

## Why Path 3

The candle fork was originally justified because llama.cpp's public
API didn't expose partial-layer forward — `llama_decode` runs all
layers or nothing. That was a real limitation in 2025, and forking
candle was a reasonable call at the time.

But the situation evolved:

1. Our user hardware profile (RX 6600 and equivalents) makes
   non-CUDA/Metal support **load-bearing, not optional**.
2. llama.cpp's internal layer loop is already there; exposing it is a
   small, well-scoped patch, not a rewrite.
3. We are already shipping llama.cpp to users via the Python shard.
   There is no new dependency, only a different use of an existing one.
4. Running two engines is a standing tax on every feature:
   maintenance, model support, bug surface, docs.

The cost of Path 3 is **a few weeks of focused engineering**. The cost
of not doing it is **ongoing ecosystem drag, a capped user base, and
worse UX for the exact users the project exists to serve**.

---

## What does NOT change

The P2P protocol is untouched by this decision:

- `crates/wire` — `Msg` enum, `ForwardHidden`, CBOR framing, 16 MiB cap
- `crates/net` — directories, (future) libp2p, Noise XX, DHT
- `crates/crypto` — Ed25519 identity, X25519 ECDH, AES-256-GCM prompts
- `crates/registry` — signed-envelope RPCs, hysteresis, heartbeats
- The chain driver, session handshake, per-step timeout, structured errors

Peer A still sends fp16/int8 hidden state bytes over TCP to Peer B.
Peer B still owns `[k..k+N)`. Peer B still sends to Peer C. Byte-for-byte
identical wire.

The bit-identical guarantee that the paper leans on — "N-peer chain
output equals single-process full forward, `max_abs_diff=0`" — must
hold on the new backend too. That is the gate for declaring the port
done (task #10).

---

## What DOES change

Inside `crates/runtime`:

- `quantized_qwen2.rs` (the candle fork) — **deleted** once ggml path
  is green.
- `hidden.rs` — gains a ggml-tensor ↔ wire-bytes side. Same layout,
  same fp16/int8, same round-trip tests.
- The `Pipelined` trait — stays as an abstraction, but its impl is
  ggml-backed instead of candle-backed.
- `device.rs` — probes via ggml (CUDA / ROCm / Metal / Vulkan / SYCL /
  CPU) instead of candle's narrower set.
- New crate `crates/ggml` — FFI bindings to our patched llama.cpp,
  runtime dlopen of the matching prebuilt library.

---

## Deployment: users do NOT build from source

This is load-bearing for the UX.

**What we do (once per release):** GitHub Actions compiles the patched
llama.cpp for each `{backend × OS × arch}`, publishes signed artifacts
to GitHub Releases. Roughly:

```
libllama-cuda-linux-x64.so
libllama-rocm-linux-x64.so
libllama-vulkan-linux-x64.so
libllama-vulkan-windows-x64.dll
libllama-metal-macos-arm64.dylib
libllama-cpu-linux-x64.so
libllama-cpu-macos-arm64.dylib
libllama-cpu-windows-x64.dll
...
```

**What the user does:** runs `intelnav`. On first launch it probes the
hardware (same mechanism the Python shard already uses), picks the
matching artifact, downloads it (~30–80 MB) into
`~/.cache/intelnav/libllama/<backend>-<version>/`, `dlopen`s it.
Subsequent runs are instant.

**What the user does NOT do:** install cmake, install clang, install a
GPU SDK, wait 15 minutes for a C++ compile, debug build errors.

The one thing we cannot ship is the GPU **runtime driver itself**
(NVIDIA proprietary driver, ROCm runtime libs, Vulkan loader). Our
`intelnav doctor` command surfaces this clearly:

```
$ intelnav doctor
✓ detected AMD GPU (gfx1032, RX 6600)
✗ ROCm runtime not found
  install:  sudo apt install rocm-libs
  or continue with CPU backend (~7× slower)
```

---

## Performance expectations (pre-implementation estimates)

| Peer hardware | Today (candle) | After Path 3 (ggml) |
|---|---|---|
| NVIDIA consumer | GPU speed | GPU speed (±) |
| Apple Silicon | GPU speed | GPU speed (±) |
| **AMD (RX 6000/7000)** | CPU speed | **GPU speed (5–20× faster)** |
| **Intel Arc** | CPU speed | **GPU speed (5–20× faster)** |
| Vulkan fallback (iGPU, older) | CPU speed | **GPU speed** |
| Pure CPU | candle CPU | ggml CPU (~2× faster from better kernels) |

Network latency: unchanged (same wire format, same RTT, same CBOR).
VRAM: slightly lower (ggml is leaner than candle for the same model).

---

## Risks and how we mitigate them

1. **FFI segfaults during bring-up.** → The `intelnav-ggml` crate is
   written defensively; every ggml handle has clear ownership; tokio
   cancellations are handled explicitly. First milestone is a passing
   bit-identical test, not a fast one.
2. **Bit-identical property fails on ggml.** → We port that test
   *first*, before anything else depends on the ggml path. If it
   fails, we debug before building further. Plan to catch byte-order
   issues, tensor-layout mismatches, quantization-dequant ordering.
3. **Upstream llama.cpp rejects the patch.** → We maintain a minimal
   fork indefinitely. The patch is small (~200–500 lines); rebasing
   on upstream releases is tractable. We still try to upstream; if
   Georgi merges, great. If not, fine.
4. **Build matrix complexity in CI.** → We don't invent this problem;
   llama.cpp's own release pipeline already builds this matrix. We
   copy their workflow and swap in our fork's URL.
5. **Users on systems with no GPU runtime installed.** → `doctor`
   surfaces it; CPU fallback still works.

---

## Milestone impact

This decision adds an implicit **M0 gate** that must pass before the
rest of the roadmap is meaningful:

> **M0 — Single-engine cross-device runtime.** Patched llama.cpp +
> `crates/ggml` FFI + ggml-backed `Pipelined` impl. Bit-identical
> chain test passes. ROCm bench on RX 6600 shows GPU speed for the
> layer-split path.

M1's "GPU perf measurement of speculative decoding" item, previously
blocked by candle not supporting our hardware, becomes measurable
after M0 completes.

See task list for the concrete breakdown.

---

## Open questions (to resolve during implementation)

1. Does ggml's graph allocator let us hand it an already-materialized
   hidden-state tensor as input, or do we need to go through a "set
   tensor data" call? → **resolved during the API survey (below):
   yes, `llama_batch_init(n_tokens, embd, n_seq_max)` + `ubatch.embd`
   already routes a pre-computed embedding tensor into the layer
   stack, bypassing `get_rows(tok_embd, …)`. Our patch reuses this
   path and adds layer-range control.**
2. Does `llama_decode`'s internal KV cache layout allow truncation
   (our `truncate_kv_to` op for speculative decoding rollback)? →
   **resolved: yes, `llama_memory_seq_rm(mem, seq_id, p0, p1)` is
   public and supports range removal. No 4th patched function
   needed — we call this from the Rust FFI crate directly.**
3. How do we avoid re-downloading libllama on every minor update? →
   artifact is versioned by our fork's release tag; cache key
   includes version; old versions garbage-collected after 30 days.

---

## llama.cpp API survey (2026-04-22)

Pulled from `ggml-org/llama.cpp` master (`include/llama.h`,
`src/llama-graph.h`, `src/llama-graph.cpp`). Confirms Path 3 is
smaller than initially scoped.

### What already exists

- **Opaque `llama_context` and `llama_model`** — clean boundary for
  adding public functions without touching ABI-surfaced structs.
- **`llama_batch` already supports embedding input.** `llama_batch_init(
  n_tokens, embd, n_seq_max)` takes an `embd` size (`0` = tokens
  path, `>0` = embeddings path). The `ubatch.token` vs `ubatch.embd`
  fields are mutually exclusive, and `build_inp_embd` already
  branches on `ubatch.token ? 0 : 1` to skip the token-embedding
  lookup when `embd` is set. **This means the "peer B receives a
  hidden tensor and feeds it into layer k" direction is already
  supported internally — we just need to expose it and add a layer
  range.**
- **KV cache management is public.** `llama_get_memory(ctx)` +
  `llama_memory_seq_rm(mem, seq_id, p0, p1)` supports range
  truncation per sequence. This satisfies our `kv_truncate_to` op
  for speculative-decoding rollback without any patch — we just
  call it from `crates/ggml`.
- **`llama_model_n_layer(model)`** — public. We read it to bound
  `[start, end)`.
- **`llama_set_embeddings(ctx, true)`** + `llama_get_embeddings_ith`
  already extracts final-layer hidden state per token. Useful for
  the head-only case (embed → layers(0..N) → head) as a
  cross-check, though we want *per-layer-boundary* hidden state,
  which `llama_set_embeddings` alone doesn't give us.

### What's missing (what the patch adds)

- **No public way to get intermediate hidden state at an arbitrary
  layer index.** `llama_get_embeddings*` gives final-layer only.
- **No public "start at layer K, stop at layer L" control.** The
  per-arch build functions iterate `for (il = 0; il < n_layer; il++)`
  unconditionally.

The patch therefore is:

1. Add two new fields to `llama_context_params` (or a new dedicated
   struct, TBD): `layer_start`, `layer_end`. When set, the per-arch
   layer loop honors them.
2. Add an internal "output-hidden-at-layer-end" mode alongside the
   existing "output-logits-at-end" mode.
3. Expose three thin public wrappers:

   ```c
   LLAMA_API int32_t llama_embed_only(
       struct llama_context * ctx,
       const llama_token *    tokens,
       int32_t                n_tokens,
       float *                out_hidden,   // [n_tokens * n_embd]
       int32_t                out_capacity);

   LLAMA_API int32_t llama_decode_layers(
       struct llama_context * ctx,
       const float *          hidden_in,    // [n_tokens * n_embd]
       int32_t                n_tokens,
       int32_t                layer_start,
       int32_t                layer_end,
       float *                hidden_out,
       int32_t                out_capacity);

   LLAMA_API int32_t llama_head_only(
       struct llama_context * ctx,
       const float *          hidden_in,
       int32_t                n_tokens,
       float *                logits_out,   // [n_tokens * n_vocab]
       int32_t                out_capacity);
   ```

   Each returns `0` on success, negative on error. No allocation on
   our side of the FFI — caller owns buffers.

### Scope estimate (pre-implementation)

- **Per-arch layer-loop patch**: the loop is inside per-arch build
  functions in `src/llama-model.cpp` (Llama, Qwen, DeepSeek,
  Mistral, Mixtral, etc., each ~100–300 LOC). Turning a loop bound
  into a parameter is ~5 lines per arch. With ~30 architectures,
  that's ~150 lines total. **We do NOT need to touch every
  architecture at once** — we land Qwen2 first (it's our test
  target), then DeepSeek (prod target), then take contributions or
  add lazily.
- **Public wrapper functions**: ~100 lines of C in
  `src/llama-impl.cpp` or a new `src/llama-layers.cpp`.
- **Header additions**: ~30 lines in `include/llama.h`.
- **Tests**: port `tests/test-backend-ops.cpp` patterns; the real
  test ("full forward == embed + decode_layers(0..N) + head") lives
  on our side in `crates/ggml`.

Ballpark: **400–600 lines of C++** for the initial patch (two archs
supported). Upstream-PR-worthy size.

### Risks refined by the survey

- The biggest unknown is whether the per-arch layer loop is
  *uniform enough across architectures* that one layer_start/end
  mechanism works for all. Qwen2 and DeepSeek-V2 have slightly
  different attention and expert-routing — our patch must not
  assume a specific variant. Mitigation: only claim support for
  archs we've tested, list them in the header docs, fail loudly on
  others.
- The attention mask and RoPE `position_ids` depend on the absolute
  position of each token in the sequence, not the relative position
  within the current call. So the caller has to pass correct
  positions even when starting mid-layer. This is already the
  contract of `llama_batch` (`batch.pos[]`); we just need to
  document it for our new functions.

---

## Concrete patch sketch (2026-04-22, from clone review)

Clone at `/home/islam/IntelNav/llama.cpp`, commit `0d0764d`.

### Architecture is on our side

Recent llama.cpp has been **refactored to one file per architecture**
in `src/models/` — 104 files, each ~100–300 LOC, each implementing a
single `llm_build_<arch>` struct with the same three-phase shape:

1. **Input** (~10 lines): `build_inp_embd`, `build_inp_pos`,
   `build_attn_inp_*`, `build_inp_out_ids`.
2. **Layer loop** (~50–250 lines, arch-dependent):
   `for (int il = 0; il < n_layer; ++il)` — identical loop bound
   convention across archs (verified in `qwen2.cpp`, `deepseek2.cpp`,
   `llama.cpp`, `qwen3.cpp`, etc.).
3. **Head** (~10 lines): `build_norm` → `t_embd` → `build_lora_mm` or
   `ggml_mul_mat` → `t_logits` → `ggml_build_forward_expand`.

**One pattern, 104 files.** Only the per-arch files' internals differ;
the wrapping phases are uniform.

### Two-step patch

**Step 1 — plumb layer-range fields through the graph-params struct.**

`src/llama-graph.h:530 struct llm_graph_params` gains three fields:

```cpp
int32_t layer_start  = 0;       // first layer to run, inclusive
int32_t layer_end    = -1;      // last+1 to run, or -1 for n_layer
bool    run_head     = true;    // apply output_norm + lm_head at end
```

`llama_context::process_ubatch(ubatch, gtype, …)` in
`src/llama-context.cpp:1171` gains three optional args forwarded into
`graph_params()` which sets the fields on `llm_graph_params`. Three
new public functions on `llama_context`:

```cpp
int32_t decode_layers(const llama_batch & batch,
                      int32_t layer_start,
                      int32_t layer_end);    // run_head=false implied
int32_t embed_only (const llama_batch & batch);  // layers [0..0), head=false, but need an embed-only path
int32_t head_only  (const llama_batch & batch);  // skip layers, run head only on provided ubatch.embd
```

These route through `process_ubatch` with the right param values.

**Step 2 — make each per-arch file honor the fields.**

Per-arch patch is ~15 lines, same pattern everywhere. Template using
`src/models/qwen2.cpp` as the canonical example:

```diff
@@ src/models/qwen2.cpp @@
-    for (int il = 0; il < n_layer; ++il) {
+    const int32_t il_first = params.layer_start;
+    const int32_t il_last  = params.layer_end >= 0 ? params.layer_end : n_layer;
+    for (int il = il_first; il < il_last; ++il) {
         // ... unchanged per-layer body ...
     }
     cur = inpL;

+    if (il_last < n_layer) {
+        // Stopped mid-stack: return hidden state, skip norm+head.
+        res->t_embd = cur;
+        ggml_build_forward_expand(gf, cur);
+        return;
+    }
+
     cur = build_norm(cur,
             model.output_norm, NULL,
             LLM_NORM_RMS, -1);
     cb(cur, "result_norm", -1);
     res->t_embd = cur;

+    if (!params.run_head) {
+        ggml_build_forward_expand(gf, cur);
+        return;
+    }
+
     // lm_head
     cur = build_lora_mm(model.output, cur);
     // ...
```

**`build_inp_out_ids` nuance:** the final-layer optimization at
`qwen2.cpp:56-59` only fetches output-positions on the last layer.
When we're a middle peer, the next peer needs *all* token positions'
hidden state, not just output positions. The patch must treat
`n_outputs == n_tokens` (force "all positions") whenever
`il_last < n_layer`. `llm_graph_params::n_outputs` already exists;
set it in the context layer when the caller requests a middle range.

**`embed_only` path:** call with `layer_start = layer_end = 0`. The
token-embed lookup runs, `inpL` is the result, layer loop is skipped,
and we return `res->t_embd = inpL` directly — no norm, no head.

**`head_only` path:** `ubatch.embd` is set (not tokens), the layer
loop is skipped entirely (`layer_start = layer_end = 0` with the
embedding-input path taking `inpL` from `ubatch.embd` via the
existing `build_inp_embd` dispatch), then fall through to norm +
head. Requires a second flag — "skip layers but still run head" —
added as `bool run_layers = true`.

Refined three fields on `llm_graph_params`:

```cpp
int32_t layer_start  = 0;
int32_t layer_end    = -1;     // -1 sentinel = n_layer
bool    run_head     = true;   // apply output_norm + lm_head
```

(`run_layers` turns out to be redundant — `layer_start == layer_end`
means "run no layers.")

### Public header additions

`include/llama.h` gains:

```c
// --- IntelNav layer-range extensions (our fork) ---

// Embed-only: tokens → embeddings, no layers, no head.
// On success, hidden state is available via llama_get_embeddings_ith.
LLAMA_API int32_t llama_embed_only(
        struct llama_context * ctx,
          struct llama_batch   batch);

// Decode a layer range. batch must carry embeddings (ubatch.embd)
// for layer_start > 0, or tokens for layer_start == 0. On success,
// hidden state is available via llama_get_embeddings_ith.
LLAMA_API int32_t llama_decode_layers(
        struct llama_context * ctx,
          struct llama_batch   batch,
                 int32_t       layer_start,
                 int32_t       layer_end);

// Head-only: hidden state (via ubatch.embd) → output_norm → lm_head.
// Logits available via llama_get_logits_ith.
LLAMA_API int32_t llama_head_only(
        struct llama_context * ctx,
          struct llama_batch   batch);
```

No new struct exposed across the ABI — fields live in internal
`llm_graph_params`. Users pass layer range as function args.

### Size estimate (post-clone review)

| Change | Where | LOC |
|---|---|---|
| Add 3 fields to `llm_graph_params` | `src/llama-graph.h` | ~5 |
| Plumb through `graph_params()` | `src/llama-context.cpp` | ~15 |
| Add `decode_layers/embed_only/head_only` methods | `src/llama-context.{h,cpp}` | ~60 |
| Public wrappers | `src/llama.cpp`, `include/llama.h` | ~40 |
| Per-arch loop patch (Qwen2) | `src/models/qwen2.cpp` | ~15 |
| Per-arch loop patch (DeepSeek2) | `src/models/deepseek2.cpp` | ~15 |
| Per-arch loop patch (Llama) | `src/models/llama.cpp` | ~15 |
| `n_outputs = n_tokens` guard for partial range | `src/llama-context.cpp` | ~10 |

**Total: ~175 LOC for the initial patch** (down from the 400–600
pre-clone estimate — the refactored per-arch file layout cut the
work significantly). Upstreamable in one PR. Remaining 101 arch
files get the same 15-line patch lazily, as users exercise them.

### What still needs to happen

1. Fork the repo to a GitHub account (user action; not automated).
2. Implement the patch in that fork (task #5).
3. Verify locally that `full_forward(tokens)` equals
   `embed_only(tokens) + decode_layers(hidden, 0, N) + head_only(hidden)`
   bit-for-bit on Qwen2.5-0.5B Q4_K_M before building the FFI crate
   on top.

---

## References

- Paper: `paper/paper.pdf` §7 (original candle-based sharding design)
- Today's status: `docs/STATUS.md`
- Full progress log: `docs/dev/PROGRESS.md`
- Wire protocol: `specs/protocol-v1.md`, `crates/wire/src/lib.rs`
- Python shard (pattern we're replicating): `python/intelnav_shard/`
