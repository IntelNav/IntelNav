# IntelNav Architecture

A tour of the crates, what each one owns, and how a prompt travels
from the user's terminal to a chain of peers and back.

---

## 1. Crate graph

```
                        ┌────────────────┐
                        │ intelnav-cli   │  ← user-facing binary
                        │  (chat REPL,   │
                        │   gateway cmd, │
                        │   operator ops)│
                        └──────┬────┬────┘
                               │    │
                ┌──────────────┘    └──────────────────┐
                ▼                                      ▼
      ┌──────────────────┐                  ┌──────────────────┐
      │ intelnav-gateway │                  │ intelnav-runtime │
      │  HTTP /v1/*,     │                  │  layer-range     │
      │  OpenAI-compat   │                  │  forward pass,   │
      │                  │                  │  chain driver,   │
      │                  │                  │  generate, spec  │
      └──────┬───────────┘                  └─────┬────────────┘
             │                                    │
             ▼                                    ▼
     ┌───────────────┐                    ┌────────────────┐
     │ intelnav-net  │                    │ intelnav-wire  │
     │  directories  │ ◄──────┬─────────► │  CBOR codecs,  │
     │  (mDNS, DHT   │        │           │  Msg enum,     │
     │   stub,       │        │           │  framing       │
     │   registry)   │        │           └─────┬──────────┘
     └─────┬─────────┘        │                 │
           │                  │                 │
           ▼                  ▼                 ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ intelnav-      │  │ intelnav-core  │  │ intelnav-      │
     │   registry     │  │  types, errors,│  │   crypto       │
     │  (HTTP service │  │  Config,       │  │  Ed25519,      │
     │   + manifest)  │  │  PeerId, etc.  │  │  X25519, AES   │
     └────────────────┘  └────────────────┘  └────────────────┘
```

Dependencies flow downward; `core` and `crypto` are leaves.
`runtime` is the only crate that depends on `candle-*` — the rest of
the workspace is inference-backend agnostic.

---

## 2. Crate responsibilities

| Crate            | Purpose                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `core`           | Shared shape-only types: `PeerId`, `SessionId`, `ModelId`, `LayerRange`, `Quant`, `Config`, `RunMode`, `LatencyTier`, `CapabilityV1`. No network or crypto deps. |
| `wire`           | `Msg` enum (paper §A), CBOR encode/decode, 16 MiB length-prefixed framing, `Dtype` for activation precision on the wire. |
| `crypto`         | `Identity` (Ed25519), ephemeral/static X25519 handshake, AES-256-GCM prompt encryption, session key derivation (blake3 XOF over shared secret). |
| `net`            | `PeerDirectory` trait + three impls: `StaticDirectory`, `MdnsDirectory`, `RegistryDirectory`. `DhtDirectory` is a stub until M2 ships libp2p. |
| `runtime`        | Layer-range forward pass on top of candle. Ships a Qwen2 fork that adds `embed / forward_range / head / head_all / truncate_kv_to`. Also: chain driver (`Chain`, `run_turn`, `run_turn_spec`), generate loop, hidden-state tensor↔bytes bridge, device picker, arch sniffer (`ModelKind`). |
| `gateway`        | Axum HTTP server: `/v1/chat/completions`, `/v1/models`, `/v1/network/peers`, `/v1/network/health`. Parses the `intelnav` request extension (paper §10). |
| `registry`       | HTTP shard-registry server. Manifest-driven; signed-envelope RPCs; cloud-seeder hysteresis; reference impl of [`specs/shard-registry-v1.md`](../specs/shard-registry-v1.md). |
| `cli`            | The `intelnav` binary. Ratatui-based chat TUI, `ask`/`gateway`/`models`/`peers`/`health`/`doctor`/`init` subcommands. |
| `python/intelnav_shard` | The contributor shard server. Auto-downloads `llama-server` for the host backend (CUDA/ROCm/Metal/Vulkan/CPU) and shells out to it; registry client; weight resolver. |

---

## 3. Data flow

### Local path (`RunMode::Local`)

```
  user input
      │
      ▼
  tui::run ──► LocalDriver ──► intelnav-runtime::generate
                                      │
                                      ▼
                               ModelHandle (candle)
                                      │
                                      ▼
                                forward (full) ──► logits ──► sample ──► token
                                      │                                    │
                                      └───────────── step loop ◄───────────┘
```

No network involved. One process owns the whole model.

### Pipeline path (`RunMode::Network` with configured peers)

```
  user input
      │
      ▼
  tui::run ──► ChainDriver
                   │
                   ▼
              run_turn(model, chain, tokens)
                   │
                   ▼
         embed → front_forward([0..s0))
                   │
                   ├─► TCP ─── ForwardHidden(prefill, dtype) ──► peer 0  ([s0..s1))
                   │                                                │
                   │◄── TCP ─── ForwardHidden ──────────────────────┘
                   │
                   ├─► TCP ─── ForwardHidden ──► peer 1 ([s1..s2)) ──►  …
                   │
                   ▼
         head_forward → sample → token
                   │
                   └───────────── step loop ◄──
```

Wire state is fp16 by default, int8 optional (per-token symmetric
quant; ~2× smaller bytes/step). Peers maintain a KV cache keyed on
`session_id`; speculative decoding uses `ForwardHidden.kv_truncate_to`
to roll back rejected draft tokens.

### Gateway path (`RunMode::Network` with upstream)

```
  HTTP POST /v1/chat/completions ──► gateway ──► upstream OpenAI-compat
                                       │           (Ollama / LM Studio / vLLM)
                                       ▼
                                  stream tokens back as SSE
```

The per-peer pipeline routing inside the gateway is not yet wired;
today the gateway is a clean pass-through with peer discovery + health
tracking.

---

## 4. Session lifecycle (paper §A)

```
  driver                              peer
  ──────                              ────
  Hello ────────────────────────────► (Hello exchange, capability match)
  SessionInit(x25519_pub, layers) ──► (derive shared secret, ack)
  ◄────────────────────────── SessionAck(shard_x25519_pub)
  Prompt(ciphertext, nonce) ────────► (entry shard only)
  ForwardHidden(prefill) ───────────► (run forward_range, reply)
  ◄──────────────────────── ForwardHidden (out of layer range)
  ForwardHidden(decode) ───► … ◄───
  Heartbeat / AbortSession as needed
```

Only the entry shard sees the plaintext prompt (post-AES). Middle /
tail shards only see hidden states — they never decrypt anything,
because they never need to.

---

## 5. Configuration surface

`Config` (in `core::config`) is loaded from, in order:

1. Compiled-in defaults.
2. `$XDG_CONFIG_HOME/intelnav/config.toml`.
3. `INTELNAV_*` environment variables.

Every field has an `Env:` comment in the source. Notable ones:

- `INTELNAV_MODE` — `auto | local | network`.
- `INTELNAV_MODELS_DIR` — where the CLI scans for GGUFs.
- `INTELNAV_GATEWAY_URL` — where the CLI talks if in network mode.
- `INTELNAV_UPSTREAM_URL` — what the gateway proxies to.
- `INTELNAV_PEERS` + `INTELNAV_SPLITS` — ad-hoc pipeline chain.
- `INTELNAV_DRAFT_MODEL` + `INTELNAV_SPEC_K` — speculative decoding.
- `INTELNAV_WIRE_DTYPE` — `fp16` or `int8`.

---

## 6. Invariants

These hold across the workspace; break them and CI (or code review)
will catch it:

- **`#![forbid(unsafe_code)]`** in every crate except `cli` (which
  drops to `libc::dup2` for the TUI's stderr redirect — clearly
  marked `#[allow(unsafe_code)]`).
- **Bit-identical layer-split.** `forward_range(0..k) → forward_range(k..N) → head`
  must equal `forward(tokens)` to floating-point zero on the same
  device. Tested on qwen2.5-0.5b q4_k_m.
- **Framing cap.** 16 MiB max frame. Refuse larger; don't allocate.
- **Signed envelopes on mutating registry RPCs.** No unauthenticated
  `assign` / `claim` / `heartbeat` / `release`.
- **One language per seam.** Rust is the inference and coordination
  plane; Python is only the shard host (for the llama.cpp bindings).

---

## 7. Pointers

- Normative wire protocol: [`specs/protocol-v1.md`](../specs/protocol-v1.md).
- Registry spec: [`specs/shard-registry-v1.md`](../specs/shard-registry-v1.md).
- Threat model: [`specs/security-v1.md`](../specs/security-v1.md).
- Deep engineering log: [`docs/dev/PROGRESS.md`](dev/PROGRESS.md).
- TUI plan: [`docs/dev/PROGRESS_TUI.md`](dev/PROGRESS_TUI.md).
