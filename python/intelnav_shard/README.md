# intelnav_shard

Python host for the contributor shard server (paper §12.1). Shells out
to `llama-server` (from `ggml-org/llama.cpp`) for inference and speaks
the CBOR wire protocol from [`specs/protocol-v1.md`](../../specs/protocol-v1.md)
to the Rust gateway.

### Why Python here

The Rust side handles the coordination plane (gateway, registry,
directories, chain driver, layer-split runtime for Qwen2). Python is
only the shard *host* — the thinnest wrapper around `llama-server`
that speaks our wire protocol. See deviation D3 in
[`docs/dev/PROGRESS.md`](../../docs/dev/PROGRESS.md).

### Binary auto-provisioning

`intelnav_shard.llama_cpp_runtime` detects the host backend
(CUDA / ROCm / Metal / Vulkan / SYCL / CPU) and downloads the matching
official `llama.cpp` release binary on first run. No compile toolchain
needed. Typical cost: ~4 s first run (32 MB Vulkan download), ~755 ms
warm start.

### Entry points

```bash
# dev install
pip install -e .

# single-shard session (serves the whole model on this peer)
python -m intelnav_shard.shard_server \
    --model  /path/to/model.gguf \
    --socket /tmp/intelnav_shard.sock

# registry-joined contributor
python -m intelnav_shard.shard_server \
    --registry http://127.0.0.1:7810 \
    --role     volunteer
```

### Smoke tests

- `smoke_client.py` — full `Hello → SessionInit → SessionAck → Prompt → Token*`
  exchange against a local shard. Verifies X25519 + AES-GCM interop
  with the Rust `crates/crypto`.
- `smoke_registry.py` — 9-check registry end-to-end: signed envelope
  accept/reject, replay window, rate limits, hysteresis, heartbeat
  eviction.

### Layout

```
intelnav_shard/
├── shard_server.py       main entry — wire protocol + llama-server bridge
├── llama_cpp_runtime.py  backend detect + binary download
├── registry_client.py    signed-envelope assign/claim/heartbeat/release
├── model_resolver.py     file:// + https:// weight fetch with sha256
└── wire.py               CBOR helpers (mirrors crates/wire)
```
