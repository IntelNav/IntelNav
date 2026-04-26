# intelnav-registry

Reference implementation of the IntelNav shard registry. Normative
spec: [`specs/shard-registry-v1.md`](../../specs/shard-registry-v1.md).

The registry is the pragmatic bootstrap for the network: it tracks
*who holds which shard of which model* while the DHT (M2) and IPFS
(M4) layers are still landing, and it coordinates the cloud-seeder
hand-off so a fresh network isn't frozen on day one.

### Routes

```
GET  /v1/shards/:model            list shards + seeders (for directory polling)
POST /v1/shards/:model/assign     ask the registry which part to claim
POST /v1/shards/:model/:part/claim      commit the claim
POST /v1/shards/:model/:part/heartbeat  keep the claim live
POST /v1/shards/:model/:part/release    voluntarily leave
GET  /v1/shards/:model/:part/peers      peers serving a specific part
```

### Guarantees

- **Signed envelopes.** Every mutating call carries an Ed25519
  signature over `CBOR(envelope)` binding `peer_id`, `model_cid`,
  `part_id`, `timestamp`, `nonce`.
- **120 s replay window** + nonce tie a signature to its call site.
- **10 min minimum uptime** before a volunteer counts toward
  `desired_k` (burst-Sybil resistance).
- **6/hour assign rate limit** per `peer_id`.
- **Hysteresis.** Cloud peers go `Standby` when `live_volunteers ≥ desired_k`
  and `Resume` when it drops — cloud capacity is parked, not drained.
- **Missed-heartbeat eviction.** Configurable timeout; `Evicted` seeders
  lose their claim.

### Modes

- `intelnav-registry init <gguf>` — read GGUF header + sha256 the file,
  emit a single-part `manifest.toml`. Removes hand-written manifests.
- `intelnav-registry serve --manifest <path> --bind <addr>` — run the
  HTTP service.

Smoke test: `python python/intelnav_shard/smoke_registry.py` — 9/9
checks including rate limits, hysteresis, eviction, signed-envelope
rejection.

`#![forbid(unsafe_code)]`.
