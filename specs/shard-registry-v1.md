# IntelNav Shard Registry — v1 (normative)

Pragmatic bootstrap layer for the IntelNav network: while the DHT and IPFS
layers (paper §7, M4) are still landing, a single HTTP service acts as the
authoritative map of *who-holds-which-shard-of-which-model*. It is also the
coordination point for the **cloud-seeder hand-off** strategy (paper §12.3
M3 — "HTTP registry").

Design goal: **cold-start the network from paid cloud capacity, decommission
cloud one part at a time as volunteer peers arrive with enough redundancy to
cover it.** The registry is the component that decides when that is safe.

The Rust implementation lives in `crates/registry`.

## 1. Entities

```
Model { cid, name, quant, total_layers, parts: [Part] }
Part  { id, layer_range, weight_url, sha256, size_bytes,
        min_vram_bytes, desired_k, seeders: [SeederEntry] }
SeederEntry { peer_id, role, joined_at, last_heartbeat, status }
```

`role ∈ { volunteer, cloud }`. `status ∈ { live, standby, draining, evicted }`.
`desired_k` is the target redundancy for a part (default 3).

A model is uniquely identified by `cid` (the IPFS CID of its canonical
GGUF — while we are pre-IPFS, any stable content hash works). The set of
parts is *fixed at model-registration time* — parts are not dynamically
split once announced.

## 2. Manifest

A registry instance is seeded from a TOML manifest file. The manifest is
the authoritative partitioning of a model; the registry refuses to start
if the manifest's `sum(part.layer_range.len)` does not equal
`model.total_layers`.

```toml
[model]
cid           = "bafybeigdyrztdeshard..."
name          = "deepseek-coder:33b"
quant         = "Q4_K_M"
total_layers  = 64

[[part]]
id              = "p1"
layer_range     = [0, 15]      # inclusive both ends
weight_url      = "https://weights.intelnav.net/ds33b-q4km-p1.gguf"
sha256          = "..."
size_bytes      = 4_100_000_000
min_vram_bytes  = 8_000_000_000
desired_k       = 3            # optional; falls back to [defaults].desired_k

[[part]]
id              = "p2"
layer_range     = [16, 31]
# ...

[defaults]
desired_k                 = 3
heartbeat_interval_s      = 20
heartbeat_miss_tolerance  = 3     # evict after N missed heartbeats
min_live_seconds          = 600   # 10 min before a volunteer counts toward k
```

## 3. HTTP API

All routes are prefixed with `/v1/shards/`. Request and response bodies
are JSON.

| Verb | Path | Purpose |
| ---- | ---- | ------- |
| GET  | `:model_cid` | Full manifest snapshot: parts, weight URLs, replication counts |
| POST | `:model_cid/assign` | Registry picks the least-replicated part and returns an assignment to the caller |
| POST | `:model_cid/:part_id/claim` | Caller commits — "I've downloaded, I'm live" |
| POST | `:model_cid/:part_id/heartbeat` | Periodic liveness ping (interval from manifest) |
| POST | `:model_cid/:part_id/release` | Graceful leave (caller drains in-flight sessions first) |
| GET  | `:model_cid/:part_id/peers` | Peers currently serving this part — consumed by the gateway |
| GET  | `health` | Registry liveness check |

### 3.1 Request authentication (`claim`, `heartbeat`, `release`)

Mutating requests are signed with the caller's Ed25519 identity (the same
key that produces the peer_id). The request body is:

```json
{
  "envelope": {
    "peer_id":    "<64-char hex pubkey>",
    "model_cid":  "<string>",
    "part_id":    "<string>",
    "timestamp":  1713600000,
    "nonce":      "<16-byte hex>"
  },
  "sig": "<128-char hex ed25519 signature over CBOR(envelope)>"
}
```

The registry MUST verify:

1. `sig` is a valid Ed25519 signature over `CBOR(envelope)` under `peer_id`.
2. `timestamp` is within ±120 s of registry wall-clock (replay window).
3. For `claim` only: `(peer_id, model_cid, part_id)` is not already `live`
   or `draining`.

Rejected requests return 401 with `{ "error": "<reason>" }`.

### 3.2 `POST :model_cid/assign`

Request: `{ peer_id, role, capability: CapabilityV1 }` (unsigned — this
is pre-claim, the registry is just offering an assignment).

Registry picks the part with the lowest `live_or_committed_volunteer_count`
that the caller can host (checks `min_vram_bytes` against
`capability.vram_bytes`). Returns:

```json
{
  "part_id":     "p1",
  "layer_range": [0, 15],
  "weight_url":  "https://weights.intelnav.net/ds33b-q4km-p1.gguf",
  "sha256":      "...",
  "size_bytes":  4_100_000_000,
  "reservation_ttl_s": 1800
}
```

The caller has `reservation_ttl_s` seconds to `claim` before the slot is
released back to the pool. A peer that fails to claim in time is not
penalized — it can re-`assign`.

### 3.3 Seeder state machine

```
  none ──claim───▶ live ──hb miss──▶ evicted
                    │                    ▲
                    │ release            │ rejoin
                    ▼                    │
                 draining ──drained────▶ none
```

A seeder transitions from `live → draining` when it POSTs `release`. The
registry immediately stops counting it toward `desired_k`, but the seeder
is expected to finish in-flight sessions before disappearing. The gateway
stops dispatching new sessions to a `draining` seeder.

### 3.4 Cloud standby / resume

Cloud-role seeders get extra status: `standby`. On each state mutation the
registry recomputes, for every part:

```
live_volunteers(p) = | { s ∈ p.seeders : s.role = volunteer ∧ s.status = live
                          ∧ now − s.joined_at ≥ min_live_seconds } |
```

Then for each cloud seeder `c` of part `p`:

* If `c.status = live` and `live_volunteers(p) ≥ p.desired_k` → signal
  `c` to enter `standby`.
* If `c.status = standby` and `live_volunteers(p) ≤ p.desired_k − 1` →
  signal `c` to resume `live`.

The signal is the return value of the cloud seeder's next `heartbeat`:

```json
{ "ack": true, "directive": "standby" | "resume" | null }
```

Hysteresis of 1 (`≥ k` to standby, `≤ k-1` to resume) prevents
thrashing when volunteer count sits right on the boundary.

## 4. Sybil mitigation

Three layers, stacked (§8 of the security spec will pick these up):

1. **Ed25519 gate.** Every `claim/heartbeat/release` verified (§3.1).
2. **Minimum continuous uptime.** A volunteer is not counted toward
   `desired_k` until `now − joined_at ≥ min_live_seconds` (default 10 min).
   A burst of ephemeral peers cannot trigger cloud standby.
3. **Rate limit on `assign`.** A peer_id gets ≤ 6 assign calls per hour.
   Prevents a single key from hoarding the least-replicated part across
   rapid reconnects.

These are intentionally modest; real economic Sybil resistance is paper §9
and M4.

## 5. Gateway integration

`CapabilityV1` gains a `role: Role` field (`Volunteer | Cloud`). The
gateway's candidate-ranking function (`crates/gateway/src/api.rs`) applies
a tiebreaker:

```
key(peer) = (
    0 if peer.role == Volunteer else 1,   // prefer volunteers
    -peer.tok_per_s,                      // then by speed
    peer.peer_id,                         // stable tiebreak
)
```

So volunteers are tried first; cloud is the fallback when no volunteer in
the right latency tier can serve.

## 6. Out of scope for v1 (explicitly)

* **Federation / gossip between registries** — single registry per
  IntelNav instance. Cross-registry coordination is M4.
* **Incentive accounting** — we persist a `service_credits` counter per
  peer (increments on heartbeat while `live`), but it is not redeemable
  for anything and has no on-chain representation. Paper §9 is M4.
* **Dynamic re-sharding** — the parts partition is fixed once a model is
  announced. Re-sharding (splitting p1 → p1a/p1b) would require a model
  version bump.
* **Weight seeding from volunteers** — weight URLs are served by the
  operator of the registry (CDN/object store). BitTorrent-style peer
  seeding of weights is M4 (IPFS).
