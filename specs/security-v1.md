# IntelNav Security Model — v1 (normative)

Derived from paper §8.  The Rust surface for this model is split between
`crates/crypto` (primitives) and `crates/gateway` (request handling).

## Identity
Ed25519 keypair generated on first start; peer ID is `multihash(pubkey)`.

## Prompt confidentiality
Ephemeral X25519 exchange between client and *entry* shard. The gateway
never sees the shared secret; prompt bytes travel AES-256-GCM-encrypted.

## Result integrity
`t`-of-`n` quorum over disjoint shard chains (paper §8.3). Default
`t=n=1` for low-stakes workloads; configurable per-session via the
`intelnav.quorum` API extension.

## Transport
libp2p Noise XX on every hop (paper §6.3). Until M2 lands, the CLI/gateway
fall back to TLS between the client and gateway with plaintext hops inside
the local network.
