# intelnav-crypto

Identity, handshake, and prompt-confidentiality primitives.

- `Identity` — long-lived Ed25519 signing key. `PeerId = multihash(pubkey)`.
  `generate()`, `from_seed(&[u8;32])`, `sign(msg)`, `peer_id()`.
- `verify(peer_pub, msg, sig)` — signature check.
- `EphemeralHandshake` / `StaticHandshake` — X25519 key exchange. Paper
  §3.3: gateway ↔ entry-shard session setup.
- `session_key(shared)` — blake3-XOF key derivation with the
  `"intelnav/v1/prompt"` domain-separation tag. 32 bytes out.
- `encrypt` / `decrypt` — AES-256-GCM over the user prompt with a
  freshly generated 96-bit nonce (paper §8.2).

Output is byte-identical to the Python shard's re-implementation — the
smoke suite in `python/intelnav_shard/smoke_client.py` verifies this.

`#![forbid(unsafe_code)]`.
