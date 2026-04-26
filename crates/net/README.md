# intelnav-net

Peer directories. A `PeerDirectory` is any source of `PeerRecord`s;
the gateway composes several at once.

- `StaticDirectory` — peers from `Config::bootstrap`.
- `MdnsDirectory` — local-network discovery via mDNS/Bonjour (paper §6.2).
- `RegistryDirectory` — polls an HTTP shard registry every 5 s and
  exposes its seeders, sorted volunteer-first (registry spec §5).
- `DhtDirectory` — in-memory stub. M2 replaces this with a real
  Kademlia DHT over libp2p; the trait boundary is the swap point.

All four implement the same `PeerDirectory` trait so the gateway
treats them interchangeably.

`#![forbid(unsafe_code)]`.
