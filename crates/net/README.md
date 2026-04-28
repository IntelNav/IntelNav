# intelnav-net

libp2p + Kademlia DHT shard index, plus two boot directories that
populate the routing table before the DHT has converged.

- `Libp2pNode` — the swarm host: TCP + DNS + Noise XX + yamux +
  identify + ping + Kademlia, all behind one command channel. Spawn
  with `spawn_libp2p_node`.
- `SwarmIndex` — sits on top of the DHT and answers "which models
  can the swarm serve me right now?" by fanning out per-`(cid,
  range)` Kademlia lookups.
- `StaticDirectory` — peers from `Config::bootstrap` (config-pinned
  seeds, mostly for self-hosted swarms / dev).
- `MdnsDirectory` — LAN discovery via mDNS/Bonjour for zero-config
  on a local network.

Both directories implement `PeerDirectory` so callers treat them
interchangeably.

`#![forbid(unsafe_code)]`.
