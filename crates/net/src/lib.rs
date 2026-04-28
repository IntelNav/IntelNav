//! `intelnav-net` — peer discovery & network substrate.
//!
//! Three layers, surfaced from least-magical to most:
//!
//! 1. **Boot directories.** [`StaticDirectory`] / [`MdnsDirectory`]
//!    populate a peer list before the DHT routing table has
//!    converged. Both implement [`PeerDirectory`] so callers treat
//!    them interchangeably.
//!
//! 2. **The swarm host.** [`Libp2pNode`] runs TCP + Noise XX + yamux
//!    + `identify` + `ping` + Kademlia behind a single command
//!    channel. Spawn it with [`spawn_libp2p_node`].
//!
//! 3. **The shard index.** [`SwarmIndex`] sits on top of the DHT and
//!    answers "which models can the swarm serve me right now?" by
//!    fanning out per-`(cid, range)` Kademlia lookups. The picker
//!    in `intelnav-app` consumes its [`SwarmModel`] output.

#![forbid(unsafe_code)]

pub mod dht;
pub mod directory;
pub mod mdns;
pub mod swarm;
pub mod swarm_index;

pub use dht::{model_key, shard_key, ModelEnvelope, ProviderRecord};
pub use directory::{PeerDirectory, PeerRecord, StaticDirectory};
pub use mdns::MdnsDirectory;
pub use swarm::{
    spawn as spawn_libp2p_node, identity_to_keypair, IdentifiedPeer, Libp2pNode,
    AGENT_VERSION, PROTOCOL_VERSION,
};
pub use swarm_index::{RangeCoverage, SwarmIndex, SwarmModel};

// Multiaddr is a libp2p type re-exported for convenience — callers that
// need to construct one for `spawn_libp2p_node` use this re-export
// rather than depending on libp2p directly. Wrapping it would require
// also wrapping PeerId and IdentifiedPeer.listen_addrs, which isn't
// worth the churn.
pub use libp2p::Multiaddr;
