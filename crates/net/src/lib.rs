//! `intelnav-net` — peer discovery & network substrate.
//!
//! The paper mandates a libp2p-based DHT (Kademlia) as the canonical
//! peer directory (§6). [`swarm`] delivers the M2 substrate
//! (TCP + Noise XX + yamux + identify + ping under `/intelnav/v1`);
//! Kademlia and NAT traversal layer onto the same swarm in later
//! M2 sub-tasks.
//!
//! Three directory implementations exist alongside the swarm — they
//! cover the boot path before the swarm has a populated routing
//! table and the operator-facing static / registry-poll cases:
//!
//! * [`StaticDirectory`] — hardcoded peers from config.
//! * [`MdnsDirectory`]   — mDNS/Bonjour local-network discovery (§6.2).
//! * [`DhtDirectory`]    — slot the libp2p substrate populates.
//!
//! All three implement [`PeerDirectory`] so the gateway treats them
//! interchangeably.

#![forbid(unsafe_code)]

pub mod directory;
pub mod mdns;
pub mod registry_dir;
pub mod swarm;

pub use directory::{PeerDirectory, PeerRecord, StaticDirectory, DhtDirectory};
pub use mdns::MdnsDirectory;
pub use registry_dir::RegistryDirectory;
pub use swarm::{
    spawn as spawn_libp2p_node, identity_to_keypair, IdentifiedPeer, IntelNavBehaviour,
    Libp2pNode, AGENT_VERSION, PROTOCOL_VERSION,
};
