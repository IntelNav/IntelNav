//! `intelnav-net` — peer discovery & network substrate.
//!
//! The paper mandates a libp2p-based DHT (Kademlia) as the canonical peer
//! directory (§6). M2 delivers that substrate in week 8. Until then the
//! gateway composes the three directories implemented here:
//!
//! * [`StaticDirectory`] — hardcoded peers from config.
//! * [`MdnsDirectory`]   — mDNS/Bonjour local-network discovery (§6.2).
//! * [`DhtDirectory`]    — placeholder the libp2p substrate will populate.
//!
//! All three implement [`PeerDirectory`] so the gateway treats them
//! interchangeably.

#![forbid(unsafe_code)]

pub mod directory;
pub mod mdns;
pub mod registry_dir;

pub use directory::{PeerDirectory, PeerRecord, StaticDirectory, DhtDirectory};
pub use mdns::MdnsDirectory;
pub use registry_dir::RegistryDirectory;
