//! Directory trait + two trivial implementations.

use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use intelnav_core::{ModelId, PeerId};
use intelnav_core::types::{CapabilityV1, Quant};

/// One entry in a peer directory — a capability advertisement plus enough
/// routing info to reach the peer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerRecord {
    pub peer_id:    PeerId,
    pub addrs:      Vec<String>, // multiaddrs or host:port
    pub capability: CapabilityV1,
    /// UNIX seconds at which this record was last observed.
    pub last_seen:  u64,
}

impl PeerRecord {
    pub fn serves(&self, model: &ModelId, quant: Quant) -> bool {
        self.capability.models.iter().any(|m| m == model)
            && self.capability.quants.contains(&quant)
    }
}

#[async_trait]
pub trait PeerDirectory: Send + Sync {
    /// List every peer currently known to this directory.
    async fn all(&self) -> Vec<PeerRecord>;

    /// Find peers that advertise `(model, quant)`.
    async fn providers(&self, model: &ModelId, quant: Quant) -> Vec<PeerRecord> {
        self.all()
            .await
            .into_iter()
            .filter(|p| p.serves(model, quant))
            .collect()
    }

    /// Human-readable name for logging / `/v1/network/peers`.
    fn name(&self) -> &'static str;
}

// ----------------------------------------------------------------------
//  StaticDirectory — peers listed in config
// ----------------------------------------------------------------------

#[derive(Default)]
pub struct StaticDirectory {
    inner: RwLock<HashMap<PeerId, PeerRecord>>,
}

impl StaticDirectory {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&self, record: PeerRecord) {
        let mut g = self.inner.write().unwrap();
        g.insert(record.peer_id, record);
    }
}

#[async_trait]
impl PeerDirectory for StaticDirectory {
    async fn all(&self) -> Vec<PeerRecord> {
        self.inner.read().unwrap().values().cloned().collect()
    }
    fn name(&self) -> &'static str { "static" }
}

// ----------------------------------------------------------------------
//  DhtDirectory — placeholder until libp2p wiring lands (M2)
// ----------------------------------------------------------------------

#[derive(Default)]
pub struct DhtDirectory {
    inner: RwLock<HashMap<PeerId, PeerRecord>>,
}

impl DhtDirectory {
    pub fn new() -> Self {
        Self::default()
    }
    /// M2 hook: libp2p Kademlia populates this from `Advertise` messages.
    pub fn feed(&self, record: PeerRecord) {
        self.inner.write().unwrap().insert(record.peer_id, record);
    }
}

#[async_trait]
impl PeerDirectory for DhtDirectory {
    async fn all(&self) -> Vec<PeerRecord> {
        self.inner.read().unwrap().values().cloned().collect()
    }
    fn name(&self) -> &'static str { "dht" }
}
