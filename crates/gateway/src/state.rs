//! Shared gateway state — config + peer directories + upstream client.

use std::sync::Arc;

use intelnav_core::Config;
use intelnav_net::{DhtDirectory, MdnsDirectory, PeerDirectory, RegistryDirectory, StaticDirectory};
use intelnav_runtime::Telemetry;

#[derive(Clone)]
pub struct GatewayState {
    pub config:       Arc<Config>,
    pub http:         reqwest::Client,
    pub static_dir:   Arc<StaticDirectory>,
    pub dht_dir:      Arc<DhtDirectory>,
    pub mdns_dir:     Option<Arc<MdnsDirectory>>,
    pub registry_dir: Option<Arc<RegistryDirectory>>,
    pub started_at:   std::time::Instant,
    /// Broadcast channel of [`intelnav_runtime::StepEvent`]. Real
    /// events come from a `Chain` driven inside this gateway (arc 6
    /// sub-D); until then, [`crate::server::run`] spawns a synth
    /// heartbeat loop that emits one event every couple seconds so
    /// the SPA's `/v1/swarm/events` SSE always has *something* to
    /// show. Each event carries `synthetic: true` until real data
    /// replaces it.
    pub telemetry:    Telemetry,
}

impl GatewayState {
    pub fn directories(&self) -> Vec<Arc<dyn PeerDirectory>> {
        let mut v: Vec<Arc<dyn PeerDirectory>> = Vec::new();
        v.push(self.static_dir.clone());
        v.push(self.dht_dir.clone());
        if let Some(m) = &self.mdns_dir {
            v.push(m.clone());
        }
        if let Some(r) = &self.registry_dir {
            v.push(r.clone());
        }
        v
    }
}
