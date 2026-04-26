//! Shared gateway state — config + peer directories + upstream client.

use std::sync::Arc;

use intelnav_core::Config;
use intelnav_net::{DhtDirectory, MdnsDirectory, PeerDirectory, RegistryDirectory, StaticDirectory};

#[derive(Clone)]
pub struct GatewayState {
    pub config:       Arc<Config>,
    pub http:         reqwest::Client,
    pub static_dir:   Arc<StaticDirectory>,
    pub dht_dir:      Arc<DhtDirectory>,
    pub mdns_dir:     Option<Arc<MdnsDirectory>>,
    pub registry_dir: Option<Arc<RegistryDirectory>>,
    pub started_at:   std::time::Instant,
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
