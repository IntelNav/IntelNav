//! mDNS / Bonjour local-network peer discovery (paper §6.2).
//!
//! Each gateway registers itself under the service type `_intelnav._tcp.local.`
//! and continuously listens for peer announcements. Advertised capabilities
//! travel via TXT records, JSON-encoded.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use tokio::task::JoinHandle;

use intelnav_core::{PeerId, types::CapabilityV1};

use crate::directory::{PeerDirectory, PeerRecord};

const SERVICE_TYPE: &str = "_intelnav._tcp.local.";
const TXT_CAPABILITY: &str = "cap";
const TXT_PEER_ID: &str    = "pid";

pub struct MdnsDirectory {
    inner:    Arc<RwLock<HashMap<PeerId, PeerRecord>>>,
    _daemon:  ServiceDaemon,
    _watcher: JoinHandle<()>,
    _self:    Option<ServiceInfo>,
}

impl MdnsDirectory {
    /// Start discovery. If `self_advertise` is `Some`, the caller's own
    /// capability is broadcast.
    pub fn spawn(self_advertise: Option<(CapabilityV1, u16)>) -> std::io::Result<Self> {
        let daemon = ServiceDaemon::new().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let receiver = daemon.browse(SERVICE_TYPE).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let inner: Arc<RwLock<HashMap<PeerId, PeerRecord>>> = Arc::default();

        // Register ourselves, if asked.
        let self_info = if let Some((cap, port)) = self_advertise {
            let host_ifs: Vec<String> = if_addrs::get_if_addrs()
                .unwrap_or_default()
                .into_iter()
                .filter(|i| !i.is_loopback())
                .map(|i| i.ip().to_string())
                .collect();
            let instance = hex::encode(&cap.peer_id.as_bytes()[..8]);
            let cap_json = serde_json::to_string(&cap).unwrap_or_default();
            let mut txt = HashMap::new();
            txt.insert(TXT_PEER_ID.into(), cap.peer_id.to_string());
            txt.insert(TXT_CAPABILITY.into(), cap_json);
            let host = format!("intelnav-{}.local.", instance);
            match ServiceInfo::new(
                SERVICE_TYPE,
                &instance,
                &host,
                host_ifs.as_slice(),
                port,
                Some(txt),
            ) {
                Ok(si) => {
                    if let Err(e) = daemon.register(si.clone()) {
                        tracing::warn!(?e, "mdns: register failed");
                    }
                    Some(si)
                }
                Err(e) => {
                    tracing::warn!(?e, "mdns: service info build failed");
                    None
                }
            }
        } else {
            None
        };

        let dir = inner.clone();
        let watcher = tokio::spawn(async move {
            loop {
                match receiver.recv_async().await {
                    Ok(ServiceEvent::ServiceResolved(info)) => {
                        if let Some(rec) = record_from_info(&info) {
                            dir.write().unwrap().insert(rec.peer_id, rec);
                        }
                    }
                    Ok(ServiceEvent::ServiceRemoved(_, fullname)) => {
                        tracing::debug!(%fullname, "mdns: service removed");
                    }
                    Ok(_) => {}
                    Err(_) => {
                        tokio::time::sleep(Duration::from_millis(250)).await;
                    }
                }
            }
        });

        Ok(Self { inner, _daemon: daemon, _watcher: watcher, _self: self_info })
    }
}

fn record_from_info(info: &ServiceInfo) -> Option<PeerRecord> {
    let props = info.get_properties();
    let cap_raw = props.get_property_val_str(TXT_CAPABILITY)?;
    let cap: CapabilityV1 = serde_json::from_str(cap_raw).ok()?;
    let addrs: Vec<String> = info
        .get_addresses()
        .iter()
        .map(|ip| format!("{}:{}", ip, info.get_port()))
        .collect();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Some(PeerRecord {
        peer_id:    cap.peer_id,
        addrs,
        capability: cap,
        last_seen:  now,
    })
}

#[async_trait]
impl PeerDirectory for MdnsDirectory {
    async fn all(&self) -> Vec<PeerRecord> {
        self.inner.read().unwrap().values().cloned().collect()
    }
    fn name(&self) -> &'static str { "mdns" }
}
