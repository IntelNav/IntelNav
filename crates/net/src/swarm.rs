//! libp2p substrate.
//!
//! [`Libp2pNode`] is the long-lived swarm host every contributor /
//! end-user starts at boot. It speaks TCP + Noise XX + yamux,
//! advertises itself as `/intelnav/v1` via `identify`, and runs a
//! Kademlia DHT keyed on `(model_cid, layer_range)` for shard
//! discovery. `ping` keeps idle connections honest.
//!
//! The Ed25519 [`Identity`] from `intelnav-crypto` is the canonical
//! keypair — [`identity_to_keypair`] hands the same 32-byte seed to
//! libp2p so the resulting `libp2p::PeerId` derives from the same key
//! the wire layer signs with.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt as _;
use libp2p::{
    identify, identity,
    kad::{
        self,
        store::MemoryStore,
        GetRecordOk, PutRecordOk, QueryId, QueryResult, Quorum, Record, RecordKey,
    },
    multiaddr::Multiaddr,
    noise, ping,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, PeerId, Swarm, SwarmBuilder,
};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use intelnav_crypto::Identity;

/// Protocol version string broadcast through `identify`. Two peers
/// that disagree here are not on the same product even if the
/// transport handshake completes.
pub const PROTOCOL_VERSION: &str = "/intelnav/v1";

/// Informational agent string broadcast through `identify`.
pub const AGENT_VERSION: &str = concat!("intelnav-net/", env!("CARGO_PKG_VERSION"));

#[derive(NetworkBehaviour)]
pub struct IntelNavBehaviour {
    pub identify: identify::Behaviour,
    pub ping:     ping::Behaviour,
    pub kad:      kad::Behaviour<MemoryStore>,
}

/// Public face of an `intelnav-net` libp2p host.
///
/// Owns no swarm state directly — a background tokio task drives
/// the swarm; [`Libp2pNode`] forwards user actions over a small
/// command channel so the API stays non-blocking and Send-safe.
pub struct Libp2pNode {
    pub(crate) tx:           mpsc::Sender<Command>,
    pub peer_id:      PeerId,
    pub listen_addrs: Vec<Multiaddr>,
}

pub(crate) enum Command {
    Dial { addr: Multiaddr, done: oneshot::Sender<Result<()>> },
    NextIdentified { done: oneshot::Sender<IdentifiedPeer> },
    AddBootstrap { peer: PeerId, addr: Multiaddr },
    Bootstrap { done: oneshot::Sender<Result<()>> },
    AnnounceShard {
        key:  RecordKey,
        meta: Vec<u8>,            // CBOR-encoded ProviderRecord
        done: oneshot::Sender<Result<()>>,
    },
    GetShardProviders {
        key:  RecordKey,
        done: oneshot::Sender<Result<Vec<Vec<u8>>>>,
    },
    Shutdown,
}

/// Snapshot of one peer the swarm has identified.
#[derive(Clone, Debug)]
pub struct IdentifiedPeer {
    pub peer_id:          PeerId,
    pub protocol_version: String,
    pub agent_version:    String,
    pub listen_addrs:     Vec<Multiaddr>,
}

/// Spawn a libp2p host bound to `listen` (typically
/// `"/ip4/0.0.0.0/tcp/0"` for an ephemeral port). Returns once the
/// swarm has surfaced its first listen address — callers can publish
/// `node.listen_addrs[0]` to bootstrap peers immediately.
pub async fn spawn(keypair: identity::Keypair, listen: Multiaddr) -> Result<Libp2pNode> {
    let peer_id = PeerId::from(keypair.public());

    let mut swarm: Swarm<IntelNavBehaviour> = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .context("libp2p tcp/noise/yamux stack")?
        // Wrap the TCP transport with DNS so callers can dial
        // `/dns4/<host>/tcp/<port>/p2p/<peer_id>` multiaddrs without
        // pre-resolving — bootstrap seeds are advertised by hostname.
        .with_dns()
        .context("libp2p dns transport")?
        .with_behaviour(|key| {
            let pid = PeerId::from(key.public());
            let mut kad_cfg = kad::Config::new(kad::PROTOCOL_NAME);
            // Cap record/provider TTLs so a peer that drops out of the
            // swarm stops being advertised after a short window. The
            // periodic announce task in the CLI re-publishes well
            // inside this budget.
            kad_cfg.set_record_ttl(Some(Duration::from_secs(30 * 60)));
            kad_cfg.set_provider_record_ttl(Some(Duration::from_secs(30 * 60)));
            kad_cfg.set_publication_interval(Some(Duration::from_secs(5 * 60)));
            kad_cfg.set_provider_publication_interval(Some(Duration::from_secs(5 * 60)));
            let mut kad_b = kad::Behaviour::with_config(pid, MemoryStore::new(pid), kad_cfg);
            // Start in client-only mode; switch to server once we have
            // confirmed reachability via at least one identify exchange.
            kad_b.set_mode(Some(kad::Mode::Server));
            Ok(IntelNavBehaviour {
                identify: identify::Behaviour::new(
                    identify::Config::new(PROTOCOL_VERSION.into(), key.public())
                        .with_agent_version(AGENT_VERSION.into()),
                ),
                ping: ping::Behaviour::default(),
                kad:  kad_b,
            })
        })
        .map_err(|e| anyhow!("building swarm behaviour: {e}"))?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(300)))
        .build();

    swarm.listen_on(listen).context("listen_on")?;

    let mut listen_addrs = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(anyhow!("timed out waiting for first listen addr"));
        }
        match tokio::time::timeout(remaining, swarm.select_next_some()).await {
            Ok(SwarmEvent::NewListenAddr { address, .. }) => {
                listen_addrs.push(address);
                break;
            }
            Ok(other) => debug!(?other, "pre-listen event"),
            Err(_) => return Err(anyhow!("timed out waiting for first listen addr")),
        }
    }

    let (tx, rx) = mpsc::channel::<Command>(32);
    tokio::spawn(drive_swarm(swarm, rx));
    Ok(Libp2pNode { tx, peer_id, listen_addrs })
}

impl Libp2pNode {
    /// Dial a remote multiaddr. Resolves once the swarm accepts the
    /// dial — connection completion arrives later as Identify events.
    pub async fn dial(&self, addr: Multiaddr) -> Result<()> {
        let (done, rx) = oneshot::channel();
        self.tx.send(Command::Dial { addr, done }).await
            .map_err(|_| anyhow!("swarm task is gone"))?;
        rx.await.map_err(|_| anyhow!("dial reply dropped"))?
    }

    /// Block until the swarm completes its next Identify exchange.
    pub async fn next_identified(&self) -> Result<IdentifiedPeer> {
        let (done, rx) = oneshot::channel();
        self.tx.send(Command::NextIdentified { done }).await
            .map_err(|_| anyhow!("swarm task is gone"))?;
        rx.await.map_err(|_| anyhow!("identify reply dropped"))
    }

    /// Add a known bootstrap peer to the Kademlia routing table.
    pub async fn add_bootstrap(&self, peer: PeerId, addr: Multiaddr) -> Result<()> {
        self.tx.send(Command::AddBootstrap { peer, addr }).await
            .map_err(|_| anyhow!("swarm task is gone"))
    }

    /// Run a Kademlia bootstrap query (populates the routing table).
    pub async fn bootstrap(&self) -> Result<()> {
        let (done, rx) = oneshot::channel();
        self.tx.send(Command::Bootstrap { done }).await
            .map_err(|_| anyhow!("swarm task is gone"))?;
        rx.await.map_err(|_| anyhow!("bootstrap reply dropped"))?
    }

    /// Stop the swarm task. Idempotent.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(Command::Shutdown).await;
    }
}

async fn drive_swarm(mut swarm: Swarm<IntelNavBehaviour>, mut rx: mpsc::Receiver<Command>) {
    use std::collections::VecDeque;
    let mut pending_id: VecDeque<IdentifiedPeer> = VecDeque::new();
    let mut id_waiters:  VecDeque<oneshot::Sender<IdentifiedPeer>> = VecDeque::new();

    // Outstanding Kademlia queries indexed by their QueryId. The
    // event loop translates the terminal QueryResult into a oneshot
    // reply on the matching channel.
    let mut put_waiters:  HashMap<QueryId, oneshot::Sender<Result<()>>> = HashMap::new();
    let mut get_waiters:  HashMap<QueryId, oneshot::Sender<Result<Vec<Vec<u8>>>>> = HashMap::new();
    // Records arrive incrementally on GetRecord queries; buffer until
    // the OutboundQueryProgressed FinishedWithNoAdditionalRecord arrives.
    let mut get_buffers:  HashMap<QueryId, Vec<Vec<u8>>> = HashMap::new();
    let mut bootstrap_waiters: HashMap<QueryId, oneshot::Sender<Result<()>>> = HashMap::new();

    loop {
        tokio::select! {
            cmd = rx.recv() => {
                let Some(cmd) = cmd else { return; };
                match cmd {
                    Command::Dial { addr, done } => {
                        let _ = done.send(swarm.dial(addr).map_err(|e| anyhow!("dial: {e}")));
                    }
                    Command::NextIdentified { done } => {
                        if let Some(peer) = pending_id.pop_front() {
                            let _ = done.send(peer);
                        } else {
                            id_waiters.push_back(done);
                        }
                    }
                    Command::AddBootstrap { peer, addr } => {
                        swarm.behaviour_mut().kad.add_address(&peer, addr);
                    }
                    Command::Bootstrap { done } => {
                        match swarm.behaviour_mut().kad.bootstrap() {
                            Ok(qid) => { bootstrap_waiters.insert(qid, done); }
                            Err(e) => { let _ = done.send(Err(anyhow!("bootstrap: {e}"))); }
                        }
                    }
                    Command::AnnounceShard { key, meta, done } => {
                        // Store the record value (CBOR-encoded ProviderRecord)
                        // under the shard key so any peer that does
                        // GetRecord on the same key sees this provider.
                        let record = Record {
                            key: key.clone(),
                            value: meta,
                            publisher: None,
                            expires: None,
                        };
                        match swarm.behaviour_mut().kad.put_record(record, Quorum::One) {
                            Ok(qid) => { put_waiters.insert(qid, done); }
                            Err(e) => { let _ = done.send(Err(anyhow!("put_record: {e}"))); }
                        }
                    }
                    Command::GetShardProviders { key, done } => {
                        let qid = swarm.behaviour_mut().kad.get_record(key);
                        get_buffers.insert(qid, Vec::new());
                        get_waiters.insert(qid, done);
                    }
                    Command::Shutdown => return,
                }
            }
            event = swarm.select_next_some() => match event {
                SwarmEvent::Behaviour(IntelNavBehaviourEvent::Identify(
                    identify::Event::Received { peer_id, info, .. }
                )) => {
                    // Feed the Kademlia routing table with addresses
                    // we just learned for this peer. Without this the
                    // DHT is starved on a fresh node — kad won't dial
                    // an unknown peer to do its own discovery.
                    for addr in &info.listen_addrs {
                        swarm.behaviour_mut().kad.add_address(&peer_id, addr.clone());
                    }
                    let peer = IdentifiedPeer {
                        peer_id,
                        protocol_version: info.protocol_version,
                        agent_version:    info.agent_version,
                        listen_addrs:     info.listen_addrs,
                    };
                    if let Some(w) = id_waiters.pop_front() {
                        let _ = w.send(peer);
                    } else {
                        pending_id.push_back(peer);
                    }
                }
                SwarmEvent::Behaviour(IntelNavBehaviourEvent::Kad(
                    kad::Event::OutboundQueryProgressed { id, result, step, .. }
                )) => match result {
                    QueryResult::PutRecord(res) => {
                        if let Some(done) = put_waiters.remove(&id) {
                            let _ = done.send(res
                                .map(|PutRecordOk { .. }| ())
                                .map_err(|e| anyhow!("put_record: {e}")));
                        }
                    }
                    QueryResult::GetRecord(res) => {
                        match res {
                            Ok(GetRecordOk::FoundRecord(rec)) => {
                                if let Some(buf) = get_buffers.get_mut(&id) {
                                    buf.push(rec.record.value);
                                }
                            }
                            Ok(GetRecordOk::FinishedWithNoAdditionalRecord { .. }) => {
                                if step.last {
                                    let buf = get_buffers.remove(&id).unwrap_or_default();
                                    if let Some(done) = get_waiters.remove(&id) {
                                        let _ = done.send(Ok(buf));
                                    }
                                }
                            }
                            Err(e) => {
                                // Even on error, return whatever records
                                // we managed to collect so the caller can
                                // make progress on a partially reachable DHT.
                                let buf = get_buffers.remove(&id).unwrap_or_default();
                                if let Some(done) = get_waiters.remove(&id) {
                                    if buf.is_empty() {
                                        let _ = done.send(Err(anyhow!("get_record: {e:?}")));
                                    } else {
                                        let _ = done.send(Ok(buf));
                                    }
                                }
                            }
                        }
                    }
                    QueryResult::Bootstrap(res) => {
                        if step.last {
                            if let Some(done) = bootstrap_waiters.remove(&id) {
                                let _ = done.send(res
                                    .map(|_| ())
                                    .map_err(|e| anyhow!("bootstrap: {e:?}")));
                            }
                        }
                    }
                    other => debug!(?other, "kad query result"),
                },
                SwarmEvent::Behaviour(IntelNavBehaviourEvent::Ping(ev)) => {
                    debug!(?ev, "ping");
                }
                // Connection errors are noisy on a healthy swarm: every
                // peer learned via mDNS/identify gets dialed back, and
                // any of them being NAT'd or shut down produces an
                // error here. Demote to debug — real failures surface
                // through the explicit `dial()` Result and the
                // bootstrap loop's own logging.
                SwarmEvent::IncomingConnectionError { error, .. } => {
                    debug!(?error, "incoming connection error");
                }
                SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                    debug!(?peer_id, ?error, "outgoing connection error");
                }
                _ => {}
            }
        }
    }
}

/// Bridge an `intelnav-crypto` Ed25519 [`Identity`] into the libp2p
/// keypair format. Both are Ed25519 — the same 32-byte seed feeds
/// both, so `libp2p::PeerId` derives from the same key the wire
/// layer signs with.
pub fn identity_to_keypair(id: &Identity) -> Result<identity::Keypair> {
    let mut seed = id.seed();
    identity::Keypair::ed25519_from_bytes(seed.as_mut())
        .map_err(|e| anyhow!("ed25519 from seed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawn two nodes on loopback, dial one to the other, and assert
    /// that Identify completes with our `/intelnav/v1` protocol
    /// version. This is the M2.a substrate gate: if it fails, no
    /// other M2 sub-task is reachable.
    #[tokio::test]
    async fn two_nodes_identify_each_other() {
        let id_a = Identity::generate();
        let id_b = Identity::generate();

        let kp_a = identity_to_keypair(&id_a).unwrap();
        let kp_b = identity_to_keypair(&id_b).unwrap();

        let listen: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
        let a = spawn(kp_a, listen.clone()).await.unwrap();
        let b = spawn(kp_b, listen).await.unwrap();

        // Dial b from a using b's first listen addr + b's PeerId so
        // the dial picks up the secure-channel handshake correctly.
        let mut dial_addr = b.listen_addrs[0].clone();
        dial_addr.push(libp2p::multiaddr::Protocol::P2p(b.peer_id));
        a.dial(dial_addr).await.unwrap();

        // Identify is symmetric — wait for both sides.
        let from_a = tokio::time::timeout(Duration::from_secs(10), a.next_identified())
            .await
            .expect("a's identify timed out")
            .unwrap();
        let from_b = tokio::time::timeout(Duration::from_secs(10), b.next_identified())
            .await
            .expect("b's identify timed out")
            .unwrap();

        assert_eq!(from_a.peer_id, b.peer_id, "a should identify b");
        assert_eq!(from_b.peer_id, a.peer_id, "b should identify a");
        assert_eq!(from_a.protocol_version, PROTOCOL_VERSION);
        assert_eq!(from_b.protocol_version, PROTOCOL_VERSION);
        assert!(from_a.agent_version.starts_with("intelnav-net/"));

        a.shutdown().await;
        b.shutdown().await;
    }

    /// Two-node Kademlia smoke: peer A announces a shard provider
    /// record under (cid="m1", 0..16); peer B (after dialing A and
    /// completing identify) does `find_shard_providers` and gets
    /// A's record back. Exercises the full PUT → walk → GET path
    /// inside a single tokio runtime.
    #[tokio::test]
    async fn dht_announce_and_find_shard() {
        use crate::dht::ProviderRecord;
        use std::time::SystemTime;

        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let kp_a = identity_to_keypair(&id_a).unwrap();
        let kp_b = identity_to_keypair(&id_b).unwrap();

        let listen: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
        let a = spawn(kp_a, listen.clone()).await.unwrap();
        let b = spawn(kp_b, listen).await.unwrap();

        // B must know about A in its routing table before kad walks
        // can find anything. Dial through identify is enough — kad
        // ingests the addrs we add via `add_address` on identify.
        let mut dial_addr = a.listen_addrs[0].clone();
        dial_addr.push(libp2p::multiaddr::Protocol::P2p(a.peer_id));
        b.add_bootstrap(a.peer_id, dial_addr.clone()).await.unwrap();
        b.dial(dial_addr).await.unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(10), b.next_identified()).await;
        let _ = tokio::time::timeout(Duration::from_secs(10), a.next_identified()).await;

        let record = ProviderRecord {
            peer_id:      a.peer_id.to_base58(),
            addrs:        a.listen_addrs.iter().map(|m| m.to_string()).collect(),
            chunks_url:   Some("127.0.0.1:8765".into()),
            manifest_cid: Some("bafkrei-test".into()),
            forward_url:  Some("127.0.0.1:7717".into()),
            minted_at:    SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };
        a.announce_shard("m1", 0, 16, record.clone()).await.unwrap();

        // Allow the put_record query to complete and propagate.
        let providers = tokio::time::timeout(
            Duration::from_secs(15),
            b.find_shard_providers("m1", 0, 16),
        )
        .await
        .expect("find_shard_providers timed out")
        .expect("find_shard_providers returned err");

        assert!(
            providers.iter().any(|p| p.peer_id == a.peer_id.to_base58()),
            "B should have discovered A's provider record on the DHT, got {providers:?}"
        );

        a.shutdown().await;
        b.shutdown().await;
    }

    /// SwarmIndex aggregator: A announces the model envelope plus
    /// two slices, B refreshes for the same cid and gets coverage
    /// numbers we can act on (n_providers per range, gaps).
    #[tokio::test]
    async fn swarm_index_refresh_one_aggregates() {
        use crate::dht::{ModelEnvelope, ProviderRecord};
        use crate::swarm_index::SwarmIndex;
        use std::sync::Arc;
        use std::time::SystemTime;

        let id_a = Identity::generate();
        let id_b = Identity::generate();
        let kp_a = identity_to_keypair(&id_a).unwrap();
        let kp_b = identity_to_keypair(&id_b).unwrap();

        let listen: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
        let a = spawn(kp_a, listen.clone()).await.unwrap();
        let b = spawn(kp_b, listen).await.unwrap();

        let mut dial_addr = a.listen_addrs[0].clone();
        dial_addr.push(libp2p::multiaddr::Protocol::P2p(a.peer_id));
        b.add_bootstrap(a.peer_id, dial_addr.clone()).await.unwrap();
        b.dial(dial_addr).await.unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(10), b.next_identified()).await;
        let _ = tokio::time::timeout(Duration::from_secs(10), a.next_identified()).await;

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let envelope = ModelEnvelope {
            cid:          "qwen2.5-0.5b-q4km".into(),
            display_name: "Qwen2.5 0.5B Instruct (Q4_K_M)".into(),
            arch:         "qwen2".into(),
            block_count:  24,
            total_bytes:  470_000_000,
            quant:        "Q4_K_M".into(),
        };
        let record = ProviderRecord {
            peer_id:      a.peer_id.to_base58(),
            addrs:        a.listen_addrs.iter().map(|m| m.to_string()).collect(),
            chunks_url:   None,
            manifest_cid: None,
            forward_url:  None,
            minted_at:    now,
        };

        a.announce_model(&envelope).await.unwrap();
        a.announce_shard(&envelope.cid, 0, 12, record.clone()).await.unwrap();
        a.announce_shard(&envelope.cid, 12, 24, record.clone()).await.unwrap();

        let index = SwarmIndex::new(Arc::new(b));
        let model = tokio::time::timeout(
            Duration::from_secs(15),
            index.refresh_one(&envelope.cid, &[(0, 12), (12, 24), (24, 36)]),
        )
        .await
        .expect("refresh_one timed out")
        .expect("refresh_one returned err");

        assert_eq!(model.envelope.as_ref().map(|e| e.block_count), Some(24));
        assert_eq!(model.ranges.len(), 3);
        assert!(!model.ranges[0].providers.is_empty(), "(0,12) should be served");
        assert!(!model.ranges[1].providers.is_empty(), "(12,24) should be served");
        assert!(model.ranges[2].providers.is_empty(),  "(24,36) should NOT be served");
        assert_eq!(model.unique_providers(), 1);
        assert!(!model.fully_served());
        assert_eq!(model.gaps(), vec![(24, 36)]);

        a.shutdown().await;
    }

    /// `intelnav-crypto::Identity` and the libp2p keypair derived
    /// from it must produce the same Ed25519 public key — otherwise
    /// the wire signature and the libp2p peer id disagree.
    #[test]
    fn identity_to_keypair_round_trip() {
        let id = Identity::generate();
        let kp = identity_to_keypair(&id).unwrap();
        let lp2p_pub_bytes = kp.public()
            .try_into_ed25519()
            .expect("derived keypair must be ed25519")
            .to_bytes();
        assert_eq!(lp2p_pub_bytes, id.public(), "libp2p ed25519 pub != intelnav-crypto pub");
    }
}
