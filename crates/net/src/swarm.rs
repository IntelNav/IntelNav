//! libp2p substrate (paper §6, milestone M2).
//!
//! [`Libp2pNode`] is the long-lived swarm host every gateway starts
//! at boot. It speaks TCP + Noise XX + yamux and advertises itself as
//! `/intelnav/v1` through the `identify` protocol, so any two peers
//! that handshake can confirm they're talking the same product
//! before anything else happens. `ping` is wired to keep idle
//! connections honest.
//!
//! M2 sub-tasks layer onto this: Kademlia DHT (provider records keyed
//! on model CID), Circuit-v2 + DCUtR for NAT traversal, gossipsub on
//! `/intelnav/v1/health`. Each behaviour is a field on
//! [`IntelNavBehaviour`]; the `NetworkBehaviour` derive picks them up.
//!
//! The Ed25519 [`Identity`] from `intelnav-crypto` is the canonical
//! keypair — [`identity_to_keypair`] hands the same 32-byte seed to
//! libp2p so the resulting `libp2p::PeerId` derives from the same key
//! the wire layer signs with.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt as _;
use libp2p::{
    identify, identity,
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
}

/// Public face of an `intelnav-net` libp2p host.
///
/// Owns no swarm state directly — a background tokio task drives
/// the swarm; [`Libp2pNode`] forwards user actions over a small
/// command channel so the API stays non-blocking and Send-safe.
pub struct Libp2pNode {
    tx:           mpsc::Sender<Command>,
    pub peer_id:      PeerId,
    pub listen_addrs: Vec<Multiaddr>,
}

enum Command {
    Dial { addr: Multiaddr, done: oneshot::Sender<Result<()>> },
    NextIdentified { done: oneshot::Sender<IdentifiedPeer> },
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
        .with_behaviour(|key| {
            Ok(IntelNavBehaviour {
                identify: identify::Behaviour::new(
                    identify::Config::new(PROTOCOL_VERSION.into(), key.public())
                        .with_agent_version(AGENT_VERSION.into()),
                ),
                ping: ping::Behaviour::default(),
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

    /// Block until the swarm completes its next Identify exchange
    /// with a remote peer. Tests use this; production code will
    /// observe identify through a broadcast channel once that lands
    /// alongside the directory feed in M2.b.
    pub async fn next_identified(&self) -> Result<IdentifiedPeer> {
        let (done, rx) = oneshot::channel();
        self.tx.send(Command::NextIdentified { done }).await
            .map_err(|_| anyhow!("swarm task is gone"))?;
        rx.await.map_err(|_| anyhow!("identify reply dropped"))
    }

    /// Stop the swarm task. Idempotent.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(Command::Shutdown).await;
    }
}

async fn drive_swarm(mut swarm: Swarm<IntelNavBehaviour>, mut rx: mpsc::Receiver<Command>) {
    use std::collections::VecDeque;
    // Identify events can arrive before any caller has registered a
    // waiter (e.g. during a fast LAN dial). Buffer the unclaimed ones
    // so `next_identified` can drain them in order.
    let mut pending: VecDeque<IdentifiedPeer> = VecDeque::new();
    let mut id_waiters: VecDeque<oneshot::Sender<IdentifiedPeer>> = VecDeque::new();

    loop {
        tokio::select! {
            cmd = rx.recv() => {
                let Some(cmd) = cmd else { return; };
                match cmd {
                    Command::Dial { addr, done } => {
                        let _ = done.send(swarm.dial(addr).map_err(|e| anyhow!("dial: {e}")));
                    }
                    Command::NextIdentified { done } => {
                        if let Some(peer) = pending.pop_front() {
                            let _ = done.send(peer);
                        } else {
                            id_waiters.push_back(done);
                        }
                    }
                    Command::Shutdown => return,
                }
            }
            event = swarm.select_next_some() => match event {
                SwarmEvent::Behaviour(IntelNavBehaviourEvent::Identify(
                    identify::Event::Received { peer_id, info, .. }
                )) => {
                    let peer = IdentifiedPeer {
                        peer_id,
                        protocol_version: info.protocol_version,
                        agent_version:    info.agent_version,
                        listen_addrs:     info.listen_addrs,
                    };
                    if let Some(w) = id_waiters.pop_front() {
                        let _ = w.send(peer);
                    } else {
                        pending.push_back(peer);
                    }
                }
                SwarmEvent::Behaviour(IntelNavBehaviourEvent::Ping(ev)) => {
                    debug!(?ev, "ping");
                }
                SwarmEvent::IncomingConnectionError { error, .. } => {
                    warn!(?error, "incoming connection error");
                }
                SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                    warn!(?peer_id, ?error, "outgoing connection error");
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
