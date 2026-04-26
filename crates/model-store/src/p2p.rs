//! Libp2p-based chunk fetcher (Phase 4 MVP).
//!
//! Mirrors the public shape of [`crate::http`]: caller supplies a
//! manifest CID + a plan, we return a [`FetchOutcome`] pointing at a
//! local cache directory the stitcher can read. The transport swaps
//! from HTTP to a libp2p swarm running Kademlia + a request-response
//! protocol over TCP + Noise + yamux.
//!
//! Discovery: each seeder announces every chunk CID it holds via
//! Kademlia provider records. A fetcher queries `get_providers(cid)`,
//! picks any provider, and opens a request-response stream asking
//! for the chunk. The responder streams the raw bytes; the fetcher
//! verifies SHA-256 against the CID before caching.
//!
//! Scope limits (documented on purpose):
//!
//! * LAN / explicit-multiaddr only. No NAT traversal, no AutoNAT,
//!   no relay. Internet-spanning swarm is a follow-up.
//! * No DHT bootstrap-seed list. Caller supplies known peer addrs.
//! * No authorization / rate limiting. Anyone who finds us can pull
//!   any chunk we've announced — chunks are content-addressed so
//!   this is public data anyway, but a peer-budget / allowlist is
//!   Phase-5 material.
//!
//! Manifest CID is published under the same provider-record scheme
//! as any other chunk; the fetcher pulls the manifest first, then
//! plans the chunk set, then pulls those.

#![cfg(feature = "p2p")]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use libp2p::{
    identify, identity, kad,
    multiaddr::Multiaddr,
    noise,
    request_response::{self, ProtocolSupport},
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, PeerId, StreamProtocol, Swarm, SwarmBuilder,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::cid::{cid_string_for, cid_string_from_sha256};
use crate::manifest::{Chunk, Manifest};
use crate::http::{FetchedManifest, FetchOutcome, FetchPlan};

// ---------- Protocol ----------

/// Wire messages for the chunk request-response protocol.
///
/// Encoded with CBOR (libp2p's built-in cbor codec — not the fastest
/// but keeps us off a custom varint protocol while we prove things
/// out). If this ever hits a throughput wall, swap to a length-
/// prefixed raw-bytes codec; the protocol ID bumps so old and new
/// peers can coexist.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChunkRequest {
    /// Please send me the bytes of chunk `cid`.
    Get { cid: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChunkResponse {
    Bytes(Vec<u8>),
    /// We do not hold (or won't serve) that chunk. Fetcher falls
    /// back to another provider.
    NotFound,
}

const CHUNK_PROTOCOL: StreamProtocol = StreamProtocol::new("/intelnav/chunk/1.0.0");

// ---------- Swarm behaviour ----------

#[derive(NetworkBehaviour)]
struct Behaviour {
    kad: kad::Behaviour<kad::store::MemoryStore>,
    reqres: request_response::cbor::Behaviour<ChunkRequest, ChunkResponse>,
    identify: identify::Behaviour,
}

// ---------- Node (seeder or fetcher) ----------

/// A running libp2p node that can seed and/or fetch IntelNav chunks.
///
/// Holds a tokio task driving the swarm; commands flow through an
/// mpsc channel so the async API stays non-blocking.
pub struct Node {
    tx:        mpsc::Sender<Command>,
    pub peer_id:   PeerId,
    pub listen_addrs: Vec<Multiaddr>,
}

enum Command {
    /// Announce that we hold this CID — callers must have the chunk
    /// bytes available via [`Node::seed_dir`].
    Announce  { cid: String, done: oneshot::Sender<Result<()>> },
    /// Fetch a chunk's bytes, verified against its CID.
    Fetch     { cid: String, expected_size: u64, done: oneshot::Sender<Result<Vec<u8>>> },
    /// Tell the swarm to dial a bootstrap multiaddr.
    Dial      { addr: Multiaddr, done: oneshot::Sender<Result<()>> },
}

/// Build a node with an ephemeral identity (for tests) or the given
/// keypair (production — the caller reuses the node's ed25519 peer
/// identity across sessions so providers accumulate across runs).
///
/// `seed_dir` is where chunk bytes are read from on an incoming
/// request — typically a cache directory produced by
/// [`crate::chunker`] or a previous HTTP fetch. `None` means
/// fetcher-only.
pub async fn spawn_node(
    keypair:  identity::Keypair,
    listen:   Multiaddr,
    seed_dir: Option<PathBuf>,
) -> Result<Node> {
    let peer_id = PeerId::from(keypair.public());
    let mut swarm: Swarm<Behaviour> = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .context("libp2p tcp/noise/yamux stack")?
        .with_behaviour(|key| {
            let store = kad::store::MemoryStore::new(key.public().to_peer_id());
            let mut kcfg = kad::Config::new(StreamProtocol::new("/intelnav/kad/1.0.0"));
            kcfg.set_query_timeout(Duration::from_secs(30));
            let kad = kad::Behaviour::with_config(key.public().to_peer_id(), store, kcfg);

            let reqres = request_response::cbor::Behaviour::<ChunkRequest, ChunkResponse>::new(
                [(CHUNK_PROTOCOL, ProtocolSupport::Full)],
                request_response::Config::default()
                    .with_request_timeout(Duration::from_secs(120)),
            );

            let identify = identify::Behaviour::new(identify::Config::new(
                "/intelnav/id/1.0.0".to_string(),
                key.public(),
            ));

            Ok(Behaviour { kad, reqres, identify })
        })
        .map_err(|e| anyhow!("building swarm behaviour: {e}"))?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(300)))
        .build();

    swarm.listen_on(listen.clone()).context("listen_on")?;
    // Kademlia defaults to client mode; flip to server so we answer
    // provider queries from other peers.
    swarm.behaviour_mut().kad.set_mode(Some(kad::Mode::Server));

    // Wait for the first `NewListenAddr` event so the caller can
    // publish the bound addr to bootstrap peers.
    let mut listen_addrs = Vec::new();
    loop {
        match swarm.next_event().await {
            SwarmEvent::NewListenAddr { address, .. } => {
                listen_addrs.push(address.clone());
                break;
            }
            other => debug!(?other, "pre-listen event"),
        }
    }

    let (tx, rx) = mpsc::channel::<Command>(32);
    let node = Node { tx, peer_id, listen_addrs: listen_addrs.clone() };
    tokio::spawn(drive_swarm(swarm, rx, seed_dir));
    Ok(node)
}

impl Node {
    pub async fn announce(&self, cid: impl Into<String>) -> Result<()> {
        let (done, rx) = oneshot::channel();
        self.tx.send(Command::Announce { cid: cid.into(), done }).await
            .context("sending announce command")?;
        rx.await.context("announce reply")?
    }

    pub async fn fetch(&self, cid: impl Into<String>, expected_size: u64) -> Result<Vec<u8>> {
        let (done, rx) = oneshot::channel();
        self.tx.send(Command::Fetch { cid: cid.into(), expected_size, done }).await
            .context("sending fetch command")?;
        rx.await.context("fetch reply")?
    }

    pub async fn dial(&self, addr: Multiaddr) -> Result<()> {
        let (done, rx) = oneshot::channel();
        self.tx.send(Command::Dial { addr, done }).await
            .context("sending dial command")?;
        rx.await.context("dial reply")?
    }
}

// ---------- Swarm driver ----------

async fn drive_swarm(
    mut swarm:   Swarm<Behaviour>,
    mut rx:      mpsc::Receiver<Command>,
    seed_dir:    Option<PathBuf>,
) {
    use std::collections::HashMap;
    // Pending fetches: CID → ( expected_size, waiter )
    struct PendingFetch {
        expected_size: u64,
        providers_found: HashSet<PeerId>,
        providers_tried: HashSet<PeerId>,
        waiter: oneshot::Sender<Result<Vec<u8>>>,
    }
    let mut pending_fetch: HashMap<String, PendingFetch> = HashMap::new();
    // Map outbound request-response id → CID so we can correlate
    // replies back to the pending fetch.
    let mut req_to_cid: HashMap<request_response::OutboundRequestId, String> = HashMap::new();
    // Map kad GetProviders query id → CID.
    let mut provider_query: HashMap<kad::QueryId, String> = HashMap::new();

    loop {
        tokio::select! {
            // ---- Commands from the node's public API ----
            cmd = rx.recv() => {
                let Some(cmd) = cmd else { return; };
                match cmd {
                    Command::Announce { cid, done } => {
                        let key = cid.clone().into_bytes();
                        match swarm.behaviour_mut().kad.start_providing(kad::RecordKey::new(&key)) {
                            Ok(_query_id) => {
                                debug!(%cid, "announce started");
                                let _ = done.send(Ok(()));
                            }
                            Err(e) => {
                                let _ = done.send(Err(anyhow!("kad start_providing: {e:?}")));
                            }
                        }
                    }
                    Command::Fetch { cid, expected_size, done } => {
                        let key = cid.clone().into_bytes();
                        let qid = swarm.behaviour_mut().kad.get_providers(kad::RecordKey::new(&key));
                        provider_query.insert(qid, cid.clone());
                        pending_fetch.insert(cid, PendingFetch {
                            expected_size,
                            providers_found: HashSet::new(),
                            providers_tried: HashSet::new(),
                            waiter: done,
                        });
                    }
                    Command::Dial { addr, done } => {
                        match swarm.dial(addr) {
                            Ok(()) => { let _ = done.send(Ok(())); }
                            Err(e) => { let _ = done.send(Err(anyhow!("dial: {e}"))); }
                        }
                    }
                }
            }

            // ---- Events from the swarm ----
            event = swarm.next_event() => {
                match event {
                    // Kad reply: list of providers for a CID.
                    SwarmEvent::Behaviour(BehaviourEvent::Kad(
                        kad::Event::OutboundQueryProgressed { id, result, .. }
                    )) => {
                        let Some(cid) = provider_query.get(&id).cloned() else { continue; };
                        match result {
                            kad::QueryResult::GetProviders(Ok(
                                kad::GetProvidersOk::FoundProviders { providers, .. }
                            )) => {
                                if let Some(pf) = pending_fetch.get_mut(&cid) {
                                    for p in &providers {
                                        pf.providers_found.insert(*p);
                                    }
                                    // Ask the first untried provider.
                                    for p in &providers {
                                        if pf.providers_tried.insert(*p) {
                                            let req_id = swarm.behaviour_mut().reqres
                                                .send_request(p, ChunkRequest::Get { cid: cid.clone() });
                                            req_to_cid.insert(req_id, cid.clone());
                                            debug!(%cid, peer = %p, "chunk request sent");
                                            break;
                                        }
                                    }
                                }
                            }
                            kad::QueryResult::GetProviders(Ok(
                                kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. }
                            )) => {
                                // Query closed. If we haven't found anyone yet, fail fast.
                                if let Some(pf) = pending_fetch.remove(&cid) {
                                    if pf.providers_found.is_empty() {
                                        let _ = pf.waiter.send(Err(anyhow!(
                                            "no providers found for chunk {cid}"
                                        )));
                                    } else {
                                        // Still waiting on the active request-response.
                                        pending_fetch.insert(cid.clone(), pf);
                                    }
                                }
                            }
                            kad::QueryResult::GetProviders(Err(e)) => {
                                if let Some(pf) = pending_fetch.remove(&cid) {
                                    let _ = pf.waiter.send(Err(anyhow!("kad get_providers: {e:?}")));
                                }
                            }
                            kad::QueryResult::StartProviding(_) => {
                                debug!(%cid, "start_providing confirmed");
                            }
                            _ => {}
                        }
                    }

                    // RequestResponse: we got asked for a chunk.
                    SwarmEvent::Behaviour(BehaviourEvent::Reqres(
                        request_response::Event::Message { message, .. }
                    )) => {
                        match message {
                            request_response::Message::Request { request, channel, .. } => {
                                let ChunkRequest::Get { cid } = request;
                                let resp = match &seed_dir {
                                    Some(dir) => load_chunk_from_dir(dir, &cid).await
                                        .map(ChunkResponse::Bytes)
                                        .unwrap_or_else(|e| {
                                            warn!(%cid, error = %e, "seed lookup failed");
                                            ChunkResponse::NotFound
                                        }),
                                    None => ChunkResponse::NotFound,
                                };
                                let _ = swarm.behaviour_mut().reqres.send_response(channel, resp);
                            }
                            request_response::Message::Response { request_id, response } => {
                                let Some(cid) = req_to_cid.remove(&request_id) else { continue; };
                                let Some(pf) = pending_fetch.remove(&cid) else { continue; };
                                match response {
                                    ChunkResponse::Bytes(bytes) => {
                                        // Size check: enforced only when the caller
                                        // knew the expected size up front (chunks do;
                                        // the manifest doesn't, it uses u64::MAX as
                                        // "don't enforce, just hash").
                                        let size_ok = pf.expected_size == u64::MAX
                                            || bytes.len() as u64 == pf.expected_size;
                                        if !size_ok {
                                            let _ = pf.waiter.send(Err(anyhow!(
                                                "chunk {cid} size mismatch: expected {}, got {}",
                                                pf.expected_size, bytes.len()
                                            )));
                                        } else {
                                            let digest = Sha256::digest(&bytes);
                                            let actual = cid_string_from_sha256(&digest);
                                            if actual != cid {
                                                let _ = pf.waiter.send(Err(anyhow!(
                                                    "chunk hash mismatch: expected {cid}, got {actual}"
                                                )));
                                            } else {
                                                let _ = pf.waiter.send(Ok(bytes));
                                            }
                                        }
                                    }
                                    ChunkResponse::NotFound => {
                                        // Ask the next untried provider, if any.
                                        let untried: Vec<PeerId> = pf.providers_found.iter()
                                            .filter(|p| !pf.providers_tried.contains(p))
                                            .copied().collect();
                                        if let Some(p) = untried.first() {
                                            let mut pf = pf;
                                            pf.providers_tried.insert(*p);
                                            let req_id = swarm.behaviour_mut().reqres
                                                .send_request(p, ChunkRequest::Get { cid: cid.clone() });
                                            req_to_cid.insert(req_id, cid.clone());
                                            pending_fetch.insert(cid, pf);
                                        } else {
                                            let _ = pf.waiter.send(Err(anyhow!(
                                                "all {} providers returned NotFound for chunk {cid}",
                                                pf.providers_tried.len()
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Request failed outright (timeout, disconnect, etc.)
                    SwarmEvent::Behaviour(BehaviourEvent::Reqres(
                        request_response::Event::OutboundFailure { request_id, error, .. }
                    )) => {
                        let Some(cid) = req_to_cid.remove(&request_id) else { continue; };
                        if let Some(pf) = pending_fetch.remove(&cid) {
                            let _ = pf.waiter.send(Err(anyhow!("reqres outbound failure: {error:?}")));
                        }
                    }

                    // Identify finishes and gives us the peer's listen addrs — feed into Kad.
                    SwarmEvent::Behaviour(BehaviourEvent::Identify(
                        identify::Event::Received { peer_id, info, .. }
                    )) => {
                        for addr in info.listen_addrs {
                            swarm.behaviour_mut().kad.add_address(&peer_id, addr);
                        }
                    }

                    _ => {}
                }
            }
        }
    }
}

async fn load_chunk_from_dir(dir: &Path, cid: &str) -> Result<Vec<u8>> {
    // Same dual-layout logic as the stitcher: accept `<cache>/chunks/<cid>.bin`
    // (chunker layout) or `<cache>/<cid>.bin` (flat).
    let candidates = [
        dir.join("chunks").join(format!("{cid}.bin")),
        dir.join(format!("{cid}.bin")),
    ];
    for p in &candidates {
        if let Ok(bytes) = fs::read(p).await {
            // Sanity-verify what we're about to serve.
            if cid_string_for(&bytes) == cid {
                return Ok(bytes);
            } else {
                return Err(anyhow!("cached {cid} fails local hash check"));
            }
        }
    }
    Err(anyhow!("chunk {cid} not in seed dir {}", dir.display()))
}

// ---------- Fetch API (matches http.rs shape) ----------

/// Options for a P2P fetch. Keeps the same knobs as `FetchOptions`
/// where they make sense; adds bootstrap peers and the listen addr.
#[derive(Clone, Debug)]
pub struct P2pFetchOptions {
    pub cache_root: PathBuf,
    pub listen: Multiaddr,
    /// Multiaddrs of known peers to dial for DHT bootstrap. At least
    /// one is needed in the MVP since we don't have a well-known
    /// rendezvous yet.
    pub bootstrap: Vec<Multiaddr>,
    pub keypair: identity::Keypair,
    pub max_manifest_bytes: u64,
}

impl Default for P2pFetchOptions {
    fn default() -> Self {
        Self {
            cache_root: crate::http::default_cache_root(),
            listen: "/ip4/0.0.0.0/tcp/0".parse().unwrap(),
            bootstrap: Vec::new(),
            keypair: identity::Keypair::generate_ed25519(),
            max_manifest_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Fetch a manifest + chunk plan from a P2P swarm using an existing
/// [`Node`] that the caller has already connected to the swarm.
///
/// Reusing one node across the manifest + chunk steps matters because
/// each new node has to re-bootstrap its routing table; we'd
/// otherwise pay the DHT warmup cost twice. It also lets the caller
/// seed its own chunks (via `seed_dir`) while fetching, which is
/// exactly how peers behave in practice — part of the swarm, not a
/// client/server split.
pub async fn fetch_manifest_and_chunks_p2p_with(
    node:         &Node,
    manifest_cid: &str,
    plan:         &FetchPlan,
    cache_root:   &Path,
    max_manifest_bytes: u64,
) -> Result<FetchOutcome> {
    let bytes = node.fetch(manifest_cid, u64::MAX).await
        .with_context(|| format!("fetching manifest {manifest_cid}"))?;
    if bytes.len() as u64 > max_manifest_bytes {
        return Err(anyhow!("manifest body exceeds cap {max_manifest_bytes} bytes"));
    }
    let manifest = Manifest::from_json_bytes(&bytes).context("parsing manifest JSON")?;
    let actual_cid = cid_string_for(&bytes);
    if actual_cid != manifest_cid {
        return Err(anyhow!("manifest CID mismatch: got {actual_cid}, expected {manifest_cid}"));
    }
    let dir = cache_root.join(manifest_cid);
    let chunks_dir = dir.join("chunks");
    fs::create_dir_all(&chunks_dir).await
        .with_context(|| format!("creating {}", chunks_dir.display()))?;
    fs::write(dir.join("manifest.json"), &bytes).await
        .context("writing manifest.json")?;
    info!(manifest_cid = %manifest_cid, dir = %dir.display(), "manifest cached (p2p)");

    let mut wanted: Vec<Chunk> = Vec::new();
    wanted.push(manifest.header_chunk.clone());
    match plan {
        FetchPlan::Full => {
            for b in &manifest.bundles {
                wanted.push(Chunk { cid: b.cid.clone(), size: b.size });
            }
        }
        FetchPlan::Bundles(names) => {
            for n in names {
                let b = manifest.bundles.iter().find(|b| &b.name == n)
                    .ok_or_else(|| anyhow!("manifest has no bundle named {n}"))?;
                wanted.push(Chunk { cid: b.cid.clone(), size: b.size });
            }
        }
    }

    let mut bytes_downloaded: u64 = 0;
    let mut bytes_reused: u64 = 0;
    for c in &wanted {
        let final_path = chunks_dir.join(format!("{}.bin", c.cid));
        if fs::try_exists(&final_path).await.unwrap_or(false) {
            let existing = fs::read(&final_path).await.unwrap_or_default();
            if existing.len() as u64 == c.size && cid_string_for(&existing) == c.cid {
                bytes_reused += c.size;
                continue;
            }
        }
        let bytes = node.fetch(&c.cid, c.size).await
            .with_context(|| format!("fetching chunk {}", c.cid))?;
        let tmp = chunks_dir.join(format!("{}.bin.tmp", c.cid));
        fs::write(&tmp, &bytes).await?;
        fs::rename(&tmp, &final_path).await?;
        bytes_downloaded += c.size;
    }

    Ok(FetchOutcome {
        dir,
        manifest,
        manifest_cid: manifest_cid.to_string(),
        bytes_downloaded,
        bytes_reused,
    })
}

/// Convenience wrapper: spawn a fresh node, dial bootstrap peers,
/// wait briefly for Kad routing to populate, then delegate to
/// [`fetch_manifest_and_chunks_p2p_with`]. This is the simplest
/// "gimme the model over P2P" entry point and what the CLI / pipe
/// peer will call.
pub async fn fetch_manifest_and_chunks_p2p(
    manifest_cid: &str,
    plan:         &FetchPlan,
    opts:         &P2pFetchOptions,
) -> Result<FetchOutcome> {
    let node = spawn_node(opts.keypair.clone(), opts.listen.clone(), None).await?;
    for addr in &opts.bootstrap {
        node.dial(addr.clone()).await.with_context(|| format!("dial bootstrap {addr}"))?;
    }
    // Let dials hand-shake + identify exchange settle so Kad sees
    // the bootstrap peers in its routing table. 500 ms is more than
    // enough on a LAN; over a WAN this might want a wait-for-event
    // style primitive. Phase 5 polish.
    tokio::time::sleep(Duration::from_millis(500)).await;
    fetch_manifest_and_chunks_p2p_with(
        &node, manifest_cid, plan, &opts.cache_root, opts.max_manifest_bytes,
    ).await
}

// ---------- Small helper: Swarm::next_event with `await` ----------
//
// libp2p 0.56 returns events through `.select_next_some()`, which
// requires StreamExt; we re-export that convenience here so the
// driver loop stays readable.
use futures_util::StreamExt as _;
#[async_trait::async_trait]
trait SwarmNext {
    async fn next_event(&mut self) -> SwarmEvent<BehaviourEvent>;
}
#[async_trait::async_trait]
impl SwarmNext for Swarm<Behaviour> {
    async fn next_event(&mut self) -> SwarmEvent<BehaviourEvent> {
        self.select_next_some().await
    }
}
