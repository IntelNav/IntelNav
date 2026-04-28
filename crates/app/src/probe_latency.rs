//! Pre-flight latency probe for forward peers.
//!
//! We want `ChainTarget::from_swarm` to rank candidates by *real*
//! latency, not just freshness. The probe opens a plain TCP connection
//! to each candidate's `forward_url`, measures the connect RTT, and
//! drops the socket. Anything > [`PROBE_TIMEOUT`] counts as unreachable.
//!
//! Results are cached in a small LRU keyed by `SocketAddr`. The cache
//! is cleared automatically after [`CACHE_TTL`] so a peer that comes
//! back online doesn't stay blacklisted.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::net::TcpStream;
use tokio::time::timeout;

const PROBE_TIMEOUT: Duration = Duration::from_millis(800);
const CACHE_TTL:     Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug)]
pub struct ProbeResult {
    /// Round-trip time of the TCP handshake. `None` means unreachable.
    pub rtt: Option<Duration>,
    pub measured_at: Instant,
}

/// Score for ranking: lower is better. Unreachable peers get a sentinel
/// `u64::MAX` so they sort last but still appear (we'd rather try a
/// stale alternate than fail outright).
pub fn score(result: &ProbeResult) -> u64 {
    match result.rtt {
        Some(d) => d.as_micros() as u64,
        None    => u64::MAX,
    }
}

static CACHE: OnceLock<Mutex<HashMap<SocketAddr, ProbeResult>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<SocketAddr, ProbeResult>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_get(addr: &SocketAddr) -> Option<ProbeResult> {
    let g = cache().lock().ok()?;
    let r = g.get(addr).copied()?;
    if r.measured_at.elapsed() < CACHE_TTL { Some(r) } else { None }
}

fn cache_put(addr: SocketAddr, r: ProbeResult) {
    if let Ok(mut g) = cache().lock() {
        g.insert(addr, r);
    }
}

/// Probe one peer. Cheap — just a TCP handshake.
pub async fn probe(addr: SocketAddr) -> ProbeResult {
    if let Some(cached) = cache_get(&addr) { return cached; }
    let start = Instant::now();
    let r = match timeout(PROBE_TIMEOUT, TcpStream::connect(addr)).await {
        Ok(Ok(_))   => ProbeResult { rtt: Some(start.elapsed()), measured_at: Instant::now() },
        Ok(Err(_))  => ProbeResult { rtt: None, measured_at: Instant::now() },
        Err(_)      => ProbeResult { rtt: None, measured_at: Instant::now() },
    };
    cache_put(addr, r);
    r
}

/// Probe many peers in parallel. Returns one result per input addr,
/// in the same order — even unreachable peers get an entry so the
/// caller can keep them as last-resort alternates.
pub async fn probe_many(addrs: &[SocketAddr]) -> Vec<ProbeResult> {
    let futs = addrs.iter().copied().map(probe);
    futures::future::join_all(futs).await
}
