//! User-space TCP shaper for IntelNav demos.
//!
//! Sits between two TCP endpoints and applies a configurable
//! impairment profile per direction:
//!
//!   * `delay_ms`   — one-way latency (mean)
//!   * `jitter_ms`  — gaussian stddev added to the mean
//!   * `bw_mbps`    — bandwidth cap, token-bucket
//!   * `loss_pct`   — chance to drop a chunk on arrival
//!   * `reorder_pct`— chance to swap a chunk with the previous one
//!
//! All knobs are live-mutable through the [`Shaper`] handle. Changes
//! take effect for the *next* chunk read off the wire — already-queued
//! bytes still drain on their original schedule, which is the same
//! semantics you get from `tc qdisc change`.
//!
//! What this isn't: a packet-level netem replacement. We forward at
//! the byte-stream level, so TCP control loops see stretched RTTs but
//! not packet drops in the kernel sense — the application sees a
//! coalesced read return short or a closed connection on `loss_pct`
//! crossing 100%. Good enough for a demo, not a research benchmark.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use bytes::BytesMut;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio::time::{sleep_until, Instant};

/// Live-mutable impairment profile for one direction of a TCP link.
///
/// Defaults render as a perfect link (no delay, no jitter, no cap, no
/// loss). Set fields explicitly for what you want to simulate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkParams {
    /// One-way delay in ms applied to every chunk before it's
    /// forwarded. The full RTT seen by the application is
    /// `forward.delay_ms + reverse.delay_ms`.
    pub delay_ms:    f64,
    /// Gaussian stddev added on top of `delay_ms`. Clipped to
    /// `[0, +inf)` so a chunk never arrives before its predecessor's
    /// schedule.
    pub jitter_ms:   f64,
    /// Bandwidth cap in megabits/sec. `0.0` means uncapped. The
    /// token bucket refills continuously and bursts up to one second
    /// worth of capacity.
    pub bw_mbps:     f64,
    /// Chance (0.0..=1.0) that a single chunk gets dropped on arrival
    /// rather than forwarded. A chunk is one socket read, not a TCP
    /// segment — the packet/segment boundary lives below us.
    pub loss_pct:    f64,
    /// Chance (0.0..=1.0) that a chunk gets its scheduled deadline
    /// nudged behind the next chunk's, simulating out-of-order
    /// delivery. The downstream reader still sees bytes in order
    /// because we serialize writes — the visible effect is a brief
    /// extra delay, same as a real TCP path with reordering.
    pub reorder_pct: f64,
}

impl Default for LinkParams {
    fn default() -> Self {
        Self {
            delay_ms:    0.0,
            jitter_ms:   0.0,
            bw_mbps:     0.0,
            loss_pct:    0.0,
            reorder_pct: 0.0,
        }
    }
}

/// Per-direction stats. Counters are monotonic since shaper start;
/// the SPA does its own delta math against the previous poll.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LinkStats {
    pub bytes:           u64,
    pub chunks:          u64,
    pub dropped_chunks:  u64,
    pub last_delay_ms:   f64,
    /// Effective bandwidth observed over the last second, in bytes.
    /// Sliding window aggregation is done by the caller — we publish
    /// the raw monotonic counters and let the HTTP layer compute
    /// deltas.
    pub bytes_recent_s:  u64,
}

/// Full shaper config: forward (`a → b`) and reverse (`b → a`)
/// independently. Most demos want the reverse leg to mirror the
/// forward leg, but having them separate matches netem's behaviour
/// (asymmetric links are a real thing — a home uplink isn't a home
/// downlink).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaperConfig {
    pub upstream: SocketAddr,
    #[serde(default)]
    pub forward:  LinkParams,
    #[serde(default)]
    pub reverse:  LinkParams,
    /// Tier label exposed via `/stats` so the SPA can render
    /// "Tokyo VPS" or "Berlin LAN" without having to invent its own
    /// classification rules.
    #[serde(default)]
    pub label:    String,
}

/// A running shaper. Cheap to clone — the live state lives behind
/// [`RwLock`] / [`AtomicU64`].
#[derive(Clone)]
pub struct Shaper {
    inner: Arc<Inner>,
}

struct Inner {
    label:   parking_mutex::Mutex<String>,
    forward: RwLock<LinkParams>,
    reverse: RwLock<LinkParams>,
    upstream: SocketAddr,
    stats_forward: parking_mutex::Mutex<StatsAccum>,
    stats_reverse: parking_mutex::Mutex<StatsAccum>,
}

/// Thin wrapper around `std::sync::Mutex` so we don't pull in
/// parking_lot. Module name is shorter than `std::sync` for the
/// callsites below.
mod parking_mutex {
    pub use std::sync::Mutex;
}

#[derive(Default)]
struct StatsAccum {
    bytes:          u64,
    chunks:         u64,
    dropped_chunks: u64,
    last_delay_ms:  f64,
    /// Ring of (instant_ms, bytes) for the last 60 seconds.
    /// Aggregation cadence is "one entry per write," which is bursty
    /// but bounded by the bandwidth cap. We trim entries older than
    /// 1 s when computing `bytes_recent_s`.
    window:         Vec<(u64, u64)>,
}

impl StatsAccum {
    fn record(&mut self, bytes: u64, delay_ms: f64) {
        self.bytes += bytes;
        self.chunks += 1;
        self.last_delay_ms = delay_ms;
        let now_ms = monotonic_ms();
        self.window.push((now_ms, bytes));
        // Keep the window cheap — drop entries older than 1 s.
        let cutoff = now_ms.saturating_sub(1000);
        if self.window.len() > 256 || self.window.first().map_or(false, |(t, _)| *t < cutoff) {
            self.window.retain(|(t, _)| *t >= cutoff);
        }
    }
    fn snapshot(&self) -> LinkStats {
        let cutoff = monotonic_ms().saturating_sub(1000);
        let recent: u64 = self.window.iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, b)| *b).sum();
        LinkStats {
            bytes:          self.bytes,
            chunks:         self.chunks,
            dropped_chunks: self.dropped_chunks,
            last_delay_ms:  self.last_delay_ms,
            bytes_recent_s: recent,
        }
    }
    fn count_drop(&mut self) {
        self.dropped_chunks += 1;
    }
}

fn monotonic_ms() -> u64 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_millis() as u64
}

/// Public summary read by the control plane.
#[derive(Clone, Debug, Serialize)]
pub struct ShaperSnapshot {
    pub label:    String,
    pub upstream: String,
    pub forward:  LinkParams,
    pub reverse:  LinkParams,
    pub forward_stats: LinkStats,
    pub reverse_stats: LinkStats,
}

impl Shaper {
    pub fn new(cfg: ShaperConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                label:   parking_mutex::Mutex::new(cfg.label),
                forward: RwLock::new(cfg.forward),
                reverse: RwLock::new(cfg.reverse),
                upstream: cfg.upstream,
                stats_forward: Default::default(),
                stats_reverse: Default::default(),
            }),
        }
    }

    /// Replace one direction's params atomically. `forward = true`
    /// targets `client → upstream`; `forward = false` is the reverse.
    pub async fn set_link(&self, forward: bool, params: LinkParams) {
        let lock = if forward { &self.inner.forward } else { &self.inner.reverse };
        *lock.write().await = params;
    }

    pub async fn set_label(&self, label: String) {
        *self.inner.label.lock().unwrap() = label;
    }

    pub async fn snapshot(&self) -> ShaperSnapshot {
        let label    = self.inner.label.lock().unwrap().clone();
        let forward  = self.inner.forward.read().await.clone();
        let reverse  = self.inner.reverse.read().await.clone();
        let fs = self.inner.stats_forward.lock().unwrap().snapshot();
        let rs = self.inner.stats_reverse.lock().unwrap().snapshot();
        ShaperSnapshot {
            label,
            upstream: self.inner.upstream.to_string(),
            forward,
            reverse,
            forward_stats: fs,
            reverse_stats: rs,
        }
    }

    /// Bind `listen` and forward every accepted connection to
    /// `upstream`, applying the live params. Runs until the listener
    /// errors or the future is dropped.
    pub async fn serve(&self, listen: SocketAddr) -> Result<()> {
        let lst = TcpListener::bind(listen).await
            .with_context(|| format!("bind shaper listener on {listen}"))?;
        tracing::info!(%listen, upstream = %self.inner.upstream, "netsim listening");
        loop {
            let (client, peer_addr) = match lst.accept().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(?e, "accept failed");
                    continue;
                }
            };
            let me = self.clone();
            tokio::spawn(async move {
                if let Err(e) = me.handle(client).await {
                    tracing::debug!(%peer_addr, ?e, "shaper conn ended");
                }
            });
        }
    }

    async fn handle(&self, client: TcpStream) -> Result<()> {
        client.set_nodelay(true).ok();
        let upstream = TcpStream::connect(self.inner.upstream).await
            .with_context(|| format!("connect upstream {}", self.inner.upstream))?;
        upstream.set_nodelay(true).ok();

        let (cr, cw) = client.into_split();
        let (ur, uw) = upstream.into_split();

        let me_f = self.clone();
        let f = tokio::spawn(async move {
            me_f.pump(cr, uw, true).await
        });
        let me_r = self.clone();
        let r = tokio::spawn(async move {
            me_r.pump(ur, cw, false).await
        });
        // Either side closing tears down the other.
        let _ = tokio::try_join!(f, r);
        Ok(())
    }

    async fn pump<R, W>(&self, mut src: R, mut dst: W, forward: bool) -> Result<()>
    where
        R: AsyncReadExt + Unpin,
        W: AsyncWriteExt + Unpin,
    {
        let mut buf = BytesMut::with_capacity(64 * 1024);
        let mut bucket = TokenBucket::new();
        let mut last_deadline: Option<Instant> = None;
        loop {
            buf.clear();
            buf.resize(64 * 1024, 0);
            let n = match src.read(&mut buf[..]).await {
                Ok(0) => break,
                Ok(n) => n,
                Err(_) => break,
            };
            buf.truncate(n);

            // Pull the live params for this chunk. Lock dance is cheap
            // — RwLock::read is one atomic on the fast path.
            let params = {
                let lock = if forward { &self.inner.forward } else { &self.inner.reverse };
                lock.read().await.clone()
            };

            // Loss: drop the chunk entirely.
            if params.loss_pct > 0.0 && rand::random::<f64>() < params.loss_pct.clamp(0.0, 1.0) {
                let stats = if forward { &self.inner.stats_forward } else { &self.inner.stats_reverse };
                stats.lock().unwrap().count_drop();
                continue;
            }

            // Bandwidth: wait until the bucket can afford the bytes.
            if params.bw_mbps > 0.0 {
                bucket.consume(n, params.bw_mbps).await;
            }

            // Delay: mean + clipped gaussian jitter. Sample a fresh
            // ThreadRng on each iteration to keep this future `Send`
            // (ThreadRng is `!Send` and survives an await otherwise).
            let mut delay_ms = params.delay_ms;
            if params.jitter_ms > 0.0 {
                delay_ms += gaussian_sample() * params.jitter_ms;
            }
            if delay_ms < 0.0 { delay_ms = 0.0; }

            let mut deadline = Instant::now() + Duration::from_micros((delay_ms * 1000.0) as u64);

            // Reorder: nudge this chunk's deadline behind the previous
            // one so it lands later. Writes are still serialized — the
            // visible effect on the receiver is a stretched
            // inter-chunk gap, the same shape kernel netem produces.
            if params.reorder_pct > 0.0 && rand::random::<f64>() < params.reorder_pct.clamp(0.0, 1.0) {
                if let Some(prev) = last_deadline {
                    let extra: f64 = rand::random();
                    deadline = prev + Duration::from_micros(((1.0 + extra * 5.0) * 1000.0) as u64);
                }
            }
            last_deadline = Some(deadline);

            sleep_until(deadline).await;

            if dst.write_all(&buf[..n]).await.is_err() {
                break;
            }
            let _ = dst.flush().await;

            let stats = if forward { &self.inner.stats_forward } else { &self.inner.stats_reverse };
            stats.lock().unwrap().record(n as u64, delay_ms);
        }
        Ok(())
    }
}

/// Box-Muller-ish gaussian sample. Coarse but fine for jitter.
/// Uses a fresh `ThreadRng` per call so the future stays `Send`.
fn gaussian_sample() -> f64 {
    let mut rng = rand::thread_rng();
    let u1: f64 = rng.gen_range(1e-9..1.0);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Continuous-fill token bucket. `consume(n, rate_mbps)` sleeps until
/// `n` bytes can be sent at the given rate. Burst capacity = 1 s.
struct TokenBucket {
    last: Instant,
    tokens: f64,
}

impl TokenBucket {
    fn new() -> Self { Self { last: Instant::now(), tokens: 0.0 } }
    async fn consume(&mut self, bytes: usize, rate_mbps: f64) {
        let bytes_per_sec = (rate_mbps * 1_000_000.0) / 8.0;
        let cap = bytes_per_sec; // 1-second burst
        loop {
            let now = Instant::now();
            let dt = now.duration_since(self.last).as_secs_f64();
            self.last = now;
            self.tokens = (self.tokens + dt * bytes_per_sec).min(cap);
            if self.tokens >= bytes as f64 {
                self.tokens -= bytes as f64;
                return;
            }
            let need = bytes as f64 - self.tokens;
            let wait = need / bytes_per_sec;
            tokio::time::sleep(Duration::from_micros((wait * 1_000_000.0) as u64)).await;
        }
    }
}

/// Convenience helper used by the binary: parse a CSV-ish tier
/// string into a [`LinkParams`]. Format: `delay=40,jitter=4,bw=100,loss=0.01`.
/// Unknown keys are ignored — we'd rather an SPA-side typo silently
/// no-op than crash the proxy mid-demo.
pub fn parse_params(s: &str) -> Result<LinkParams> {
    let mut p = LinkParams::default();
    if s.trim().is_empty() {
        return Ok(p);
    }
    for kv in s.split(',') {
        let (k, v) = kv.split_once('=').ok_or_else(|| anyhow!("expected key=value, got `{kv}`"))?;
        let k = k.trim();
        let v: f64 = v.trim().parse().with_context(|| format!("parsing {k}"))?;
        match k {
            "delay" | "delay_ms"     => p.delay_ms = v,
            "jitter"| "jitter_ms"    => p.jitter_ms = v,
            "bw"    | "bw_mbps"      => p.bw_mbps = v,
            "loss"  | "loss_pct"     => p.loss_pct = v,
            "reorder"|"reorder_pct"  => p.reorder_pct = v,
            _ => tracing::warn!(unknown = %k, "ignoring netsim param"),
        }
    }
    Ok(p)
}
