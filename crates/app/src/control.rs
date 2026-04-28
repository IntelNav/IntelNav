//! Control RPC between `intelnav` (chat client) and `intelnav-node` (daemon).
//!
//! Transport: a Unix socket at `<data_dir>/control.sock`, mode `0600`,
//! JSON-lines (one Request per line, one Response per line). The daemon
//! is the server; the chat client is the client.
//!
//! This is intentionally tiny — no protobuf, no HTTP server. Auth is
//! "you have access to my data dir," which is the same trust boundary
//! as `peer.key`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, warn};

// ----------------------------------------------------------------------
//  Wire types
// ----------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum Request {
    /// Daemon status: peer id, uptime, total chain count.
    Status,
    /// List every (cid, range) the daemon is currently hosting.
    ListHosted,
    /// Add a slice to hosting. Caller has already written the
    /// `kept_ranges.json` sidecar; this just kicks the announce loop.
    Join { cid: String, start: u16, end: u16 },
    /// Begin graceful drain of a slice. Stops re-announces immediately,
    /// refuses new forward connections, lets in-flight chains finish.
    Leave { cid: String, start: u16, end: u16 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Response {
    Status(StatusReply),
    Hosted { slices: Vec<HostedSlice> },
    Joined,
    Leaving,
    Error { message: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatusReply {
    pub peer_id: String,
    pub uptime_seconds: u64,
    pub total_active_chains: u32,
    pub hosted_slices: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostedSlice {
    pub cid: String,
    pub display_name: String,
    pub start: u16,
    pub end: u16,
    pub state: SliceStatus,
    pub active_chains: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceStatus {
    Announcing,
    Draining,
    Stopped,
}

// ----------------------------------------------------------------------
//  Server-side state
// ----------------------------------------------------------------------

/// Shared mutable state held by the daemon's control server.
///
/// The state machine for each slice (#35) writes here; the announce
/// loop (in [`crate::swarm_node`]) reads here to decide whether to
/// republish; the forward listener reads here to decide whether to
/// accept a new connection.
pub struct ControlState {
    pub peer_id: String,
    pub started_at: SystemTime,
    pub slices: Mutex<BTreeMap<SliceKey, SliceEntry>>,
}

pub type SliceKey = (String, u16, u16); // (cid, start, end)

#[derive(Clone, Debug)]
pub struct SliceEntry {
    pub display_name: String,
    pub state: SliceStatus,
    pub active_chains: u32,
    /// Wall-clock instant at which this slice transitioned to Draining.
    /// Used by the drain-timeout watchdog (#36) to force-stop a slice
    /// whose grace period has elapsed even if chains are still attached.
    pub draining_since: Option<std::time::Instant>,
}

impl ControlState {
    pub fn new(peer_id: String) -> Arc<Self> {
        Arc::new(Self {
            peer_id,
            started_at: SystemTime::now(),
            slices: Mutex::new(BTreeMap::new()),
        })
    }

    pub async fn snapshot_status(&self) -> StatusReply {
        let slices = self.slices.lock().await;
        let total_active: u32 = slices.values().map(|s| s.active_chains).sum();
        StatusReply {
            peer_id: self.peer_id.clone(),
            uptime_seconds: self.started_at.elapsed().unwrap_or(Duration::ZERO).as_secs(),
            total_active_chains: total_active,
            hosted_slices: slices.len() as u32,
        }
    }

    pub async fn snapshot_hosted(&self) -> Vec<HostedSlice> {
        let slices = self.slices.lock().await;
        slices.iter().map(|((cid, start, end), entry)| HostedSlice {
            cid: cid.clone(),
            display_name: entry.display_name.clone(),
            start: *start,
            end: *end,
            state: entry.state,
            active_chains: entry.active_chains,
        }).collect()
    }

    /// Mark a slice as Announcing. Used on daemon boot to seed state
    /// from the kept_ranges.json sidecars.
    pub async fn upsert_announcing(&self, key: SliceKey, display_name: String) {
        let mut slices = self.slices.lock().await;
        slices.entry(key)
            .and_modify(|e| {
                e.state = SliceStatus::Announcing;
                e.display_name = display_name.clone();
            })
            .or_insert(SliceEntry {
                display_name,
                state: SliceStatus::Announcing,
                active_chains: 0,
                draining_since: None,
            });
    }

    pub async fn begin_drain(&self, key: &SliceKey) -> bool {
        let mut slices = self.slices.lock().await;
        if let Some(entry) = slices.get_mut(key) {
            if entry.state == SliceStatus::Announcing {
                entry.state = SliceStatus::Draining;
                entry.draining_since = Some(std::time::Instant::now());
                return true;
            }
        }
        false
    }

    /// Return every slice currently in Draining whose grace period has
    /// elapsed. Returns the keys *and* the SliceEntry so callers can
    /// log the active_chains count for telemetry.
    pub async fn expired_draining(&self, grace: Duration) -> Vec<SliceKey> {
        let slices = self.slices.lock().await;
        slices.iter()
            .filter_map(|(k, e)| {
                if e.state != SliceStatus::Draining { return None; }
                let since = e.draining_since?;
                if since.elapsed() >= grace { Some(k.clone()) } else { None }
            })
            .collect()
    }

    pub async fn is_announcing(&self, key: &SliceKey) -> bool {
        let slices = self.slices.lock().await;
        slices.get(key).map(|e| e.state == SliceStatus::Announcing).unwrap_or(false)
    }

    /// Try to admit a new chain through `key`. Returns a [`ChainGuard`]
    /// whose Drop decrements `active_chains`. Refuses with `None` if
    /// the slice is Draining or Stopped — the forward listener should
    /// close the connection with a "draining, retry elsewhere" error
    /// so the consumer fails over to its alternate provider.
    pub async fn accept_chain(self: &Arc<Self>, key: &SliceKey) -> Option<ChainGuard> {
        let mut slices = self.slices.lock().await;
        let entry = slices.get_mut(key)?;
        if entry.state != SliceStatus::Announcing {
            return None;
        }
        entry.active_chains = entry.active_chains.saturating_add(1);
        Some(ChainGuard {
            state: self.clone(),
            key: key.clone(),
            released: false,
        })
    }

    /// Internal: called from `ChainGuard::drop`.
    async fn release_chain(&self, key: &SliceKey) -> Option<SliceStatus> {
        let mut slices = self.slices.lock().await;
        let entry = slices.get_mut(key)?;
        entry.active_chains = entry.active_chains.saturating_sub(1);
        // If we were draining and the last chain just hung up, flip to
        // Stopped so the slice can be safely de-listed on disk.
        if entry.state == SliceStatus::Draining && entry.active_chains == 0 {
            entry.state = SliceStatus::Stopped;
        }
        Some(entry.state)
    }

    /// Force every Draining slice with no active chains to Stopped.
    /// Returns the keys that flipped — caller persists the disable.
    pub async fn drain_idle(&self) -> Vec<SliceKey> {
        let mut slices = self.slices.lock().await;
        let mut flipped = Vec::new();
        for (key, entry) in slices.iter_mut() {
            if entry.state == SliceStatus::Draining && entry.active_chains == 0 {
                entry.state = SliceStatus::Stopped;
                flipped.push(key.clone());
            }
        }
        flipped
    }

    /// Force-stop any Draining slice whose grace period has elapsed,
    /// regardless of how many chains are still attached. Used by the
    /// drain-timeout watchdog (#36) so a wedged consumer can't pin a
    /// host slice forever.
    pub async fn force_stop_all_draining(&self) -> Vec<SliceKey> {
        let mut slices = self.slices.lock().await;
        let mut flipped = Vec::new();
        for (key, entry) in slices.iter_mut() {
            if entry.state == SliceStatus::Draining {
                entry.state = SliceStatus::Stopped;
                entry.active_chains = 0;
                flipped.push(key.clone());
            }
        }
        flipped
    }
}

/// RAII guard for an in-flight chain. Decrement on Drop. The guard is
/// held by the forward listener for as long as the connection is up;
/// when it drops, ControlState is updated and (if Draining hit 0) the
/// slice transitions to Stopped.
pub struct ChainGuard {
    state: Arc<ControlState>,
    key: SliceKey,
    released: bool,
}

impl ChainGuard {
    /// Explicit release — same as Drop but lets you observe the new state.
    pub async fn release(mut self) -> Option<SliceStatus> {
        self.released = true;
        self.state.release_chain(&self.key).await
    }
}

impl Drop for ChainGuard {
    fn drop(&mut self) {
        if self.released { return; }
        // Drop runs in sync context; use try_lock fast path, fall back
        // to a detached release if the lock is contended.
        let state = self.state.clone();
        let key = self.key.clone();
        tokio::spawn(async move {
            let _ = state.release_chain(&key).await;
        });
    }
}

// ----------------------------------------------------------------------
//  Server
// ----------------------------------------------------------------------

/// Default control socket path: `<data_dir>/control.sock`.
pub fn default_socket_path() -> PathBuf {
    directories::ProjectDirs::from("io", "intelnav", "intelnav")
        .map(|p| p.data_dir().join("control.sock"))
        .unwrap_or_else(|| PathBuf::from("./intelnav-control.sock"))
}

/// Spawn the control server. Returns a handle whose Drop aborts the
/// listener task and removes the socket file.
pub fn spawn_server(state: Arc<ControlState>, sock_path: PathBuf) -> Result<ControlServer> {
    if let Some(p) = sock_path.parent() {
        std::fs::create_dir_all(p)
            .with_context(|| format!("create {}", p.display()))?;
    }
    // Stale socket from a previous crash blocks bind — remove it.
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path)
        .with_context(|| format!("bind {}", sock_path.display()))?;
    set_socket_perms(&sock_path)?;

    let path_for_drop = sock_path.clone();
    let state_t = state.clone();
    let task = tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let st = state_t.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_client(st, stream).await {
                            debug!(?e, "control client closed");
                        }
                    });
                }
                Err(e) => {
                    warn!(?e, "control listener accept failed");
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }
    });
    Ok(ControlServer { _task: Some(task), socket_path: path_for_drop })
}

pub struct ControlServer {
    _task: Option<JoinHandle<()>>,
    socket_path: PathBuf,
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        if let Some(t) = self._task.take() {
            t.abort();
        }
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

#[cfg(unix)]
fn set_socket_perms(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_socket_perms(_path: &std::path::Path) -> Result<()> { Ok(()) }

async fn handle_client(state: Arc<ControlState>, stream: UnixStream) -> Result<()> {
    let (read, mut write) = stream.into_split();
    let mut lines = BufReader::new(read).lines();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() { continue; }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(req) => dispatch(&state, req).await,
            Err(e)  => Response::Error { message: format!("bad request: {e}") },
        };
        let mut bytes = serde_json::to_vec(&response)?;
        bytes.push(b'\n');
        write.write_all(&bytes).await?;
        write.flush().await?;
    }
    Ok(())
}

async fn dispatch(state: &Arc<ControlState>, req: Request) -> Response {
    match req {
        Request::Status => Response::Status(state.snapshot_status().await),
        Request::ListHosted => Response::Hosted { slices: state.snapshot_hosted().await },
        Request::Join { cid, start, end } => {
            // The caller has already written kept_ranges.json + chunks; we
            // just flip our view of state. The next announce tick picks it up.
            state.upsert_announcing((cid, start, end), String::new()).await;
            Response::Joined
        }
        Request::Leave { cid, start, end } => {
            let key = (cid, start, end);
            if state.begin_drain(&key).await {
                Response::Leaving
            } else {
                Response::Error { message: "slice not currently announcing".into() }
            }
        }
    }
}

// ----------------------------------------------------------------------
//  Client (used by the chat TUI)
// ----------------------------------------------------------------------

/// One-shot RPC: connect, write request, read one response, close.
pub async fn call(sock_path: &std::path::Path, req: Request) -> Result<Response> {
    let stream = UnixStream::connect(sock_path).await
        .with_context(|| format!("connect {}", sock_path.display()))?;
    let (read, mut write) = stream.into_split();
    let mut bytes = serde_json::to_vec(&req)?;
    bytes.push(b'\n');
    write.write_all(&bytes).await?;
    write.flush().await?;
    write.shutdown().await.ok();

    let mut lines = BufReader::new(read).lines();
    let line = lines.next_line().await?
        .ok_or_else(|| anyhow::anyhow!("daemon closed without responding"))?;
    Ok(serde_json::from_str(&line)?)
}

/// Convenience: is the daemon running and reachable right now?
pub async fn ping(sock_path: &std::path::Path) -> bool {
    matches!(call(sock_path, Request::Status).await, Ok(Response::Status(_)))
}
