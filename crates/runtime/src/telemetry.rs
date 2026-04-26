//! In-process pub/sub of chain step events for live demos.
//!
//! Scope: when a `Chain` is driving inference, every per-peer step
//! produces a [`StepEvent`] — start, encode/decode/transport timings,
//! bytes on the wire. A [`Telemetry`] handle wraps a
//! `tokio::sync::broadcast` so the gateway (or a CLI watcher) can
//! subscribe at any time and see events without back-pressuring the
//! producer.
//!
//! Out of scope (today): persistence, sampling, multi-process. The
//! channel is in-process only — when the gateway drives its own
//! chain (arc 6 sub-D), publisher and subscriber live in the same
//! axum process; CLI-driven chains expose the data through their own
//! gateway in the same way.
//!
//! When no chain is running, the gateway can still synthesize a
//! heartbeat event from time to time so the SPA's "live" badge has
//! something to react to. The `synthetic: true` flag on the event
//! lets the UI dim those values relative to real measurements.

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Pipeline phase the event refers to. Mirrors `intelnav_wire::Phase`
/// but kept independent so the runtime crate doesn't have to spell
/// the wire crate in every `match`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepPhase {
    /// First forward of a turn (full prompt feeds in).
    Prefill,
    /// Subsequent token-by-token decode.
    Decode,
    /// Pseudo-event: gateway is alive but no chain is running.
    Heartbeat,
}

/// One event from the chain. Always relates to a single peer hop in
/// a single chain step. The gateway broadcasts these to SSE
/// subscribers; the SPA renders them as edge animations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepEvent {
    /// Sequence number — monotonically increasing per-Telemetry.
    pub seq:           u64,
    /// Unix epoch milliseconds.
    pub at_ms:         u64,
    /// Which peer in the chain this event refers to (0-indexed in the
    /// peer list as the gateway sees it). `peer_id` is the bs58-short
    /// rendered form so the SPA can match against
    /// `/v1/swarm/topology` ids without an extra lookup.
    pub peer_index:    usize,
    pub peer_id:       String,
    pub phase:         StepPhase,
    /// Total round-trip ms for this hop (encode + send + remote work
    /// + recv + decode). 0 for heartbeats.
    pub rtt_ms:        f32,
    /// Bytes uploaded to the peer for this step.
    pub bytes_up:      u64,
    /// Bytes received from the peer for this step.
    pub bytes_down:    u64,
    /// True when the value above came from a synth source rather than
    /// a real measurement. Lets the SPA mark synth data visibly.
    pub synthetic:     bool,
}

/// Cheaply-clonable producer/subscribe handle. Cloning shares the
/// underlying broadcast channel, so all emitters fan out to the same
/// subscribers. A subscriber that lags past the channel capacity sees
/// `RecvError::Lagged` and can resubscribe; we lean on that rather
/// than buffering history because the SPA only cares about live state.
#[derive(Clone)]
pub struct Telemetry {
    tx:  broadcast::Sender<StepEvent>,
    /// Shared monotonic counter — every emit takes the next value.
    seq: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Telemetry {
    /// Create a new channel with the given capacity. The capacity
    /// determines how far a slow subscriber can lag before missing
    /// events; 256 covers about 5 seconds of decode at 50 tok/s with
    /// one event per peer per token.
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity.max(16));
        Self {
            tx,
            seq: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Subscribe — every subscriber sees every event from this point
    /// forward until they drop the receiver or lag too far.
    pub fn subscribe(&self) -> broadcast::Receiver<StepEvent> {
        self.tx.subscribe()
    }

    /// True if any subscriber would receive the event. Producers can
    /// use this to skip building an event payload when no one is
    /// listening — keeps the chain hot path free of allocator churn
    /// when the demo isn't being watched.
    pub fn has_subscribers(&self) -> bool {
        self.tx.receiver_count() > 0
    }

    /// Number of currently-attached subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Publish an event. Returns the assigned `seq` so callers can
    /// reference it in logs without rereading the value back from a
    /// receiver. Drops silently if no subscribers — the broadcast
    /// channel handles fan-out for us.
    pub fn emit(&self, mut ev: StepEvent) -> u64 {
        let seq = self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ev.seq = seq;
        if ev.at_ms == 0 {
            ev.at_ms = unix_ms();
        }
        let _ = self.tx.send(ev);
        seq
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new(256)
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn emit_assigns_monotonic_seq() {
        let t = Telemetry::new(16);
        let mut rx = t.subscribe();
        for i in 0..5 {
            t.emit(StepEvent {
                seq: 999, at_ms: 0,           // overwritten
                peer_index: i, peer_id: format!("p{i}"),
                phase: StepPhase::Decode,
                rtt_ms: 1.0, bytes_up: 0, bytes_down: 0,
                synthetic: false,
            });
        }
        for expected in 0..5 {
            let ev = tokio::time::timeout(Duration::from_millis(100), rx.recv())
                .await.unwrap().unwrap();
            assert_eq!(ev.seq, expected);
        }
    }

    #[tokio::test]
    async fn no_subscribers_is_not_an_error() {
        let t = Telemetry::new(16);
        // No subscribers — emit must not panic, must increment seq.
        let s0 = t.emit(StepEvent {
            seq: 0, at_ms: 0,
            peer_index: 0, peer_id: "p".into(),
            phase: StepPhase::Heartbeat,
            rtt_ms: 0.0, bytes_up: 0, bytes_down: 0,
            synthetic: true,
        });
        assert_eq!(s0, 0);
        assert!(!t.has_subscribers());
    }

    #[tokio::test]
    async fn fanout_reaches_every_subscriber() {
        let t = Telemetry::new(16);
        let mut a = t.subscribe();
        let mut b = t.subscribe();
        t.emit(StepEvent {
            seq: 0, at_ms: 0,
            peer_index: 0, peer_id: "p".into(),
            phase: StepPhase::Prefill,
            rtt_ms: 1.0, bytes_up: 100, bytes_down: 50,
            synthetic: false,
        });
        let ea = a.recv().await.unwrap();
        let eb = b.recv().await.unwrap();
        assert_eq!(ea.seq, eb.seq);
        assert_eq!(ea.peer_id, "p");
        assert_eq!(eb.bytes_up, 100);
    }
}
