//! Latency tiers — paper §5.4 table.
//!
//! The gateway's chain builder refuses to emit a route that violates the
//! selected tier's overhead budget; see [`LatencyTier::decode_budget`].

use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LatencyTier {
    /// T1 — LAN. Typical RTT < 5 ms / hop. Preferred route.
    Lan,
    /// T2 — same-continent. Typical RTT < 80 ms / hop. Fallback.
    Continent,
    /// T3 — cross-continent. Typical RTT ≥ 80 ms / hop. Opt-in only.
    Wan,
}

impl LatencyTier {
    pub fn typical_rtt(self) -> Duration {
        match self {
            Self::Lan       => Duration::from_millis(5),
            Self::Continent => Duration::from_millis(80),
            Self::Wan       => Duration::from_millis(150),
        }
    }
    pub fn decode_budget(self) -> Option<Duration> {
        match self {
            Self::Lan       => Some(Duration::from_millis(20)),
            Self::Continent => Some(Duration::from_millis(100)),
            Self::Wan       => None, // no hard budget
        }
    }
    pub fn display(self) -> &'static str {
        match self {
            Self::Lan       => "T1 LAN",
            Self::Continent => "T2 CONT",
            Self::Wan       => "T3 WAN",
        }
    }
    /// Classify a measured per-hop RTT into a tier.
    pub fn classify(rtt: Duration) -> Self {
        let ms = rtt.as_millis();
        if ms < 5 {
            Self::Lan
        } else if ms < 80 {
            Self::Continent
        } else {
            Self::Wan
        }
    }
}
