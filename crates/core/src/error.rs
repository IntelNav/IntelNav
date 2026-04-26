//! Error taxonomy. `NoViableRoute` is a first-class citizen — paper §5.4.

use thiserror::Error;

use crate::ids::ModelId;
use crate::tier::LatencyTier;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no viable route for model {model} at tier {tier:?}: {reason}")]
    NoViableRoute {
        model:  ModelId,
        tier:   LatencyTier,
        reason: String,
    },

    #[error("insufficient capacity: missing layers {missing:?}")]
    InsufficientCapacity { missing: (u16, u16) },

    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("http: {0}")]
    Http(String),

    #[error("serde: {0}")]
    Serde(String),

    #[error("parse: {0}")]
    Parse(String),

    #[error("config: {0}")]
    Config(String),

    #[error("gateway refused: {0}")]
    GatewayRefused(String),

    #[error("quorum disagreement: {disagreeing} of {total} chains disagreed")]
    QuorumFailed { disagreeing: usize, total: usize },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e.to_string())
    }
}
