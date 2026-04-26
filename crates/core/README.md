# intelnav-core

Shared, shape-only types that every other crate depends on. No network
or crypto primitives — those layer on top.

Exports:

- `PeerId`, `SessionId`, `ModelId` — identity newtypes.
- `LayerRange`, `Quant`, `Backend`, `Role`, `CapabilityV1`, `ShardRoute`.
- `LatencyTier` — `Lan | Continent | Wan` (paper §6.5).
- `Config`, `RunMode` — the full runtime config surface; loads from
  `$XDG_CONFIG_HOME/intelnav/config.toml` and overlays `INTELNAV_*`
  env vars.
- `Error`, `Result` — canonical error type for the workspace.

`#![forbid(unsafe_code)]`.
