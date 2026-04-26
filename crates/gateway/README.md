# intelnav-gateway

OpenAI-compatible HTTP surface (paper §10). Axum-based. Serves as the
local entry point for any existing chat app, IDE plugin, or script.

| Method | Path                   | Purpose                               |
| ------ | ---------------------- | ------------------------------------- |
| POST   | `/v1/chat/completions` | Streaming + non-streaming chat.       |
| GET    | `/v1/models`           | Union of upstream + P2P-discovered.   |
| GET    | `/v1/network/peers`    | Every known peer across directories.  |
| GET    | `/v1/network/health`   | Gateway liveness + counts.            |
| GET    | `/`                    | Friendly banner.                      |

The `intelnav` request extension (paper §10) — `tier`, `allow_wan`,
`quorum`, `min_reputation`, `speculative` — is parsed and surfaced to
the route planner. `allow_wan` is enforced today (returns
`NoViableRoute` when T3 is requested without opt-in); the rest are
honored best-effort and logged until the relevant M2/M3 machinery
lands.

Currently proxies chat to the configured upstream (`INTELNAV_UPSTREAM_URL`).
Per-peer pipeline routing inside the gateway comes online once M2 ships.

`#![forbid(unsafe_code)]`.

Run: `intelnav gateway` (or `cargo run -p intelnav-cli --release -- gateway`).
