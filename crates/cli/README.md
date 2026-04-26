# intelnav-cli

The user-facing `intelnav` binary. Ratatui-based chat TUI plus a set of
operator subcommands.

### Subcommands

```
intelnav                 # default: chat (interactive TUI)
intelnav chat            # explicit
intelnav ask [prompt]    # non-interactive one-shot (stdin if no arg)
intelnav gateway         # run the local OpenAI-compatible gateway
intelnav node            # bridge the Rust side to the Python shard
intelnav models          # list models reachable (local + network)
intelnav peers           # list peers across directories
intelnav health          # gateway + upstream + network snapshot
intelnav doctor          # preflight checks (gateway, identity, mDNS)
intelnav init            # write default config + generate peer identity
```

### Config + env

Config lives at `$XDG_CONFIG_HOME/intelnav/config.toml`. Every field
is overridable via `INTELNAV_*` env vars; see
[`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md#5-configuration-surface)
for the full list. Key ones:

- `INTELNAV_MODE` — `auto | local | network`.
- `INTELNAV_MODELS_DIR` — where to scan for local GGUFs.
- `INTELNAV_PEERS` + `INTELNAV_SPLITS` — ad-hoc pipeline chain.
- `INTELNAV_DRAFT_MODEL` + `INTELNAV_SPEC_K` — speculative decoding.
- `INTELNAV_WIRE_DTYPE` — `fp16 | int8`.

### In-REPL slash commands

- `/models` — browser fusing local + network + Hugging Face catalog.
- `/peers host:port,... splits,...` — route this session through a chain.
- `/draft <path> [k]` — enable speculative decoding.
- `/wire fp16|int8` — switch activation dtype.
- `/doctor` — preflight inline.
- `Shift+Home` / `Shift+End` — jump to top / re-engage tail follow.

### TUI plan

The TUI is still maturing; the design study + open bugs live in
[`docs/dev/PROGRESS_TUI.md`](../../docs/dev/PROGRESS_TUI.md).

### Logging

When the TUI is active, tracing output + raw FD 2 are redirected to
`$XDG_STATE_HOME/intelnav/intelnav.log` — native deps can't paint over
the Ratatui canvas. `-v` = debug, `-vv` = trace.

`#![deny(unsafe_code)]` except for the `libc::dup2` stderr redirect,
which is inline-`#[allow(unsafe_code)]` with justification.
