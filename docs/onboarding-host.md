# Onboarding — host a slice

You want to commit hardware to the swarm. This is what happens.

The user-facing flow is `intelnav` (TUI) → `/models` → press `c` →
`/service install`. There's no `systemctl` step, no separate forward
or chunk listener to launch — it's all one daemon (`intelnav-node`).

## Prereqs

- `intelnav` and `intelnav-node` on your PATH. Either the `curl | sh`
  installer or `cargo build --release -p intelnav-cli -p intelnav-node`.
- A libllama tarball auto-discovered at
  `~/.cache/intelnav/libllama/bin/`. The installer fetches one for
  your platform; from source, run `bash scripts/install-libllama.sh`.
- `intelnav doctor` verifies both.

## Pick what to host

```bash
intelnav        # opens the TUI
/models         # opens the picker
```

Three row kinds:

- **Hub row** (e.g. *Qwen 2.5 · 7B · Instruct*). Pressing `c` downloads
  the full GGUF, runs the chunker, drops a `kept_ranges.json` listing
  the catalog's first standard split. Use this when you have the
  disk + bandwidth for a fresh model.
- **Swarm row** (a model another peer is already serving). Pressing
  `c` pulls *just* the chunks for one range — no full GGUF download.
  Use this when you just want to fill a coverage gap or join a
  popular model.
- **Local row** (a `.gguf` you already had — only for catalog-known
  models). Pressing `c` chunks it in place.

After the contribute flow completes, `<models_dir>/.shards/<cid>/`
exists with `manifest.json` + `chunks/` + `kept_ranges.json`.

## Install the service

```
/service install
```

`pkexec` pops once for `loginctl enable-linger <user>`. Everything
else runs as you, no further root. The TUI prints:

```
service: installed and started.
```

The install verifies the unit reached `Active` before reporting
success — if the daemon failed to start, the TUI surfaces the
journal tail in the message instead of pretending it worked.

The daemon now runs forever, including across reboots. You can close
`intelnav` — your slices keep being announced.

`/service status` reports `Active` once everything is up.

## What the daemon does

- Spawns the libp2p swarm on `libp2p_listen` (defaults to a free TCP
  port on `0.0.0.0`).
- Re-announces every kept `(cid, range)` to the DHT every 5 min.
- Hosts a multi-shard chunk HTTP server on `chunks_addr`
  (auto-picked port on first run).
- Hosts the inference forward TCP listener on `forward_addr`
  (auto-picked port). Lazy-loads each slice's GGUF on first inbound
  chain. Stitches subsets from chunks if the full GGUF isn't on
  disk.
- Listens for control RPCs on `~/.local/share/intelnav/control.sock`
  so the chat client can drive Join/Leave/Status without an IPC
  framework.
- Honours SIGINT and SIGTERM with a graceful drain: every Announcing
  slice flips to Draining, in-flight chains keep streaming for up to
  30 s, then the process exits cleanly. Lets a logout or reboot
  finish serving instead of dropping streams.

You can configure ports manually in `~/.config/intelnav/config.toml`
if you have NAT/router preferences:

```toml
chunks_addr  = "0.0.0.0:8765"
forward_addr = "0.0.0.0:7717"
```

## Joining additional slices

Run `intelnav` again, `/models`, press `c` on another row. The
contribute flow writes a new `kept_ranges.json` sidecar and the
daemon picks it up on the next announce tick (~5 min, or trigger a
restart with `systemctl --user restart intelnav-node` for instant
pickup).

## Leaving a slice

```
/hosting                              what you host + active chains
/leave                                pick a slice (numbered list)
/leave <n>                            drain row n from the listing
/leave <cid> <start> <end>            drain by full coords (power user)
```

Drain protocol:

1. **Announcing → Draining**: the daemon stops re-publishing your
   provider record, so consumers don't pick you for new chains.
2. The forward listener refuses new sessions for that slice with a
   clean abort message; consumers fail over to their alternate.
3. In-flight chains keep streaming until they finish.
4. **Draining → Stopped** when `active_chains` hits 0, or after a
   5-minute grace timeout (force-stop) if a chain is wedged.
5. The kept_ranges entry is added to `disabled_ranges.json` next to
   the manifest, so a daemon restart honours the leave.
6. Chunks stay on disk — re-joining is instant, no re-download.

## Uninstall

```
/service uninstall
```

Stops the unit, removes the file, leaves your
`~/.local/share/intelnav/` data + identity untouched. To wipe
everything:

```bash
rm -rf ~/.local/share/intelnav ~/.config/intelnav ~/.cache/intelnav
```

That nukes your peer key (you'll get a new peer ID next launch).
The daemon's identity is content-addressed for the DHT, so other
peers will see you as a fresh peer — no name collision risk.
