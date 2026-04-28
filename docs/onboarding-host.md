# Onboarding — host a slice

You want to commit hardware to the swarm. This is what happens.

The user-facing flow is `intelnav` (TUI) → `/models` → press `c` → 
`/service install`. There's no `systemctl` step, no `pipe_peer` to
launch separately, no chunk-server to spin up — it's all one
daemon (`intelnav-node`).

## Prereqs

- A built `intelnav` and `intelnav-node`
  (`cargo build --release -p intelnav-cli -p intelnav-node`).
- A libllama tarball + companion ggml backends at
  `INTELNAV_LIBLLAMA_DIR/bin/`. `intelnav doctor` verifies.

## Pick what to host

```bash
intelnav        # opens the TUI
/models         # opens the picker
```

Three row kinds in the picker:

- **Hub row** (e.g. *Qwen 2.5 · 7B · Instruct*). Pressing `c`
  downloads the full GGUF, runs the chunker, drops a
  `kept_ranges.json` listing the catalog's first standard split.
  Use this when you have the disk + bandwidth for a fresh model.
- **Swarm row** (a model another peer is already serving).
  Pressing `c` here pulls *just* the chunks for one range — no
  full GGUF download. Use this when you just want to fill a
  coverage gap or join a popular model.
- **Local row** (a `.gguf` you already had). Pressing `c` chunks
  it in place.

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

The daemon now runs forever, including across reboots. You can
close `intelnav` — your slices keep being announced.

`/service status` reports `Active` once everything is up.

## What the daemon does

- Spawns the libp2p swarm on `libp2p_listen` (defaults to a free
  TCP port on `0.0.0.0`).
- Re-announces every kept (cid, range) to the DHT every 5 min.
- Hosts a multi-shard chunk HTTP server on `chunks_addr`
  (auto-picked port on first run).
- Hosts the inference forward TCP listener on `forward_addr`
  (auto-picked port). Lazy-loads each slice's GGUF on first
  inbound chain. Stitches subsets from chunks if the full GGUF
  isn't on disk.
- Listens for control RPCs on `~/.local/share/intelnav/control.sock`
  so the chat client can drive Join/Leave/Status without an IPC
  framework.

You can configure ports manually in `~/.config/intelnav/config.toml`
if you have NAT/router preferences:

```toml
chunks_addr  = "0.0.0.0:8765"
forward_addr = "0.0.0.0:7717"
```

## Joining additional slices

Run `intelnav` again, `/models`, press `c` on another row. The
contribute flow writes a new `kept_ranges.json` sidecar and tells
the daemon (via control RPC) to add it to the announce loop. No
restart.

## Leaving a slice

```
/hosting                          # see what you host + active chains
/leave <cid> <start> <end>        # graceful drain
```

Drain protocol:

1. **Announcing → Draining**: the daemon stops re-publishing your
   provider record, so consumers don't pick you for new chains.
2. The forward listener refuses new sessions for that slice with
   a clean abort message; consumers fail over to their alternate.
3. In-flight chains keep streaming until they finish.
4. **Draining → Stopped** when active_chains hits 0, or after a
   5-minute grace timeout (force-stop) if a chain is wedged.
5. The kept_ranges entry is added to `disabled_ranges.json` next
   to it, so a daemon restart honours the leave.
6. Chunks stay on disk — re-joining is instant, no re-download.

## Uninstall

```
/service uninstall
```

Stops the unit, removes the file, leaves your `~/.local/share/intelnav/`
data + identity untouched. To wipe everything:

```bash
rm -rf ~/.local/share/intelnav ~/.config/intelnav
```
