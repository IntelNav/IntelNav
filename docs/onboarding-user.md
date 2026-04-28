# Onboarding — first run

You just downloaded `intelnav`. You don't know any of the file
paths or commands. This is what running it for the first time
looks like, and how to choose between **hosting a slice** and
**relay-only mode**.

## What's required

- A built `intelnav` + `intelnav-node` binary
  (`cargo build --release -p intelnav-cli -p intelnav-node`).
- A libllama tarball at `INTELNAV_LIBLLAMA_DIR/bin/`.
  `intelnav doctor` will tell you if it's missing.

You do **not** need to:

- Edit `~/.config/intelnav/config.toml` — the TUI generates it.
- Run `intelnav init` — the TUI runs it for you.
- Type `systemctl` — `/service install` from the TUI does it.
- Run `intelnav-chunk` or `pipe_peer` — they're folded into
  `intelnav-node`.

## First launch

```bash
intelnav
```

The TUI:

1. Writes a default `config.toml` with auto-picked free ports.
2. Generates `~/.local/share/intelnav/peer.key` (your peer
   identity, 0600).
3. Fetches the bootstrap seed list from the project's GitHub
   release, caches it locally.
4. Probes your hardware and shows the **contribution gate**:

```
IntelNav requires every peer to contribute.

  1. Host a slice — recommended for your hardware:
       Qwen 2.5 · 3B · Instruct  layers [0..9)  (comfortable)
  2. Relay only — daemon participates in the DHT but runs no
       inference.
```

You pick one. Chat doesn't unlock until you do.

## Path A — host a slice

Inside the TUI:

```
/models             # opens the three-source picker
                    # highlight a row, press `c` to contribute
```

After the contribute flow finishes (download + split for hub rows,
or swarm pull for swarm rows), the TUI prompts to install the
daemon as a service:

```
intelnav-node is not yet a system service.
Install with /service install (one pkexec prompt).
```

Run `/service install`. `pkexec` pops once, asks for your password,
and runs `loginctl enable-linger <user>`. After that the daemon
runs forever, even across reboots — no further sudo.

You can verify with `/service status` and inspect what you're
hosting with `/hosting`.

## Path B — relay only

If your hardware can't host a slice (or you don't want to), set:

```bash
INTELNAV_RELAY_ONLY=1 intelnav
```

…or add `relay_only = true` to `~/.config/intelnav/config.toml`.

The daemon still participates in the Kademlia DHT, so you're
contributing routing, just not inference. Chat is unlocked.

## Chat against the swarm

Once gated through:

```
/models                   # three-source picker
                          # highlight a `swarm · ready` row → Enter
> hi, what can you do?
```

Tokens stream back through the chain. If a hop goes down mid-turn
the chain driver swaps in the next-best provider for that hop
without dropping your stream — you'll see one short
`[swarm] hop 2 unreachable, swapping to backup` line in the
transcript.

## Managing your hosting

```
/hosting                          # list slices, active chains, state
/leave <cid> <start> <end>        # graceful drain
                                  # in-flight chains finish; new ones go elsewhere
```

A drain transitions through `Announcing → Draining → Stopped`.
While Draining, the daemon stops re-publishing the provider
record (so consumers don't pick you for new chains) and refuses
new forward connections. Existing chains stream until they finish.
After 5 min of Draining the daemon force-stops, in case a chain
is wedged.

The chunks stay on disk so re-joining the same slice later costs
zero bandwidth.

## When something doesn't work

- `swarm: offline` in the status bar → bootstrap fetch failed or
  no peers reachable. The cached manifest is used as a fallback;
  `/service status` will show whether the daemon is live.
- `daemon not reachable` from `/hosting` → daemon isn't running.
  Run `/service install` (or just `intelnav-node` in another
  terminal for ad-hoc).
- `intelnav doctor` shows missing libllama → unpack the tarball
  and set `INTELNAV_LIBLLAMA_DIR=<dir>/bin`.
