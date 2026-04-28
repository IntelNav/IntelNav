# Onboarding — first run

You just downloaded `intelnav`. This is what running it for the first
time looks like, and how to choose between **hosting a slice** and
**relay-only mode**.

## What's required

- The `intelnav` and `intelnav-node` binaries on your PATH. Either
  - the `curl | sh` installer:

    ```bash
    curl -fsSL https://intelnav.net/install.sh | sh
    ```

  - or build from source:

    ```bash
    bash scripts/provision.sh
    cargo build --release -p intelnav-cli -p intelnav-node
    ```

- A libllama tarball. The installer fetches one for your platform +
  GPU vendor automatically. From source: `bash scripts/install-libllama.sh`.

That's it. `intelnav doctor` runs every preflight and tells you what's
missing if anything is.

## What you do **not** need to do

- Edit `~/.config/intelnav/config.toml` — `intelnav` writes it on
  first run.
- Run `intelnav init` — first run does this for you.
- Set `INTELNAV_LIBLLAMA_DIR` — auto-discovered from
  `~/.cache/intelnav/libllama/bin/`.
- Set `HSA_OVERRIDE_GFX_VERSION` for an AMD card — auto-set when
  needed by the GPU probe.
- Type `systemctl` — `/service install` does it from inside the TUI.

## First launch

```bash
intelnav
```

The TUI:

1. Writes a default `config.toml` with auto-picked free ports.
2. Generates `~/.local/share/intelnav/peer.key` (your Ed25519
   identity, mode 0600).
3. Auto-discovers libllama at `~/.cache/intelnav/libllama/bin/`.
4. Fetches the bootstrap seed list from the project's GitHub
   release, caches it locally for 7 days.
5. Probes your hardware and shows the **contribution gate**:

```
IntelNav requires every peer to contribute.

Your hardware is plenty for hosting. Please host a slice.
The network only works because capable peers commit their hardware.

  Suggested:  Qwen 2.5 · 7B · Instruct  layers [0..7)
```

The exact wording depends on your hardware tier — capable cards see
"Please host a slice" with no relay-only paragraph; modest cards see
both options; hardware below the catalog floor sees relay-only as
the recommended path.

Chat doesn't unlock until you've picked one of the two.

## Path A — host a slice

```bash
INTELNAV_RELAY_ONLY=1 intelnav   # one-time, just to reach the TUI
```

Inside the TUI:

```
/models      open the three-source picker
             highlight a row, press `c` to start chunking
```

After "splitting Qwen 2.5 · … → shards in …" appears, the chunker
wrote a sidecar to `<models_dir>/.shards/<cid>/`. Now make it
permanent:

```
/service install
```

`pkexec` pops once, asks for your password, runs
`loginctl enable-linger <user>`. After that `intelnav-node` runs
forever — even across reboots — and you can drop the
`INTELNAV_RELAY_ONLY=1` env var.

Verify:

```
/service status        # should report Active
/hosting               # the slice you just contributed shows up
```

## Path B — relay only

If your hardware can't host a slice:

```bash
INTELNAV_RELAY_ONLY=1 intelnav
```

…or, to make it permanent, add `relay_only = true` to
`~/.config/intelnav/config.toml`. The daemon still participates in
the Kademlia DHT (which the network needs for peer discovery), it
just doesn't run inference layers.

## Chat against the swarm

Once gated through:

```
/models                   # three-source picker
                          # highlight a `swarm · ready` row → Enter
hi, what can you do?
```

Tokens stream back through the chain. If a hop goes down mid-turn
the chain driver swaps in the next-best provider for that hop
without dropping your stream — you'll see one short
`[swarm] hop 2 unreachable, swapping to backup` line in the
transcript.

## Managing your hosting

```
/hosting                              what slices you host + active chain count
/leave                                pick a slice to drain (numbered list)
/leave <n>                            drain row n from the last listing
/leave <cid> <start> <end>            drain by full coords (power user)
```

A drain transitions `Announcing → Draining → Stopped`. While
Draining, the daemon stops re-publishing the provider record (so
consumers stop picking you for new chains) and refuses new forward
sessions. Existing chains keep streaming until they finish. After
5 min of Draining with chains still attached, the daemon
force-stops; this is the safety valve against a wedged consumer.

The chunks stay on disk, so re-joining the same slice later costs
zero bandwidth.

## Keybindings

`/keybindings` inside the TUI prints the canonical list. The ones
worth remembering:

- `Esc Esc` — clear input (double-tap within 600 ms)
- `\` + `Enter` — newline (alongside `Shift+Enter`)
- `Ctrl+G` — edit current input in `$EDITOR`
- `Ctrl+Shift+_` — undo last input edit
- `Alt+P` — cycle to next cached model
- `Ctrl+L` — clear transcript
- `Ctrl+C` — cancel stream / clear input. Twice within 1.5s = quit.

## When something doesn't work

- **Bootstrap fetch failed.** Logged at startup. Cached manifest is
  used as fallback. mDNS still finds peers on the same LAN.
- **`daemon not reachable` from `/hosting`.** `intelnav-node` isn't
  running. Run `/service install` (or `intelnav-node` in another
  terminal for ad-hoc).
- **`intelnav doctor` reports missing libllama.** Run
  `bash scripts/install-libllama.sh` (or pass
  `INTELNAV_LIBLLAMA_DIR=/path/to/bin` if you already have one).
- **ROCm error: invalid device function.** Your card's `gfx` arch
  isn't covered by the libllama tarball. The runtime auto-sets
  `HSA_OVERRIDE_GFX_VERSION` for known RDNA2/3/3.5 cards; for
  anything more exotic, set it manually to the nearest neighbor
  arch. Open an issue with the output of `rocminfo`.
- **Logs.** Everything goes to `~/.local/state/intelnav/intelnav.log`
  while the TUI is up. `tail -f` it in another terminal.
