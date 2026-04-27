# Onboarding — host a slice

You want to contribute hardware to the swarm. This is what happens.

## Prereqs

- A built `intelnav-node` binary (`cargo build --release -p intelnav-node`).
- A built `intelnav` binary, used once to pick the slice you want to host.
- A libllama tarball + companion ggml backends, pointed at by
  `INTELNAV_LIBLLAMA_DIR`. Run `intelnav doctor` to verify.
- A peer identity (`intelnav init` writes one to `~/.local/share/intelnav/peer.key`).

## Pick what to host

```bash
intelnav        # opens the TUI
/models         # opens the picker
```

Highlight a row. Two interesting kinds:

- **Hub row** (e.g. *Qwen 2.5 · 7B · Instruct*). Pressing `c` here
  downloads the full GGUF, runs the chunker, drops a
  `kept_ranges.json` listing the catalog's first standard split.
  Use this when you have the disk + bandwidth for the full model.

- **Swarm row** (e.g. a model another peer is already serving).
  Pressing `c` here pulls *just* the chunks for one range — no full
  GGUF download. Use this when you just want to fill a coverage
  gap or join a model that's already popular.

After either path completes, `<models_dir>/.shards/<cid>/` exists
with `manifest.json` + `chunks/` + `kept_ranges.json`.

## Run the daemon

```bash
intelnav-node
```

Logs to stderr. Look for:

```
INFO libp2p node up peer_id=… listen_addrs=…
INFO DHT announces published n=1
INFO intelnav-node ready — Ctrl+C to stop
```

That's it. Every 5 minutes the daemon republishes your provider
records so the Kademlia TTL doesn't expire you out.

## Set up `chunks_addr` and `forward_addr`

For other peers to actually use your slice, they need to dial
something. Two endpoints:

- **`chunks_addr`** — the `host:port` of your chunk-server, run
  separately as `intelnav-chunk serve --bind <host:port>`. Lets
  other peers pull your bundles on the *swarm pre-split* path.
- **`forward_addr`** — the `host:port` of your inference TCP
  listener (`pipe_peer`). Lets other peers include you in a
  Network-mode chain.

Put both in your config (or env: `INTELNAV_CHUNKS_ADDR`,
`INTELNAV_FORWARD_ADDR`) before running the daemon. Without them
your provider records still publish but nobody can route work to
you.

## Run as a service

The daemon is intentionally a plain blocking process so any
service manager will do. A minimal systemd user unit:

```ini
[Unit]
Description=IntelNav swarm node
After=network-online.target

[Service]
Type=simple
ExecStart=%h/.cargo/bin/intelnav-node
Restart=on-failure
Environment=INTELNAV_LIBLLAMA_DIR=%h/.cache/intelnav/libllama/bin
Environment=INTELNAV_CHUNKS_ADDR=0.0.0.0:8765
Environment=INTELNAV_FORWARD_ADDR=0.0.0.0:7717

[Install]
WantedBy=default.target
```

`systemctl --user enable --now intelnav-node` and you're seeding.
