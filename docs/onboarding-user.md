# Onboarding — chat without hosting

You just want to use the swarm. You don't have spare hardware to
contribute. This is what happens.

## Prereqs

- A built `intelnav` binary (`cargo build --release -p intelnav-cli`).
- A libllama tarball at `INTELNAV_LIBLLAMA_DIR/bin/`. Run
  `intelnav doctor` to confirm.
- An identity (`intelnav init` if you've never run it before).

You do **not** need to host slices, run `intelnav-node`, or download
any GGUFs.

## Chat against the swarm

```bash
intelnav
```

The TUI opens. The first lines tell you whether the swarm is
reachable:

```
swarm: peer 12D3KooW… reading the DHT (run `intelnav-node` to host).
```

That means the chat client spawned a *client-only* libp2p host.
It dials your bootstrap peers, populates a routing table, and is
ready to query the DHT — but it does **not** publish anything.
Closing this window doesn't take any slices off the network.

## Pick a model

```
/models
```

Three sources show up:

- **`local`** — GGUFs you've cached (likely none on your first run).
- **`swarm`** — what the DHT advertises right now. Rows with `· ready`
  are end-to-end serveable; partial rows are dimmed.
- **`hub`** — HuggingFace models you can install. With "+N swarm peers"
  if the same cid is also on the DHT.

Highlight a swarm row that says `· ready` and press `Enter`. The
chat client greedy-picks one provider per slice, builds a
`ChainTarget`, and hands it to the local chain driver. Your front-half
runs in process; the rest streams through the swarm.

## Type a prompt

```
> hi, what can you do?
```

Tokens stream back through the chain. Latency depends on the slowest
hop, not the slowest peer — you'll feel ~80–120 ms per token on a
LAN-ish chain, more on WAN.

## What to set in config

Bare minimum for a chat-only setup:

```toml
# ~/.config/intelnav/config.toml
mode = "network"
default_model = "qwen2.5-7b-instruct-q4"
bootstrap = [
  "/dns4/seed1.intelnav.io/tcp/4001/p2p/12D3KooW…",
  "/dns4/seed2.intelnav.io/tcp/4001/p2p/12D3KooW…",
]
```

`bootstrap` is the seed list — a small set of long-lived peers the
swarm has agreed on. Without it your routing table starts empty and
the first `/models` open returns nothing.

## When to switch to hosting

If you find yourself running `intelnav` more than a couple times a
day, the marginal cost of also running `intelnav-node` is tiny — see
[onboarding-host.md](onboarding-host.md). The swarm is healthier
with more hosts and your latency improves when slices live near you.
