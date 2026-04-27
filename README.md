# IntelNav

**Decentralized, pipeline-parallel LLM inference.**

IntelNav splits a model into layer-range slices, scatters them across
volunteer hardware, and streams hidden states through the chain to
answer a prompt. No single peer holds the whole model. Slices are
addressed on a Kademlia DHT, prompts are encrypted end-to-end, and
the only thing a contributor commits to is the slice they have RAM
for.

```
   prompt ──► [you: layers 0..k) ──► peer A: [k..m) ──► peer B: [m..N) ──► tokens
```

## Two binaries

| binary          | what it does                                                         |
| --------------- | -------------------------------------------------------------------- |
| `intelnav`      | Chat client. Reads the DHT to find models the swarm can serve. No host duties. |
| `intelnav-node` | Host daemon. Holds slices, serves chunks, accepts inference forwards. Long-running. |

You run `intelnav` whenever you want to chat. You run `intelnav-node`
in the background (systemd unit, screen session, whatever) for as long
as you want to contribute. They share the same identity (`~/.local/share/intelnav/peer.key`)
and the same `models_dir`, so they cooperate without IPC.

## Quickstart — chat against the swarm

```bash
bash scripts/provision.sh                # system deps + rust
cargo build --release -p intelnav-cli    # build the chat binary
./target/release/intelnav                # opens the TUI
```

Inside the TUI: `/models` lists three things — what you have cached,
what the swarm is serving, and what you can pull from HuggingFace.
`Enter` runs / joins. `c` (on a hub or swarm row) starts a contribute
flow that hands off to your `intelnav-node`.

## Quickstart — host a slice

```bash
cargo build --release -p intelnav-node
./target/release/intelnav-node           # runs forever
```

The node scans `<models_dir>/.shards/*/kept_ranges.json` to learn
which slices it owns, dials the bootstrap peers from your config,
publishes provider records to the DHT every 5 minutes, and accepts
inbound chunk and forward connections.

## Layout

```
intelnav/
├── crates/
│   ├── core/             shared types, config, errors
│   ├── wire/             CBOR codecs for the protocol
│   ├── crypto/           Ed25519, X25519, AES-256-GCM
│   ├── ggml/             libllama loader + GPU probe
│   ├── runtime/          layer-range inference (ggml-backed)
│   ├── model-store/      GGUF chunker, stitcher, fetcher, optional serve
│   ├── net/              libp2p + Kademlia DHT shard index, mDNS, registry
│   ├── app/              substantive code: TUI, drivers, contribute paths
│   ├── cli/              `intelnav` — chat client (thin binary over `app`)
│   ├── node/             `intelnav-node` — host daemon (thin binary over `app`)
│   └── registry/         optional bootstrap coordinator
├── docs/
│   ├── architecture.md     workspace + protocol overview
│   ├── onboarding-host.md  how to host slices
│   └── onboarding-user.md  how to chat without hosting
└── specs/                wire protocol + registry specs
```

## License

Apache-2.0.
