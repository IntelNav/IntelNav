# IntelNav, in plain English

## What it is

IntelNav is a way for people with ordinary computers to run big AI
language models together — without anyone needing the expensive
hardware it normally takes. You connect your computer to the network,
and your machine contributes a slice of its memory and compute to
running a model that's too large for any single participant to host
alone.

Think of it like carpooling, but for AI: nobody has to own a whole
bus, everyone contributes a seat.

## The problem

Large language models like GPT-4, Claude, or DeepSeek-Coder 33B need
tens of gigabytes of fast memory to run. Most people's laptops and
gaming PCs have 8–16 GB of GPU memory — not enough. So they pay a
cloud provider (OpenAI, Anthropic, etc.), which:

- Costs money every time you send a prompt.
- Sends your conversation through someone else's servers.
- Doesn't work offline or on a local network.
- Concentrates the ability to run powerful AI in a few companies.

Meanwhile, there are millions of idle gaming GPUs sitting around the
world doing nothing most of the time.

## What IntelNav does

IntelNav chops a big model into pieces and spreads the pieces across
several computers. When you send a prompt:

1. Your prompt enters through a local **gateway** on your machine
   (it looks just like an OpenAI-compatible API, so existing tools
   work unchanged).
2. The gateway finds a chain of peers on the network, each of whom
   is holding a different slice of the model.
3. The prompt flows through the chain: peer A runs the first third
   of the model's layers, hands the intermediate result to peer B,
   who runs the middle third, who hands it to peer C, who runs the
   final third and produces the answer.
4. The answer streams back to you token by token.

The key trick is that **no single peer ever sees the whole model**,
and your prompt is encrypted so middle peers can't read it — they
just do math on numbers.

## The parts of the system

- **Gateway** — the local HTTP server you talk to. Presents an
  OpenAI-compatible API so any existing chat app, IDE plugin, or
  script can use IntelNav without modification.
- **Shard** — a program running on each peer's computer that holds
  a slice of the model and answers requests from other peers.
- **Registry** — a directory service that tracks who's online, what
  slice they hold, and how to reach them. Think of it as the phone
  book.
- **Wire protocol** — the rules everyone agrees on for talking:
  message shapes, encryption, session handshakes.

## Who's running the peers?

Two kinds of participants:

- **Volunteers** — regular users who dedicate some of their GPU time
  in exchange for being part of the network. Similar in spirit to
  running a BitTorrent client or a Tor relay.
- **Cloud seeders** — the project can run paid cloud peers during
  the cold-start phase to ensure the network is always usable, and
  gracefully step them down as volunteers come online. This means
  the network works from day one, even before it's popular.

## What's built today

The full plumbing for a working network is in place:

- A **gateway** you can actually run — it already speaks the
  OpenAI API format and proxies to a local model today.
- A **registry service** that tracks peers, verifies their identity
  with cryptographic signatures, and prevents fake peers from
  taking over.
- An **encrypted wire format** so prompts stay private end-to-end.
- A **single-peer shard** that runs models on its own using
  llama.cpp, with automatic download of the right GPU drivers for
  your hardware.
- As of this week, the **layer-splitting inference engine** — the
  part that lets a peer run just "layers 0 through 12" of a model
  and pass the result to another peer. We proved it produces
  mathematically identical output to running the whole model on
  one machine.
- An **`intelnav-registry init`** command that reads a model file
  and generates the configuration for you, so operators don't
  hand-write config.

## What's not built yet

- **Connecting two peers end-to-end.** The layer-splitter works in
  isolation; now we need to wire it across two real processes
  talking over the network.
- **Peer-to-peer discovery at internet scale.** Today peers find
  each other on a local network (mDNS) or via a registry. The real
  design uses a distributed hash table (DHT) over libp2p so peers
  find each other across the internet without a central server.
- **Batching and fault tolerance.** When several users are
  chatting at once, the network should merge their requests into a
  single pass through the model. And when a peer goes offline
  mid-session, the system should route around it.
- **A one-line installer.** Today running IntelNav requires
  `cargo build` and some tinkering. The goal is `curl … | sh` and
  you're serving models.
- **A public testnet.** A bootstrap set of peers that anyone can
  join.

## Why this matters

- **Cost**: running AI on your own hardware + a network of peers is
  much cheaper than paying per-token cloud bills, especially for
  people who use AI heavily.
- **Privacy**: your prompts are encrypted and split across peers
  who only see a fragment of the computation. No single company
  logs your conversations.
- **Resilience**: the network has no central point of failure. If
  one peer disappears, another takes its place.
- **Access**: a student with a gaming GPU can run a frontier
  coding model that normally requires $40,000 of hardware, as long
  as a few other people are online to share the load.

## The long-term pitch

AI should feel more like email or BitTorrent — a protocol that
belongs to everyone — than like a service you rent from one of
three companies. IntelNav is one attempt to make that concrete.
