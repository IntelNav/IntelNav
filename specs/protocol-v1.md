# IntelNav Protocol — v1 (normative)

This document pins the wire protocol specified by `paper/paper.pdf` §A. The
Rust implementation lives in `crates/wire`.

## Framing

Every message is length-prefixed with a big-endian `u32` byte count, followed
by a CBOR-encoded `Msg` body. A single frame is capped at 16 MiB.

## Messages

The message taxonomy is the `Msg` enum in `crates/wire/src/lib.rs`. Paper §A
is authoritative; the Rust enum is *derived from* it. Adding a variant
requires bumping `proto_ver` in `Msg::Hello`.

## DHT keys

Provider lookup uses

```
/intelnav/v1/model/<MODEL_ID>/<QUANT>
```

matching paper §7.1.
