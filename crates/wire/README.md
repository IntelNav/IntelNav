# intelnav-wire

CBOR codecs for the IntelNav wire protocol. Normative spec:
[`specs/protocol-v1.md`](../../specs/protocol-v1.md), paper §A.

- `Msg` enum — every on-wire message (`Hello`, `SessionInit`/`Ack`,
  `Prompt`, `ForwardHidden`, `Token`, `Heartbeat`, `AbortSession`,
  `Advertise`, `Gossip`).
- `encode` / `decode` — single-message CBOR.
- `encode_frame` / `decode_frame` — length-prefixed (big-endian `u32`)
  framing with a 16 MiB safety cap.
- `Dtype` — activation dtype on the chain wire (`Fp16`, `Bf16`, `Int8`).
- `Phase` — `Prefill | Decode`.
- `dht_provider_key(model, quant)` — canonical DHT lookup key.

Protocol additions are backward-compatible: use
`#[serde(default, skip_serializing_if = "Option::is_none")]` on new
fields so proto-v1 peers still decode.

`#![forbid(unsafe_code)]`.
