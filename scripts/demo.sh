#!/usr/bin/env bash
# IntelNav demo bring-up — spawns 3 localhost pipe_peers + a gateway,
# all wired together so the SPA at http://127.0.0.1:8787 shows a live
# 3-node swarm.
#
# Usage:
#
#   scripts/demo.sh            # default: Qwen2.5-0.5B (24 layers, 8/8/8 split)
#   GGUF=/path/to/model.gguf scripts/demo.sh
#   N_LAYERS=24 SPLITS=8,16 scripts/demo.sh   # explicit split points
#
# Tear down with Ctrl+C — the trap stops every child process.

set -Eeuo pipefail

# ---------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF="${GGUF:-/home/islam/IntelNav/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
LIBLLAMA_DIR="${INTELNAV_LIBLLAMA_DIR:-/home/islam/IntelNav/llama.cpp/build/bin}"
N_LAYERS="${N_LAYERS:-24}"      # Qwen2.5-0.5B has 24 transformer blocks.
# SPLITS is the chain protocol's convention: one entry per peer,
# specifying that peer's start layer. The gateway owns the prefix
# [0..splits[0]) locally (embed + first slice + head); each peer i
# owns [splits[i]..splits[i+1]), and the tail peer owns
# [splits[N-1]..N_LAYERS). For Qwen2.5-0.5B (24 layers) the default
# 4/9/14/19 puts ~5 layers on each peer plus 4 on the gateway prefix.
SPLITS="${SPLITS:-4,9,14,19}"
# Real peer ports. Each peer is fronted by a netsim that listens on
# NETSIM_PORTS[i] — the gateway only ever talks to those, never
# directly to PORTS[i].
PORTS=(7717 7718 7719 7720)
NETSIM_PORTS=(7817 7818 7819 7820)
NETSIM_CTRL_PORTS=(9117 9118 9119 9120)
# Per-peer netsim tier. Each entry is a `label|forward|reverse` triple
# the demo passes to intelnav-netsim. The forward leg is gateway →
# peer; reverse is peer → gateway. Numbers loosely model:
#   * peer-1: a fast LAN box on the same switch
#   * peer-2: a metro-area volunteer 25 ms away
#   * peer-3: a transcontinental WAN node
#   * peer-4: same WAN distance but a flakier link with packet loss
NETSIM_TIERS=(
    "LAN|delay=2,jitter=0.5,bw=1000|delay=2,jitter=0.5,bw=1000"
    "Metro|delay=22,jitter=4,bw=300|delay=22,jitter=4,bw=300"
    "WAN|delay=72,jitter=10,bw=120|delay=72,jitter=10,bw=120"
    "Lossy WAN|delay=110,jitter=18,bw=40,loss=0.01,reorder=0.005|delay=110,jitter=18,bw=40,loss=0.01,reorder=0.005"
)
GATEWAY_PORT="${GATEWAY_PORT:-8787}"
LOG_DIR="${LOG_DIR:-$ROOT/target/demo-logs}"

# Resolve binaries — prefer the freshest build.
INTELNAV_BIN="${INTELNAV_BIN:-$ROOT/target/debug/intelnav}"
PIPE_PEER_BIN="${PIPE_PEER_BIN:-$ROOT/target/debug/examples/pipe_peer}"
CHUNK_BIN="${CHUNK_BIN:-$ROOT/target/debug/intelnav-chunk}"
NETSIM_BIN="${NETSIM_BIN:-$ROOT/target/debug/intelnav-netsim}"

# Path B (stitched-subset) is the default: each peer downloads + loads
# only its layer slice, not the full GGUF. Set STITCHED=0 to fall back
# to "every peer mmaps the full file" mode for debugging.
STITCHED="${STITCHED:-1}"
CHUNK_PORT="${CHUNK_PORT:-9099}"

# Set NETSIM=0 to skip the shaper layer entirely (gateway talks
# directly to peers). Useful when testing without simulated latency.
NETSIM="${NETSIM:-1}"

# ---------------------------------------------------------------------
# Sanity checks — fail fast and tell the user what to fix.
# ---------------------------------------------------------------------
die() { echo "demo: $*" >&2; exit 1; }

[[ -f "$GGUF" ]]                || die "GGUF not found at $GGUF (override with GGUF=…)"
[[ -d "$LIBLLAMA_DIR" ]]        || die "libllama dir not found at $LIBLLAMA_DIR (override with INTELNAV_LIBLLAMA_DIR=…)"
[[ -f "$INTELNAV_BIN" ]]        || die "intelnav binary not at $INTELNAV_BIN (cargo build -p intelnav-cli)"
[[ -f "$PIPE_PEER_BIN" ]]       || die "pipe_peer not at $PIPE_PEER_BIN (cargo build -p intelnav-runtime --example pipe_peer)"
if [[ "$STITCHED" == "1" ]]; then
    [[ -f "$CHUNK_BIN" ]] || die "intelnav-chunk not at $CHUNK_BIN (cargo build -p intelnav-model-store --features serve --bin intelnav-chunk)"
fi
if [[ "$NETSIM" == "1" ]]; then
    [[ -f "$NETSIM_BIN" ]] || die "intelnav-netsim not at $NETSIM_BIN (cargo build -p intelnav-netsim)"
    [[ "${#NETSIM_PORTS[@]}"      -eq "${#PORTS[@]}" ]] || die "NETSIM_PORTS length must match PORTS"
    [[ "${#NETSIM_CTRL_PORTS[@]}" -eq "${#PORTS[@]}" ]] || die "NETSIM_CTRL_PORTS length must match PORTS"
    [[ "${#NETSIM_TIERS[@]}"      -eq "${#PORTS[@]}" ]] || die "NETSIM_TIERS length must match PORTS"
fi

mkdir -p "$LOG_DIR"

# Parse SPLITS (chain protocol convention: one entry per peer = that
# peer's start layer). With SPLITS="4,9,14,19" and N_LAYERS=24 the
# gateway owns [0..4) locally, peer-1 owns [4..9), peer-2 [9..14),
# peer-3 [14..19), peer-4 [19..24).
IFS=',' read -ra split_arr <<< "$SPLITS"
peer_ranges=()
for i in "${!split_arr[@]}"; do
    start="${split_arr[$i]}"
    next_idx=$((i + 1))
    if (( next_idx < ${#split_arr[@]} )); then
        end="${split_arr[$next_idx]}"
    else
        end="$N_LAYERS"
    fi
    peer_ranges+=("$start:$end")
done

[[ "${#peer_ranges[@]}" -eq "${#PORTS[@]}" ]] \
    || die "SPLITS=$SPLITS produced ${#peer_ranges[@]} ranges but we have ${#PORTS[@]} ports"

# ---------------------------------------------------------------------
# Lifecycle: start, stream logs to disk, kill everything on exit.
# ---------------------------------------------------------------------
declare -a CHILDREN=()
declare -a CHILD_LABELS=()

cleanup() {
    echo
    echo "demo: shutting down…"
    for i in "${!CHILDREN[@]}"; do
        local pid="${CHILDREN[$i]}"
        local label="${CHILD_LABELS[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            echo "demo:   stop $label (pid $pid)"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Give them a beat to drain, then SIGKILL stragglers.
    sleep 1
    for pid in "${CHILDREN[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "demo: done"
}
trap cleanup INT TERM EXIT

start_child() {
    local label="$1"; shift
    local out="$LOG_DIR/$label.out"
    local err="$LOG_DIR/$label.err"
    echo "demo: start $label" >&2
    "$@" >"$out" 2>"$err" &
    local pid=$!
    CHILDREN+=("$pid")
    CHILD_LABELS+=("$label")
    echo "$pid"
}

wait_for_port() {
    local port="$1"
    local label="$2"
    local deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            return 0
        fi
        sleep 0.2
    done
    die "$label did not start listening on :$port within 30s — see $LOG_DIR/$label.err"
}

# ---------------------------------------------------------------------
# Spawn the peers.
# ---------------------------------------------------------------------
echo "demo: model     = $GGUF"
echo "demo: libllama  = $LIBLLAMA_DIR"
echo "demo: peers     = ${#PORTS[@]} on ports ${PORTS[*]}"
echo "demo: gateway   = http://127.0.0.1:$GATEWAY_PORT"
echo "demo: stitched  = $STITCHED"
echo "demo: netsim    = $NETSIM (ports ${NETSIM_PORTS[*]})"
echo "demo: logs      = $LOG_DIR"
echo

export INTELNAV_LIBLLAMA_DIR="$LIBLLAMA_DIR"

# ---------------------------------------------------------------------
# Path B: chunk the GGUF once, host it, point each peer at the manifest
# so they download + load only their layer slice. This is the whole
# reason the project exists — a peer with 8 GiB doesn't keep 19 GiB of
# weights warm.
# ---------------------------------------------------------------------
CHUNK_DIR=""
MANIFEST_URL=""
if [[ "$STITCHED" == "1" ]]; then
    gguf_stem="$(basename "$GGUF" .gguf)"
    CHUNK_DIR="$ROOT/target/demo-chunks/$gguf_stem"
    if [[ -f "$CHUNK_DIR/manifest.json" ]]; then
        echo "demo: chunks   reusing $CHUNK_DIR (delete to re-chunk)"
    else
        echo "demo: chunks   chunking $GGUF -> $CHUNK_DIR"
        mkdir -p "$CHUNK_DIR"
        "$CHUNK_BIN" chunk "$GGUF" "$CHUNK_DIR" --overwrite >"$LOG_DIR/chunk.out" 2>"$LOG_DIR/chunk.err" \
            || die "intelnav-chunk failed — see $LOG_DIR/chunk.err"
    fi
    start_child "chunk-server" \
        "$CHUNK_BIN" serve "$CHUNK_DIR" --bind "127.0.0.1:$CHUNK_PORT" >/dev/null
    wait_for_port "$CHUNK_PORT" "chunk-server"
    MANIFEST_URL="http://127.0.0.1:$CHUNK_PORT/manifest.json"
    chunk_size="$(du -sh "$CHUNK_DIR" 2>/dev/null | cut -f1 || echo '?')"
    echo "demo:   chunk-server ready · $MANIFEST_URL · $chunk_size on disk"
    echo
fi

# Each peer owns one layer slice and binds its own port. In stitched
# mode (the default) it fetches its bundles from the chunk-server and
# only mmaps its own slice — Path B end-to-end.
PEER_ADDRS=()
NETSIM_CTRLS=()
for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    range="${peer_ranges[$i]}"
    start="${range%:*}"
    end="${range#*:}"
    label="peer-$((i+1))"
    if [[ "$STITCHED" == "1" ]]; then
        peer_cache="$LOG_DIR/$label-cache"
        mkdir -p "$peer_cache"
        start_child "$label" \
            "$PIPE_PEER_BIN" \
            --manifest "$MANIFEST_URL" \
            --chunk-cache "$peer_cache" \
            --start "$start" \
            --end "$end" \
            --bind "127.0.0.1:$port" \
            --device cpu >/dev/null
    else
        start_child "$label" \
            "$PIPE_PEER_BIN" \
            --gguf "$GGUF" \
            --start "$start" \
            --end "$end" \
            --bind "127.0.0.1:$port" \
            --device cpu >/dev/null
    fi
    wait_for_port "$port" "$label"
    if [[ "$STITCHED" == "1" ]]; then
        slice_size="$(du -sh "$LOG_DIR/$label-cache" 2>/dev/null | cut -f1 || echo '?')"
        echo "demo:   $label ready · layers [$start..$end) · 127.0.0.1:$port · stitched · slice $slice_size"
    else
        echo "demo:   $label ready · layers [$start..$end) · 127.0.0.1:$port · full-gguf"
    fi

    # Front the peer with a netsim if enabled. The gateway will see
    # only the netsim port; per-link delay/jitter/bandwidth/loss come
    # from NETSIM_TIERS[$i] and can be re-tuned live via
    # http://127.0.0.1:$ctrl_port/config.
    if [[ "$NETSIM" == "1" ]]; then
        ns_port="${NETSIM_PORTS[$i]}"
        ns_ctrl="${NETSIM_CTRL_PORTS[$i]}"
        IFS='|' read -r ns_label ns_fwd ns_rev <<< "${NETSIM_TIERS[$i]}"
        ns_name="netsim-$((i+1))"
        start_child "$ns_name" \
            "$NETSIM_BIN" \
            --bind     "127.0.0.1:$ns_port" \
            --upstream "127.0.0.1:$port" \
            --control  "127.0.0.1:$ns_ctrl" \
            --label    "$ns_label" \
            --forward  "$ns_fwd" \
            --reverse  "$ns_rev" >/dev/null
        wait_for_port "$ns_port" "$ns_name"
        wait_for_port "$ns_ctrl" "$ns_name-ctrl"
        echo "demo:   $ns_name ready · 127.0.0.1:$ns_port → 127.0.0.1:$port · $ns_label"
        PEER_ADDRS+=("127.0.0.1:$ns_port")
        NETSIM_CTRLS+=("127.0.0.1:$ns_ctrl")
    else
        PEER_ADDRS+=("127.0.0.1:$port")
    fi
done
echo

# ---------------------------------------------------------------------
# Spawn the gateway with the three peers pre-registered as static
# directory entries so the SPA at /v1/swarm/topology shows them.
# ---------------------------------------------------------------------
PEERS_CSV="$(IFS=,; echo "${PEER_ADDRS[*]}")"
SPLITS_CSV="$SPLITS"

# Export env vars Config picks up — registers the 3 peers in the
# gateway's static directory so they show up in /v1/swarm/topology
# and tells the gateway to drive the chain itself for chat
# completions (vs proxying to upstream).
export INTELNAV_PEERS="$PEERS_CSV"
export INTELNAV_SPLITS="$SPLITS_CSV"
export INTELNAV_GATEWAY_MODEL="$GGUF"
export INTELNAV_MODELS_SEARCH="$(dirname "$GGUF")"
if [[ "$NETSIM" == "1" ]]; then
    NETSIM_CTRLS_CSV="$(IFS=,; echo "${NETSIM_CTRLS[*]}")"
    export INTELNAV_NETSIMS="$NETSIM_CTRLS_CSV"
fi

start_child "gateway" \
    "$INTELNAV_BIN" gateway \
    --bind "127.0.0.1:$GATEWAY_PORT" \
    --no-mdns >/dev/null
wait_for_port "$GATEWAY_PORT" "gateway"
echo "demo:   gateway ready · http://127.0.0.1:$GATEWAY_PORT"
echo

# Drop a hint for the operator.
cat <<EOF
demo: setup live ─ open http://127.0.0.1:$GATEWAY_PORT in a browser.
demo: tail logs   tail -F $LOG_DIR/{peer-1,peer-2,peer-3,gateway}.{out,err}
demo: stop        Ctrl+C
EOF

# Hold the script so the trap fires on Ctrl+C; surface child crashes.
while true; do
    for i in "${!CHILDREN[@]}"; do
        if ! kill -0 "${CHILDREN[$i]}" 2>/dev/null; then
            echo "demo: ${CHILD_LABELS[$i]} (pid ${CHILDREN[$i]}) exited unexpectedly" >&2
            echo "demo: tail of $LOG_DIR/${CHILD_LABELS[$i]}.err:" >&2
            tail -n 20 "$LOG_DIR/${CHILD_LABELS[$i]}.err" >&2 || true
            exit 1
        fi
    done
    sleep 1
done
