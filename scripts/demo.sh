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
SPLITS="${SPLITS:-8,16}"        # peer-1 owns [0..8), peer-2 [8..16), peer-3 [16..24).
PORTS=(7717 7718 7719)
GATEWAY_PORT="${GATEWAY_PORT:-8787}"
LOG_DIR="${LOG_DIR:-$ROOT/target/demo-logs}"

# Resolve binaries — prefer the freshest build.
INTELNAV_BIN="${INTELNAV_BIN:-$ROOT/target/debug/intelnav}"
PIPE_PEER_BIN="${PIPE_PEER_BIN:-$ROOT/target/debug/examples/pipe_peer}"

# ---------------------------------------------------------------------
# Sanity checks — fail fast and tell the user what to fix.
# ---------------------------------------------------------------------
die() { echo "demo: $*" >&2; exit 1; }

[[ -f "$GGUF" ]]                || die "GGUF not found at $GGUF (override with GGUF=…)"
[[ -d "$LIBLLAMA_DIR" ]]        || die "libllama dir not found at $LIBLLAMA_DIR (override with INTELNAV_LIBLLAMA_DIR=…)"
[[ -f "$INTELNAV_BIN" ]]        || die "intelnav binary not at $INTELNAV_BIN (cargo build -p intelnav-cli)"
[[ -f "$PIPE_PEER_BIN" ]]       || die "pipe_peer not at $PIPE_PEER_BIN (cargo build -p intelnav-runtime --example pipe_peer)"

mkdir -p "$LOG_DIR"

# Parse the SPLITS list into peer ranges. With N_LAYERS=24 and
# SPLITS="8,16" we get ranges "0..8", "8..16", "16..24" — three peers,
# eight layers each.
IFS=',' read -ra split_arr <<< "$SPLITS"
peer_ranges=()
prev=0
for s in "${split_arr[@]}"; do
    peer_ranges+=("$prev:$s")
    prev=$s
done
peer_ranges+=("$prev:$N_LAYERS")

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
echo "demo: logs      = $LOG_DIR"
echo

export INTELNAV_LIBLLAMA_DIR="$LIBLLAMA_DIR"

# Each peer owns one layer slice and binds its own port.
PEER_ADDRS=()
for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    range="${peer_ranges[$i]}"
    start="${range%:*}"
    end="${range#*:}"
    label="peer-$((i+1))"
    start_child "$label" \
        "$PIPE_PEER_BIN" \
        --gguf "$GGUF" \
        --start "$start" \
        --end "$end" \
        --bind "127.0.0.1:$port" \
        --device cpu >/dev/null
    PEER_ADDRS+=("127.0.0.1:$port")
    wait_for_port "$port" "$label"
    echo "demo:   $label ready · layers [$start..$end) · 127.0.0.1:$port"
done
echo

# ---------------------------------------------------------------------
# Spawn the gateway with the three peers pre-registered as static
# directory entries so the SPA at /v1/swarm/topology shows them.
# ---------------------------------------------------------------------
PEERS_CSV="$(IFS=,; echo "${PEER_ADDRS[*]}")"
SPLITS_CSV="$SPLITS"

# Export env vars Config picks up — registers the 3 peers in the
# gateway's static directory so they show up in /v1/swarm/topology.
export INTELNAV_PEERS="$PEERS_CSV"
export INTELNAV_SPLITS="$SPLITS_CSV"

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
