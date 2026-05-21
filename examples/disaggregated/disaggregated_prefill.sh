#!/usr/bin/env bash
# This file demonstrates the example usage of disaggregated prefilling.
# We launch 2 vLLM instances (1 for prefill and 1 for decode), transfer the
# KV cache between them, and route two requests through a local proxy.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$SCRIPT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "vllm" ]]; then
    echo "Activate conda environment 'vllm' before running this script." >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "python was not found in the active environment." >&2
    exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm CLI was not found in the active environment." >&2
    exit 1
fi

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

DEFAULT_MODEL=/shared/models/hf/Meta-Llama-3-8B-Instruct
MODEL_NAME=${MODEL_NAME:-${HF_MODEL_NAME:-$DEFAULT_MODEL}}
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}
PREFILL_PORT=${PREFILL_PORT:-8100}
DECODE_PORT=${DECODE_PORT:-8200}
PROXY_PORT=${PROXY_PORT:-8000}
DISCOVERY_PORT=${DISCOVERY_PORT:-30001}
PREFILL_KV_PORT=${PREFILL_KV_PORT:-14579}
DECODE_KV_PORT=${DECODE_KV_PORT:-14580}
KV_SEND_TYPE=${KV_SEND_TYPE:-GET}
PREFILL_KV_BUFFER_SIZE=${PREFILL_KV_BUFFER_SIZE:-8e9}
DECODE_KV_BUFFER_SIZE=${DECODE_KV_BUFFER_SIZE:-8e9}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-1200}

if [[ "$MODEL_NAME" == /* && ! -e "$MODEL_NAME" ]]; then
    echo "Model path does not exist: $MODEL_NAME" >&2
    exit 1
fi

if [[ -z "${VLLM_HOST_IP:-}" || "${VLLM_HOST_IP}" == "localhost" ]]; then
    export VLLM_HOST_IP=127.0.0.1
elif [[ "${VLLM_HOST_IP}" != "127.0.0.1" ]]; then
    echo "VLLM_HOST_IP must stay on loopback for this example. Use 127.0.0.1." >&2
    exit 1
fi

RUNTIME_DIR="$SCRIPT_DIR/.runtime/disaggregated_prefill"
TMP_DIR="$RUNTIME_DIR/tmp"
RPC_DIR="$RUNTIME_DIR/rpc"
RPC_SOCKET_DIR="$SCRIPT_DIR/.r"
CACHE_DIR="$RUNTIME_DIR/cache"
CONFIG_DIR="$RUNTIME_DIR/config"
HF_DIR="$RUNTIME_DIR/hf"
XDG_CACHE_DIR="$RUNTIME_DIR/xdg-cache"

mkdir -p "$TMP_DIR" "$RPC_DIR" "$CACHE_DIR" "$CONFIG_DIR" "$HF_DIR" "$XDG_CACHE_DIR"

rm -rf "$RPC_SOCKET_DIR"
ln -s "$RPC_DIR" "$RPC_SOCKET_DIR"

export TMPDIR="$TMP_DIR"
export VLLM_RPC_BASE_PATH="$RPC_SOCKET_DIR"
export VLLM_CACHE_ROOT="$CACHE_DIR"
export VLLM_CONFIG_ROOT="$CONFIG_DIR"
export HF_HOME="$HF_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"

PREFILL_LOG="$RUNTIME_DIR/prefill.log"
DECODE_LOG="$RUNTIME_DIR/decode.log"
PROXY_LOG="$RUNTIME_DIR/proxy.log"
RESPONSE_ONE_FILE="$RUNTIME_DIR/response1.json"
RESPONSE_TWO_FILE="$RUNTIME_DIR/response2.json"
PROXY_SCRIPT="$REPO_ROOT/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py"

rm -f \
    "$PREFILL_LOG" \
    "$DECODE_LOG" \
    "$PROXY_LOG" \
    "$RESPONSE_ONE_FILE" \
    "$RESPONSE_TWO_FILE"

if [[ ! -f "$PROXY_SCRIPT" ]]; then
    echo "Proxy script not found: $PROXY_SCRIPT" >&2
    exit 1
fi

PIDS=()
CLEANUP_DONE=0

show_logs() {
    echo "Log files:" >&2
    echo "  prefiller: $PREFILL_LOG" >&2
    echo "  decoder:   $DECODE_LOG" >&2
    echo "  proxy:     $PROXY_LOG" >&2
}

ensure_no_stale_processes() {
    local matches

    matches=$(pgrep -af "$PROXY_SCRIPT|vllm serve .*--port ($PREFILL_PORT|$DECODE_PORT)" || true)
    if [[ -n "$matches" ]]; then
        echo "Found existing disaggregated prefill demo process using target ports:" >&2
        echo "$matches" >&2
        echo "Stop the stale process or override PREFILL_PORT/DECODE_PORT/PROXY_PORT/DISCOVERY_PORT." >&2
        exit 1
    fi
}

stop_pid() {
    local pid=$1
    local attempt

    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return
    fi

    kill "$pid" 2>/dev/null || true
    for attempt in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return
        fi
        sleep 1
    done

    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local pid

    if (( CLEANUP_DONE )); then
        return
    fi
    CLEANUP_DONE=1

    for pid in "${PIDS[@]}"; do
        stop_pid "$pid"
    done

    if [[ -L "$RPC_SOCKET_DIR" || -e "$RPC_SOCKET_DIR" ]]; then
        rm -rf "$RPC_SOCKET_DIR"
    fi
}

on_exit() {
    local exit_code=$?

    cleanup
    if (( exit_code != 0 )); then
        echo "disaggregated_prefill.sh failed with exit code $exit_code" >&2
        show_logs
    fi
    exit "$exit_code"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_http() {
    local url=$1
    local service_name=$2
    local pid=$3
    local start_time now

    start_time=$(date +%s)
    echo "Waiting for $service_name at $url ..."

    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$service_name exited before becoming ready." >&2
            return 1
        fi

        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "$service_name is ready."
            return 0
        fi

        now=$(date +%s)
        if (( now - start_time >= STARTUP_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for $service_name." >&2
            return 1
        fi

        sleep 1
    done
}

validate_response() {
    local response_file=$1
    local label=$2
    local expected_text=${3:-}

    python - "$response_file" "$label" "$expected_text" <<'PY'
import json
import pathlib
import sys

response_path = pathlib.Path(sys.argv[1])
label = sys.argv[2]
expected_text = sys.argv[3].lower()
payload = json.loads(response_path.read_text())

choices = payload.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit(f"{label}: missing completion choices")

choice = choices[0]
text = choice.get("text")
if text is None:
    message = choice.get("message")
    if isinstance(message, dict):
        text = message.get("content")
if not isinstance(text, str):
    raise SystemExit(f"{label}: response text is missing or invalid")
if not text.strip():
    raise SystemExit(f"{label}: response text is empty")
if expected_text and expected_text not in text.lower():
    raise SystemExit(
        f"{label}: expected {expected_text!r} in response text, got {text!r}"
    )

print(f"{label}: response text={text!r}")
PY
}

ensure_ports_available() {
    python - "$@" <<'PY'
import socket
import sys

busy = []
for port_arg in sys.argv[1:]:
    port = int(port_arg)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            busy.append(f"127.0.0.1:{port} ({exc})")

if busy:
    raise SystemExit("Ports are already in use: " + ", ".join(busy))
PY
}

echo "Using active conda environment: ${CONDA_DEFAULT_ENV}"
echo "Using model: $MODEL_NAME"
echo "Using loopback host: $VLLM_HOST_IP"
echo "Runtime directory: $RUNTIME_DIR"

ensure_no_stale_processes
ensure_ports_available \
    "$PROXY_PORT" \
    "$PREFILL_PORT" \
    "$DECODE_PORT" \
    "$DISCOVERY_PORT" \
    "$PREFILL_KV_PORT" \
    "$DECODE_KV_PORT"

# The P2P connector indexes tensors by request_id, so both independent vLLM
# engines must keep the proxy-supplied id unchanged.
VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
vllm serve "$MODEL_NAME" \
    --enforce-eager \
    --host 127.0.0.1 \
    --port "$PREFILL_PORT" \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2,\"kv_buffer_size\":\"$PREFILL_KV_BUFFER_SIZE\",\"kv_port\":\"$PREFILL_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$VLLM_HOST_IP\",\"proxy_port\":\"$DISCOVERY_PORT\",\"http_port\":\"$PREFILL_PORT\",\"send_type\":\"$KV_SEND_TYPE\"}}" \
    >"$PREFILL_LOG" 2>&1 &
PREFILL_PID=$!
PIDS+=("$PREFILL_PID")

VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
vllm serve "$MODEL_NAME" \
    --enforce-eager \
    --host 127.0.0.1 \
    --port "$DECODE_PORT" \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_parallel_size\":2,\"kv_buffer_size\":\"$DECODE_KV_BUFFER_SIZE\",\"kv_port\":\"$DECODE_KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$VLLM_HOST_IP\",\"proxy_port\":\"$DISCOVERY_PORT\",\"http_port\":\"$DECODE_PORT\",\"send_type\":\"$KV_SEND_TYPE\"}}" \
    >"$DECODE_LOG" 2>&1 &
DECODE_PID=$!
PIDS+=("$DECODE_PID")

python "$PROXY_SCRIPT" \
    --port "$PROXY_PORT" \
    --prefill-url "http://127.0.0.1:$PREFILL_PORT" \
    --decode-url "http://127.0.0.1:$DECODE_PORT" \
    --kv-host "$VLLM_HOST_IP" \
    --prefill-kv-port "$PREFILL_KV_PORT" \
    --decode-kv-port "$DECODE_KV_PORT" \
    --discovery-port "$DISCOVERY_PORT" \
    >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
PIDS+=("$PROXY_PID")

wait_for_http "http://127.0.0.1:${PREFILL_PORT}/v1/models" "prefill server" "$PREFILL_PID"
wait_for_http "http://127.0.0.1:${DECODE_PORT}/v1/models" "decode server" "$DECODE_PID"
wait_for_http "http://127.0.0.1:${PROXY_PORT}/healthz" "proxy server" "$PROXY_PID"

output1=$(curl -fsS -X POST "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
"model": "'"$MODEL_NAME"'",
"messages": [{"role": "user", "content": "Answer in one short sentence: What is the capital of France?"}],
"max_tokens": 20,
"temperature": 0
}')

output2=$(curl -fsS -X POST "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
"model": "'"$MODEL_NAME"'",
"messages": [{"role": "user", "content": "Answer in one short sentence: Name one color of the sky on a clear day."}],
"max_tokens": 20,
"temperature": 0
}')

printf '%s\n' "$output1" >"$RESPONSE_ONE_FILE"
printf '%s\n' "$output2" >"$RESPONSE_TWO_FILE"

validate_response "$RESPONSE_ONE_FILE" "Request 1" "paris"
validate_response "$RESPONSE_TWO_FILE" "Request 2" "blue"

echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"
echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
