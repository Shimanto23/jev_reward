#!/usr/bin/env bash
set -euo pipefail

: "${OPENJEV_MODEL_DIR:?Set OPENJEV_MODEL_DIR to the downloaded openjev-FP8 directory}"
: "${OPENJEV_HELPER_PATH:?Set OPENJEV_HELPER_PATH to helper/shim.py from the OpenJEV main repository}"

PYTHON_BIN="${OPENJEV_PYTHON:-python}"
VLLM_PORT="${OPENJEV_VLLM_PORT:-8000}"
API_PORT="${OPENJEV_PORT:-3000}"
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
API_BASE_URL="http://127.0.0.1:${API_PORT}"
STARTUP_TIMEOUT="${OPENJEV_STARTUP_TIMEOUT:-900}"
MAX_MODEL_LEN="${OPENJEV_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${OPENJEV_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${OPENJEV_MAX_NUM_SEQS:-1}"

"$PYTHON_BIN" -m vllm serve "$OPENJEV_MODEL_DIR" \
  --host 127.0.0.1 \
  --served-model-name qwen \
  --port "$VLLM_PORT" \
  --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt '{"image":1}' \
  --trust-remote-code \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-logprobs 64 \
  --gdn-prefill-backend triton \
  --quantization fp8 &
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for OpenJEV's vLLM backend at ${VLLM_BASE_URL}..."
elapsed=0
until curl --silent --fail "${VLLM_BASE_URL}/models" >/dev/null; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "OpenJEV vLLM exited before becoming ready." >&2
    exit 1
  fi
  if (( elapsed >= STARTUP_TIMEOUT )); then
    echo "Timed out waiting for OpenJEV vLLM." >&2
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

echo "Starting the OpenJEV decision helper at ${API_BASE_URL}..."
VLLM="$VLLM_BASE_URL" \
TOKENIZER="$OPENJEV_MODEL_DIR" \
READOUT_T=0.85 \
READOUT_NOUL_T=1.829074 \
READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 \
READOUT_INSTR_STYLE=pyrepr \
SHIM_STAGGER=1 \
"$PYTHON_BIN" "$OPENJEV_HELPER_PATH" --host 127.0.0.1 --port "$API_PORT"
