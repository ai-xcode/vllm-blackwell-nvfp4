#!/bin/bash
# start-nvfp4.sh — launch any NVFP4 model in vLLM with all Blackwell-PCIe + curand fixes baked in.
#
# Encapsulates the gotchas we hit on RTX PRO 6000 Blackwell + CUDA 13:
#   - flashinfer's FP4 GEMM JIT compile can't find curand_kernel.h
#     because /usr/local/cuda-13.0/include/ is missing the curand-dev package.
#     We point NVCC_PREPEND_FLAGS / CPATH at the pip-installed nvidia/cu13/include
#     where the headers actually live.
#   - --quantization modelopt_fp4 is required for NVFP4 weights (modelopt format).
#   - For TP>1: NCCL_P2P_DISABLE=1 + --disable-custom-all-reduce per
#     vllm-blackwell-tp2-fix (PCIe-only Blackwell deadlocks).
#
# Usage:
#   start-nvfp4.sh <model_dir> [options]
#     --port N        (default 8011)
#     --tp N          (default 1; for TP=2, also applies Blackwell PCIe fixes)
#     --util F        (default 0.30 — gpu-memory-utilization)
#     --max-len N     (default 4096)
#     --max-seqs N    (default 8)
#     --name STR      (default: basename of model_dir)
#     -- ...          (any further args passed verbatim to `vllm serve`)
#
# Example:
#   start-nvfp4.sh ~/vLLM_Servers/models_awq/Qwen3-4B-NVFP4
#   start-nvfp4.sh ~/vLLM_Servers/models_awq/Qwen3-14B-NVFP4 --port 8012 --max-len 8192

set -e

MODEL=""
PORT=8011
TP=1
UTIL=0.30
MAX_LEN=4096
MAX_SEQS=8
SERVED_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)     PORT="$2"; shift 2 ;;
        --tp)       TP="$2"; shift 2 ;;
        --util)     UTIL="$2"; shift 2 ;;
        --max-len)  MAX_LEN="$2"; shift 2 ;;
        --max-seqs) MAX_SEQS="$2"; shift 2 ;;
        --name)     SERVED_NAME="$2"; shift 2 ;;
        --)         shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*)
            echo "unknown option: $1"; exit 1 ;;
        *)
            [ -z "$MODEL" ] && MODEL="$1" || EXTRA_ARGS+=("$1")
            shift ;;
    esac
done

[ -z "$MODEL" ] && { echo "ERROR: model directory required"; echo "Try: $0 --help"; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: model directory not found: $MODEL"; exit 1; }
[ -z "$SERVED_NAME" ] && SERVED_NAME=$(basename "$MODEL")

VENV="$HOME/vLLM_Servers/vllm_env"
[ -f "$VENV/bin/activate" ] || { echo "ERROR: vLLM venv not found at $VENV"; exit 1; }

PIP_NV_INC="$VENV/lib/python3.12/site-packages/nvidia/cu13/include"
[ -f "$PIP_NV_INC/curand_kernel.h" ] || {
    echo "ERROR: curand_kernel.h not found at $PIP_NV_INC"
    echo "       (pip nvidia-curand-cu13 missing or in different location?)"
    exit 1
}

source "$VENV/bin/activate"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH

# ─── NVFP4 fix: point compiler at the pip-installed nvidia headers ──────────
export NVCC_PREPEND_FLAGS="-I$PIP_NV_INC"
export CPATH="$PIP_NV_INC"

# Pip-installed cu13 runtime libs
_NV="$VENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${_NV}/cu13/lib:${_NV}/cublas/lib:${_NV}/cuda_nvrtc/lib:${_NV}/cuda_runtime/lib:${_NV}/cusolver/lib:${_NV}/cusparse/lib:${_NV}/cufft/lib:${_NV}/curand/lib:${_NV}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

TP_ARGS=()
if [ "$TP" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="0,1"
    # Blackwell PCIe TP>1 fixes (see vllm-blackwell-tp2-fix repo)
    export NCCL_P2P_DISABLE=1
    TP_ARGS+=(--disable-custom-all-reduce)
else
    export CUDA_VISIBLE_DEVICES="0"
fi

echo "[start-nvfp4] $SERVED_NAME → http://127.0.0.1:${PORT}/v1   TP=$TP util=$UTIL"
echo "[start-nvfp4] model: $MODEL"

exec vllm serve "$MODEL" \
    --quantization modelopt_fp4 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$UTIL" \
    --trust-remote-code \
    --port "$PORT" --host 127.0.0.1 \
    --served-model-name "$SERVED_NAME" \
    "${TP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
