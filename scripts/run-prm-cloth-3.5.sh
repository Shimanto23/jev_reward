#!/bin/bash
set -e

PROJECT_DIR="/home/ssa393/Milestone-4/jev_reward"

cd "$PROJECT_DIR"

# ============================================================
# GPU
# ============================================================
export CUDA_VISIBLE_DEVICES=0
export gpu_id=0

# ============================================================
# PyFlex / SoftGym
# ============================================================
export PYFLEXROOT="$PROJECT_DIR/softgym/PyFlex"

export PYTHONPATH="$PYFLEXROOT/bindings/build:$PROJECT_DIR/softgym:$PROJECT_DIR"

export LD_LIBRARY_PATH="/.singularity.d/libs:$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:/usr/local/cuda/lib64"

# ============================================================
# Headless EGL
# ============================================================
export __EGL_VENDOR_LIBRARY_FILENAMES="$PROJECT_DIR/egl_vendor/10_nvidia.json"

unset DISPLAY
unset MUJOCO_GL
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM

# ============================================================
# OpenJEV vLLM
# ============================================================
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
SERVED_MODEL_ID="$(curl --silent --show-error --fail "$VLLM_BASE_URL/models" | python -c 'import json,sys; models=json.load(sys.stdin).get("data", []); print(models[0]["id"] if models else "")')"
if [[ -z "$SERVED_MODEL_ID" ]]; then
  echo "No model ID returned by $VLLM_BASE_URL/models"
  exit 1
fi
export VLLM_MODEL="${OPENJEV_VLLM_MODEL:-$SERVED_MODEL_ID}"
export FRM_LLM_MODEL="$VLLM_MODEL"
export RRM_LABEL_BACKEND="openjev"
export RRM_ACTIVE_QUERY="1"
export RRM_ACTIVE_WARMUP="30"
export RRM_ACTIVE_CANDIDATE_FACTOR="4"
export OPENJEV_BASE_URL="${OPENJEV_BASE_URL:-http://127.0.0.1:3000}"
export OPENJEV_MODEL="${OPENJEV_MODEL:-openjev}"
if ! curl --silent --show-error --fail "$OPENJEV_BASE_URL/v1/version" >/dev/null; then
  echo "OpenJEV helper is unavailable at $OPENJEV_BASE_URL"
exit 1
fi

echo "=========================================="
echo "Working directory : $(pwd)"
echo "Python            : $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYFLEXROOT=$PYFLEXROOT"
echo "EGL vendor=$__EGL_VENDOR_LIBRARY_FILENAMES"
echo "VLLM_BASE_URL=$VLLM_BASE_URL"
echo "VLLM_MODEL=$VLLM_MODEL"
echo "FRM periodic code generation is disabled for this JEV preference run"
echo "RRM active query selection starts after 30 random comparisons"
echo "=========================================="

# ============================================================
# Check vLLM server
# ============================================================
echo "Checking vLLM server..."
curl -s http://127.0.0.1:8000/v1/models
echo
echo "=========================================="

# ============================================================
# Train
# BT-FRM + Pairwise RRM + PRM
# ============================================================
python "$PROJECT_DIR/train_CoRe.py" \
  env=softgym_ClothFoldDiagonal \
  agent=sac_cloth \
  seed=0 \
  experiment=openjev_rrm_prm_cloth_15k \
  segment=3 \
  num_train_steps=15000 \
  replay_buffer_capacity=15000 \
  num_seed_steps=250 \
  num_pre_steps=250 \
  num_interact=1000 \
  max_feedback=300 \
  RRM_batch=10 \
  RRM_update=25 \
  eval_frequency=250 \
  num_eval_episodes=10 \
  reward_alpha=0.5 \
  FRM_align_step=2400 \
  FRM_align_max_steps=0 \
  use_FRM_online=False \
  use_PRM=true \
  PRM_beta=0.1
