#!/bin/bash
set -e

PROJECT_DIR="/home/ssa393/Milestone-4/jev_reward"

cd "$PROJECT_DIR"

# ============================================================
# GPU
# ============================================================
export CUDA_VISIBLE_DEVICES=0
export gpu_id=0

# Reduce PyTorch CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ============================================================
# Python paths
# ============================================================
export PYTHONPATH="$PROJECT_DIR/softgym:$PROJECT_DIR/metaworld:$PROJECT_DIR:$PYTHONPATH"

# ============================================================
# MetaWorld / MuJoCo Headless Rendering
# ============================================================
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export __EGL_VENDOR_LIBRARY_FILENAMES="$PROJECT_DIR/egl_vendor/10_nvidia.json"

unset DISPLAY

# ============================================================
# Libraries
# ============================================================
export LD_LIBRARY_PATH="/.singularity.d/libs:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# ============================================================
# Qwen3.5 vLLM
# ============================================================
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_MODEL="Qwen/Qwen3.5-9B"
export FRM_LLM_MODEL="${FRM_LLM_MODEL:-$VLLM_MODEL}"
export RRM_LABEL_BACKEND="openjev"
export OPENJEV_BASE_URL="${OPENJEV_BASE_URL:-http://127.0.0.1:3000}"
export OPENJEV_MODEL="${OPENJEV_MODEL:-openjev}"
if ! curl --silent --show-error --fail "$OPENJEV_BASE_URL/v1/version" >/dev/null; then
  echo "OpenJEV helper is unavailable at $OPENJEV_BASE_URL"
exit 1
fi

# ============================================================
# Information
# ============================================================
echo "=========================================="
echo "MetaWorld Button Press Training"
echo "BT-FRM + Pairwise RRM + PRM"
echo "=========================================="
echo "Script            : run-prm-button-3.5.sh"
echo "Working directory : $(pwd)"
echo "Python            : $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "PYTHONPATH=$PYTHONPATH"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"
echo "EGL vendor=$__EGL_VENDOR_LIBRARY_FILENAMES"
echo "VLLM_BASE_URL=$VLLM_BASE_URL"
echo "VLLM_MODEL=$VLLM_MODEL"
echo "=========================================="

# ============================================================
# Check MetaWorld
# ============================================================
echo "Checking MetaWorld..."

python -c "import metaworld; print('MetaWorld import OK')"

echo "=========================================="

# ============================================================
# Check Qwen3.5 vLLM server
# ============================================================
echo "Checking vLLM server..."

/usr/bin/curl -s http://127.0.0.1:8000/v1/models

echo
echo "=========================================="

# ============================================================
# Training Configuration
#
# Environment:
#   MetaWorld Button Press Topdown
#
# Reward:
#   BT-FRM
#   + Pairwise RRM
#   + PRM
#
# Training:
#   500,000 environment steps
#
# Important settings:
#   segment                  = 20
#   RRM_batch                = 10
#   replay_buffer_capacity   = 20,000
#   max_feedback             = 500
#   PRM_beta                 = 0.1
# ============================================================

echo "Training configuration:"
echo "Environment            = metaworld_button-press-topdown-v2"
echo "Training steps         = 500000"
echo "Replay buffer capacity = 20000"
echo "Segment                = 20"
echo "RRM batch              = 10"
echo "RRM update             = 10"
echo "Max feedback           = 500"
echo "PRM beta               = 0.1"
echo "=========================================="

# ============================================================
# Train MetaWorld Button Press
# ============================================================

python "$PROJECT_DIR/train_CoRe.py" \
  env=metaworld_button-press-topdown-v2 \
  agent=sac \
  seed=0 \
  experiment=qwen35_btfrm_rrm_prm_button_press_500k \
  segment=20 \
  num_train_steps=500000 \
  replay_buffer_capacity=20000 \
  num_seed_steps=1000 \
  num_pre_steps=19000 \
  num_interact=4000 \
  max_feedback=500 \
  RRM_batch=10 \
  RRM_update=10 \
  eval_frequency=5000 \
  num_eval_episodes=10 \
  reward_alpha=0.5 \
  FRM_align_step=25000 \
  FRM_align_max_steps=100000 \
  use_FRM_online=False \
  use_PRM=true \
  PRM_beta=0.1

echo "=========================================="
echo "Training completed successfully."
echo "=========================================="
