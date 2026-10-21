#!/bin/bash
set -e

# ============================================================
# Project
# ============================================================
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
# CPU / RAM Optimisation
# These do NOT change the learning methodology
# ============================================================
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4


# ============================================================
# Disk-Backed Image Buffers
#
# PassWater uses large image buffers.
# Store RRM trajectory images and replay-buffer images
# on HPC scratch instead of keeping them fully in RAM.
# ============================================================

# Prefer HPC-provided SCRATCH_DIR.
# Fall back to TMPDIR if SCRATCH_DIR is unavailable.
if [ -n "$SCRATCH_DIR" ]; then
    MEMMAP_DIR="$SCRATCH_DIR"
elif [ -n "$TMPDIR" ]; then
    MEMMAP_DIR="$TMPDIR"
else
    MEMMAP_DIR="/scratch/jobs/$USER/$SLURM_JOB_ID"
fi

mkdir -p "$MEMMAP_DIR"

# RRM 720x720 trajectory image buffer
export RRM_MEMMAP_DIR="$MEMMAP_DIR"

# Replay-buffer image storage
export REPLAY_MEMMAP_DIR="$MEMMAP_DIR"


# ============================================================
# SoftGym / PyFlex
# ============================================================
export PYFLEXROOT="$PROJECT_DIR/softgym/PyFlex"

# PyFlex Python module is inside bindings/build
export PYTHONPATH="$PYFLEXROOT/bindings/build:$PROJECT_DIR/softgym:$PROJECT_DIR/metaworld:$PROJECT_DIR:$PYTHONPATH"


# ============================================================
# Headless EGL Rendering
# ============================================================
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export __EGL_VENDOR_LIBRARY_FILENAMES="$PROJECT_DIR/egl_vendor/10_nvidia.json"

unset DISPLAY


# ============================================================
# Libraries
# ============================================================
export LD_LIBRARY_PATH="/.singularity.d/libs:$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"


# ============================================================
# Local Qwen3.5 vLLM
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
echo "============================================================"
echo " CoRe + BT-FRM + RRM + PRM"
echo " Task          : Pass Water"
echo " Environment   : softgym_PassWater"
echo " VLM           : Qwen/Qwen3.5-9B"
echo " Train Steps   : 100050"
echo " RRM Batch     : 5"
echo " RRM Interval  : 2500"
echo " Max Feedback  : 200"
echo " PRM Beta      : 0.1"
echo " Alpha         : 0.5"
echo "============================================================"

echo ""
echo "PROJECT_DIR:"
echo "$PROJECT_DIR"

echo ""
echo "PYFLEXROOT:"
echo "$PYFLEXROOT"

echo ""
echo "RRM_MEMMAP_DIR:"
echo "$RRM_MEMMAP_DIR"

echo ""
echo "REPLAY_MEMMAP_DIR:"
echo "$REPLAY_MEMMAP_DIR"

echo ""
echo "Scratch filesystem:"
df -h "$MEMMAP_DIR"


# ============================================================
# Check PyFlex
# ============================================================
echo ""
echo "Checking PyFlex installation..."

python -c "import pyflex; print('SUCCESS: PyFlex imported successfully')"


# ============================================================
# Check Python Files
# ============================================================
echo ""
echo "Checking RRM/reward_model.py syntax..."

python -m py_compile RRM/reward_model.py

echo "SUCCESS: reward_model.py syntax is valid."


echo ""
echo "Checking replay_buffer.py syntax..."

python -m py_compile replay_buffer.py

echo "SUCCESS: replay_buffer.py syntax is valid."


# ============================================================
# Start Training
# ============================================================
echo ""
echo "All checks passed."
echo "Starting Pass Water training..."
echo ""


python train_CoRe.py \
    env=softgym_PassWater \
    agent=sac \
    seed=0 \
    experiment=qwen35_btfrm_rrm_prm_pass_water_100k \
    segment=3 \
    num_train_steps=100050 \
    num_seed_steps=1000 \
    num_pre_steps=9000 \
    num_interact=2500 \
    max_feedback=200 \
    RRM_lr=1e-4 \
    RRM_batch=5 \
    RRM_update=30 \
    resnet=1 \
    eval_frequency=5000 \
    num_eval_episodes=10 \
    reward_alpha=0.5 \
    FRM_align_step=24975 \
    FRM_align_max_steps=100000 \
    use_FRM_online=False \
    use_PRM=true \
    PRM_beta=0.1


# ============================================================
# Finished
# ============================================================
echo ""
echo "============================================================"
echo " Pass Water training finished successfully."
echo "============================================================"
