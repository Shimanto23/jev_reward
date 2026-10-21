#!/bin/bash
set -e

# ============================================================
# Project
# ============================================================

PROJECT_DIR="/home/ssa393/Milestone-4/jev_reward"

cd "$PROJECT_DIR"


# ============================================================
# Conda
# ============================================================

source "/home/ssa393/miniconda3/etc/profile.d/conda.sh"
conda activate core


# ============================================================
# GPU
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export gpu_id=0

# Reduce PyTorch CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


# ============================================================
# CPU / RAM
# ============================================================

export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false


# ============================================================
# MetaWorld / MuJoCo Headless EGL
# ============================================================

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0

# Force EGL to use NVIDIA instead of Mesa/software EGL.
export __EGL_VENDOR_LIBRARY_FILENAMES="$PROJECT_DIR/egl_vendor/10_nvidia.json"

unset DISPLAY


# ============================================================
# Libraries
#
# Singularity/NVIDIA libraries come BEFORE Conda libraries.
# ============================================================

export LD_LIBRARY_PATH="/.singularity.d/libs:/usr/local/cuda/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"


# ============================================================
# Python Path
# ============================================================

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/metaworld:${PYTHONPATH:-}"


# ============================================================
# Hydra
# ============================================================

export HYDRA_FULL_ERROR=1


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
# Experiment Settings
#
# IMPORTANT:
# Do NOT change EXPERIMENT or SEED between GPU sessions.
# ============================================================

ENV_NAME="metaworld_sweep-into-v2"
EXPERIMENT="qwen35_btfrm_rrm_prm_sweep_into_250k"
SEED=0


# ============================================================
# Persistent Experiment / Hydra Run Directory
#
# IMPORTANT:
# This is the SAME directory for every resume.
#
# Everything for Seed 0 will use:
#
# .../seed0/
# ============================================================

RUN_DIR="$PROJECT_DIR/exp/$EXPERIMENT/$ENV_NAME/inter4000_maxfeed500_seg20_Rlr0.0003_Rbatch20_Rupdate10/seed${SEED}"

mkdir -p "$RUN_DIR"


# ============================================================
# Checkpoint / Resume
# ============================================================

export CORE_CHECKPOINT_INTERVAL=10000
export CORE_CHECKPOINT_KEEP=2

export CORE_CHECKPOINT_DIR="$PROJECT_DIR/resume_checkpoints/$EXPERIMENT/$ENV_NAME/seed$SEED"

mkdir -p "$CORE_CHECKPOINT_DIR"


# ============================================================
# Replay Buffer Disk Storage
#
# Replay images are stored on disk instead of filling RAM.
# ============================================================

export REPLAY_MEMMAP_DIR="$PROJECT_DIR/runtime_memmap/${EXPERIMENT}_seed${SEED}"

mkdir -p "$REPLAY_MEMMAP_DIR"


# ============================================================
# Remove Old Temporary LIVE Replay Memmaps
#
# This does NOT delete checkpoint image snapshots.
# ============================================================

find "$REPLAY_MEMMAP_DIR" \
    -maxdepth 1 \
    -type f \
    -name "core_replay_images_*.dat" \
    -delete 2>/dev/null || true


# ============================================================
# Information
# ============================================================

echo ""
echo "============================================================"
echo " CoRe + FRM + RRM + PRM"
echo " Sweep Into - MetaWorld"
echo "============================================================"
echo ""

echo "Project:"
echo "$PROJECT_DIR"
echo ""

echo "Environment:"
echo "$ENV_NAME"
echo ""

echo "Experiment:"
echo "$EXPERIMENT"
echo ""

echo "Seed:"
echo "$SEED"
echo ""

echo "VLM:"
echo "$VLLM_MODEL"
echo ""

echo "Persistent experiment directory:"
echo "$RUN_DIR"
echo ""

echo "Checkpoint directory:"
echo "$CORE_CHECKPOINT_DIR"
echo ""

echo "Replay memmap directory:"
echo "$REPLAY_MEMMAP_DIR"
echo ""

echo "============================================================"
echo ""


# ============================================================
# Check GPU
# ============================================================

echo "Checking GPU..."
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free \
    --format=csv,noheader

echo ""


# ============================================================
# Check NVIDIA EGL Vendor File
# ============================================================

echo "Checking NVIDIA EGL vendor file..."
echo ""

if [ ! -f "$__EGL_VENDOR_LIBRARY_FILENAMES" ]; then

    echo "ERROR:"
    echo "NVIDIA EGL vendor file not found:"
    echo "$__EGL_VENDOR_LIBRARY_FILENAMES"

    exit 1
fi

echo "SUCCESS:"
echo "$__EGL_VENDOR_LIBRARY_FILENAMES"
echo ""


# ============================================================
# Test MuJoCo EGL
#
# This catches EGL problems BEFORE starting 250K training.
# ============================================================

echo "Testing MuJoCo EGL..."
echo ""

python - <<'PY'
import os

print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))

print(
    "__EGL_VENDOR_LIBRARY_FILENAMES =",
    os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES")
)

import mujoco

print("SUCCESS: MuJoCo imported.")

from mujoco import GLContext

ctx = GLContext(64, 64)
ctx.make_current()

print("SUCCESS: MuJoCo EGL context created.")
PY

echo ""
echo "MuJoCo EGL check passed."
echo ""


# ============================================================
# Check Python Files
# ============================================================

echo "Checking Python syntax..."
echo ""

python -m py_compile train_CoRe.py
python -m py_compile replay_buffer.py
python -m py_compile RRM/reward_model.py
python -m py_compile PRM/progress_model.py
python -m py_compile FRM/reward_code.py

echo ""
echo "SUCCESS: Python syntax checks passed."
echo ""


# ============================================================
# Check Existing Checkpoint
# ============================================================

if [ -f "$CORE_CHECKPOINT_DIR/latest.txt" ]; then

    echo "============================================================"
    echo " Existing checkpoint found."
    echo " Training will automatically RESUME."
    echo "============================================================"
    echo ""

    echo "Latest checkpoint:"
    cat "$CORE_CHECKPOINT_DIR/latest.txt"

    echo ""

else

    echo "============================================================"
    echo " No previous checkpoint found."
    echo " Training will start from step 0."
    echo "============================================================"
    echo ""

fi


# ============================================================
# Check Persistent Run Directory
# ============================================================

echo "============================================================"
echo " PERSISTENT RUN DIRECTORY"
echo "============================================================"
echo ""
echo "$RUN_DIR"
echo ""

if [ -d "$RUN_DIR" ]; then
    echo "Run directory ready."
fi

echo ""
echo "All future resumes of Seed $SEED will use this same folder."
echo ""
echo "============================================================"
echo ""


# ============================================================
# Sweep Into Training Information
#
# MetaWorld / CoRe-style settings:
#
# Train Steps             = 250000
# Random / Seed Steps     = 1000
# Initial FR Steps        = 19000
# First RRM Learning      = 20000
# VLM Query Frequency     = 4000
# Queries Per Session     = 20
# Segment Length          = 20
# Max Feedback            = 500
# FRM Update Frequency    = 25000
#
# PRM is our extension.
# ============================================================

echo "============================================================"
echo " SWEEP INTO TRAINING CONFIGURATION"
echo "============================================================"
echo " Task                 : Sweep Into"
echo " Environment          : $ENV_NAME"
echo " Train Steps          : 250000"
echo " Replay Capacity      : 10000"
echo " Seed Steps           : 1000"
echo " Pre Steps            : 19000"
echo " First RRM Step       : 20000"
echo " RRM Query Interval   : 4000"
echo " Queries / Session    : 20"
echo " Segment              : 20"
echo " Max Feedback         : 500"
echo " RRM Learning Rate    : 3e-4"
echo " RRM Batch            : 20"
echo " RRM Updates          : 10"
echo " Image Size           : 300"
echo " Reward Alpha         : 0.5"
echo " PRM Beta             : 0.1"
echo " Evaluation Frequency : 5000"
echo " Eval Episodes        : 10"
echo " FRM Alignment        : every 25000"
echo " FRM Alignment End    : 100000"
echo " Checkpoint Interval  : 10000"
echo " Checkpoints Kept     : 2"
echo ""
echo " Persistent Run Dir:"
echo " $RUN_DIR"
echo "============================================================"
echo ""


# ============================================================
# Start Training
#
# IMPORTANT:
# hydra.run.dir forces every restart/resume of this seed
# to use the SAME experiment folder.
# ============================================================

python train_CoRe.py \
    hydra.run.dir="$RUN_DIR" \
    env="$ENV_NAME" \
    agent=sac \
    seed="$SEED" \
    experiment="$EXPERIMENT" \
    num_train_steps=250000 \
    replay_buffer_capacity=10000 \
    num_seed_steps=1000 \
    num_pre_steps=19000 \
    num_interact=4000 \
    max_feedback=500 \
    segment=20 \
    ensemble_size=3 \
    RRM_lr=3e-4 \
    RRM_batch=20 \
    RRM_update=10 \
    resnet=0 \
    image_size=300 \
    reward_alpha=0.5 \
    eval_frequency=5000 \
    num_eval_episodes=10 \
    FRM_align_step=25000 \
    FRM_align_max_steps=100000 \
    use_FRM_online=False \
    use_PRM=true \
    PRM_beta=0.1 \
    PRM_gamma=0.99 \
    PRM_clip=0.5 \
    PRM_lr=1e-4 \
    PRM_batch=32 \
    PRM_update=5


# ============================================================
# Finished
# ============================================================

echo ""
echo "============================================================"
echo " Sweep Into training finished."
echo ""
echo "Persistent results directory:"
echo "$RUN_DIR"
echo ""
echo "Final checkpoints:"
echo "$CORE_CHECKPOINT_DIR"
echo "============================================================"
echo ""
