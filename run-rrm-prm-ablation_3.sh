#!/bin/bash
set -e


# ============================================================
# CoRe + RRM + PRM (Clean FRM Ablation)
#
# Usage:
#
# bash run-rrm-prm-ablation-full.sh water 0
# bash run-rrm-prm-ablation-full.sh water 1
# bash run-rrm-prm-ablation-full.sh water 2
#
# bash run-rrm-prm-ablation-full.sh peg 0
# bash run-rrm-prm-ablation-full.sh hammer 0
#
# Tasks:
#   soccer
#   sweep
#   drawer
#   button
#   dial
#   hammer
#   peg
#   cloth
#   rope
#   water
#
# Seeds:
#   0
#   1
#   2
# ============================================================



# ============================================================
# INPUT
# ============================================================

TASK="${1:-}"

SEED="${2:-0}"


if [ -z "$TASK" ]; then

    echo ""
    echo "ERROR: No task supplied."
    echo ""
    echo "Example:"
    echo ""
    echo "bash run-rrm-prm-ablation-full.sh water 0"
    echo ""
    echo "Available tasks:"
    echo ""
    echo "soccer sweep drawer button dial hammer peg cloth rope water"
    echo ""

    exit 1

fi


if [[ "$SEED" != "0" && "$SEED" != "1" && "$SEED" != "2" ]]; then

    echo ""
    echo "ERROR: Seed must be 0, 1, or 2."
    echo ""

    exit 1

fi



# ============================================================
# PROJECT
# ============================================================

PROJECT_DIR="/home/ssa393/Milestone-4/jev_reward"

cd "$PROJECT_DIR"



# ============================================================
# CHECK CONDA ENVIRONMENT
# ============================================================

echo ""
echo "Current Conda environment: ${CONDA_DEFAULT_ENV:-not activated}"
echo ""


if [ "${CONDA_DEFAULT_ENV:-}" != "core" ]; then

    echo ""
    echo "ERROR:"
    echo ""
    echo "Activate the core environment first:"
    echo ""
    echo "conda activate core"
    echo "cd $PROJECT_DIR"
    echo ""

    exit 1

fi



# ============================================================
# GPU
# ============================================================

export CUDA_VISIBLE_DEVICES=0

export gpu_id=0


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
# LOCAL QWEN3.5 / vLLM
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

export MY_API_KEY="EMPTY"



# ============================================================
# HYDRA
# ============================================================

export HYDRA_FULL_ERROR=1



# ============================================================
# DEFAULT METAWORLD CoRe SETTINGS
# ============================================================

PLATFORM="metaworld"


AGENT="sac"


SEGMENT=20


# Original CoRe MetaWorld = 500000
# Your MetaWorld experiment = 250000

TRAIN_STEPS=500000


SEED_STEPS=1000


PRE_STEPS=19000


INTERACT=4000


MAX_FEEDBACK=500


RRM_LR="0.0003"


RRM_BATCH=20


RRM_UPDATE=10


RESNET=0


IMAGE_SIZE=300


# Original CoRe evaluation configuration

EVAL_FREQUENCY=25000


NUM_EVAL_EPISODES=10


FRM_ALIGN_STEP=25000


FRM_ALIGN_MAX=100000



# ============================================================
# TASK CONFIGURATION
# ============================================================

case "$TASK" in



    # ========================================================
    # 1. SOCCER
    # ========================================================

    soccer)

        LABEL="Soccer"

        ENV_NAME="metaworld_soccer-v2"

        EXPERIMENT="qwen35_rrm_prm_clean_ablation_soccer_500k"

        ;;



    # ========================================================
    # 2. SWEEP INTO
    # ========================================================

    sweep)

        LABEL="Sweep Into"

        ENV_NAME="metaworld_sweep-into-v2"

        EXPERIMENT="qwen35_rrm_prm_clean_ablation_sweep_into_500k"

        ;;



    # ========================================================
    # 3. DRAWER OPEN
    # ========================================================

    drawer)

        LABEL="Drawer Open"

        ENV_NAME="metaworld_drawer-open-v2"

        EXPERIMENT="qwen35_rrm_prm_clean_ablation_drawer_open_500k"

        ;;



    # ========================================================
    # 4. BUTTON PRESS
    # ========================================================

    button)

        LABEL="Button Press"

        ENV_NAME="metaworld_button-press-topdown-v2"

        EXPERIMENT="qwen35_rrm_prm_clean_ablation_button_press_500k"

        ;;



    # ========================================================
    # 5. DIAL TURN
    # ========================================================

    dial)

        LABEL="Dial Turn"

        ENV_NAME="metaworld_dial-turn-v2"

        EXPERIMENT="qwen35_rrm_prm_clean_ablation_dial_turn_500k"

        ;;



    # ========================================================
    # 6. HAMMER
    # ========================================================

    hammer)

        LABEL="Hammer"

        ENV_NAME="metaworld_hammer-v2"

        EXPERIMENT="qwen35_rrm_prm_clean_ablation_hammer_500k"

        ;;



    # ========================================================
    # 7. PEG INSERT
    # ========================================================

    peg)

        LABEL="Peg Insert"

        ENV_NAME="metaworld_peg-insert-side-v2"

        EXPERIMENT="qwen35_rrm_prm_clean_ablation_peg_insert_500k"

        ;;



    # ========================================================
    # 8. FOLD CLOTH
    #
    # Original CoRe:
    # 15,000 steps
    # ========================================================

    cloth)

        LABEL="Fold Cloth"


        PLATFORM="softgym"


        ENV_NAME="softgym_ClothFoldDiagonal"


        EXPERIMENT="qwen35_rrm_prm_clean_ablation_cloth_15k"


        AGENT="sac_cloth"


        SEGMENT=3


        TRAIN_STEPS=15000


        SEED_STEPS=250


        PRE_STEPS=250


        INTERACT=1000


        MAX_FEEDBACK=150


        RRM_LR="0.0003"


        RRM_BATCH=10


        RRM_UPDATE=25


        RESNET=1


        IMAGE_SIZE=360


        EVAL_FREQUENCY=250


        FRM_ALIGN_STEP=2400


        FRM_ALIGN_MAX=9600

        ;;



    # ========================================================
    # 9. STRAIGHTEN ROPE
    #
    # CoRe:
    # approximately 100K steps
    # ========================================================

    rope)

        LABEL="Straighten Rope"


        PLATFORM="softgym"


        ENV_NAME="softgym_RopeFlattenEasy"


        EXPERIMENT="qwen35_rrm_prm_clean_ablation_rope_101k"


        AGENT="sac"


        SEGMENT=3


        TRAIN_STEPS=101000


        SEED_STEPS=1000


        PRE_STEPS=9000


        INTERACT=5000


        MAX_FEEDBACK=200


        RRM_LR="0.0001"


        RRM_BATCH=10


        RRM_UPDATE=30


        RESNET=1


        IMAGE_SIZE=240


        # Keep CoRe-style evaluation interval

        EVAL_FREQUENCY=10000


        FRM_ALIGN_STEP=25000


        FRM_ALIGN_MAX=100000

        ;;



    # ========================================================
    # 10. PASS WATER
    #
    # Original CoRe:
    #
    # Train Steps       = 100050
    # VLM Query         = 5000
    # Max Feedback      = 200
    # Evaluation        = 10000
    #
    # Evaluation condition is fixed below so that a 10K
    # boundary is not missed simply because the episode
    # finishes slightly after the exact step.
    # ========================================================

    water)

        LABEL="Pass Water"


        PLATFORM="softgym"


        ENV_NAME="softgym_PassWater"


        EXPERIMENT="qwen35_rrm_prm_clean_ablation_pass_water_100k"


        AGENT="sac"


        SEGMENT=3


        TRAIN_STEPS=100050


        SEED_STEPS=1000


        PRE_STEPS=9000


        INTERACT=5000


        MAX_FEEDBACK=200


        RRM_LR="0.0001"


        RRM_BATCH=10


        RRM_UPDATE=30


        RESNET=1


        IMAGE_SIZE=360


        # Keep original CoRe-style evaluation frequency

        EVAL_FREQUENCY=10000


        FRM_ALIGN_STEP=24975


        FRM_ALIGN_MAX=100000

        ;;



    *)

        echo ""
        echo "ERROR: Unknown task: $TASK"
        echo ""
        echo "Available tasks:"
        echo ""
        echo "soccer"
        echo "sweep"
        echo "drawer"
        echo "button"
        echo "dial"
        echo "hammer"
        echo "peg"
        echo "cloth"
        echo "rope"
        echo "water"
        echo ""

        exit 1

        ;;

esac


# ============================================================
# DISABLE FRM ALIGNMENT / ITERATION FOR RRM + PRM ABLATION
#
# train_CoRe.py enters the FRM alignment block only while:
#     self.step <= FRM_align_max_steps
#
# With -1, this condition is false from step 0 onward.
# This prevents FRM_generate() and FRM replay relabeling.
# ============================================================

FRM_ALIGN_MAX=-1


# ============================================================
# HEADLESS EGL
# ============================================================

export __EGL_VENDOR_LIBRARY_FILENAMES="$PROJECT_DIR/egl_vendor/10_nvidia.json"


unset DISPLAY



# ============================================================
# METAWORLD SETUP
# ============================================================

if [ "$PLATFORM" = "metaworld" ]; then


    export MUJOCO_GL=egl


    export PYOPENGL_PLATFORM=egl


    export MUJOCO_EGL_DEVICE_ID=0


    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/metaworld:${PYTHONPATH:-}"


    export LD_LIBRARY_PATH="/.singularity.d/libs:/usr/local/cuda/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"


fi



# ============================================================
# SOFTGYM SETUP
# ============================================================

if [ "$PLATFORM" = "softgym" ]; then


    export PYFLEXROOT="$PROJECT_DIR/softgym/PyFlex"


    export PYTHONPATH="$PYFLEXROOT/bindings/build:$PROJECT_DIR/softgym:$PROJECT_DIR:$PROJECT_DIR/metaworld:${PYTHONPATH:-}"


    export LD_LIBRARY_PATH="/.singularity.d/libs:/usr/local/cuda/lib64:$PYFLEXROOT/external/SDL2-2.0.4/lib/x64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"


fi



# ============================================================
# PERSISTENT EXP DIRECTORY
#
# No timestamp.
#
# Each task/seed keeps using the same directory.
#
# seed0
# seed1
# seed2
#
# This allows resume after the HPC time limit.
# ============================================================

RUN_DIR="$PROJECT_DIR/exp/$EXPERIMENT/$ENV_NAME/inter${INTERACT}_maxfeed${MAX_FEEDBACK}_seg${SEGMENT}_Rlr${RRM_LR}_Rbatch${RRM_BATCH}_Rupdate${RRM_UPDATE}/seed${SEED}"


mkdir -p "$RUN_DIR"



# ============================================================
# CHECKPOINT / RESUME
# ============================================================

export CORE_CHECKPOINT_INTERVAL=25000


export CORE_CHECKPOINT_KEEP=2


export CORE_CHECKPOINT_DIR="$PROJECT_DIR/resume_checkpoints/$EXPERIMENT/$ENV_NAME/seed${SEED}"


mkdir -p "$CORE_CHECKPOINT_DIR"



# ============================================================
# DISK-BACKED REPLAY BUFFER
# ============================================================

export REPLAY_MEMMAP_DIR="$PROJECT_DIR/runtime_memmap/${EXPERIMENT}_replay_seed${SEED}"


mkdir -p "$REPLAY_MEMMAP_DIR"



# ============================================================
# DISK-BACKED RRM IMAGE BUFFER
# ============================================================

export RRM_MEMMAP_DIR="$PROJECT_DIR/runtime_memmap/${EXPERIMENT}_rrm_seed${SEED}"


mkdir -p "$RRM_MEMMAP_DIR"



# ============================================================
# INFORMATION
# ============================================================

echo ""
echo "============================================================"
echo " CoRe + RRM + PRM (Clean FRM Ablation)"
echo "============================================================"
echo ""

echo "Task                 : $LABEL"
echo "Environment          : $ENV_NAME"
echo "Platform             : $PLATFORM"
echo "Experiment           : $EXPERIMENT"
echo "Ablation             : RRM + PRM only (FRM reward = 0; FRM iteration disabled)"
echo "Seed                 : $SEED"

echo ""

echo "Train Steps          : $TRAIN_STEPS"
echo "Seed Steps           : $SEED_STEPS"
echo "Pre Steps            : $PRE_STEPS"

echo ""

echo "VLM Query Frequency  : $INTERACT"
echo "Segment Length       : $SEGMENT"
echo "Max Feedback         : $MAX_FEEDBACK"

echo ""

echo "RRM Learning Rate    : $RRM_LR"
echo "RRM Batch            : $RRM_BATCH"
echo "RRM Updates          : $RRM_UPDATE"

echo ""

echo "FRM Reward Weight    : 0.0"
echo "FRM Online           : false"
echo "FRM Alignment        : disabled"
echo "FRM Alignment Max    : $FRM_ALIGN_MAX"

echo ""

echo "RRM Reward Weight    : 1.0"
echo "PRM Enabled          : true"
echo "PRM Beta             : 0.1"
echo "PRM Gamma            : 0.99"

echo ""

echo "Evaluation Frequency : $EVAL_FREQUENCY"
echo "Evaluation Episodes  : $NUM_EVAL_EPISODES"

echo ""

echo "Checkpoint Interval  : $CORE_CHECKPOINT_INTERVAL"

echo ""

echo "VLM                  : $VLLM_MODEL"

echo ""

echo "Run directory:"
echo "$RUN_DIR"

echo ""

echo "Checkpoint directory:"
echo "$CORE_CHECKPOINT_DIR"

echo ""

echo "Replay memmap:"
echo "$REPLAY_MEMMAP_DIR"

echo ""

echo "RRM memmap:"
echo "$RRM_MEMMAP_DIR"

echo ""

echo "============================================================"
echo ""



# ============================================================
# GPU CHECK
# ============================================================

echo "GPU:"
echo ""


nvidia-smi \
    --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader


echo ""



# ============================================================
# EGL CHECK
# ============================================================

if [ ! -f "$__EGL_VENDOR_LIBRARY_FILENAMES" ]; then

    echo ""
    echo "ERROR:"
    echo ""
    echo "NVIDIA EGL vendor file not found:"
    echo ""
    echo "$__EGL_VENDOR_LIBRARY_FILENAMES"
    echo ""

    exit 1

fi


# ============================================================
# PYTHON SYNTAX CHECK
# ============================================================

echo ""
echo "Checking Python files..."
echo ""


python -m py_compile train_CoRe.py


python -m py_compile replay_buffer.py


python -m py_compile RRM/reward_model.py


python -m py_compile PRM/progress_model.py


python -m py_compile FRM/reward_code.py


echo ""
echo "Python syntax OK."
echo ""



# ============================================================
# CHECK FOR EXISTING CHECKPOINT
# ============================================================

if [ -f "$CORE_CHECKPOINT_DIR/latest.txt" ]; then

    echo ""
    echo "============================================================"
    echo " EXISTING CHECKPOINT FOUND"
    echo "============================================================"
    echo ""

    echo "Training will RESUME."

    echo ""

    echo "Latest checkpoint:"

    echo ""

    cat "$CORE_CHECKPOINT_DIR/latest.txt"

    echo ""
    echo ""


else

    echo ""
    echo "============================================================"
    echo " NO CHECKPOINT FOUND"
    echo "============================================================"
    echo ""

    echo "Starting from step 0."

    echo ""

    echo "Task : $LABEL"

    echo "Seed : $SEED"

    echo ""

fi



# ============================================================
# START TRAINING
#
# save_model_step is intentionally NOT passed.
#
# Your custom checkpoint system handles checkpoint/resume.
#
# CORE_CHECKPOINT_INTERVAL=25000
# ============================================================

echo ""
echo "============================================================"
echo " STARTING TRAINING"
echo "============================================================"
echo ""

echo "Task              : $LABEL"

echo "Seed              : $SEED"

echo "Environment Steps : $TRAIN_STEPS"

echo "Eval Frequency    : $EVAL_FREQUENCY"

echo "Eval Episodes     : $NUM_EVAL_EPISODES"

echo ""



python train_CoRe.py \
    hydra.run.dir="$RUN_DIR" \
    env="$ENV_NAME" \
    agent="$AGENT" \
    seed="$SEED" \
    experiment="$EXPERIMENT" \
    segment="$SEGMENT" \
    num_train_steps="$TRAIN_STEPS" \
    num_seed_steps="$SEED_STEPS" \
    num_pre_steps="$PRE_STEPS" \
    num_interact="$INTERACT" \
    max_feedback="$MAX_FEEDBACK" \
    RRM_lr="$RRM_LR" \
    RRM_batch="$RRM_BATCH" \
    RRM_update="$RRM_UPDATE" \
    resnet="$RESNET" \
    image_size="$IMAGE_SIZE" \
    ensemble_size=3 \
    reward_alpha=1.0 \
    eval_frequency="$EVAL_FREQUENCY" \
    num_eval_episodes="$NUM_EVAL_EPISODES" \
    save_eval_video=False \
    FRM_align_step="$FRM_ALIGN_STEP" \
    FRM_align_max_steps="$FRM_ALIGN_MAX" \
    use_FRM_online=False \
    use_PRM=true \
    PRM_beta=0.1 \
    PRM_gamma=0.99 \
    PRM_clip=0.5 \
    PRM_lr=1e-4 \
    PRM_batch=32 \
    PRM_update=5



# ============================================================
# FINISHED
# ============================================================

echo ""
echo "============================================================"
echo " TRAINING FINISHED"
echo "============================================================"
echo ""

echo "Task:"
echo "$LABEL"

echo ""

echo "Seed:"
echo "$SEED"

echo ""

echo "Results:"
echo "$RUN_DIR"

echo ""

echo "Checkpoint:"
echo "$CORE_CHECKPOINT_DIR"

echo ""

echo "============================================================"
echo ""
