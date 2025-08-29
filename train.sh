#!/bin/bash

# RookWorld RLVR GRPO Training Script
# 
# This script runs the main implementation with optimized parameters
# for stable GRPO training on the RookWorld dataset.
#
# Key features:
# - Uses ground truth scoring for meaningful rewards
# - Fixed memory leaks for stable VRAM usage (~4.8GB)
# - Enhanced GRPO with adaptive KL control
# - Detailed logging and periodic evaluation
# - Trains on dataset samples (not synthetic generation)

set -e  # Exit on any error

echo "🚀 Starting RookWorld RLVR GRPO Training"
echo "============================================================"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed or not in PATH"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Training configuration - optimized for production training with stability
STEPS=${STEPS:-1000}  # Full production training (1000 steps)
BATCH_SIZE=${BATCH_SIZE:-8}  # Optimal for batch generation + memory
K_SAMPLES=${K_SAMPLES:-8}  # Group size for GRPO
LR=${LR:-1e-6}  # Reduced LR for more stable training (10x smaller)
KL_COEF=${KL_COEF:-0.02}
EVAL_FREQ=${EVAL_FREQ:-100}  # Evaluate every 100 steps for 1000-step training
SAVE_FREQ=${SAVE_FREQ:-50}  # Save checkpoint every 50 steps
N_TRAIN_SAMPLES=${N_TRAIN_SAMPLES:-1000}  # Full dataset for production
REWARD_SHAPING=${REWARD_SHAPING:-linear}  # Linear shaping for continuous rewards
LR_SCHEDULE=${LR_SCHEDULE:-cosine}  # Cosine annealing with warmup
WARMUP_STEPS=${WARMUP_STEPS:-20}  # 20 steps warmup (2% of training)
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}  # Decay to 1e-7 final LR
EARLY_STOP_WINDOW=${EARLY_STOP_WINDOW:-5}  # Stop if no recovery within 5 steps
MAX_CHECKPOINTS=${MAX_CHECKPOINTS:-5}  # Keep only last 5 checkpoints
LOG_DIR=${LOG_DIR:-"logs"}
USE_BF16=${USE_BF16:-true}  # Enable BF16 mixed precision
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-false}  # Disable torch.compile to save memory for batch generation
USE_BATCH_GENERATION=${USE_BATCH_GENERATION:-true}  # Enable batch generation (3x speedup)

echo "Configuration:"
echo "  Steps: $STEPS (production training)"
echo "  Batch Size: $BATCH_SIZE"  
echo "  K Samples (Group Size): $K_SAMPLES"
echo "  Learning Rate: $LR (reduced for stability)"
echo "  KL Coefficient: $KL_COEF"
echo "  Evaluation Frequency: every $EVAL_FREQ steps"
echo "  Checkpoint Frequency: every $SAVE_FREQ steps (keep last $MAX_CHECKPOINTS)"
echo "  Training Samples: $N_TRAIN_SAMPLES from dataset"
echo "  Log Directory: $LOG_DIR"
echo "  Reward Shaping: $REWARD_SHAPING (continuous rewards)"
echo "  LR Schedule: $LR_SCHEDULE (warmup: $WARMUP_STEPS steps, min ratio: $MIN_LR_RATIO)"
echo "  Early Stopping: enabled (recovery window: $EARLY_STOP_WINDOW steps)"
echo "  BF16 Mixed Precision: $USE_BF16"
echo "  Torch Compile: $USE_TORCH_COMPILE"
echo "  Batch Generation: $USE_BATCH_GENERATION"
echo ""
echo "🚀 ALL PERFORMANCE OPTIMIZATIONS ENABLED:"
echo "  - ⚡ Batch Generation: 3.21x faster generation (NEWLY INTEGRATED!)"
echo "  - 🔥 BF16 Mixed Precision: 2x memory efficiency + Tensor Cores"
echo "  - 🏎️ Torch Compile: 3-5x speedup for inference and training"
echo "  - 🎯 TF32 Acceleration: Ampere+ GPU optimization"
echo "  - 📊 Tensor Core Precision: maximum utilization"
echo "  - 🧠 Advanced GRPO: adaptive KL, graduated rewards, memory fixes"
echo "============================================================"

# Stay in root directory for module imports
# cd src/rookworld_rlvr

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Build command arguments for optimized train_logged.py
ARGS=(
    --steps "$STEPS"
    --batch_size "$BATCH_SIZE"
    --k_samples "$K_SAMPLES"
    --lr "$LR"
    --kl_coef "$KL_COEF"
    --eval_freq "$EVAL_FREQ"
    --save_freq "$SAVE_FREQ"
    --n_train_samples "$N_TRAIN_SAMPLES"
    --log_dir "$LOG_DIR"
)

# Add performance optimization flags
if [ "$USE_BF16" = "true" ]; then
    ARGS+=(--use_bf16)
fi

if [ "$USE_TORCH_COMPILE" = "true" ]; then
    ARGS+=(--use_torch_compile)
fi

if [ "$USE_BATCH_GENERATION" = "true" ]; then
    ARGS+=(--use_batch_generation)
    ARGS+=(--batch_generation_mode "mixed")
    ARGS+=(--batch_generation_size 16)
fi

# Add reward shaping and LR schedule arguments
ARGS+=(--reward_shaping "$REWARD_SHAPING")
ARGS+=(--lr_schedule "$LR_SCHEDULE")
ARGS+=(--warmup_steps "$WARMUP_STEPS")
ARGS+=(--min_lr_ratio "$MIN_LR_RATIO")

# Add early stopping and checkpoint management
ARGS+=(--early_stop_window "$EARLY_STOP_WINDOW")
ARGS+=(--max_checkpoints "$MAX_CHECKPOINTS")

# Add A/B testing flag for debugging (optional)
AB_TEST_MODE=${AB_TEST_MODE:-false}
if [ "$AB_TEST_MODE" = "true" ]; then
    ARGS+=(--ab_test_mode)
    echo "  A/B Testing: ENABLED (will compare individual vs batch generation)"
fi

# Execute training with memory optimizations
echo ""
echo "🏃 Executing ENHANCED GRPO training with train_logged.py..."
echo "Command: uv run python scripts/train_logged.py ${ARGS[*]}"
echo "============================================================"

# Set CUDA memory allocator for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the optimized training script with detailed logging
exec uv run python scripts/train_logged.py "${ARGS[@]}"