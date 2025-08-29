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

# Training configuration - parameters supported by mini implementation
STEPS=${STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-8}
K_SAMPLES=${K_SAMPLES:-8}  # Group size for GRPO
LR=${LR:-1e-5}
KL_COEF=${KL_COEF:-0.02}
EVAL_FREQ=${EVAL_FREQ:-100}  # Evaluate every 100 steps
SAVE_FREQ=${SAVE_FREQ:-1000}  # Save checkpoint every 1000 steps
N_TRAIN_SAMPLES=${N_TRAIN_SAMPLES:-1000}  # Number of samples from dataset
LOG_DIR=${LOG_DIR:-"logs"}

echo "Configuration:"
echo "  Steps: $STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  K Samples (Group Size): $K_SAMPLES"
echo "  Learning Rate: $LR"
echo "  KL Coefficient: $KL_COEF"
echo "  Evaluation Frequency: every $EVAL_FREQ steps"
echo "  Checkpoint Frequency: every $SAVE_FREQ steps"
echo "  Training Samples: $N_TRAIN_SAMPLES from dataset"
echo "  Log Directory: $LOG_DIR"
echo ""
echo "Built-in features (src/rookworld_rlvr/config.py):"
echo "  - Temperature: 0.8"
echo "  - Clip Range: 0.2"
echo "  - Max New Tokens: 144"
echo "  - KL Type: forward"
echo "  - Adaptive KL: enabled (target=0.01)"
echo "  - Baseline: group_mean"
echo "  - GAE: enabled (lambda=0.95)"
echo "  - Reward: graduated with ground truth scoring"
echo "  - Continuous: FEN similarity (exponential), evaluations (linear)"
echo "  - Memory: Fixed leaks, stable ~4.8GB VRAM"
echo "============================================================"

# Stay in root directory for module imports
# cd src/rookworld_rlvr

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Build command arguments - parameters supported by train_logged.py
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

# Execute training with memory optimizations
echo ""
echo "🏃 Executing GRPO training..."
echo "Command: uv run python scripts/train_logged.py ${ARGS[*]}"
echo "============================================================"

# Set CUDA memory allocator for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training script with detailed logging
exec uv run python scripts/train_logged.py "${ARGS[@]}"