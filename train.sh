#!/bin/bash

# RookWorld GRPO Training Script with Optimized Parameters
# 
# This script contains the best hyperparameters discovered through comprehensive
# training stability improvements and hyperparameter search.
#
# Key improvements implemented:
# - KL warmup to prevent early training divergence
# - Graduated reward system with partial credit
# - Device placement fixes for multi-GPU setup
# - Reward normalization and smoothing
# - Higher KL divergence thresholds for tolerance

set -e  # Exit on any error

echo "🚀 Starting RookWorld GRPO Training with Optimized Parameters"
echo "============================================================"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed or not in PATH"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Training configuration
STEPS=${STEPS:-1000}
LR=${LR:-5e-6}
KL_COEF=${KL_COEF:-0.001}
CLIP_RANGE=${CLIP_RANGE:-0.1}
TEMPERATURE=${TEMPERATURE:-0.5}
BATCH_POSITIONS=${BATCH_POSITIONS:-4}
GROUP_SIZE=${GROUP_SIZE:-8}
MIX_ENV_RATIO=${MIX_ENV_RATIO:-0.2}  # Enable environment tasks for testing
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-100}  # Increased from 64 to allow complete policy analysis
MAX_NEW_TOKENS_ENV=${MAX_NEW_TOKENS_ENV:-150}  # Increased from 80 to prevent truncation

# Improved stability parameters
KL_DIVERGENCE_THRESHOLD=${KL_DIVERGENCE_THRESHOLD:-50.0}
KL_WARMUP_STEPS=${KL_WARMUP_STEPS:-500}
KL_WARMUP_FACTOR=${KL_WARMUP_FACTOR:-0.0}
REWARD_WARMUP_STEPS=${REWARD_WARMUP_STEPS:-100}

# Additional options
USE_DATASET=${USE_DATASET:-true}
NEW_RUN=${NEW_RUN:-true}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-false}

echo "Configuration:"
echo "  Steps: $STEPS"
echo "  Learning Rate: $LR" 
echo "  KL Coefficient: $KL_COEF"
echo "  Clip Range: $CLIP_RANGE"
echo "  Temperature: $TEMPERATURE"
echo "  Batch Positions: $BATCH_POSITIONS"
echo "  Group Size: $GROUP_SIZE"
echo "  Mix Env Ratio: $MIX_ENV_RATIO"
echo "  Max New Tokens: $MAX_NEW_TOKENS"
echo "  Max New Tokens Env: $MAX_NEW_TOKENS_ENV"
echo "  KL Divergence Threshold: $KL_DIVERGENCE_THRESHOLD"
echo "  KL Warmup Steps: $KL_WARMUP_STEPS"
echo "  KL Warmup Factor: $KL_WARMUP_FACTOR"
echo "  Reward Warmup Steps: $REWARD_WARMUP_STEPS"
echo "  Use Dataset: $USE_DATASET"
echo "  New Run: $NEW_RUN"
echo "  Use Torch Compile: $USE_TORCH_COMPILE"
echo "============================================================"

# Build command arguments
ARGS=(
    --steps "$STEPS"
    --lr "$LR"
    --kl-coef "$KL_COEF"
    --clip-range "$CLIP_RANGE"
    --temperature "$TEMPERATURE"
    --batch-positions "$BATCH_POSITIONS"
    --group-size "$GROUP_SIZE"
    --mix-env-ratio "$MIX_ENV_RATIO"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-new-tokens-env "$MAX_NEW_TOKENS_ENV"
    
    # Improved stability parameters
    --kl-divergence-threshold "$KL_DIVERGENCE_THRESHOLD"
    --kl-warmup-steps "$KL_WARMUP_STEPS"
    --kl-warmup-factor "$KL_WARMUP_FACTOR"
    --reward-warmup-steps "$REWARD_WARMUP_STEPS"
)

# Add conditional flags
if [ "$USE_DATASET" = "true" ]; then
    ARGS+=(--use-dataset)
fi

if [ "$NEW_RUN" = "true" ]; then
    ARGS+=(--new-run)
fi

if [ "$USE_TORCH_COMPILE" = "false" ]; then
    ARGS+=(--no-torch-compile)
fi

# Execute training with memory optimizations
echo "🏃 Executing training command..."
echo "uv run python train_rookworld_grpo.py ${ARGS[*]}"
echo "============================================================"

# Set CUDA memory allocator for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec uv run python train_rookworld_grpo.py "${ARGS[@]}"