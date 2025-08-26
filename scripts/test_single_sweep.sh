#!/bin/bash

# Test single hyperparameter sweep run to diagnose OOM issues

set -e

echo "🧪 Testing Single Hyperparameter Sweep Run"
echo "============================================"

# Base configuration from hyperparameter_sweep.sh
BASE_STEPS=100  # Shorter for testing
BASE_BATCH_POSITIONS=2
BASE_GROUP_SIZE=4
BASE_CLIP_RANGE=0.1
BASE_TEMPERATURE=0.5
BASE_MIX_ENV_RATIO=0.2
BASE_MAX_NEW_TOKENS=144
BASE_MAX_NEW_TOKENS_ENV=150
BASE_KL_DIVERGENCE_THRESHOLD=50.0
BASE_REWARD_WARMUP_STEPS=100

# Single parameter combination
kl_warmup_steps=50
kl_warmup_factor=0.0
lr=1e-6
kl_coef=0.0005

echo "Parameters:"
echo "  KL Warmup Steps: $kl_warmup_steps"
echo "  KL Warmup Factor: $kl_warmup_factor"
echo "  Learning Rate: $lr"
echo "  KL Coefficient: $kl_coef"
echo "  Effective Batch Size: $((BASE_BATCH_POSITIONS * BASE_GROUP_SIZE))"
echo "============================================"

# Run with same environment as hyperparameter sweep
timeout 300 env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    STEPS="$BASE_STEPS" \
    BATCH_POSITIONS="$BASE_BATCH_POSITIONS" \
    GROUP_SIZE="$BASE_GROUP_SIZE" \
    CLIP_RANGE="$BASE_CLIP_RANGE" \
    TEMPERATURE="$BASE_TEMPERATURE" \
    MIX_ENV_RATIO="$BASE_MIX_ENV_RATIO" \
    MAX_NEW_TOKENS="$BASE_MAX_NEW_TOKENS" \
    MAX_NEW_TOKENS_ENV="$BASE_MAX_NEW_TOKENS_ENV" \
    KL_DIVERGENCE_THRESHOLD="$BASE_KL_DIVERGENCE_THRESHOLD" \
    REWARD_WARMUP_STEPS="$BASE_REWARD_WARMUP_STEPS" \
    EVAL_EVERY="9999" \
    KL_WARMUP_STEPS="$kl_warmup_steps" \
    KL_WARMUP_FACTOR="$kl_warmup_factor" \
    LR="$lr" \
    KL_COEF="$kl_coef" \
    ./train.sh

echo "Test completed successfully!"