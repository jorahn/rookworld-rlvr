#!/bin/bash

# Quick test of the mini implementation training script
echo "🧪 Testing mini implementation training with small parameters..."
echo "This will run 5 steps to verify the setup is working."
echo ""

# Small test parameters
STEPS=5 \
BATCH_SIZE=4 \
K_SAMPLES=4 \
N_TRAIN_SAMPLES=50 \
EVAL_FREQ=2 \
SAVE_FREQ=10 \
LOG_DIR="logs_test" \
./train.sh