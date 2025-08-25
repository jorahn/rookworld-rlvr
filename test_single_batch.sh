#!/bin/bash

# Single Batch Training Test Script
# 
# This script runs the single batch training test with the exact configuration
# requested: 2 samples (1 policy, 1 environment), group size 2, 1 step.
# 
# It logs comprehensive information for every component:
# - Prompt, completion, expected completion
# - Format validation results
# - Individual reward components  
# - Total reward and loss

set -e  # Exit on any error

echo "🧪 Single Batch Training Test"
echo "=============================================="
echo "Configuration:"
echo "  Samples: 2 (1 policy P:, 1 environment A:)"
echo "  Group size: 2"
echo "  Steps: 1" 
echo "  Comprehensive logging: enabled"
echo "=============================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed or not in PATH"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Falling back to direct python execution..."
    python run_single_batch_test.py
else
    echo "✅ Using uv for dependency management"
    echo "🏃 Executing single batch test..."
    uv run python run_single_batch_test.py
fi

echo "=============================================="
echo "✅ Single batch test completed!"
echo "Check the output above for detailed sample logs"