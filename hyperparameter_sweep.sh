#!/bin/bash

# RookWorld GRPO Hyperparameter Sweep for KL Stability
# 
# This script performs a sparse, wide grid search on the most impactful parameters
# for preventing KL divergence instability in GRPO training.
#
# Target: Find optimal balance between exploration and stability for 500-step runs

set -e

# Handle Ctrl-C gracefully to exit sweep instead of continuing to next test
trap 'echo ""; echo "❌ Hyperparameter sweep interrupted by user"; exit 1' INT

echo "🔬 Starting RookWorld GRPO Hyperparameter Sweep"
echo "============================================================"
echo "Target: KL Divergence Stability Optimization"
echo "Run Length: 500 steps per experiment"
echo "Focus: Exploration vs Stability Balance"
echo "============================================================"

# Check dependencies
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed or not in PATH"
    exit 1
fi

# Base configuration (fixed parameters)
BASE_STEPS=500
BASE_BATCH_POSITIONS=4
BASE_GROUP_SIZE=8
BASE_CLIP_RANGE=0.1
BASE_TEMPERATURE=0.5
BASE_MIX_ENV_RATIO=0.2
BASE_MAX_NEW_TOKENS=100
BASE_MAX_NEW_TOKENS_ENV=150
BASE_KL_DIVERGENCE_THRESHOLD=50.0
BASE_REWARD_WARMUP_STEPS=100

# Sparse, Wide Parameter Grid (4D grid = 4×4×3×3 = 144 total combinations)

# 1. KL Warmup Steps - How long to delay KL penalties
KL_WARMUP_STEPS_VALUES=(50 150 300 500)  # Very short to very long warmup

# 2. KL Warmup Factor - KL penalty strength during warmup  
KL_WARMUP_FACTOR_VALUES=(0.0 0.1 0.3 0.5)  # None to half-strength penalty

# 3. Learning Rate - Update magnitude
LR_VALUES=(1e-6 3e-6 8e-6)  # Conservative to moderate

# 4. KL Coefficient - KL penalty strength after warmup
KL_COEF_VALUES=(0.0005 0.001 0.002)  # Weak to moderate penalty

# Results tracking
RESULTS_DIR="hyperparameter_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
RESULTS_CSV="$RESULTS_DIR/results.csv"

# Create CSV header
echo "run_id,kl_warmup_steps,kl_warmup_factor,learning_rate,kl_coefficient,status,steps_completed,final_kl_mean,final_kl_95pct,final_reward,training_time,diverged_at_step" > "$RESULTS_CSV"

echo "📊 Parameter Grid Configuration:"
echo "  Base Batch Size: $BASE_BATCH_POSITIONS positions × $BASE_GROUP_SIZE group = $((BASE_BATCH_POSITIONS * BASE_GROUP_SIZE)) effective batch size"
echo "  KL Warmup Steps: ${KL_WARMUP_STEPS_VALUES[*]}"
echo "  KL Warmup Factor: ${KL_WARMUP_FACTOR_VALUES[*]}"  
echo "  Learning Rate: ${LR_VALUES[*]}"
echo "  KL Coefficient: ${KL_COEF_VALUES[*]}"
echo "  Total Combinations: $((${#KL_WARMUP_STEPS_VALUES[@]} * ${#KL_WARMUP_FACTOR_VALUES[@]} * ${#LR_VALUES[@]} * ${#KL_COEF_VALUES[@]}))"
echo "  Results Directory: $RESULTS_DIR"
echo "============================================================"

# Counter for progress tracking
run_count=0
total_runs=$((${#KL_WARMUP_STEPS_VALUES[@]} * ${#KL_WARMUP_FACTOR_VALUES[@]} * ${#LR_VALUES[@]} * ${#KL_COEF_VALUES[@]}))

# Nested loops for grid search
for kl_warmup_steps in "${KL_WARMUP_STEPS_VALUES[@]}"; do
    for kl_warmup_factor in "${KL_WARMUP_FACTOR_VALUES[@]}"; do
        for lr in "${LR_VALUES[@]}"; do
            for kl_coef in "${KL_COEF_VALUES[@]}"; do
                run_count=$((run_count + 1))
                
                # Generate unique run ID
                run_id="sweep_$(date +%Y%m%d_%H%M%S)_${run_count}"
                
                echo ""
                echo "🚀 Run $run_count/$total_runs: $run_id"
                echo "   KL Warmup Steps: $kl_warmup_steps"
                echo "   KL Warmup Factor: $kl_warmup_factor"
                echo "   Learning Rate: $lr"  
                echo "   KL Coefficient: $kl_coef"
                echo "   ----------------------------------------"
                
                # Start time tracking
                start_time=$(date +%s)
                
                # Clear GPU memory before training
                echo "   🧹 Clearing GPU cache..."
                uv run python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                
                # Run training with current parameter combination
                # Capture output and extract metrics
                if timeout 1800 env \
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
                    KL_WARMUP_STEPS="$kl_warmup_steps" \
                    KL_WARMUP_FACTOR="$kl_warmup_factor" \
                    LR="$lr" \
                    KL_COEF="$kl_coef" \
                    ./train.sh > "$RESULTS_DIR/${run_id}.log" 2>&1; then
                    
                    # Training completed successfully
                    status="SUCCESS"
                    steps_completed="$BASE_STEPS"
                    diverged_at_step=""
                    
                    # Extract final metrics from log
                    final_kl_mean=$(grep "Final Learning Rate" "$RESULTS_DIR/${run_id}.log" -B 20 | grep -E "KL: [0-9.]+" | tail -1 | grep -oE "KL: [0-9.]+" | cut -d' ' -f2 || echo "N/A")
                    final_kl_95pct=$(grep "Final Learning Rate" "$RESULTS_DIR/${run_id}.log" -B 20 | grep -E "\(95%: [0-9.]+\)" | tail -1 | grep -oE "[0-9.]+" || echo "N/A")
                    final_reward=$(grep "Final Learning Rate" "$RESULTS_DIR/${run_id}.log" -B 20 | grep -E "Reward: [0-9.]+" | tail -1 | grep -oE "Reward: [0-9.]+" | cut -d' ' -f2 || echo "N/A")
                    
                    echo "   ✅ SUCCESS: Completed $BASE_STEPS steps"
                    echo "   📊 Final KL: $final_kl_mean (95%: $final_kl_95pct), Reward: $final_reward"
                    
                else
                    # Training failed or timed out
                    if grep -q "Training diverged" "$RESULTS_DIR/${run_id}.log"; then
                        status="DIVERGED"
                        
                        # Extract divergence step and final KL
                        divergence_line=$(grep "Training diverged" "$RESULTS_DIR/${run_id}.log" | head -1)
                        final_kl_mean=$(echo "$divergence_line" | grep -oE "mean=[0-9.]+" | cut -d'=' -f2 || echo "N/A")
                        final_kl_95pct="N/A"
                        final_reward="N/A"
                        
                        # Estimate steps completed from log
                        steps_completed=$(grep -E "Step\s+[0-9]+/[0-9]+" "$RESULTS_DIR/${run_id}.log" | tail -1 | grep -oE "Step\s+[0-9]+" | grep -oE "[0-9]+" || echo "N/A")
                        diverged_at_step="$steps_completed"
                        
                        echo "   💥 DIVERGED: KL=$final_kl_mean at step $diverged_at_step"
                    elif grep -q "CUDA out of memory" "$RESULTS_DIR/${run_id}.log"; then
                        status="OOM"
                        
                        # Extract last completed step from OOM logs
                        steps_completed=$(grep -E "Step\s+[0-9]+/[0-9]+" "$RESULTS_DIR/${run_id}.log" | tail -1 | grep -oE "Step\s+[0-9]+" | grep -oE "[0-9]+" || echo "N/A")
                        
                        # Extract last KL if available
                        final_kl_mean=$(grep "KL:" "$RESULTS_DIR/${run_id}.log" | tail -1 | grep -oE "KL: [0-9.]+" | cut -d' ' -f2 || echo "N/A")
                        final_kl_95pct="N/A"
                        final_reward="N/A"
                        diverged_at_step=""
                        
                        echo "   💾 OOM: GPU memory exhausted at step $steps_completed, KL=$final_kl_mean"
                    else
                        status="TIMEOUT"
                        
                        # Extract last completed step for actual timeouts
                        steps_completed=$(grep -E "Step\s+[0-9]+/[0-9]+" "$RESULTS_DIR/${run_id}.log" | tail -1 | grep -oE "Step\s+[0-9]+" | grep -oE "[0-9]+" || echo "N/A")
                        
                        # Extract last KL if available
                        final_kl_mean=$(grep "KL:" "$RESULTS_DIR/${run_id}.log" | tail -1 | grep -oE "KL: [0-9.]+" | cut -d' ' -f2 || echo "N/A")
                        final_kl_95pct="N/A"
                        final_reward="N/A"
                        diverged_at_step=""
                        
                        echo "   ⏱️  TIMEOUT: Exceeded 30 minutes at step $steps_completed, KL=$final_kl_mean"
                    fi
                fi
                
                # Calculate training time
                end_time=$(date +%s)
                training_time=$((end_time - start_time))
                
                # Log results to CSV
                echo "$run_id,$kl_warmup_steps,$kl_warmup_factor,$lr,$kl_coef,$status,$steps_completed,$final_kl_mean,$final_kl_95pct,$final_reward,$training_time,$diverged_at_step" >> "$RESULTS_CSV"
                
                echo "   ⏱️  Training Time: ${training_time}s"
                
                # Brief pause between runs for system stability
                sleep 2
            done
        done
    done
done

echo ""
echo "============================================================"
echo "🎯 HYPERPARAMETER SWEEP COMPLETED"
echo "============================================================"
echo "Total Runs: $total_runs"
echo "Results Directory: $RESULTS_DIR"
echo "Results CSV: $RESULTS_CSV"
echo ""

# Generate quick summary
successful_runs=$(grep -c "SUCCESS" "$RESULTS_CSV" || echo "0")
diverged_runs=$(grep -c "DIVERGED" "$RESULTS_CSV" || echo "0")
timeout_runs=$(grep -c "TIMEOUT" "$RESULTS_CSV" || echo "0")

echo "📊 Quick Summary:"
echo "  Successful: $successful_runs"
echo "  Diverged: $diverged_runs" 
echo "  Timeouts: $timeout_runs"
echo ""

if [ "$successful_runs" -gt 0 ]; then
    echo "🏆 Top 5 Most Stable Configurations:"
    echo "Rank,KL_Warmup_Steps,KL_Warmup_Factor,Learning_Rate,KL_Coefficient,Final_KL"
    grep "SUCCESS" "$RESULTS_CSV" | sort -t',' -k8 -n | head -5 | nl -w2 -s'. ' | while IFS= read -r line; do
        rank=$(echo "$line" | cut -d'.' -f1)
        data=$(echo "$line" | cut -d'.' -f2- | tr -s ' ')
        kl_warmup_steps=$(echo "$data" | cut -d',' -f2)
        kl_warmup_factor=$(echo "$data" | cut -d',' -f3)
        lr=$(echo "$data" | cut -d',' -f4)
        kl_coef=$(echo "$data" | cut -d',' -f5) 
        final_kl=$(echo "$data" | cut -d',' -f8)
        echo "$rank,$kl_warmup_steps,$kl_warmup_factor,$lr,$kl_coef,$final_kl"
    done
    echo ""
fi

echo "💡 Analysis Commands:"
echo "  View results: cat $RESULTS_CSV"
echo "  Success rate: grep SUCCESS $RESULTS_CSV | wc -l" 
echo "  Divergence analysis: grep DIVERGED $RESULTS_CSV | sort -t',' -k12 -nr"
echo ""
echo "Next steps: Analyze $RESULTS_CSV to identify optimal parameter combinations!"