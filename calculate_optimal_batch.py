#!/usr/bin/env python3
"""
Calculate optimal batch size for RookWorld GRPO training on RTX 4090
Based on observed memory usage patterns during training.
"""

import torch
import subprocess
import sys
import json
import os

def get_gpu_info():
    """Get GPU memory information"""
    try:
        # Get GPU memory info
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        memory_info = []
        for line in result.stdout.strip().split('\n'):
            total, free, used = map(int, line.split(', '))
            memory_info.append({
                'total_mb': total,
                'free_mb': free, 
                'used_mb': used,
                'total_gb': total / 1024,
                'free_gb': free / 1024,
                'used_gb': used / 1024
            })
        
        return memory_info
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")
        return [{'total_gb': 24, 'free_gb': 20, 'used_gb': 4}]  # RTX 4090 defaults

def calculate_model_memory():
    """Calculate base model memory usage"""
    # RookWorld-LM 124M parameters
    model_params = 124_439_808
    
    # Memory per parameter in different precisions
    memory_per_param = {
        'fp32': 4,  # bytes
        'fp16': 2,
        'bf16': 2,
    }
    
    # We use BF16 mixed precision
    precision = 'bf16'
    bytes_per_param = memory_per_param[precision]
    
    # Model weights memory
    model_memory_bytes = model_params * bytes_per_param
    model_memory_gb = model_memory_bytes / (1024**3)
    
    # We have 2 models (training + reference on different GPUs)
    training_model_gb = model_memory_gb
    ref_model_gb = model_memory_gb
    
    return {
        'model_params': model_params,
        'precision': precision,
        'training_model_gb': training_model_gb,
        'ref_model_gb': ref_model_gb,
        'total_model_gb': training_model_gb + ref_model_gb
    }

def calculate_batch_memory(batch_positions, group_size, max_tokens=100):
    """Calculate memory usage for a given batch configuration"""
    
    effective_batch_size = batch_positions * group_size
    
    # Token memory (input_ids, attention_mask, etc.)
    # Sequence length is variable but let's estimate
    avg_seq_len = 200  # Conservative estimate including prompt + generation
    
    # Memory per token in BF16 (2 bytes)
    token_memory_bytes = effective_batch_size * avg_seq_len * 2
    
    # Activations and gradients (more realistic estimate for GRPO)
    # GRPO needs: forward pass activations + backward pass gradients + KL computation
    # Rule of thumb: ~2-4x model parameters in activations for training
    model_params = 124_439_808
    
    # Base activation memory (forward + backward)
    base_activation_gb = (model_params * 2 * 4) / (1024**3)  # BF16 forward + FP32 gradients
    
    # Batch scaling factor (larger batches need more activation memory)
    batch_scale = min(2.0, 1.0 + (effective_batch_size / 16.0))
    activation_memory_bytes = base_activation_gb * batch_scale * (1024**3)
    
    # KL divergence computation (requires storing logprobs)
    kl_memory_bytes = effective_batch_size * avg_seq_len * 4  # float32 logprobs
    
    total_memory_bytes = token_memory_bytes + activation_memory_bytes + kl_memory_bytes
    total_memory_gb = total_memory_bytes / (1024**3)
    
    return {
        'batch_positions': batch_positions,
        'group_size': group_size,
        'effective_batch_size': effective_batch_size,
        'avg_seq_len': avg_seq_len,
        'token_memory_gb': token_memory_bytes / (1024**3),
        'activation_memory_gb': activation_memory_bytes / (1024**3),
        'kl_memory_gb': kl_memory_bytes / (1024**3),
        'total_memory_gb': total_memory_gb,
        'batch_scale': batch_scale
    }

def main():
    print("🔍 Calculating Optimal Batch Size for RookWorld GRPO Training")
    print("=" * 70)
    
    # Get system info
    gpu_info = get_gpu_info()
    model_info = calculate_model_memory()
    
    print(f"GPU Memory Information:")
    for i, gpu in enumerate(gpu_info):
        print(f"  GPU {i}: {gpu['total_gb']:.1f}GB total, {gpu['free_gb']:.1f}GB free")
    
    print(f"\nModel Memory Usage:")
    print(f"  Model Parameters: {model_info['model_params']:,}")
    print(f"  Precision: {model_info['precision']}")
    print(f"  Training Model (GPU 0): {model_info['training_model_gb']:.2f}GB")
    print(f"  Reference Model (GPU 1): {model_info['ref_model_gb']:.2f}GB")
    
    # Calculate optimal batch sizes
    print(f"\nBatch Size Analysis:")
    print(f"{'Batch Pos':<10} {'Group Size':<11} {'Effective':<9} {'Memory (GB)':<12} {'Utilization':<12} {'Status'}")
    print("-" * 70)
    
    # Based on observed memory usage: ~0.51GB allocated, ~0.60GB reserved during training
    # This suggests current batch_positions=2, group_size=2 is quite conservative
    
    gpu_0_total = gpu_info[0]['total_gb']
    gpu_1_total = gpu_info[1]['total_gb'] if len(gpu_info) > 1 else gpu_0_total
    
    # Leave safety margin
    safe_memory_gpu_0 = gpu_0_total * 0.85  # 85% utilization max
    safe_memory_gpu_1 = gpu_1_total * 0.85
    
    # Model memory is fixed
    available_gpu_0 = safe_memory_gpu_0 - model_info['training_model_gb']
    available_gpu_1 = safe_memory_gpu_1 - model_info['ref_model_gb']
    
    configurations = [
        (1, 2), (1, 4), (1, 8), (1, 16),
        (2, 2), (2, 4), (2, 8), (2, 16), (2, 32),
        (4, 2), (4, 4), (4, 8), (4, 16), (4, 32),
        (8, 2), (8, 4), (8, 8), (8, 16),
        (16, 2), (16, 4), (16, 8),
        (32, 2), (32, 4),
        (64, 2)
    ]
    
    optimal_configs = []
    current_config = None
    
    for batch_pos, group_size in configurations:
        batch_info = calculate_batch_memory(batch_pos, group_size)
        
        # Training happens on GPU 0
        gpu_0_usage = model_info['training_model_gb'] + batch_info['total_memory_gb']
        # Reference model inference uses minimal additional memory on GPU 1
        gpu_1_usage = model_info['ref_model_gb'] + 0.1  # Small inference overhead
        
        utilization_0 = (gpu_0_usage / gpu_0_total) * 100
        utilization_1 = (gpu_1_usage / gpu_1_total) * 100
        
        # Determine status
        if gpu_0_usage <= safe_memory_gpu_0 and gpu_1_usage <= safe_memory_gpu_1:
            if utilization_0 > 40:  # Good utilization (lowered threshold for smaller model)
                status = "✅ OPTIMAL"
                optimal_configs.append((batch_pos, group_size, utilization_0))
            elif utilization_0 > 10:
                status = "✅ GOOD"
                optimal_configs.append((batch_pos, group_size, utilization_0))
            else:
                status = "✅ SAFE"
        elif gpu_0_usage <= gpu_0_total and gpu_1_usage <= gpu_1_total:
            status = "⚠️  RISKY"
        else:
            status = "❌ OOM"
        
        # Check if this is current config
        if batch_pos == 2 and group_size == 2:
            current_config = (batch_pos, group_size, utilization_0, status)
            status += " (CURRENT)"
        
        print(f"{batch_pos:<10} {group_size:<11} {batch_info['effective_batch_size']:<9} "
              f"{gpu_0_usage:.2f}GB{'':<4} {utilization_0:.1f}%{'':<6} {status}")
    
    print("\n" + "=" * 70)
    print("📊 RECOMMENDATIONS")
    print("=" * 70)
    
    if current_config:
        print(f"Current Configuration: batch_positions={current_config[0]}, group_size={current_config[1]}")
        print(f"Current GPU Utilization: {current_config[2]:.1f}% - {current_config[3]}")
    
    if optimal_configs:
        # Sort by utilization (highest first)
        optimal_configs.sort(key=lambda x: x[2], reverse=True)
        best_config = optimal_configs[0]
        
        print(f"\n🏆 RECOMMENDED OPTIMAL CONFIG:")
        print(f"   batch_positions={best_config[0]}, group_size={best_config[1]}")
        print(f"   Effective batch size: {best_config[0] * best_config[1]}")
        print(f"   GPU utilization: {best_config[2]:.1f}%")
        
        print(f"\n🚀 TRAINING COMMANDS:")
        print(f"   # Conservative (safe for all scenarios)")
        if len(optimal_configs) >= 3:
            safe_config = optimal_configs[-1]  # Lowest utilization among optimal
            print(f"   BATCH_POSITIONS={safe_config[0]} GROUP_SIZE={safe_config[1]} ./train.sh")
        
        print(f"   # Optimal (maximum performance)")
        print(f"   BATCH_POSITIONS={best_config[0]} GROUP_SIZE={best_config[1]} ./train.sh")
        
        print(f"   # Hyperparameter sweep with optimal batch")
        print(f"   # Edit hyperparameter_sweep.sh and change:")
        print(f"   # BASE_BATCH_POSITIONS={best_config[0]}")
        print(f"   # BASE_GROUP_SIZE={best_config[1]}")
    
    print(f"\n💡 MEMORY INSIGHTS:")
    print(f"   - Current memory usage is very conservative ({current_config[2]:.1f}%)")
    print(f"   - RTX 4090 has excellent memory bandwidth - higher batch sizes will improve throughput")  
    print(f"   - Multi-GPU setup allows larger batches since reference model is on separate GPU")
    print(f"   - BF16 mixed precision provides good memory efficiency")
    print(f"   - Consider torch.compile benefits scale with larger batch sizes")

if __name__ == "__main__":
    main()