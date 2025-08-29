#!/usr/bin/env python3
"""
Test memory leak using nvidia-smi to get ACTUAL GPU memory usage.
torch.cuda.memory_allocated() doesn't capture all memory usage,
so we use nvidia-smi to get the real picture.
"""

import torch
import sys
import os
import time
import gc
import numpy as np
import subprocess
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.model import GPT2Model, GPT2Config as ModelConfig
from rookworld_rlvr.grpo import compute_log_probs, ReferenceModel, grpo_loss, compute_advantages, create_prompt_mask
from rookworld_rlvr.loader import load_rookworld_model

def get_gpu_memory_nvidia_smi():
    """
    Get actual GPU memory usage using nvidia-smi.
    This captures ALL memory used by the process, not just PyTorch allocations.
    """
    try:
        # Get memory usage for the current process
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        memory_mb = float(result.stdout.strip())
        return memory_mb / 1024  # Convert to GB
    except:
        # Fallback to torch if nvidia-smi fails
        return torch.cuda.memory_allocated() / 1024**3

def get_process_gpu_memory():
    """
    Get GPU memory used specifically by this Python process.
    More accurate than global GPU memory.
    """
    try:
        pid = os.getpid()
        # Query memory for specific process
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output to find our process
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    proc_pid, mem_mb = parts
                    if int(proc_pid.strip()) == pid:
                        return float(mem_mb.strip()) / 1024  # Convert to GB
        
        # If process not found, fall back to total GPU memory
        return get_gpu_memory_nvidia_smi()
    except:
        return get_gpu_memory_nvidia_smi()

def simulate_training_step(
    model, 
    ref_model, 
    optimizer,
    config,
    step,
    sequences,
    attention_masks,
    rewards,
    prompt_lengths
):
    """Simulate a complete training step including all operations that could leak memory."""
    
    model.train()
    
    # 1. Compute advantages
    advantages = compute_advantages(
        rewards,
        group_size=config.k_samples,
        baseline_type="group_mean",
        baseline_tracker=None
    )
    
    # 2. Compute policy log probs (POTENTIAL LEAK POINT)
    policy_log_probs = compute_log_probs(
        model,
        sequences,
        attention_masks,
        chunk_size=config.log_prob_chunk_size
    )
    
    # 3. Compute reference log probs (POTENTIAL LEAK POINT)
    with torch.no_grad():
        ref_log_probs_cpu = ref_model.compute_log_probs(
            sequences,
            attention_masks,
            return_on_cpu=True,
            chunk_size=config.log_prob_chunk_size
        )
        ref_log_probs = ref_log_probs_cpu.to(config.device)
    
    # 4. Create prompt mask
    prompt_mask = create_prompt_mask(sequences, prompt_lengths)
    
    # 5. Compute loss
    loss, metrics = grpo_loss(
        policy_log_probs,
        ref_log_probs,
        advantages,
        prompt_mask,
        kl_coef=config.kl_coef,
        clip_range=config.clip_range,
        kl_type="forward"
    )
    
    # 6. Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    
    # 7. Cleanup
    loss = loss.detach()
    
    # Clear gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    
    # Delete tensors
    del loss, policy_log_probs, ref_log_probs, ref_log_probs_cpu
    del advantages, prompt_mask
    
    # Force cleanup
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

def test_memory_leak_with_nvidia_smi():
    """
    Test memory leak using nvidia-smi to get real GPU memory usage.
    """
    
    print("=" * 70)
    print("MEMORY LEAK TEST WITH NVIDIA-SMI")
    print("=" * 70)
    print("Using nvidia-smi to capture ACTUAL GPU memory usage")
    print("(not just PyTorch allocations)")
    print()
    
    # Setup
    config = GRPOConfig(
        batch_size=8,
        k_samples=8,
        max_new_tokens=144,
        log_prob_chunk_size=16,
        learning_rate=1e-5,
        kl_coef=0.02,
        clip_range=0.2,
        grad_clip=1.0
    )
    
    print("Loading model...")
    try:
        model = load_rookworld_model("jrahn/RookWorld-LM-124M", device=config.device)
        print("✓ Loaded RookWorld-LM-124M weights")
    except:
        print("⚠ Using random weights")
        model_config = ModelConfig()
        model = GPT2Model(model_config)
        model.to(config.device)
    
    ref_model = ReferenceModel(model, cache_size=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95)
    )
    
    # Parameters
    batch_size = config.batch_size * config.k_samples  # 64
    seq_len = 200
    num_steps = 200  # Run long enough to detect leak
    
    print(f"\nConfiguration:")
    print(f"  Total sequences: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Chunk size: {config.log_prob_chunk_size}")
    print(f"  Test steps: {num_steps}")
    print()
    
    # Wait a moment for GPU to settle
    time.sleep(2)
    
    # Get initial memory using nvidia-smi
    initial_nvidia_mem = get_process_gpu_memory()
    initial_torch_mem = torch.cuda.memory_allocated() / 1024**3
    
    print(f"Initial memory:")
    print(f"  nvidia-smi (actual): {initial_nvidia_mem:.3f} GB")
    print(f"  torch.cuda (allocated): {initial_torch_mem:.3f} GB")
    print()
    
    # Memory tracking
    nvidia_memory_history = []
    torch_memory_history = []
    
    print("Running training simulation...")
    print("-" * 50)
    
    for step in range(1, num_steps + 1):
        # Create data
        sequences = torch.randint(0, 50257, (batch_size, seq_len), device=config.device)
        attention_masks = torch.ones_like(sequences)
        rewards = torch.rand(batch_size, device=config.device) * 2 - 0.3
        prompt_lengths = torch.tensor([50] * batch_size, device=config.device)
        
        # Run step
        metrics = simulate_training_step(
            model, ref_model, optimizer, config, step,
            sequences, attention_masks, rewards, prompt_lengths
        )
        
        # Cleanup
        del sequences, attention_masks, rewards, prompt_lengths
        
        # Get memory after step
        torch.cuda.synchronize()
        nvidia_mem = get_process_gpu_memory()
        torch_mem = torch.cuda.memory_allocated() / 1024**3
        
        nvidia_memory_history.append(nvidia_mem)
        torch_memory_history.append(torch_mem)
        
        # Log progress
        if step == 1 or step % 20 == 0 or step == num_steps:
            nvidia_increase = (nvidia_mem - initial_nvidia_mem) * 1024  # MB
            torch_increase = (torch_mem - initial_torch_mem) * 1024  # MB
            nvidia_per_step = nvidia_increase / step
            torch_per_step = torch_increase / step
            
            print(f"Step {step:3d}:")
            print(f"  nvidia-smi: {nvidia_mem:.3f}GB (+{nvidia_increase:.1f}MB, {nvidia_per_step:.2f}MB/step)")
            print(f"  torch.cuda: {torch_mem:.3f}GB (+{torch_increase:.1f}MB, {torch_per_step:.2f}MB/step)")
            
            # Warning if high leak
            if nvidia_per_step > 10:
                print(f"  ⚠️  HIGH LEAK DETECTED: {nvidia_per_step:.1f}MB/step!")
    
    print("-" * 50)
    
    # Analysis
    final_nvidia = nvidia_memory_history[-1]
    final_torch = torch_memory_history[-1]
    
    nvidia_total_increase = (final_nvidia - initial_nvidia_mem) * 1024  # MB
    torch_total_increase = (final_torch - initial_torch_mem) * 1024  # MB
    
    nvidia_avg_leak = nvidia_total_increase / num_steps
    torch_avg_leak = torch_total_increase / num_steps
    
    # Linear regression for trend
    steps = np.arange(len(nvidia_memory_history))
    nvidia_trend = np.polyfit(steps, nvidia_memory_history, 1)[0] * 1024  # MB/step
    torch_trend = np.polyfit(steps, torch_memory_history, 1)[0] * 1024  # MB/step
    
    # Extrapolate
    expected_nvidia_600 = initial_nvidia_mem + (nvidia_trend * 600 / 1024)  # GB
    expected_torch_600 = initial_torch_mem + (torch_trend * 600 / 1024)  # GB
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nNVIDIA-SMI (ACTUAL MEMORY):")
    print(f"  Initial: {initial_nvidia_mem:.3f} GB")
    print(f"  Final: {final_nvidia:.3f} GB")
    print(f"  Total increase: {nvidia_total_increase:.1f} MB")
    print(f"  Average leak: {nvidia_avg_leak:.2f} MB/step")
    print(f"  Linear trend: {nvidia_trend:.2f} MB/step")
    print(f"  Extrapolated to 600 steps: {expected_nvidia_600:.1f} GB")
    print(f"  Would cause OOM (>22GB): {'YES!' if expected_nvidia_600 > 22 else 'NO'}")
    
    print(f"\nTORCH.CUDA (ALLOCATED ONLY):")
    print(f"  Initial: {initial_torch_mem:.3f} GB")
    print(f"  Final: {final_torch:.3f} GB") 
    print(f"  Total increase: {torch_total_increase:.1f} MB")
    print(f"  Average leak: {torch_avg_leak:.2f} MB/step")
    print(f"  Linear trend: {torch_trend:.2f} MB/step")
    
    print(f"\nDISCREPANCY:")
    print(f"  Untracked memory: {nvidia_total_increase - torch_total_increase:.1f} MB")
    print(f"  This is memory PyTorch doesn't track (computation graphs, etc.)")
    print()
    
    # Pass/Fail
    if nvidia_avg_leak < 1.0:
        print("✅ PASS: No significant memory leak (<1 MB/step)")
        return True
    elif nvidia_avg_leak < 5.0:
        print("⚠️  MARGINAL: Small leak detected (1-5 MB/step)")
        print(f"   At {nvidia_avg_leak:.1f}MB/step, would use ~{nvidia_avg_leak*10000/1024:.1f}GB for 10k steps")
        return True
    else:
        print(f"❌ FAIL: Memory leak detected ({nvidia_avg_leak:.1f} MB/step)")
        print(f"   Original leak was 31.6 MB/step")
        print(f"   This would cause OOM after ~{int(20000/nvidia_avg_leak)} steps")
        return False

if __name__ == "__main__":
    success = test_memory_leak_with_nvidia_smi()
    sys.exit(0 if success else 1)