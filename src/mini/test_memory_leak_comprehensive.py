#!/usr/bin/env python3
"""
Comprehensive memory leak test that simulates actual training conditions.
This test runs for enough iterations to detect the ~31.6 MB/step leak that
was causing OOM after ~600 steps.
"""

import torch
import sys
import os
import time
import gc
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import GRPOConfig
from model import GPT2Model, GPT2Config as ModelConfig
from grpo import compute_log_probs, ReferenceModel, grpo_loss, compute_advantages, create_prompt_mask
from loader import load_rookworld_model

def get_memory_stats():
    """Get detailed memory statistics."""
    torch.cuda.synchronize()
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**3,
        'reserved': torch.cuda.memory_reserved() / 1024**3,
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,
    }

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
    
    # This mimics the exact operations in train_logged.py
    model.train()
    
    # 1. Compute advantages (this creates tensors)
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
    
    # 5. Compute loss (POTENTIAL LEAK POINT)
    loss, metrics = grpo_loss(
        policy_log_probs,
        ref_log_probs,
        advantages,
        prompt_mask,
        kl_coef=config.kl_coef,
        clip_range=config.clip_range,
        kl_type="forward"
    )
    
    # 6. Backward pass (MAJOR POTENTIAL LEAK POINT)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    
    # 7. Cleanup (THIS IS CRITICAL)
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

def test_memory_leak_comprehensive():
    """
    Run a comprehensive test that simulates real training conditions.
    The original leak was ~31.6 MB/step, leading to OOM after ~600 steps.
    """
    
    print("=" * 70)
    print("COMPREHENSIVE MEMORY LEAK TEST")
    print("=" * 70)
    print("This test simulates actual training conditions to detect memory leaks.")
    print("Original leak: ~31.6 MB/step, causing OOM after ~600 steps")
    print()
    
    # Setup configuration matching actual training
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
    
    # Load actual model weights
    print("Loading model...")
    
    try:
        model = load_rookworld_model("jrahn/RookWorld-LM-124M", device=config.device)
        print("✓ Loaded RookWorld-LM-124M weights")
    except:
        print("⚠ Using random weights (couldn't load RookWorld-LM)")
        model_config = ModelConfig()
        model = GPT2Model(model_config)
        model.to(config.device)
    
    # Create reference model
    ref_model = ReferenceModel(model, cache_size=0)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95)
    )
    
    # Test parameters
    batch_size = config.batch_size * config.k_samples  # 64
    seq_len = 200  # Typical sequence length
    num_steps = 200  # Run longer to be absolutely sure (would leak ~6GB with original bug)
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  K samples: {config.k_samples}")
    print(f"  Total sequences: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Chunk size: {config.log_prob_chunk_size}")
    print(f"  Test steps: {num_steps}")
    print()
    
    # Memory tracking
    memory_history = []
    initial_memory = None
    
    print("Running training simulation...")
    print("-" * 50)
    
    for step in range(1, num_steps + 1):
        # Create realistic dummy data
        sequences = torch.randint(0, 50257, (batch_size, seq_len), device=config.device)
        attention_masks = torch.ones_like(sequences)
        rewards = torch.rand(batch_size, device=config.device) * 2 - 0.3  # -0.3 to 1.7 range
        prompt_lengths = torch.tensor([50] * batch_size, device=config.device)  # Typical prompt length
        
        # Get memory before step
        mem_before = get_memory_stats()
        
        # Run training step
        metrics = simulate_training_step(
            model, ref_model, optimizer, config, step,
            sequences, attention_masks, rewards, prompt_lengths
        )
        
        # Cleanup data
        del sequences, attention_masks, rewards, prompt_lengths
        
        # Get memory after step  
        mem_after = get_memory_stats()
        
        if initial_memory is None:
            initial_memory = mem_after['allocated']
        
        memory_history.append(mem_after['allocated'])
        
        # Log progress
        if step == 1 or step % 10 == 0 or step == num_steps:
            mem_increase = (mem_after['allocated'] - initial_memory) * 1024  # MB
            leak_per_step = mem_increase / step if step > 0 else 0
            
            print(f"Step {step:3d}: "
                  f"Mem={mem_after['allocated']:.3f}GB "
                  f"(+{mem_increase:.1f}MB total, "
                  f"+{leak_per_step:.1f}MB/step avg)")
            
            # Check for dangerous leak
            if leak_per_step > 10:  # More than 10MB/step average
                print(f"  ⚠️  WARNING: High memory leak detected! {leak_per_step:.1f}MB/step")
    
    print("-" * 50)
    
    # Analyze results
    final_memory = memory_history[-1]
    total_increase = (final_memory - initial_memory) * 1024  # MB
    avg_leak_per_step = total_increase / num_steps
    
    # Calculate trend using linear regression
    steps = np.arange(len(memory_history))
    coeffs = np.polyfit(steps, memory_history, 1)
    trend_mb_per_step = coeffs[0] * 1024  # Convert GB to MB
    
    # Extrapolate to 600 steps
    expected_at_600 = initial_memory + (trend_mb_per_step * 600 / 1024)  # GB
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Initial memory: {initial_memory:.3f} GB")
    print(f"Final memory: {final_memory:.3f} GB")
    print(f"Total increase: {total_increase:.1f} MB over {num_steps} steps")
    print(f"Average leak: {avg_leak_per_step:.2f} MB/step")
    print(f"Linear trend: {trend_mb_per_step:.2f} MB/step")
    print()
    print(f"Extrapolation to 600 steps:")
    print(f"  Expected memory: {expected_at_600:.1f} GB")
    print(f"  Would OOM at 22GB: {'YES' if expected_at_600 > 22 else 'NO'}")
    print()
    
    # Pass/Fail criteria
    # The original leak was ~31.6 MB/step
    # We should aim for less than 1 MB/step to be safe
    if avg_leak_per_step < 1.0:
        print("✅ PASS: Memory leak is under control (<1 MB/step)")
        return True
    elif avg_leak_per_step < 5.0:
        print("⚠️  MARGINAL: Small leak detected (1-5 MB/step)")
        print("   This might be acceptable for shorter training runs")
        return True
    else:
        print(f"❌ FAIL: Significant memory leak detected ({avg_leak_per_step:.1f} MB/step)")
        print(f"   Original leak was 31.6 MB/step")
        return False

if __name__ == "__main__":
    success = test_memory_leak_comprehensive()
    sys.exit(0 if success else 1)