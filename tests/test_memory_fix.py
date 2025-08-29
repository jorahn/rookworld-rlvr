#!/usr/bin/env python3
"""Test script to verify memory leak fixes."""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.model import GPT2Model
from rookworld_rlvr.grpo import compute_log_probs, ReferenceModel
import time

def test_memory_usage():
    """Test memory usage with chunked processing."""
    
    print("Testing memory usage with chunked log_probs computation...")
    
    # Setup
    config = GRPOConfig(
        batch_size=8,
        k_samples=8,
        max_new_tokens=144,
        log_prob_chunk_size=16  # Use chunked processing
    )
    
    # Create model with GPT-2 config
    from rookworld_rlvr.model import GPT2Config as ModelConfig
    model_config = ModelConfig()
    model = GPT2Model(model_config)
    model.to(config.device)
    model.eval()
    
    # Create reference model
    ref_model = ReferenceModel(model, cache_size=0)
    
    # Simulate sequences (batch_size * k_samples = 64 sequences)
    batch_size = config.batch_size * config.k_samples
    seq_len = 200  # Typical sequence length
    
    print(f"\nTesting with {batch_size} sequences of length {seq_len}")
    print(f"Chunk size: {config.log_prob_chunk_size}")
    
    # Track memory over multiple iterations
    memory_usage = []
    
    for step in range(10):
        # Create dummy sequences
        sequences = torch.randint(0, 50257, (batch_size, seq_len), device=config.device)
        attention_masks = torch.ones_like(sequences)
        
        # Measure memory before
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        
        # Compute log probs with chunking
        with torch.no_grad():
            policy_log_probs = compute_log_probs(
                model,
                sequences, 
                attention_masks,
                chunk_size=config.log_prob_chunk_size
            )
            
            ref_log_probs = ref_model.compute_log_probs(
                sequences,
                attention_masks,
                return_on_cpu=True,
                chunk_size=config.log_prob_chunk_size
            )
        
        # Cleanup
        del policy_log_probs, ref_log_probs, sequences, attention_masks
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Measure memory after
        mem_after = torch.cuda.memory_allocated() / 1024**3
        memory_usage.append(mem_after)
        
        print(f"Step {step+1}: Before={mem_before:.3f}GB, After={mem_after:.3f}GB, Delta={mem_after-mem_before:.3f}GB")
        
        # Check for memory leak
        if step > 0 and mem_after > memory_usage[0] + 0.1:  # More than 100MB increase
            print(f"⚠️  WARNING: Memory increased by {(mem_after - memory_usage[0])*1024:.1f}MB")
    
    # Check overall trend
    initial_mem = memory_usage[0]
    final_mem = memory_usage[-1]
    
    print(f"\n=== RESULTS ===")
    print(f"Initial memory: {initial_mem:.3f}GB")
    print(f"Final memory: {final_mem:.3f}GB")
    print(f"Memory increase: {(final_mem - initial_mem)*1024:.1f}MB")
    
    if final_mem - initial_mem < 0.01:  # Less than 10MB increase
        print("✅ PASS: No significant memory leak detected")
        return True
    else:
        print(f"❌ FAIL: Memory leak detected ({(final_mem - initial_mem)*1024:.1f}MB over 10 steps)")
        return False

if __name__ == "__main__":
    success = test_memory_usage()
    sys.exit(0 if success else 1)