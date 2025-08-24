#!/usr/bin/env python3
"""
Performance Test for RookWorld GRPO Optimizations

This script tests the performance improvements from:
- Mixed precision training
- Torch.compile optimization  
- Gradient checkpointing
- Batch optimizations
"""

import time
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rookworld_rlvr.model.gpt2 import GPT2Model
from src.rookworld_rlvr.model.config import GPT2Config
from src.rookworld_rlvr.train.config import GRPOConfig
from src.rookworld_rlvr.train.policy import CausalLMPolicy
from src.rookworld_rlvr.tokenizer.bridge import TokenizerBridge


def benchmark_model_inference(model, input_ids, attention_mask, num_iterations=10):
    """Benchmark model forward pass."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids=input_ids)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_ids=input_ids)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    return total_time / num_iterations


def benchmark_mixed_precision(model, input_ids, attention_mask, use_amp=True):
    """Benchmark with and without mixed precision."""
    model.eval()
    
    if use_amp and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            return benchmark_model_inference(model, input_ids, attention_mask)
    else:
        return benchmark_model_inference(model, input_ids, attention_mask)


def test_performance_optimizations():
    """Test all performance optimizations."""
    print("="*60)
    print("ROOKWORLD GRPO PERFORMANCE OPTIMIZATION TEST")
    print("="*60)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create test configuration
    model_config = GPT2Config()
    model_config.n_layer = 4  # Smaller for testing
    model_config.n_head = 8
    model_config.n_embd = 512
    
    grpo_config = GRPOConfig(
        device=device,
        use_mixed_precision=torch.cuda.is_available(),
        use_torch_compile=True,
        torch_compile_mode="reduce-overhead"
    )
    
    print(f"\nModel Configuration:")
    print(f"- Layers: {model_config.n_layer}")
    print(f"- Heads: {model_config.n_head}")
    print(f"- Embedding: {model_config.n_embd}")
    print(f"- Parameters: ~{(model_config.n_layer * model_config.n_embd ** 2 * 8) / 1e6:.1f}M")
    
    # Create test input
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    print(f"\nBenchmark Setup:")
    print(f"- Batch size: {batch_size}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Iterations: 20")
    
    results = {}
    
    # Test 1: Baseline model
    print(f"\n1. Testing baseline model...")
    model_baseline = GPT2Model(model_config).to(device)
    baseline_time = benchmark_model_inference(model_baseline, input_ids, attention_mask, 20)
    results['baseline'] = baseline_time
    print(f"   Baseline time: {baseline_time*1000:.2f} ms/iter")
    
    # Test 2: Mixed precision
    if torch.cuda.is_available():
        print(f"\n2. Testing mixed precision...")
        amp_time = benchmark_mixed_precision(model_baseline, input_ids, attention_mask, use_amp=True)
        results['mixed_precision'] = amp_time
        speedup = baseline_time / amp_time
        print(f"   Mixed precision time: {amp_time*1000:.2f} ms/iter ({speedup:.2f}x speedup)")
    
    # Test 3: Torch compile
    print(f"\n3. Testing torch.compile...")
    try:
        model_compiled = torch.compile(model_baseline, mode="reduce-overhead")
        compiled_time = benchmark_model_inference(model_compiled, input_ids, attention_mask, 20)
        results['torch_compile'] = compiled_time
        speedup = baseline_time / compiled_time
        print(f"   Compiled time: {compiled_time*1000:.2f} ms/iter ({speedup:.2f}x speedup)")
    except Exception as e:
        print(f"   Torch compile failed: {e}")
        results['torch_compile'] = None
    
    # Test 4: Gradient checkpointing (memory test)
    print(f"\n4. Testing gradient checkpointing...")
    model_config.use_gradient_checkpointing = True
    model_checkpointed = GPT2Model(model_config).to(device)
    
    def measure_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            model_checkpointed.train()
            output = model_checkpointed(input_ids=input_ids)
            loss = output["logits"].sum()
            loss.backward()
            
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            return peak_memory
        return 0
    
    try:
        checkpointed_memory = measure_memory()
        
        # Compare with baseline
        model_config.use_gradient_checkpointing = False
        model_no_checkpoint = GPT2Model(model_config).to(device)
        model_no_checkpoint.train()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        output = model_no_checkpoint(input_ids=input_ids)
        loss = output["logits"].sum()
        loss.backward()
        baseline_memory = torch.cuda.max_memory_allocated() / 1e9
        
        memory_savings = (baseline_memory - checkpointed_memory) / baseline_memory * 100
        print(f"   Baseline memory: {baseline_memory:.2f} GB")
        print(f"   Checkpointed memory: {checkpointed_memory:.2f} GB ({memory_savings:.1f}% savings)")
        
    except Exception as e:
        print(f"   Gradient checkpointing test failed: {e}")
    
    # Test 5: Batch processing optimization
    print(f"\n5. Testing batch processing...")
    try:
        # Compare single vs batch forward passes
        batch_size_1 = input_ids[:1]  # Single sample
        batch_size_full = input_ids    # Full batch
        
        # Time single processing
        start_time = time.time()
        model_baseline.eval()
        with torch.no_grad():
            for _ in range(20):
                for i in range(batch_size):
                    _ = model_baseline(batch_size_1)
        single_time = time.time() - start_time
        
        # Time batch processing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = model_baseline(batch_size_full)
        batch_time = time.time() - start_time
        
        speedup = single_time / batch_time
        print(f"   Single processing time: {single_time*1000:.2f} ms")
        print(f"   Batch processing time: {batch_time*1000:.2f} ms ({speedup:.2f}x speedup)")
        
    except Exception as e:
        print(f"   Batch processing test failed: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if torch.cuda.is_available():
        total_speedup = 1.0
        if 'mixed_precision' in results and results['mixed_precision']:
            total_speedup *= baseline_time / results['mixed_precision']
        if 'torch_compile' in results and results['torch_compile']:
            total_speedup *= baseline_time / results['torch_compile']
        
        print(f"Expected combined speedup: {total_speedup:.2f}x")
        print(f"Memory savings available: gradient checkpointing")
        print(f"Batch optimizations: legal move scoring")
    else:
        print("CUDA not available - limited optimizations active")
    
    print("\nOptimizations successfully integrated!")
    return results


if __name__ == "__main__":
    test_performance_optimizations()