#!/usr/bin/env python3
"""
BF16 Memory Benefits Test

Tests the primary benefit of BF16: reduced memory usage enabling larger batch sizes
which should lead to higher MFU through better GPU utilization.
"""

import torch
import time
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.grpo import compute_log_probs, ReferenceModel, grpo_loss, compute_advantages, create_prompt_mask


def measure_memory_usage(model, batch_size, seq_len, use_bf16=False, include_gradients=True):
    """Measure peak memory usage for a given configuration."""
    
    # Clear memory and reset stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create test data
    torch.manual_seed(42)
    sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(sequences)
    rewards = torch.rand(batch_size, device="cuda")
    prompt_lengths = torch.randint(seq_len//4, seq_len//2, (batch_size,))
    
    model.train()
    
    try:
        if include_gradients:
            # Forward pass
            policy_log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
            
            # Create reference (always FP32)
            ref_model = ReferenceModel(model)
            with torch.no_grad():
                ref_log_probs = ref_model.compute_log_probs(sequences, attention_mask, use_bf16=False)
                ref_log_probs = ref_log_probs.to("cuda")
            
            # Compute loss
            advantages = compute_advantages(rewards, group_size=min(8, batch_size))
            prompt_mask = create_prompt_mask(sequences, prompt_lengths)
            
            loss, metrics = grpo_loss(
                policy_log_probs,
                ref_log_probs,
                advantages,
                prompt_mask,
                kl_coef=0.02,
                clip_range=0.2
            )
            
            # Backward pass to measure gradient memory
            loss.backward()
        else:
            # Just forward pass
            with torch.no_grad():
                policy_log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
        
        # Get peak memory usage
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        
        return peak_memory_gb, True  # Success
        
    except torch.cuda.OutOfMemoryError:
        return None, False  # OOM
    finally:
        # Cleanup
        torch.cuda.empty_cache()


def test_bf16_memory_scaling():
    """Test memory usage scaling with BF16 vs FP32."""
    print("📊 Testing BF16 Memory Scaling Benefits")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    seq_len = 150  # Typical sequence length
    
    # Test various batch sizes
    batch_sizes = [4, 8, 16, 24, 32, 48]
    results = {
        'FP32': {},
        'BF16': {}
    }
    
    print(f"Testing with sequence length: {seq_len}")
    print("\nMemory Usage by Batch Size:")
    print(f"{'Batch Size':<12} {'FP32 (GB)':<12} {'BF16 (GB)':<12} {'Memory Saved':<15} {'Status'}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        for precision, use_bf16 in [("FP32", False), ("BF16", True)]:
            memory_gb, success = measure_memory_usage(
                model, batch_size, seq_len, use_bf16=use_bf16, include_gradients=True
            )
            
            results[precision][batch_size] = {
                'memory_gb': memory_gb,
                'success': success
            }
        
        # Report results for this batch size
        fp32_mem = results['FP32'][batch_size]['memory_gb'] 
        bf16_mem = results['BF16'][batch_size]['memory_gb']
        fp32_success = results['FP32'][batch_size]['success']
        bf16_success = results['BF16'][batch_size]['success']
        
        if fp32_mem is not None and bf16_mem is not None:
            memory_saved_gb = fp32_mem - bf16_mem
            memory_saved_pct = (memory_saved_gb / fp32_mem) * 100
            status = "✅ Both OK"
        elif bf16_mem is not None and fp32_mem is None:
            memory_saved_gb = "N/A"  
            memory_saved_pct = "N/A"
            status = "🎯 BF16 Only"
        elif fp32_mem is not None and bf16_mem is None:
            memory_saved_gb = "N/A"
            memory_saved_pct = "N/A" 
            status = "⚠️ BF16 OOM"
        else:
            memory_saved_gb = "N/A"
            memory_saved_pct = "N/A"
            status = "❌ Both OOM"
        
        fp32_str = f"{fp32_mem:.2f}" if fp32_mem else "OOM"
        bf16_str = f"{bf16_mem:.2f}" if bf16_mem else "OOM"
        saved_str = f"{memory_saved_gb:.2f}GB ({memory_saved_pct:.1f}%)" if isinstance(memory_saved_gb, float) else str(memory_saved_gb)
        
        print(f"{batch_size:<12} {fp32_str:<12} {bf16_str:<12} {saved_str:<15} {status}")
    
    # Find maximum viable batch sizes
    max_fp32_batch = max([bs for bs in batch_sizes if results['FP32'][bs]['success']], default=0)
    max_bf16_batch = max([bs for bs in batch_sizes if results['BF16'][bs]['success']], default=0)
    
    print(f"\n📈 Batch Size Scaling:")
    print(f"   Max FP32 batch size: {max_fp32_batch}")
    print(f"   Max BF16 batch size: {max_bf16_batch}")
    
    if max_bf16_batch > max_fp32_batch:
        batch_improvement = max_bf16_batch / max_fp32_batch
        print(f"   BF16 allows {batch_improvement:.1f}x larger batches!")
        print(f"   🎯 This should significantly improve MFU")
    else:
        print(f"   ⚠️ No batch size improvement (GPU memory may be sufficient)")
    
    return results, max_fp32_batch, max_bf16_batch


def test_mfu_with_larger_batches():
    """Test MFU improvement using larger batch sizes enabled by BF16."""
    print("\n🎯 Testing MFU with BF16-Enabled Larger Batches")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    seq_len = 150
    
    # Test configurations
    configs = [
        {"name": "FP32 Small", "batch_size": 8, "use_bf16": False},
        {"name": "BF16 Small", "batch_size": 8, "use_bf16": True},
        {"name": "BF16 Large", "batch_size": 16, "use_bf16": True},
        {"name": "BF16 XLarge", "batch_size": 24, "use_bf16": True},
    ]
    
    print(f"Testing MFU at sequence length: {seq_len}")
    print(f"\n{'Configuration':<15} {'Batch Size':<12} {'Time (s)':<10} {'Memory (GB)':<12} {'MFU (%)':<10} {'TFLOPS':<10}")
    print("-" * 80)
    
    # Calculate theoretical model FLOPs
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_per_forward = 6 * n_params * seq_len
    peak_gpu_flops = 165e12  # RTX 4090 peak
    
    for config in configs:
        batch_size = config["batch_size"]
        use_bf16 = config["use_bf16"]
        name = config["name"]
        
        try:
            # Measure memory and timing
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create test data
            torch.manual_seed(42)
            sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
            attention_mask = torch.ones_like(sequences)
            
            # Time the computation
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Forward pass only for fair comparison
            with torch.no_grad():
                log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
            
            torch.cuda.synchronize() 
            elapsed_time = time.time() - start_time
            
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            
            # Calculate MFU
            total_flops = flops_per_forward * batch_size
            actual_flops_per_sec = total_flops / elapsed_time
            mfu = (actual_flops_per_sec / peak_gpu_flops) * 100
            actual_tflops = actual_flops_per_sec / 1e12
            
            print(f"{name:<15} {batch_size:<12} {elapsed_time:<10.3f} {peak_memory_gb:<12.2f} {mfu:<10.2f} {actual_tflops:<10.2f}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"{name:<15} {batch_size:<12} {'OOM':<10} {'OOM':<12} {'OOM':<10} {'OOM':<10}")
    
    print("\n✅ BF16 memory scaling test complete")


def test_tensor_core_optimization():
    """Test TF32 and Tensor Core optimization effects."""
    print("\n🔧 Testing TF32 and Tensor Core Optimizations")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    
    # Test different optimization combinations
    optimization_configs = [
        {"name": "Baseline", "tf32": False, "precision": "highest"},
        {"name": "TF32 Only", "tf32": True, "precision": "highest"},  
        {"name": "Tensor Core", "tf32": False, "precision": "high"},
        {"name": "TF32 + TC", "tf32": True, "precision": "high"},
    ]
    
    batch_size, seq_len = 16, 150
    torch.manual_seed(42)
    sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(sequences)
    
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"\n{'Configuration':<15} {'TF32':<8} {'TC Prec':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    baseline_time = None
    
    for config in optimization_configs:
        # Set optimizations
        torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
        torch.backends.cudnn.allow_tf32 = config["tf32"]
        torch.set_float32_matmul_precision(config["precision"])
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = compute_log_probs(model, sequences, attention_mask, use_bf16=True)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            with torch.no_grad():
                log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=True)
        
        torch.cuda.synchronize()
        elapsed = (time.time() - start_time) / 10  # Average per iteration
        elapsed_ms = elapsed * 1000
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup_str = "1.00x"
        else:
            speedup = baseline_time / elapsed
            speedup_str = f"{speedup:.2f}x"
        
        tf32_str = "✅" if config["tf32"] else "❌"
        
        print(f"{config['name']:<15} {tf32_str:<8} {config['precision']:<10} {elapsed_ms:<12.2f} {speedup_str:<10}")
    
    print("\n✅ Tensor Core optimization test complete")


if __name__ == "__main__":
    print("🚀 BF16 Memory Benefits and Optimization Test")
    print("=" * 70)
    
    try:
        # Test 1: Memory scaling benefits  
        results, max_fp32, max_bf16 = test_bf16_memory_scaling()
        
        # Test 2: MFU with larger batches
        test_mfu_with_larger_batches()
        
        # Test 3: TF32 and Tensor Core optimizations
        test_tensor_core_optimization()
        
        print("\n" + "=" * 70)
        print("🎉 ALL BF16 MEMORY BENEFIT TESTS COMPLETE")
        
        if max_bf16 > max_fp32:
            print(f"🎯 KEY INSIGHT: BF16 enables {max_bf16/max_fp32:.1f}x larger batches!")
            print("   This should significantly improve training MFU.")
        else:
            print("💡 NOTE: Current GPU has sufficient memory for tested batch sizes.")
            print("   BF16 benefits will be more apparent with larger models or longer sequences.")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise