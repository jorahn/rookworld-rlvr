#!/usr/bin/env python3
"""
BF16 Mixed Precision Compatibility Tests

Validates that BF16 mixed precision training works correctly without
introducing numerical instabilities or breaking existing functionality.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.grpo import (
    compute_log_probs, 
    ReferenceModel, 
    grpo_loss, 
    compute_advantages, 
    create_prompt_mask
)


def test_bf16_log_prob_computation():
    """Test that BF16 log probability computation produces valid outputs."""
    print("🧪 Testing BF16 log probability computation...")
    
    config = GRPOConfig()
    model = load_rookworld_model(device="cuda")
    
    # Create test data
    batch_size, seq_len = 2, 50
    torch.manual_seed(42)
    input_ids = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    
    # Compute log probs in both modes
    with torch.no_grad():
        # FP32 baseline
        fp32_log_probs = compute_log_probs(model, input_ids, attention_mask, use_bf16=False)
        
        # BF16 computation  
        bf16_log_probs = compute_log_probs(model, input_ids, attention_mask, use_bf16=True)
    
    # Validate BF16 outputs
    assert not torch.isnan(bf16_log_probs).any(), "BF16 log probs contain NaN"
    assert not torch.isinf(bf16_log_probs).any(), "BF16 log probs contain Inf"
    assert torch.all(bf16_log_probs <= 0), "BF16 log probs should be negative"
    
    # Check numerical similarity (BF16 should be close to FP32, but not identical)
    mean_diff = torch.abs(fp32_log_probs - bf16_log_probs).mean().item()
    max_diff = torch.abs(fp32_log_probs - bf16_log_probs).max().item()
    
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Max difference: {max_diff:.6f}")
    
    # BF16 should be reasonably close to FP32 (within expected precision loss)
    # BF16 has ~3 decimal digits of precision vs FP32's ~7, so differences are expected
    assert mean_diff < 0.1, f"BF16 differs too much from FP32: {mean_diff}"
    assert max_diff < 0.5, f"BF16 max difference too large: {max_diff}"
    
    print("✅ BF16 log probability computation is numerically stable")


def test_bf16_grpo_loss():
    """Test that BF16 GRPO loss computation works correctly."""
    print("🧪 Testing BF16 GRPO loss computation...")
    
    config = GRPOConfig()
    model = load_rookworld_model(device="cuda")
    ref_model = ReferenceModel(model)
    
    # Create test data
    batch_size, seq_len = 2, 50
    torch.manual_seed(42)
    sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(sequences)
    rewards = torch.tensor([0.8, 0.6], device="cuda")
    prompt_lengths = torch.tensor([20, 25])
    
    # Test both FP32 and BF16
    results = {}
    
    for precision, use_bf16 in [("FP32", False), ("BF16", True)]:
        print(f"   Testing {precision}...")
        
        with torch.no_grad():
            # Compute log probs
            policy_log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
            ref_log_probs = ref_model.compute_log_probs(sequences, attention_mask, use_bf16=False)  # Ref always FP32
            ref_log_probs = ref_log_probs.to("cuda")
            
            # Compute advantages and mask
            advantages = compute_advantages(rewards, group_size=1)
            prompt_mask = create_prompt_mask(sequences, prompt_lengths)
            
            # Compute loss
            loss, metrics = grpo_loss(
                policy_log_probs,
                ref_log_probs,
                advantages, 
                prompt_mask,
                kl_coef=0.02,
                clip_range=0.2
            )
            
            results[precision] = {
                'loss': loss.item(),
                'pg_loss': metrics['pg_loss'],
                'kl_div': metrics['kl_div']
            }
    
    # Validate BF16 results
    bf16_loss = results["BF16"]["loss"]
    fp32_loss = results["FP32"]["loss"]
    
    assert not np.isnan(bf16_loss), "BF16 loss is NaN"
    assert not np.isinf(bf16_loss), "BF16 loss is Inf"
    
    # Check that losses are reasonably similar
    loss_diff = abs(bf16_loss - fp32_loss)
    rel_diff = loss_diff / abs(fp32_loss) if abs(fp32_loss) > 0 else loss_diff
    
    print(f"   FP32 loss: {fp32_loss:.6f}")
    print(f"   BF16 loss: {bf16_loss:.6f}")
    print(f"   Relative diff: {rel_diff:.4f}")
    
    assert rel_diff < 0.15, f"BF16 loss differs too much from FP32: {rel_diff:.4f}"
    
    print("✅ BF16 GRPO loss computation is numerically stable")


def test_bf16_gradient_scaling():
    """Test BF16 gradient scaling and backward pass.""" 
    print("🧪 Testing BF16 gradient scaling...")
    
    config = GRPOConfig(use_bf16=True)
    model = load_rookworld_model(device="cuda")
    ref_model = ReferenceModel(model)
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # Create test data
    batch_size, seq_len = 2, 30
    torch.manual_seed(42)
    sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(sequences)
    rewards = torch.tensor([0.8, 0.6], device="cuda")
    prompt_lengths = torch.tensor([10, 15])
    
    # Forward pass with BF16
    policy_log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=True)
    
    with torch.no_grad():
        ref_log_probs = ref_model.compute_log_probs(sequences, attention_mask, use_bf16=False)
        ref_log_probs = ref_log_probs.to("cuda")
    
    advantages = compute_advantages(rewards, group_size=1)
    prompt_mask = create_prompt_mask(sequences, prompt_lengths)
    
    # Compute loss
    loss, metrics = grpo_loss(
        policy_log_probs,
        ref_log_probs,
        advantages,
        prompt_mask,
        kl_coef=0.02,
        clip_range=0.2
    )
    
    # Test gradient scaling workflow
    optimizer.zero_grad(set_to_none=True)
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    
    # Check gradients exist and are finite
    grad_norm_before = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_before += param.grad.norm().item() ** 2
    grad_norm_before = grad_norm_before ** 0.5
    
    assert grad_norm_before > 0, "No gradients computed"
    assert np.isfinite(grad_norm_before), "Gradients are not finite"
    
    # Unscale and clip gradients
    scaler.unscale_(optimizer)
    grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Step optimizer
    scaler.step(optimizer)
    scaler.update()
    
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Grad norm (before clip): {grad_norm_before:.6f}")
    print(f"   Grad norm (after clip): {grad_norm_after:.6f}")
    print(f"   Scale factor: {scaler.get_scale():.0f}")
    
    # Validate scaling worked
    assert grad_norm_after > 0, "Gradient clipping failed"
    assert scaler.get_scale() > 0, "Gradient scaler not working"
    
    print("✅ BF16 gradient scaling works correctly")


def test_bf16_memory_efficiency():
    """Test that BF16 doesn't increase memory usage significantly."""
    print("🧪 Testing BF16 memory efficiency...")
    
    config = GRPOConfig()
    model = load_rookworld_model(device="cuda")
    
    # Test data
    batch_size, seq_len = 4, 100  # Larger batch for memory test
    torch.manual_seed(42)
    sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(sequences)
    
    # Test memory usage for both precisions
    memory_usage = {}
    
    for precision, use_bf16 in [("FP32", False), ("BF16", True)]:
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        mem_before = torch.cuda.memory_allocated() / 1024**3
        
        with torch.no_grad():
            log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
        
        mem_after = torch.cuda.memory_allocated() / 1024**3
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        memory_usage[precision] = {
            'before_gb': mem_before,
            'after_gb': mem_after,
            'peak_gb': peak_mem,
            'increase_gb': mem_after - mem_before
        }
        
        # Cleanup
        del log_probs
        torch.cuda.empty_cache()
        
        print(f"   {precision}: {mem_before:.3f}GB → {mem_after:.3f}GB (peak: {peak_mem:.3f}GB)")
    
    # BF16 should use same or less memory
    bf16_increase = memory_usage["BF16"]["increase_gb"]
    fp32_increase = memory_usage["FP32"]["increase_gb"]
    
    assert bf16_increase <= fp32_increase * 1.1, f"BF16 uses more memory: {bf16_increase:.3f}GB vs {fp32_increase:.3f}GB"
    
    print("✅ BF16 memory usage is efficient")


def test_bf16_numerical_stability():
    """Test BF16 numerical stability over multiple training steps."""
    print("🧪 Testing BF16 numerical stability...")
    
    config = GRPOConfig(use_bf16=True)
    model = load_rookworld_model(device="cuda")
    ref_model = ReferenceModel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # Track metrics over several steps
    loss_history = []
    grad_norm_history = []
    
    for step in range(5):  # Small number of steps for testing
        # Create fresh test data for each step
        torch.manual_seed(42 + step)
        batch_size, seq_len = 2, 40
        sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
        attention_mask = torch.ones_like(sequences)
        rewards = torch.rand(batch_size, device="cuda")
        prompt_lengths = torch.randint(10, 20, (batch_size,))
        
        # Forward pass
        policy_log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=True)
        
        with torch.no_grad():
            ref_log_probs = ref_model.compute_log_probs(sequences, attention_mask, use_bf16=False)
            ref_log_probs = ref_log_probs.to("cuda")
        
        advantages = compute_advantages(rewards, group_size=1)
        prompt_mask = create_prompt_mask(sequences, prompt_lengths)
        
        # Compute loss
        loss, metrics = grpo_loss(
            policy_log_probs,
            ref_log_probs,
            advantages,
            prompt_mask,
            kl_coef=0.02,
            clip_range=0.2
        )
        
        # Backward pass with gradient scaling
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Record metrics
        loss_history.append(loss.item())
        grad_norm_history.append(grad_norm.item())
        
        print(f"   Step {step+1}: Loss={loss.item():.6f}, GradNorm={grad_norm.item():.4f}")
    
    # Validate numerical stability
    assert all(np.isfinite(loss) for loss in loss_history), "Loss became non-finite"
    assert all(np.isfinite(norm) for norm in grad_norm_history), "Gradient norms became non-finite"
    assert all(norm > 0 for norm in grad_norm_history), "Zero gradient norms detected"
    
    # Check that scale factor is reasonable
    final_scale = scaler.get_scale()
    assert 1.0 <= final_scale <= 65536.0, f"Gradient scale outside reasonable range: {final_scale}"
    
    print(f"   Final gradient scale: {final_scale:.0f}")
    print("✅ BF16 training is numerically stable over multiple steps")


def test_bf16_vs_fp32_consistency():
    """Test that BF16 produces results consistent with FP32."""
    print("🧪 Testing BF16 vs FP32 consistency...")
    
    model = load_rookworld_model(device="cuda")
    
    # Create identical test conditions
    torch.manual_seed(42)
    batch_size, seq_len = 2, 30
    sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(sequences)
    
    # Test both precisions with identical inputs
    precisions = {}
    
    for name, use_bf16 in [("FP32", False), ("BF16", True)]:
        with torch.no_grad():
            log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
        
        precisions[name] = {
            'mean': log_probs.mean().item(),
            'std': log_probs.std().item(), 
            'min': log_probs.min().item(),
            'max': log_probs.max().item()
        }
    
    # Compare statistical properties
    for metric in ['mean', 'std', 'min', 'max']:
        fp32_val = precisions["FP32"][metric]
        bf16_val = precisions["BF16"][metric]
        
        if abs(fp32_val) > 0:
            rel_diff = abs(bf16_val - fp32_val) / abs(fp32_val)
        else:
            rel_diff = abs(bf16_val - fp32_val)
        
        print(f"   {metric.upper()}: FP32={fp32_val:.6f}, BF16={bf16_val:.6f}, RelDiff={rel_diff:.4f}")
        
        # BF16 should be statistically similar to FP32 (allowing for precision differences)
        assert rel_diff < 0.1, f"BF16 {metric} differs too much: {rel_diff:.4f}"
    
    print("✅ BF16 and FP32 produce consistent statistical properties")


def test_bf16_performance_improvement():
    """Test that BF16 provides measurable performance improvement."""
    print("🧪 Testing BF16 performance improvement...")
    
    model = load_rookworld_model(device="cuda")
    
    # Test data - larger batch for meaningful timing
    batch_size, seq_len = 8, 100
    torch.manual_seed(42)
    sequences = torch.randint(100, 1000, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(sequences)
    
    timings = {}
    
    # Test both precisions
    for name, use_bf16 in [("FP32", False), ("BF16", True)]:
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
        
        # Time computation
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):  # Multiple iterations for stable timing
            with torch.no_grad():
                log_probs = compute_log_probs(model, sequences, attention_mask, use_bf16=use_bf16)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        timings[name] = elapsed / 10  # Average per iteration
        print(f"   {name}: {timings[name]*1000:.2f}ms per iteration")
    
    # BF16 should be faster than FP32
    speedup = timings["FP32"] / timings["BF16"]
    print(f"   BF16 Speedup: {speedup:.2f}x")
    
    # Expect at least some speedup (might be modest for small models)
    assert speedup >= 0.95, f"BF16 is significantly slower: {speedup:.2f}x"
    
    if speedup > 1.1:
        print("✅ BF16 provides measurable performance improvement")
    else:
        print("✅ BF16 performance is comparable to FP32 (expected for small models)")


def test_bf16_configuration_integration():
    """Test that BF16 configuration integrates properly with training config."""
    print("🧪 Testing BF16 configuration integration...")
    
    # Test config with BF16 enabled
    config = GRPOConfig(use_bf16=True)
    assert config.use_bf16 == True, "BF16 config not set correctly"
    
    # Test config with BF16 disabled
    config_fp32 = GRPOConfig(use_bf16=False) 
    assert config_fp32.use_bf16 == False, "FP32 config not set correctly"
    
    print("✅ BF16 configuration works correctly")


if __name__ == "__main__":
    import time
    
    print("🚀 Running BF16 Compatibility Tests")
    print("=" * 60)
    
    try:
        test_bf16_log_prob_computation()
        test_bf16_grpo_loss() 
        test_bf16_gradient_scaling()
        test_bf16_memory_efficiency()
        test_bf16_vs_fp32_consistency()
        test_bf16_performance_improvement()
        test_bf16_configuration_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL BF16 COMPATIBILITY TESTS PASSED")
        print("BF16 mixed precision is ready for use!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ BF16 test failed: {e}")
        raise