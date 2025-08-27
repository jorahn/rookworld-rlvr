"""
Test script for enhanced GRPO features

Tests the new KL divergence measures, advanced baselines, and value function.
"""

import torch
import numpy as np
from grpo import (
    compute_kl_divergence, 
    compute_advantages,
    AdaptiveKLController,
    ValueFunction,
    grpo_loss
)


def test_kl_divergence_types():
    """Test different KL divergence measures."""
    print("Testing KL divergence types...")
    
    # Create mock data
    batch_size, seq_len = 4, 20
    policy_log_probs = torch.randn(batch_size, seq_len) * 0.1
    ref_log_probs = torch.randn(batch_size, seq_len) * 0.1  
    completion_mask = torch.ones(batch_size, seq_len)
    
    # Test all KL types
    for kl_type in ["forward", "reverse", "symmetric"]:
        kl_div = compute_kl_divergence(
            policy_log_probs, ref_log_probs, completion_mask, kl_type
        )
        print(f"  {kl_type}: {kl_div.mean().item():.4f}")
        assert kl_div.shape == (batch_size,)
        assert torch.isfinite(kl_div).all()
    
    print("✓ KL divergence types working correctly")


def test_advanced_baselines():
    """Test advanced baseline computation methods."""
    print("\nTesting advanced baseline methods...")
    
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4])
    group_size = 4
    
    # Test different baseline types
    for baseline_type in ["group_mean", "ema", "adaptive"]:
        advantages = compute_advantages(
            rewards,
            group_size=group_size,
            baseline_type=baseline_type
        )
        print(f"  {baseline_type}: mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}")
        assert advantages.shape == rewards.shape
        assert torch.isfinite(advantages).all()
    
    print("✓ Advanced baselines working correctly")


def test_adaptive_kl_controller():
    """Test adaptive KL coefficient control."""
    print("\nTesting adaptive KL controller...")
    
    controller = AdaptiveKLController(
        init_kl_coef=0.02,
        target_kl=0.01,
        horizon=1000
    )
    
    initial_kl_coef = controller.kl_coef
    
    # Simulate high KL divergence
    for _ in range(5):
        new_coef = controller.update(0.05)  # High KL
    
    print(f"  Initial KL coef: {initial_kl_coef:.4f}")
    print(f"  After high KL: {new_coef:.4f}")
    assert new_coef > initial_kl_coef, "KL coefficient should increase with high KL"
    
    # Simulate low KL divergence
    for _ in range(10):
        new_coef = controller.update(0.001)  # Very low KL
    
    print(f"  After low KL: {new_coef:.4f}")
    print("✓ Adaptive KL controller working correctly")


def test_value_function():
    """Test value function."""
    print("\nTesting value function...")
    
    batch_size, seq_len, hidden_size = 2, 10, 768
    value_fn = ValueFunction(hidden_size)
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    values = value_fn(hidden_states)
    
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {values.shape}")
    print(f"  Value range: [{values.min().item():.3f}, {values.max().item():.3f}]")
    
    assert values.shape == (batch_size, seq_len)
    assert torch.isfinite(values).all()
    
    print("✓ Value function working correctly")


def test_enhanced_grpo_loss():
    """Test enhanced GRPO loss with all features."""
    print("\nTesting enhanced GRPO loss...")
    
    batch_size, seq_len = 8, 20
    
    # Create mock data
    policy_log_probs = torch.randn(batch_size, seq_len) * 0.1
    ref_log_probs = torch.randn(batch_size, seq_len) * 0.1
    advantages = torch.randn(batch_size) * 0.5
    prompt_mask = torch.zeros(batch_size, seq_len)
    prompt_mask[:, :seq_len//2] = 1  # First half is prompt
    
    # Test with value function
    values = torch.randn(batch_size)
    value_targets = torch.randn(batch_size)
    
    for kl_type in ["forward", "reverse", "symmetric"]:
        loss, metrics = grpo_loss(
            policy_log_probs,
            ref_log_probs,
            advantages,
            prompt_mask,
            kl_coef=0.02,
            clip_range=0.2,
            kl_type=kl_type,
            values=values,
            value_targets=value_targets,
            value_loss_coef=0.1,
            entropy_coef=0.01
        )
        
        print(f"  {kl_type} - Loss: {loss.item():.4f}")
        print(f"    PG: {metrics['pg_loss']:.4f}, KL: {metrics['kl_div']:.4f}, Value: {metrics['value_loss']:.4f}")
        
        assert torch.isfinite(loss)
        assert "kl_forward" in metrics
        assert "kl_reverse" in metrics
        assert "kl_symmetric" in metrics
        assert "ratio_outliers" in metrics
    
    print("✓ Enhanced GRPO loss working correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ENHANCED GRPO FEATURES")
    print("=" * 60)
    
    test_kl_divergence_types()
    test_advanced_baselines()
    test_adaptive_kl_controller()
    test_value_function()
    test_enhanced_grpo_loss()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("Enhanced GRPO features are working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()