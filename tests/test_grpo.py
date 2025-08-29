"""
Test script for GRPO implementation

Quick tests to verify the GRPO algorithm works correctly.
"""

import torch
import tiktoken
import numpy as np

# Mini modules
from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.grpo import (
    compute_log_probs,
    compute_advantages,
    grpo_loss,
    create_prompt_mask,
    ReferenceModel
)
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples


def test_log_probs():
    """Test log probability computation."""
    print("Testing log probability computation...")
    
    # Small test
    config = GRPOConfig()
    model = load_rookworld_model(device="cpu")  # CPU for testing
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # Compute log probs
    log_probs = compute_log_probs(model, input_ids, attention_mask)
    
    # Check shape
    assert log_probs.shape == (batch_size, seq_len - 1), f"Wrong shape: {log_probs.shape}"
    
    # Check values are log probs (negative, not NaN)
    assert torch.all(log_probs <= 0), "Log probs should be <= 0"
    assert not torch.any(torch.isnan(log_probs)), "NaN in log probs"
    
    print("✓ Log probability computation works")


def test_advantages():
    """Test advantage computation."""
    print("Testing advantage computation...")
    
    # Test data
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9, 0.3])
    group_size = 3
    
    # Compute advantages
    advantages = compute_advantages(rewards, group_size)
    
    # Check properties
    assert advantages.shape == rewards.shape
    assert abs(advantages.mean()) < 1e-6, "Advantages should be centered"
    assert abs(advantages.std() - 1.0) < 0.1, "Advantages should be normalized"
    
    print("✓ Advantage computation works")


def test_grpo_loss():
    """Test GRPO loss computation."""
    print("Testing GRPO loss...")
    
    # Create dummy data
    batch_size, seq_len = 4, 20
    policy_log_probs = torch.randn(batch_size, seq_len - 1) * 0.1 - 2.0  # Negative values
    ref_log_probs = torch.randn(batch_size, seq_len - 1) * 0.1 - 2.0
    advantages = torch.randn(batch_size)
    prompt_lengths = torch.tensor([5, 6, 5, 7])
    prompt_mask = create_prompt_mask(torch.zeros(batch_size, seq_len), prompt_lengths)
    
    # Compute loss
    loss, metrics = grpo_loss(
        policy_log_probs,
        ref_log_probs,
        advantages,
        prompt_mask,
        kl_coef=0.02,
        clip_range=0.2
    )
    
    # Check outputs
    assert isinstance(loss, torch.Tensor), "Loss should be tensor"
    assert loss.shape == (), "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert "pg_loss" in metrics, "Missing pg_loss in metrics"
    assert "kl_div" in metrics, "Missing kl_div in metrics"
    
    print(f"✓ GRPO loss works - Loss: {loss.item():.4f}")


def test_reference_model():
    """Test reference model creation and freezing."""
    print("Testing reference model...")
    
    config = GRPOConfig()
    model = load_rookworld_model(device="cpu")
    
    # Create reference
    ref_model = ReferenceModel(model)
    
    # Check parameters are frozen
    for param in ref_model.model.parameters():
        assert not param.requires_grad, "Reference model should be frozen"
    
    # Test log prob computation
    input_ids = torch.randint(0, 1000, (2, 10))
    log_probs = ref_model.compute_log_probs(input_ids)
    
    assert log_probs.shape == (2, 9), f"Wrong shape: {log_probs.shape}"
    assert not torch.any(torch.isnan(log_probs)), "NaN in reference log probs"
    
    print("✓ Reference model works")


def test_data_integration():
    """Test integration with dataset and reward scorer."""
    print("Testing data integration...")
    
    # Load small dataset
    samples = load_and_prepare_samples(n_samples=5, seed=42)
    
    # Check we have both task types
    p_tasks = [s for s in samples if s[0] == 'P']
    a_tasks = [s for s in samples if s[0] == 'A']
    
    assert len(p_tasks) > 0, "No P: tasks found"
    assert len(a_tasks) > 0, "No A: tasks found"
    
    # Test reward computation
    from rookworld_rlvr.reward_scorer import compute_grpo_rewards
    
    prompts = [s[1] for s in samples[:2]]
    completions = [s[2] for s in samples[:2]]  # Use ground truth
    
    advantages, details = compute_grpo_rewards(
        prompts,
        completions,
        group_size=1,
        reward_shaping="graduated",
        verbose=False
    )
    
    assert len(advantages) == len(prompts)
    assert all(-3 <= a <= 3 for a in advantages), "Advantages out of expected range"
    
    print("✓ Data integration works")


def test_full_pipeline():
    """Test a mini training step."""
    print("Testing full pipeline...")
    
    # Mini config
    config = GRPOConfig()
    config.n_train_samples = 2
    config.batch_size = 1
    config.k_samples = 2
    
    # Setup
    device = torch.device("cpu")  # CPU for testing
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_rookworld_model(device=device)
    ref_model = ReferenceModel(model)
    
    # Load data
    samples = load_and_prepare_samples(n_samples=2, seed=42)
    
    # Generate completions (simplified)
    prompt = samples[0][1]
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=20,  # Short for testing
            temperature=0.8
        )
    
    # Compute log probs
    policy_log_probs = compute_log_probs(model, generated)
    ref_log_probs = ref_model.compute_log_probs(generated)
    
    # Create dummy advantages and masks
    advantages = torch.tensor([0.5], device=device)
    prompt_mask = create_prompt_mask(generated, torch.tensor([len(prompt_ids)]))
    
    # Compute loss
    loss, metrics = grpo_loss(
        policy_log_probs,
        ref_log_probs,
        advantages,
        prompt_mask
    )
    
    # Check we can backward
    loss.backward()
    
    print(f"✓ Full pipeline works - Loss: {loss.item():.4f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING GRPO IMPLEMENTATION")
    print("=" * 60)
    print()
    
    try:
        test_log_probs()
        test_advantages()
        test_grpo_loss()
        test_reference_model()
        test_data_integration()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()