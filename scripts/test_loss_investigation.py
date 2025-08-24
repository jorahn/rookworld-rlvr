#!/usr/bin/env python3
"""
Loss Investigation Test

Investigates why the GRPO loss becomes negative and the KL divergence
becomes extremely negative, which indicates a bug in the implementation.
"""

import torch
import torch.nn.functional as F
import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rookworld_rlvr.model.config import ROOKWORLD_CONFIG
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch

def investigate_loss_computation():
    """Investigate the loss computation step by step"""
    
    print("="*80)
    print("LOSS COMPUTATION INVESTIGATION")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple test case
    tokenizer = TokenizerBridge()
    model = GPT2Model(ROOKWORLD_CONFIG).to(device)
    ref_model = GPT2Model(ROOKWORLD_CONFIG).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    
    # Simple test batch
    text1 = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4"
    text2 = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: h2h4"
    
    tokens1 = tokenizer.encode(text1)
    tokens2 = tokenizer.encode(text2)
    
    # Create batch
    max_len = max(len(tokens1), len(tokens2))
    input_ids = torch.tensor([
        tokens1 + [tokenizer.pad_token_id] * (max_len - len(tokens1)),
        tokens2 + [tokenizer.pad_token_id] * (max_len - len(tokens2))
    ], device=device)
    
    attention_mask = torch.tensor([
        [1] * len(tokens1) + [0] * (max_len - len(tokens1)),
        [1] * len(tokens2) + [0] * (max_len - len(tokens2))
    ], device=device)
    
    target_start_indices = torch.tensor([len(tokens1)-1, len(tokens2)-1], device=device)
    rewards = torch.tensor([1.0, 0.2], device=device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target indices: {target_start_indices}")
    print(f"Rewards: {rewards}")
    print("")
    
    # Create trainer
    config = GRPOConfig(
        lr=1e-4,
        group_size=2,
        use_mixed_precision=False,
        use_torch_compile=False,
        kl_coef=0.01,
        device=str(device)
    )
    trainer = GRPOTrainer(model, ref_model, config)
    
    # Step 1: Get initial reference logprobs
    print("Step 1: Initial Reference Logprobs")
    print("-" * 40)
    with torch.no_grad():
        initial_ref_logprobs = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
    print(f"Initial ref logprobs: {initial_ref_logprobs}")
    print("")
    
    # Step 2: After some training iterations, get current logprobs
    print("Step 2: Simulating Training Effect")
    print("-" * 40)
    
    # Simulate several training steps to see the effect
    grpo_batch = GRPOBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_start_indices=target_start_indices,
        old_logprobs=initial_ref_logprobs,
        rewards=rewards,
        position_fen="test",
        task_type="policy"
    )
    
    for iteration in range(10):
        # Get current logprobs
        current_logprobs = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        # Get reference logprobs (should stay constant)
        ref_logprobs = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
        
        print(f"Iteration {iteration}:")
        print(f"  Current logprobs: {current_logprobs.tolist()}")
        print(f"  Reference logprobs: {ref_logprobs.tolist()}")
        print(f"  Old logprobs: {grpo_batch.old_logprobs.tolist()}")
        
        # Manual GRPO loss computation to debug
        print(f"  Manual GRPO Loss Computation:")
        
        # Step 1: Compute baseline (group mean reward)
        baseline = rewards.mean()
        advantages = rewards - baseline
        print(f"    Rewards: {rewards.tolist()}")
        print(f"    Baseline: {baseline.item():.6f}")
        print(f"    Advantages: {advantages.tolist()}")
        
        # Step 2: Compute ratios
        # CRITICAL: Which logprobs should we compare?
        # Option A: current vs old (from when we sampled)
        ratio_vs_old = torch.exp(current_logprobs - grpo_batch.old_logprobs)
        
        # Option B: current vs reference (current reference model)
        ratio_vs_ref = torch.exp(current_logprobs - ref_logprobs)
        
        print(f"    Current vs Old ratio: {ratio_vs_old.tolist()}")
        print(f"    Current vs Ref ratio: {ratio_vs_ref.tolist()}")
        
        # Step 3: PPO clipping (using old logprobs)
        clipped_ratio = torch.clamp(ratio_vs_old, 1.0 - 0.2, 1.0 + 0.2)
        
        unclipped_obj = ratio_vs_old * advantages
        clipped_obj = clipped_ratio * advantages
        
        policy_objective = torch.min(unclipped_obj, clipped_obj).mean()
        policy_loss = -policy_objective  # Negate to minimize
        
        print(f"    Unclipped objective: {unclipped_obj.tolist()}")
        print(f"    Clipped objective: {clipped_obj.tolist()}")
        print(f"    Policy objective: {policy_objective.item():.6f}")
        print(f"    Policy loss: {policy_loss.item():.6f}")
        
        # Step 4: KL penalty
        # CRITICAL: KL with respect to what?
        kl_div_vs_ref = (current_logprobs - ref_logprobs).mean()
        kl_loss = config.kl_coef * kl_div_vs_ref
        
        total_loss = policy_loss + kl_loss
        
        print(f"    KL div (current vs ref): {kl_div_vs_ref.item():.6f}")
        print(f"    KL loss: {kl_loss.item():.6f}")
        print(f"    Total loss: {total_loss.item():.6f}")
        
        # Compare with trainer computation
        trainer_loss, trainer_metrics = trainer.compute_grpo_loss(grpo_batch)
        print(f"  Trainer loss: {trainer_loss.item():.6f}")
        print(f"  Trainer policy loss: {trainer_metrics['policy_loss']:.6f}")
        print(f"  Trainer KL loss: {trainer_metrics['kl_loss']:.6f}")
        print("")
        
        # Training step
        trainer.optimizer.zero_grad()
        trainer_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        trainer.optimizer.step()
        
        # Stop if loss becomes very negative
        if total_loss.item() < -1.0:
            print(f"⚠️  Loss became very negative ({total_loss.item():.3f}), stopping investigation")
            break
    
    print("")
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    
    print("Key observations:")
    print("1. Reference model logprobs should remain constant")
    print("2. Current model logprobs should change with training")
    print("3. PPO uses 'old_logprobs' from when actions were sampled")
    print("4. KL penalty should be small and positive usually")
    print("")
    
    print("Potential issues:")
    print("1. Are we using the right reference for KL divergence?")
    print("2. Are old_logprobs being updated incorrectly?")
    print("3. Is the model diverging too far from reference?")
    print("4. Are we computing KL divergence correctly?")

if __name__ == "__main__":
    investigate_loss_computation()