#!/usr/bin/env python3
"""
GRPO Loss Debug Test

Examines the exact GRPO loss computation to understand catastrophic divergence.
"""

import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rookworld_rlvr.model.config import ROOKWORLD_CONFIG
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch

def debug_grpo_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    tokenizer = TokenizerBridge()
    model = GPT2Model(ROOKWORLD_CONFIG).to(device)
    ref_model = GPT2Model(ROOKWORLD_CONFIG).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    
    config = GRPOConfig(lr=1e-4, group_size=2, kl_coef=0.01, device=str(device))
    trainer = GRPOTrainer(model, ref_model, config)
    
    # Create test batch
    texts = [
        "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",
        "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: a2a3"
    ]
    
    all_tokens = []
    target_start_indices = []
    
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.append(tokens)
        
        # Find target start (after "M:")
        target_start_idx = 0
        for j in range(len(tokens) - 1):
            current_decoded = tokenizer.decode([tokens[j]]).strip()
            next_decoded = tokenizer.decode([tokens[j + 1]]).strip()
            if current_decoded == 'M' and next_decoded == ':':
                target_start_idx = j + 2
                break
            elif current_decoded.endswith('M') and next_decoded == ':':
                target_start_idx = j + 2
                break
            elif current_decoded == 'M:':
                target_start_idx = j + 1
                break
        target_start_indices.append(target_start_idx)
    
    # Pad to same length
    max_len = max(len(tokens) for tokens in all_tokens)
    input_ids = []
    attention_mask = []
    
    for tokens in all_tokens:
        padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
        mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        input_ids.append(padded)
        attention_mask.append(mask)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    target_start_indices = torch.tensor(target_start_indices, device=device)
    
    # Get initial logprobs from reference model (simulates data generation policy)
    with torch.no_grad():
        initial_logprobs = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
    
    rewards = torch.tensor([1.0, 0.3], dtype=torch.float32, device=device)
    
    print("=" * 80)
    print("GRPO LOSS DEBUG")
    print("=" * 80)
    print(f"Target start indices: {target_start_indices.tolist()}")
    print(f"Initial logprobs: {initial_logprobs.tolist()}")
    print(f"Rewards: {rewards.tolist()}")
    
    # Create GRPO batch
    batch = GRPOBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_start_indices=target_start_indices,
        old_logprobs=initial_logprobs.clone(),
        rewards=rewards,
        position_fen="test",
        task_type="policy"
    )
    
    print("\nSimulating 3 training steps to see divergence...")
    
    for step in range(3):
        # Current logprobs
        current_logprobs = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        # Reference logprobs
        ref_logprobs = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
        
        print(f"\nStep {step}:")
        print(f"  Current logprobs: {current_logprobs.tolist()}")
        print(f"  Reference logprobs: {ref_logprobs.tolist()}")
        print(f"  Old logprobs: {batch.old_logprobs.tolist()}")
        
        # Manual GRPO computation
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        # PPO ratios
        ratios = torch.exp(current_logprobs - batch.old_logprobs)
        clipped_ratios = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2)
        
        # Policy objective
        unclipped_obj = ratios * advantages
        clipped_obj = clipped_ratios * advantages
        policy_objective = torch.min(unclipped_obj, clipped_obj).mean()
        policy_loss = -policy_objective
        
        # KL penalty
        kl_div = (current_logprobs - ref_logprobs).mean()
        kl_loss = 0.01 * kl_div
        
        total_loss = policy_loss + kl_loss
        
        print(f"  Baseline: {baseline:.3f}")
        print(f"  Advantages: {advantages.tolist()}")
        print(f"  PPO ratios: {ratios.tolist()}")
        print(f"  Clipped ratios: {clipped_ratios.tolist()}")
        print(f"  Unclipped obj: {unclipped_obj.tolist()}")
        print(f"  Clipped obj: {clipped_obj.tolist()}")
        print(f"  Policy objective: {policy_objective:.6f}")
        print(f"  Policy loss: {policy_loss:.6f}")
        print(f"  KL divergence: {kl_div:.6f}")
        print(f"  KL loss: {kl_loss:.6f}")
        print(f"  Total loss: {total_loss:.6f}")
        
        # Check for problematic ratios
        if torch.any(ratios > 10.0) or torch.any(ratios < 0.1):
            print(f"  ⚠️ WARNING: Extreme PPO ratios detected!")
            
        if abs(kl_div.item()) > 2.0:
            print(f"  ⚠️ WARNING: Large KL divergence detected!")
        
        # Training step
        trainer.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        trainer.optimizer.step()

if __name__ == "__main__":
    debug_grpo_loss()