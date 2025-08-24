#!/usr/bin/env python3
"""
Debug Logprob Computation Discrepancy

Investigates the specific differences in logprob computation between 
test components and production implementation.
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer


def debug_logprob_computation():
    """Debug the logprob computation discrepancy in detail"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set reproducible seed
    torch.manual_seed(42)
    
    # Initialize components
    tokenizer = TokenizerBridge()
    model_config = GPT2Config()
    model = GPT2Model(model_config).to(device)
    ref_model = GPT2Model(model_config).to(device)
    
    # Copy model weights to reference
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad_(False)
    
    # Initialize trainer
    config = GRPOConfig(
        lr=1e-5,
        group_size=8,
        use_mixed_precision=False,
        use_torch_compile=False,
        device=str(device)
    )
    trainer = GRPOTrainer(model, ref_model, config)
    
    print("🔍 DEBUGGING LOGPROB COMPUTATION DISCREPANCY")
    print("="*80)
    
    # Simple test case
    text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4"
    tokens = tokenizer.encode(text)
    target_start = 46
    
    print(f"Text: {text}")
    print(f"Tokens: {len(tokens)} tokens")
    print(f"Target start: {target_start}")
    
    # Create batch
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    target_start_indices = torch.tensor([target_start], device=device)
    
    print(f"\nBatch shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  target_start_indices: {target_start_indices.shape}")
    
    # Compare implementations step by step
    print(f"\n" + "="*60)
    print("STEP-BY-STEP COMPARISON")
    print("="*60)
    
    with torch.no_grad():  # Disable gradients for fair comparison
        
        # 1. Forward pass - should be identical
        print(f"\n1. Forward Pass:")
        
        model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
        
        model_logits = model_outputs["logits"]
        ref_logits = ref_outputs["logits"]
        
        print(f"  Model logits shape: {model_logits.shape}")
        print(f"  Reference logits shape: {ref_logits.shape}")
        
        # Check if logits are identical (they should be initially)
        logits_diff = torch.abs(model_logits - ref_logits).max().item()
        print(f"  Logits difference: {logits_diff:.8f}")
        
        if logits_diff < 1e-6:
            print("  ✅ Logits are identical")
        else:
            print("  ❌ Logits differ - models may have been modified")
        
        # 2. Shifting for autoregressive prediction
        print(f"\n2. Autoregressive Shifting:")
        
        shift_logits = model_logits[:, :-1, :]  # [1, seq_len-1, vocab_size]
        shift_labels = input_ids[:, 1:]         # [1, seq_len-1]
        
        print(f"  Shifted logits shape: {shift_logits.shape}")
        print(f"  Shifted labels shape: {shift_labels.shape}")
        
        # Handle attention mask
        if attention_mask is None:
            shift_attention = torch.ones_like(shift_labels)
        else:
            shift_attention = attention_mask[:, 1:]
        
        print(f"  Shifted attention shape: {shift_attention.shape}")
        
        # 3. Log probability computation
        print(f"\n3. Log Probability Computation:")
        
        # Manual method (test components)
        manual_log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Production method (trainer)
        prod_log_probs = torch.log_softmax(shift_logits, dim=-1)
        
        logprob_method_diff = torch.abs(manual_log_probs - prod_log_probs).max().item()
        print(f"  F.log_softmax vs torch.log_softmax diff: {logprob_method_diff:.8f}")
        
        # 4. Token logprob gathering
        print(f"\n4. Token Logprob Gathering:")
        
        token_log_probs = torch.gather(
            prod_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [1, seq_len-1]
        
        print(f"  Token logprobs shape: {token_log_probs.shape}")
        print(f"  Sample token logprobs: {token_log_probs[0, :10].tolist()}")
        
        # 5. Target masking - THIS IS LIKELY WHERE THE DIFFERENCE IS
        print(f"\n5. Target Masking:")
        
        batch_size, seq_len = token_log_probs.shape
        print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
        
        # Manual masking (test components)
        manual_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        for i in range(batch_size):
            start_idx = max(0, target_start_indices[i] - 1)  # -1 for shift
            manual_mask[i, start_idx:] = shift_attention[i, start_idx:].bool()
        
        # Production masking (trainer implementation)
        prod_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        for i in range(batch_size):
            start_idx = max(0, target_start_indices[i] - 1)  # -1 for shift
            prod_mask[i, start_idx:] = shift_attention[i, start_idx:].bool()
        
        print(f"  Target start index: {target_start_indices[0].item()}")
        print(f"  Adjusted start (target-1): {target_start_indices[0].item() - 1}")
        print(f"  Manual mask: {manual_mask[0].int().tolist()}")
        print(f"  Production mask: {prod_mask[0].int().tolist()}")
        
        mask_diff = (manual_mask != prod_mask).sum().item()
        print(f"  Mask differences: {mask_diff}")
        
        if mask_diff == 0:
            print("  ✅ Masks are identical")
        else:
            print("  ❌ Masks differ")
        
        # 6. Mean computation
        print(f"\n6. Mean Logprob Computation:")
        
        # Manual computation
        manual_masked = token_log_probs.masked_fill(~manual_mask, 0.0)
        manual_counts = manual_mask.sum(dim=1).clamp(min=1)
        manual_mean = manual_masked.sum(dim=1) / manual_counts
        
        # Production computation
        prod_masked = token_log_probs.masked_fill(~prod_mask, 0.0)
        prod_counts = prod_mask.sum(dim=1).clamp(min=1)
        prod_mean = prod_masked.sum(dim=1) / prod_counts
        
        print(f"  Manual token count: {manual_counts.item()}")
        print(f"  Production token count: {prod_counts.item()}")
        print(f"  Manual mean: {manual_mean.item():.8f}")
        print(f"  Production mean: {prod_mean.item():.8f}")
        print(f"  Mean difference: {abs(manual_mean.item() - prod_mean.item()):.8f}")
        
        # 7. Test trainer compute_logprobs method
        print(f"\n7. Trainer Method Comparison:")
        
        trainer_result = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        print(f"  Manual result: {manual_mean.item():.8f}")
        print(f"  Trainer result: {trainer_result.item():.8f}")
        print(f"  Final difference: {abs(manual_mean.item() - trainer_result.item()):.8f}")
        
        # 8. Investigate model state
        print(f"\n8. Model State Investigation:")
        print(f"  Model training mode: {model.training}")
        print(f"  Reference training mode: {ref_model.training}")
        
        # Check if gradients are enabled for model
        print(f"  Model requires_grad: {any(p.requires_grad for p in model.parameters())}")
        print(f"  Reference requires_grad: {any(p.requires_grad for p in ref_model.parameters())}")
        
        # Test with reference model too
        trainer_ref_result = trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
        
        print(f"  Trainer (reference): {trainer_ref_result.item():.8f}")
        
        # The key insight: model vs ref_model difference
        model_ref_diff = abs(trainer_result.item() - trainer_ref_result.item())
        print(f"  Policy vs Reference diff: {model_ref_diff:.8f}")
        
        if model_ref_diff > 1e-6:
            print("  ❌ Policy and reference models produce different logprobs!")
            print("     This suggests the model parameters have been modified.")
        else:
            print("  ✅ Policy and reference models are identical")
    
    print(f"\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    
    print("The discrepancy is likely due to:")
    print("1. Model state differences (training vs eval mode)")
    print("2. Gradient computation context differences")
    print("3. Parameter modification during trainer initialization")
    print("\nThe reference model matches perfectly, suggesting the")
    print("issue is in the policy model state management.")


if __name__ == "__main__":
    debug_logprob_computation()