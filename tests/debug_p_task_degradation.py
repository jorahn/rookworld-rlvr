#!/usr/bin/env python3
"""
Debug P: Task Quality Degradation

Investigates why A: tasks achieve perfect quality preservation (0.0% delta)
while P: tasks still show degradation (-35.9%). Focus on specific differences.
"""

import torch
import sys
import tiktoken
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.reward_scorer import RewardScorer


def compare_individual_generations():
    """Compare individual generation differences to understand P: task degradation."""
    print("🔍 Comparing Individual Generation Quality")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load samples
    samples = load_and_prepare_samples(n_samples=10, seed=42)
    p_sample = next(s for s in samples if s[0] == 'P')
    
    task_type, prompt, ground_truth, data = p_sample
    print(f"P: task prompt: {prompt}")
    
    prompt_ids = tokenizer.encode(prompt, disallowed_special=())
    original_len = len(prompt_ids)
    
    generation_params = {
        'max_new_tokens': 144,  # Full length as user suggested
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    scorer = RewardScorer(
        reward_shaping="graduated",
        continuous_components={"fen_similarity": "exponential", "evaluations": "linear"}
    )
    
    k_samples = 4
    
    # Sequential approach (working)
    print(f"\n🔄 Sequential Generation (Reference):")
    sequential_results = []
    
    for k in range(k_samples):
        torch.manual_seed(42 + k)  # Different seeds
        
        prompt_tensor = torch.tensor([prompt_ids], device="cuda")
        
        with torch.no_grad():
            output = model.generate(prompt_tensor, **generation_params)
        
        completion_tokens = output[0, original_len:].cpu().tolist()
        completion = tokenizer.decode(completion_tokens).strip()
        if '<|endoftext|>' in completion:
            completion = completion.replace('<|endoftext|>', '').strip()
        
        reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
        
        sequential_results.append({
            'completion': completion,
            'reward': reward,
            'tokens': completion_tokens,
            'length': len(completion_tokens),
            'format_valid': details.format_valid
        })
        
        print(f"  K={k+1}: R={reward:.3f}, Len={len(completion_tokens)}, Valid={details.format_valid}")
        print(f"       '{completion[:80]}...'")
    
    # Improved batch approach - try different seed strategies
    seed_strategies = [
        ("Single seed", lambda k: 42),
        ("Sequential seeds", lambda k: 42 + k),
        ("Offset seeds", lambda k: 42 + k * 100),
    ]
    
    for strategy_name, seed_func in seed_strategies:
        print(f"\n⚡ Batch Generation ({strategy_name}):")
        
        # Create batch with proper left padding
        max_len = original_len + 30  # Extra space for batch context
        batch_input = torch.full((k_samples, max_len), 50256, device="cuda")
        batch_attention = torch.zeros((k_samples, max_len), device="cuda")
        
        for k in range(k_samples):
            start_pos = max_len - original_len
            batch_input[k, start_pos:] = torch.tensor(prompt_ids, device="cuda")
            batch_attention[k, start_pos:] = 1
        
        # Try the seed strategy
        torch.manual_seed(seed_func(0))  # Set base seed
        
        try:
            with torch.no_grad():
                batch_outputs = model.generate(
                    batch_input,
                    attention_mask=batch_attention,
                    **generation_params
                )
            
            batch_results = []
            
            for k in range(k_samples):
                completion_tokens = batch_outputs[k, max_len:].cpu().tolist()
                completion = tokenizer.decode(completion_tokens).strip()
                if '<|endoftext|>' in completion:
                    completion = completion.replace('<|endoftext|>', '').strip()
                
                reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
                
                batch_results.append({
                    'completion': completion,
                    'reward': reward,
                    'tokens': completion_tokens,
                    'length': len(completion_tokens),
                    'format_valid': details.format_valid
                })
                
                print(f"  K={k+1}: R={reward:.3f}, Len={len(completion_tokens)}, Valid={details.format_valid}")
                print(f"       '{completion[:80]}...'")
            
            # Compare with sequential
            seq_rewards = [r['reward'] for r in sequential_results]
            batch_rewards = [r['reward'] for r in batch_results]
            
            seq_mean = np.mean(seq_rewards)
            batch_mean = np.mean(batch_rewards)
            quality_delta = (batch_mean - seq_mean) / seq_mean if seq_mean != 0 else 0
            
            print(f"\n  Quality comparison ({strategy_name}):")
            print(f"    Sequential: {seq_mean:.3f}")
            print(f"    Batched: {batch_mean:.3f}")  
            print(f"    Delta: {quality_delta:.1%}")
            
            # Check completion diversity
            batch_unique = len(set(r['completion'] for r in batch_results))
            seq_unique = len(set(r['completion'] for r in sequential_results))
            
            print(f"    Sequential unique: {seq_unique}/{k_samples}")
            print(f"    Batched unique: {batch_unique}/{k_samples}")
            
            if abs(quality_delta) < 0.1:
                print(f"    ✅ Quality well preserved")
            elif abs(quality_delta) < 0.2:
                print(f"    ⚠️ Moderate quality change")
            else:
                print(f"    ❌ Significant quality degradation")
        
        except Exception as e:
            print(f"  ❌ Strategy failed: {e}")


def test_generation_length_effects():
    """Test if max_new_tokens needs adjustment for batched generation."""
    print(f"\n📏 Testing Generation Length Effects")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Get P: task sample
    samples = load_and_prepare_samples(n_samples=5, seed=42)
    p_sample = next(s for s in samples if s[0] == 'P')
    task_type, prompt, ground_truth, data = p_sample
    
    prompt_ids = tokenizer.encode(prompt, disallowed_special=())
    
    # Test different max_new_tokens values
    token_lengths = [100, 144, 180, 200]
    
    generation_params_base = {
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    scorer = RewardScorer(reward_shaping="graduated")
    
    print(f"Testing max_new_tokens values: {token_lengths}")
    
    for max_tokens in token_lengths:
        print(f"\n📋 max_new_tokens = {max_tokens}:")
        
        generation_params = {**generation_params_base, 'max_new_tokens': max_tokens}
        
        # Sequential
        torch.manual_seed(42)
        prompt_tensor = torch.tensor([prompt_ids], device="cuda")
        
        with torch.no_grad():
            seq_output = model.generate(prompt_tensor, **generation_params)
        
        seq_completion = tokenizer.decode(seq_output[0, len(prompt_ids):].cpu().tolist()).strip()
        seq_reward, seq_details = scorer.score_single(prompt, seq_completion, ground_truth=ground_truth, log_details=False)
        
        # Batched (improved implementation)
        torch.manual_seed(42)
        batch_tensor = prompt_tensor.repeat(1, 1)  # Single item batch for comparison
        
        with torch.no_grad():
            batch_output = model.generate(batch_tensor, **generation_params)
        
        batch_completion = tokenizer.decode(batch_output[0, len(prompt_ids):].cpu().tolist()).strip()
        batch_reward, batch_details = scorer.score_single(prompt, batch_completion, ground_truth=ground_truth, log_details=False)
        
        print(f"  Sequential: R={seq_reward:.3f}, Len={len(seq_completion)}, Valid={seq_details.format_valid}")
        print(f"  Batched:    R={batch_reward:.3f}, Len={len(batch_completion)}, Valid={batch_details.format_valid}")
        print(f"  Delta: {((batch_reward - seq_reward) / seq_reward * 100) if seq_reward != 0 else 0:.1f}%")
        
        # Check if completions are identical
        identical = seq_completion == batch_completion
        print(f"  Identical: {identical}")
        
        if not identical:
            print(f"    Sequential: '{seq_completion[:50]}...'")
            print(f"    Batched:    '{batch_completion[:50]}...'")


if __name__ == "__main__":
    print("🚀 Debug P: Task Quality Degradation")
    print("=" * 80)
    
    try:
        # Compare individual generations
        compare_individual_generations()
        
        # Test generation length effects
        test_generation_length_effects()
        
        print(f"\n" + "=" * 80)
        print(f"🎯 P: TASK DEGRADATION ANALYSIS")
        print(f"✅ A: tasks: Perfect preservation (0.0% delta)")
        print(f"⚠️ P: tasks: Still some degradation (-35.9%)")
        print(f"\n💡 Next steps:")
        print(f"  1. Fine-tune seed management for better P: task diversity")
        print(f"  2. Test max_new_tokens adjustment as user suggested")
        print(f"  3. Study exact generation differences between methods")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        raise