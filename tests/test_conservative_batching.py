#!/usr/bin/env python3
"""
Conservative Batching Strategy Test

Tests batching across multiple prompts (not k_samples) to avoid diversity issues
while still achieving speedup. Groups prompts by similar length to avoid padding problems.
"""

import torch
import sys
import tiktoken
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.reward_scorer import RewardScorer


def group_prompts_by_length(samples, max_length_diff=5):
    """Group samples by similar prompt lengths to enable efficient batching."""
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Calculate prompt lengths
    samples_with_lengths = []
    for sample in samples:
        task_type, prompt, ground_truth, data = sample
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        samples_with_lengths.append((sample, len(prompt_ids)))
    
    # Sort by length
    samples_with_lengths.sort(key=lambda x: x[1])
    
    # Group by similar lengths
    groups = []
    current_group = []
    current_length = None
    
    for sample, length in samples_with_lengths:
        if current_length is None or abs(length - current_length) <= max_length_diff:
            current_group.append(sample)
            current_length = length
        else:
            if current_group:
                groups.append(current_group)
            current_group = [sample]
            current_length = length
    
    if current_group:
        groups.append(current_group)
    
    return groups


def test_sequential_vs_conservative_batching():
    """Compare fully sequential vs conservative batching approach."""
    print("🚀 Testing Conservative Batching Strategy")
    print("=" * 70)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load samples and group by length
    samples = load_and_prepare_samples(n_samples=12, seed=42)
    length_groups = group_prompts_by_length(samples, max_length_diff=3)
    
    print(f"Loaded {len(samples)} samples, grouped into {len(length_groups)} length groups:")
    for i, group in enumerate(length_groups):
        lengths = [len(tokenizer.encode(s[1], disallowed_special=())) for s in group]
        print(f"  Group {i+1}: {len(group)} samples, lengths {min(lengths)}-{max(lengths)} tokens")
    
    generation_params = {
        'max_new_tokens': 144,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    k_samples = 4
    scorer = RewardScorer(reward_shaping="graduated")
    
    # Method 1: Fully sequential (current approach)
    print(f"\n🔄 Method 1: Fully Sequential")
    sequential_start = time.time()
    sequential_results = []
    
    for sample in samples:
        task_type, prompt, ground_truth, data = sample
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        
        sample_rewards = []
        for k in range(k_samples):
            torch.manual_seed(42 + len(sequential_results) * k_samples + k)
            
            prompt_tensor = torch.tensor([prompt_ids], device="cuda")
            
            with torch.no_grad():
                output = model.generate(prompt_tensor, **generation_params)
            
            completion = tokenizer.decode(output[0, len(prompt_ids):].cpu().tolist()).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            reward, _ = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            sample_rewards.append(reward)
        
        sequential_results.extend(sample_rewards)
    
    sequential_time = time.time() - sequential_start
    
    # Method 2: Conservative batching (batch prompts, keep k_samples sequential)  
    print(f"\n⚡ Method 2: Conservative Batching")
    batched_start = time.time()
    batched_results = []
    
    for group in length_groups:
        if len(group) < 2:  # Skip single-item groups
            # Process individually
            for sample in group:
                task_type, prompt, ground_truth, data = sample
                prompt_ids = tokenizer.encode(prompt, disallowed_special=())
                
                sample_rewards = []
                for k in range(k_samples):
                    torch.manual_seed(42 + len(batched_results) + k)
                    
                    prompt_tensor = torch.tensor([prompt_ids], device="cuda")
                    
                    with torch.no_grad():
                        output = model.generate(prompt_tensor, **generation_params)
                    
                    completion = tokenizer.decode(output[0, len(prompt_ids):].cpu().tolist()).strip()
                    if '<|endoftext|>' in completion:
                        completion = completion.replace('<|endoftext|>', '').strip()
                    
                    reward, _ = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
                    sample_rewards.append(reward)
                
                batched_results.extend(sample_rewards)
            continue
        
        # Batch process this group (similar length prompts with padding)
        print(f"  Processing group of {len(group)} similar-length prompts")
        
        all_prompt_ids = [tokenizer.encode(sample[1], disallowed_special=()) for sample in group]
        max_group_length = max(len(ids) for ids in all_prompt_ids)
        
        print(f"    Max length in group: {max_group_length} tokens")
        
        # Create padded batch tensor for this group
        group_batch = torch.full((len(group), max_group_length), 50256, device="cuda")
        group_attention = torch.zeros((len(group), max_group_length), device="cuda")
        
        for i, prompt_ids in enumerate(all_prompt_ids):
            # Left padding within group
            start_pos = max_group_length - len(prompt_ids)
            group_batch[i, start_pos:] = torch.tensor(prompt_ids, device="cuda")
            group_attention[i, start_pos:] = 1
        
        # For each k_sample iteration, batch all prompts in group
        for k in range(k_samples):
            torch.manual_seed(42 + len(batched_results) + k)
            
            # Generate for entire group at once
            with torch.no_grad():
                batch_outputs = model.generate(
                    group_batch, 
                    attention_mask=group_attention,
                    **generation_params
                )
            
            # Process each prompt's output
            for i, (task_type, prompt, ground_truth, data) in enumerate(group):
                original_length = len(all_prompt_ids[i])
                
                # Extract completion from correct position (after original prompt)
                completion_tokens = batch_outputs[i, max_group_length:].cpu().tolist()
                completion = tokenizer.decode(completion_tokens).strip()
                if '<|endoftext|>' in completion:
                    completion = completion.replace('<|endoftext|>', '').strip()
                
                reward, _ = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
                batched_results.append(reward)
    
    batched_time = time.time() - batched_start
    
    # Compare results
    print(f"\n📊 Conservative Batching Results:")
    print(f"  Sequential time: {sequential_time:.3f}s")
    print(f"  Conservative batched time: {batched_time:.3f}s")
    print(f"  Speedup: {sequential_time/batched_time:.1f}x")
    
    # Quality comparison
    sequential_valid = [r for r in sequential_results if r > -0.1]  # Rough format validity
    batched_valid = [r for r in batched_results if isinstance(r, (int, float)) and r > -0.1]
    
    seq_mean = np.mean(sequential_results)
    batch_mean = np.mean([r for r in batched_results if isinstance(r, (int, float))])
    
    print(f"  Sequential mean reward: {seq_mean:.3f}")
    print(f"  Batched mean reward: {batch_mean:.3f}")
    print(f"  Quality delta: {((batch_mean - seq_mean) / seq_mean * 100) if seq_mean != 0 else 0:.1f}%")
    
    if abs((batch_mean - seq_mean) / seq_mean) < 0.05:  # Within 5%
        print(f"✅ Conservative batching preserves quality!")
        return True
    else:
        print(f"⚠️ Quality change in conservative batching")
        return False


if __name__ == "__main__":
    print("🧪 Conservative Batching Strategy Validation")  
    print("=" * 80)
    print("Tests batching across prompts while preserving k_samples diversity")
    print()
    
    try:
        quality_preserved = test_sequential_vs_conservative_batching()
        
        print(f"\n" + "=" * 80)
        print(f"🎯 CONSERVATIVE BATCHING ASSESSMENT")
        
        if quality_preserved:
            print(f"✅ Strategy successful: Quality preserved with speedup")
            print(f"✅ Ready to implement in training pipeline")
        else:
            print(f"⚠️ Still investigating optimal batching approach")
        
        print(f"\n💡 Key insight: Batch prompts, not k_samples")
        print(f"This preserves the diversity GRPO needs while enabling speedup")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Conservative batching test failed: {e}")
        raise