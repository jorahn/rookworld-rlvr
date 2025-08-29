#!/usr/bin/env python3
"""
Generation Quality Isolation Test

Focuses solely on understanding why batched generation produces different 
quality outputs than sequential generation. Tests with real dataset samples
and mixed P:/A: task types.
"""

import torch
import sys
import tiktoken
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.reward_scorer import RewardScorer


def test_single_prompt_batch_vs_sequential():
    """Test same prompt generated sequentially vs in batch."""
    print("🔬 Testing Single Prompt: Sequential vs Batch Generation")
    print("=" * 70)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load real dataset samples
    samples = load_and_prepare_samples(n_samples=10, seed=42)
    
    # Test both P: and A: task types
    p_sample = next(s for s in samples if s[0] == 'P')
    a_sample = next(s for s in samples if s[0] == 'A') 
    
    test_prompts = [
        ("P: task", p_sample[1], p_sample[2]),
        ("A: task", a_sample[1], a_sample[2])
    ]
    
    generation_params = {
        'max_new_tokens': 144,
        'temperature': 0.8,  # Realistic training parameter
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    scorer = RewardScorer(
        reward_shaping="graduated", 
        continuous_components={"fen_similarity": "exponential", "evaluations": "linear"}
    )
    
    for task_name, prompt, ground_truth in test_prompts:
        print(f"\n📋 {task_name}: {prompt[:50]}...")
        
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        prompt_tensor = torch.tensor([prompt_ids], device="cuda")
        k_samples = 4
        
        print(f"Prompt length: {len(prompt_ids)} tokens")
        
        # Sequential generation (current working method)
        print("\n🔄 Sequential Generation:")
        sequential_completions = []
        sequential_rewards = []
        seq_start = time.time()
        
        for k in range(k_samples):
            torch.manual_seed(42 + k)  # Different seed per k for diversity
            
            with torch.no_grad():
                output = model.generate(prompt_tensor, **generation_params)
            
            completion_tokens = output[0, len(prompt_ids):].cpu().tolist()
            completion = tokenizer.decode(completion_tokens).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            
            sequential_completions.append(completion)
            sequential_rewards.append(reward)
            
            print(f"  K={k+1}: R={reward:.3f}, Format={details.format_valid}, Len={len(completion_tokens)}")
            print(f"       '{completion[:60]}...'")
        
        seq_time = time.time() - seq_start
        
        # Batched generation (test method)
        print(f"\n⚡ Batched Generation:")
        batch_start = time.time()
        
        # Create batch of identical prompts
        batch_prompts = prompt_tensor.repeat(k_samples, 1)
        
        # Try different seed strategies
        torch.manual_seed(42)  # Single seed for batch
        
        with torch.no_grad():
            batch_outputs = model.generate(batch_prompts, **generation_params)
        
        batch_completions = []
        batch_rewards = []
        
        for k in range(k_samples):
            completion_tokens = batch_outputs[k, len(prompt_ids):].cpu().tolist()
            completion = tokenizer.decode(completion_tokens).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            
            batch_completions.append(completion)
            batch_rewards.append(reward)
            
            print(f"  K={k+1}: R={reward:.3f}, Format={details.format_valid}, Len={len(completion_tokens)}")
            print(f"       '{completion[:60]}...'")
        
        batch_time = time.time() - batch_start
        
        # Compare results
        print(f"\n📊 Comparison for {task_name}:")
        print(f"  Sequential mean reward: {np.mean(sequential_rewards):.3f}")
        print(f"  Batched mean reward: {np.mean(batch_rewards):.3f}")
        print(f"  Reward difference: {np.mean(batch_rewards) - np.mean(sequential_rewards):.3f}")
        
        print(f"  Sequential time: {seq_time:.3f}s")
        print(f"  Batched time: {batch_time:.3f}s") 
        print(f"  Speedup: {seq_time/batch_time:.1f}x")
        
        # Check diversity
        seq_unique = len(set(sequential_completions))
        batch_unique = len(set(batch_completions))
        print(f"  Sequential unique completions: {seq_unique}/{k_samples}")
        print(f"  Batched unique completions: {batch_unique}/{k_samples}")
        
        if np.mean(batch_rewards) < np.mean(sequential_rewards) * 0.9:
            print(f"  ⚠️ Quality regression detected in batched generation")
        else:
            print(f"  ✅ Quality maintained in batched generation")


def test_mixed_task_batch_generation():
    """Test batched generation with mixed P: and A: tasks in same batch."""
    print(f"\n🔀 Testing Mixed Task Batch Generation")
    print("=" * 70)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load real dataset samples
    samples = load_and_prepare_samples(n_samples=20, seed=42)
    
    # Create mixed batch with both P: and A: tasks
    p_samples = [s for s in samples if s[0] == 'P'][:3]
    a_samples = [s for s in samples if s[0] == 'A'][:3]
    mixed_samples = p_samples + a_samples  # 6 total: 3 P: + 3 A:
    
    print(f"Mixed batch composition:")
    for i, (task_type, prompt, ground_truth, data) in enumerate(mixed_samples):
        prompt_len = len(tokenizer.encode(prompt, disallowed_special=()))
        print(f"  {i+1}. {task_type} task: {prompt_len} tokens - {prompt[:40]}...")
    
    generation_params = {
        'max_new_tokens': 144,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    scorer = RewardScorer(reward_shaping="graduated")
    
    # Sequential processing (reference method)
    print(f"\n🔄 Sequential Processing:")
    sequential_results = []
    seq_total_time = 0
    
    for i, (task_type, prompt, ground_truth, data) in enumerate(mixed_samples):
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        prompt_tensor = torch.tensor([prompt_ids], device="cuda")
        
        torch.manual_seed(42 + i)  # Consistent seed per sample
        seq_start = time.time()
        
        with torch.no_grad():
            output = model.generate(prompt_tensor, **generation_params)
        
        seq_time = time.time() - seq_start
        seq_total_time += seq_time
        
        completion_tokens = output[0, len(prompt_ids):].cpu().tolist()
        completion = tokenizer.decode(completion_tokens).strip()
        if '<|endoftext|>' in completion:
            completion = completion.replace('<|endoftext|>', '').strip()
        
        reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
        
        sequential_results.append({
            'task_type': task_type,
            'completion': completion,
            'reward': reward,
            'format_valid': details.format_valid,
            'generation_time': seq_time
        })
        
        print(f"  {i+1}. {task_type}: R={reward:.3f}, Valid={details.format_valid}, Time={seq_time:.3f}s")
    
    # Batched processing (test method)
    print(f"\n⚡ Batched Processing:")
    
    # Prepare batch with proper padding and attention masks
    all_prompt_ids = [tokenizer.encode(prompt, disallowed_special=()) for _, prompt, _, _ in mixed_samples]
    max_prompt_len = max(len(ids) for ids in all_prompt_ids)
    
    # Create padded batch with attention masks
    batch_size = len(mixed_samples)
    padded_prompts = torch.full((batch_size, max_prompt_len), 50256, device="cuda")  # Pad with EOS
    attention_mask = torch.zeros((batch_size, max_prompt_len), device="cuda")
    
    for i, prompt_ids in enumerate(all_prompt_ids):
        # Left padding for GPT-2
        start_pos = max_prompt_len - len(prompt_ids)
        padded_prompts[i, start_pos:] = torch.tensor(prompt_ids, device="cuda")
        attention_mask[i, start_pos:] = 1
    
    print(f"Batch shape: {padded_prompts.shape}, Max prompt length: {max_prompt_len}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Generate batch
    torch.manual_seed(42)  # Single seed for fair comparison
    batch_start = time.time()
    
    try:
        with torch.no_grad():
            batch_outputs = model.generate(
                padded_prompts,
                attention_mask=attention_mask,
                **generation_params
            )
        
        batch_total_time = time.time() - batch_start
        
        # Process batch results
        batch_results = []
        
        for i in range(batch_size):
            task_type, prompt, ground_truth, data = mixed_samples[i]
            original_prompt_len = len(all_prompt_ids[i])
            
            # Extract completion (account for padding)
            start_pos = max_prompt_len - original_prompt_len
            completion_tokens = batch_outputs[i, max_prompt_len:].cpu().tolist()  # Everything after original prompt
            completion = tokenizer.decode(completion_tokens).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            
            batch_results.append({
                'task_type': task_type,
                'completion': completion,
                'reward': reward,
                'format_valid': details.format_valid
            })
            
            print(f"  {i+1}. {task_type}: R={reward:.3f}, Valid={details.format_valid}")
        
        # Compare sequential vs batch
        print(f"\n📊 Mixed Batch Quality Comparison:")
        
        seq_rewards = [r['reward'] for r in sequential_results]
        batch_rewards = [r['reward'] for r in batch_results]
        
        seq_p_rewards = [r['reward'] for r in sequential_results if r['task_type'] == 'P']
        batch_p_rewards = [r['reward'] for r in batch_results if r['task_type'] == 'P']
        seq_a_rewards = [r['reward'] for r in sequential_results if r['task_type'] == 'A']
        batch_a_rewards = [r['reward'] for r in batch_results if r['task_type'] == 'A']
        
        print(f"  Overall:")
        print(f"    Sequential mean: {np.mean(seq_rewards):.3f}")
        print(f"    Batched mean: {np.mean(batch_rewards):.3f}")
        print(f"    Difference: {np.mean(batch_rewards) - np.mean(seq_rewards):.3f}")
        
        print(f"  P: tasks:")
        print(f"    Sequential mean: {np.mean(seq_p_rewards):.3f}")
        print(f"    Batched mean: {np.mean(batch_p_rewards):.3f}")
        
        print(f"  A: tasks:")
        print(f"    Sequential mean: {np.mean(seq_a_rewards):.3f}")
        print(f"    Batched mean: {np.mean(batch_a_rewards):.3f}")
        
        print(f"  Performance:")
        print(f"    Sequential time: {seq_total_time:.3f}s")
        print(f"    Batched time: {batch_total_time:.3f}s")
        print(f"    Speedup: {seq_total_time/batch_total_time:.1f}x")
        
        # Quality assessment
        quality_preserved = (
            np.mean(batch_rewards) >= np.mean(seq_rewards) * 0.9 and  # Within 10%
            np.mean(batch_p_rewards) >= np.mean(seq_p_rewards) * 0.9 and
            np.mean(batch_a_rewards) >= np.mean(seq_a_rewards) * 0.9
        )
        
        if quality_preserved:
            print(f"✅ Mixed batch generation preserves quality")
        else:
            print(f"❌ Mixed batch generation degrades quality")
            
        return quality_preserved
        
    except Exception as e:
        print(f"❌ Batched generation failed: {e}")
        return False


def test_attention_mask_effects():
    """Test if attention masking affects generation quality."""
    print(f"\n🎭 Testing Attention Mask Effects")
    print("=" * 70)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Use a simple prompt for focused testing
    samples = load_and_prepare_samples(n_samples=5, seed=42)
    test_prompt = next(s[1] for s in samples if s[0] == 'P')  # Get a P: task
    
    print(f"Test prompt: {test_prompt}")
    
    prompt_ids = tokenizer.encode(test_prompt, disallowed_special=())
    prompt_len = len(prompt_ids)
    
    generation_params = {
        'max_new_tokens': 144,
        'temperature': 0.8,
        'top_k': 50, 
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    # Test 1: Single prompt without attention mask
    print(f"\n1. Single prompt (no attention mask):")
    torch.manual_seed(42)
    prompt_tensor = torch.tensor([prompt_ids], device="cuda")
    
    with torch.no_grad():
        output1 = model.generate(prompt_tensor, **generation_params)
    
    completion1 = tokenizer.decode(output1[0, prompt_len:].cpu().tolist()).strip()
    print(f"   Output: '{completion1[:80]}...'")
    
    # Test 2: Single prompt with explicit attention mask
    print(f"\n2. Single prompt (with attention mask):")
    torch.manual_seed(42)  # Same seed
    attention_mask = torch.ones_like(prompt_tensor, device="cuda")
    
    with torch.no_grad():
        output2 = model.generate(prompt_tensor, attention_mask=attention_mask, **generation_params)
    
    completion2 = tokenizer.decode(output2[0, prompt_len:].cpu().tolist()).strip()
    print(f"   Output: '{completion2[:80]}...'")
    
    # Test 3: Padded prompt in batch (simulating batch behavior)
    print(f"\n3. Padded prompt in batch:")
    torch.manual_seed(42)  # Same seed
    
    # Create padded version (like in mixed batch)
    max_len = prompt_len + 10  # Add some padding
    padded_prompt = torch.full((1, max_len), 50256, device="cuda")
    padded_prompt[0, -prompt_len:] = torch.tensor(prompt_ids, device="cuda")  # Right padding this time
    padded_attention = torch.zeros((1, max_len), device="cuda")
    padded_attention[0, -prompt_len:] = 1
    
    with torch.no_grad():
        output3 = model.generate(padded_prompt, attention_mask=padded_attention, **generation_params)
    
    completion3 = tokenizer.decode(output3[0, max_len:].cpu().tolist()).strip()  # Skip padding
    print(f"   Output: '{completion3[:80]}...'")
    
    # Compare all three
    print(f"\n📊 Attention Mask Effect Analysis:")
    print(f"  Method 1 == Method 2: {completion1 == completion2}")
    print(f"  Method 1 == Method 3: {completion1 == completion3}")
    print(f"  Method 2 == Method 3: {completion2 == completion3}")
    
    if completion1 == completion2 == completion3:
        print(f"✅ Attention masking and padding don't affect generation")
    else:
        print(f"❌ Attention masking or padding affects generation quality")
        print(f"This could be the source of batch generation issues")
    
    return completion1 == completion2 == completion3


if __name__ == "__main__":
    print("🚀 Generation Quality Isolation Test")
    print("=" * 80)
    print("Focused test to isolate batched generation quality issues")
    print()
    
    try:
        # Test 1: Single prompt comparison
        test_single_prompt_batch_vs_sequential()
        
        # Test 2: Mixed task batch
        mixed_quality_ok = test_mixed_task_batch_generation()
        
        # Test 3: Attention mask effects
        mask_consistent = test_attention_mask_effects()
        
        print(f"\n" + "=" * 80)
        print(f"🎯 GENERATION QUALITY ANALYSIS SUMMARY")
        print(f"  Mixed batch quality preserved: {'✅' if mixed_quality_ok else '❌'}")
        print(f"  Attention masking consistent: {'✅' if mask_consistent else '❌'}")
        
        if mixed_quality_ok and mask_consistent:
            print(f"\n✅ Batched generation fundamentals work correctly")
            print(f"Issue may be in specific implementation details")
        else:
            print(f"\n❌ Fundamental issues with batched generation detected")
            print(f"Need to fix core generation batching before optimization")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Generation quality test failed: {e}")
        raise