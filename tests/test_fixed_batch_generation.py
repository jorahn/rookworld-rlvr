#!/usr/bin/env python3
"""
Fixed Batch Generation Test

Implements proper left-padding and token extraction based on debugging findings.
Tests if corrected implementation achieves quality preservation.
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


def test_corrected_batch_generation():
    """Test batch generation with fixed implementation."""
    print("🔧 Testing Corrected Batch Generation Implementation")
    print("=" * 70)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load real dataset samples
    samples = load_and_prepare_samples(n_samples=10, seed=42)
    
    # Get mixed samples
    p_sample = next(s for s in samples if s[0] == 'P')
    a_sample = next(s for s in samples if s[0] == 'A')
    test_samples = [p_sample, a_sample]
    
    generation_params = {
        'max_new_tokens': 100,  # Reasonable length
        'temperature': 0.8,     # Realistic training parameter
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    scorer = RewardScorer(
        reward_shaping="graduated",
        continuous_components={"fen_similarity": "exponential", "evaluations": "linear"}
    )
    
    for task_type, prompt, ground_truth, data in test_samples:
        print(f"\n📋 Testing {task_type} task: {prompt[:50]}...")
        
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        original_prompt_len = len(prompt_ids)
        
        k_samples = 4
        
        # Sequential generation (reference)
        print(f"\n🔄 Sequential (Reference):")
        sequential_completions = []
        sequential_rewards = []
        
        for k in range(k_samples):
            torch.manual_seed(42 + k)  # Different seeds for diversity
            
            prompt_tensor = torch.tensor([prompt_ids], device="cuda")
            
            with torch.no_grad():
                output = model.generate(prompt_tensor, **generation_params)
            
            completion_tokens = output[0, original_prompt_len:].cpu().tolist()
            completion = tokenizer.decode(completion_tokens).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            
            sequential_completions.append(completion)
            sequential_rewards.append(reward)
            
            print(f"  K={k+1}: R={reward:.3f}, Valid={details.format_valid}")
        
        # Corrected batched generation
        print(f"\n⚡ Corrected Batched:")
        
        # Create properly left-padded batch
        max_len = original_prompt_len + 20  # Add some padding for batch context
        
        batch_input = torch.full((k_samples, max_len), 50256, device="cuda")  # Fill with EOS
        batch_attention = torch.zeros((k_samples, max_len), device="cuda")
        
        for k in range(k_samples):
            # Left padding: put actual prompt at the END of the sequence
            start_pos = max_len - original_prompt_len
            batch_input[k, start_pos:] = torch.tensor(prompt_ids, device="cuda")
            batch_attention[k, start_pos:] = 1
        
        print(f"  Batch shape: {batch_input.shape}")
        print(f"  Prompt starts at position: {start_pos}")
        print(f"  Attention active tokens: {batch_attention[0].sum().item()}")
        
        # Generate batch with consistent seed approach
        torch.manual_seed(42)  # Base seed for reproducibility
        
        try:
            with torch.no_grad():
                batch_outputs = model.generate(
                    batch_input,
                    attention_mask=batch_attention,
                    **generation_params
                )
            
            print(f"  Generated shape: {batch_outputs.shape}")
            
            batch_completions = []
            batch_rewards = []
            
            for k in range(k_samples):
                # Extract completion: everything after the padded input
                completion_tokens = batch_outputs[k, max_len:].cpu().tolist()
                completion = tokenizer.decode(completion_tokens).strip()
                if '<|endoftext|>' in completion:
                    completion = completion.replace('<|endoftext|>', '').strip()
                
                reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
                
                batch_completions.append(completion)
                batch_rewards.append(reward)
                
                print(f"  K={k+1}: R={reward:.3f}, Valid={details.format_valid}")
                print(f"       '{completion[:60]}...'")
        
        except Exception as e:
            print(f"  ❌ Batched generation failed: {e}")
            continue
        
        # Compare quality
        print(f"\n📊 Quality Comparison:")
        seq_mean = np.mean(sequential_rewards)
        batch_mean = np.mean(batch_rewards) 
        quality_delta = (batch_mean - seq_mean) / seq_mean if seq_mean != 0 else 0
        
        print(f"  Sequential mean reward: {seq_mean:.3f}")
        print(f"  Batched mean reward: {batch_mean:.3f}")
        print(f"  Quality delta: {quality_delta:.1%}")
        
        if abs(quality_delta) < 0.1:  # Within 10%
            print(f"✅ Quality preserved in batch generation")
        else:
            print(f"⚠️ Quality change: {quality_delta:.1%}")


def test_simple_same_length_batching():
    """Test batching with prompts of exactly same length (no padding needed)."""
    print(f"\n📏 Testing Same-Length Batching (No Padding)")
    print("=" * 70)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create prompts of exactly same length by truncating/padding the text content
    base_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    base_ids = tokenizer.encode(base_prompt, disallowed_special=())
    target_len = len(base_ids)
    
    # Load samples and find ones with similar length  
    samples = load_and_prepare_samples(n_samples=20, seed=42)
    
    similar_length_samples = []
    for sample in samples:
        prompt_ids = tokenizer.encode(sample[1], disallowed_special=())
        if abs(len(prompt_ids) - target_len) <= 2:  # Within 2 tokens
            similar_length_samples.append(sample)
        if len(similar_length_samples) >= 3:  # Get 3 samples
            break
    
    if len(similar_length_samples) < 2:
        print("  ⚠️ Not enough similar-length samples found")
        return
    
    print(f"Found {len(similar_length_samples)} similar-length samples:")
    for i, (task_type, prompt, _, _) in enumerate(similar_length_samples):
        prompt_len = len(tokenizer.encode(prompt, disallowed_special=()))
        print(f"  {i+1}. {task_type} ({prompt_len} tokens): {prompt[:40]}...")
    
    # Sequential processing
    print(f"\n🔄 Sequential Processing:")
    sequential_results = []
    
    for i, (task_type, prompt, ground_truth, data) in enumerate(similar_length_samples):
        torch.manual_seed(42 + i)
        
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        prompt_tensor = torch.tensor([prompt_ids], device="cuda")
        
        with torch.no_grad():
            output = model.generate(prompt_tensor, **generation_params)
        
        completion = tokenizer.decode(output[0, len(prompt_ids):].cpu().tolist()).strip()
        
        scorer = RewardScorer(reward_shaping="graduated")
        reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
        
        sequential_results.append({
            'task_type': task_type,
            'completion': completion,
            'reward': reward,
            'format_valid': details.format_valid
        })
        
        print(f"  {i+1}. {task_type}: R={reward:.3f}, Valid={details.format_valid}")
    
    # Batched processing (same lengths - no padding needed!)
    print(f"\n⚡ Same-Length Batched Processing:")
    
    all_prompt_ids = [tokenizer.encode(sample[1], disallowed_special=()) for sample in similar_length_samples]
    
    # Verify all same length
    lengths = [len(ids) for ids in all_prompt_ids]
    if len(set(lengths)) == 1:
        print(f"  ✅ All prompts exactly {lengths[0]} tokens - no padding needed")
        
        # Create batch tensor directly (no padding!)
        batch_tensor = torch.tensor(all_prompt_ids, device="cuda")
        
        torch.manual_seed(42)  # Single seed for comparison
        
        with torch.no_grad():
            batch_outputs = model.generate(batch_tensor, **generation_params)
        
        batch_results = []
        
        for i in range(len(similar_length_samples)):
            task_type, prompt, ground_truth, data = similar_length_samples[i]
            original_len = len(all_prompt_ids[i])
            
            completion = tokenizer.decode(batch_outputs[i, original_len:].cpu().tolist()).strip()
            
            scorer = RewardScorer(reward_shaping="graduated")
            reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            
            batch_results.append({
                'task_type': task_type,
                'completion': completion,
                'reward': reward,
                'format_valid': details.format_valid
            })
            
            print(f"  {i+1}. {task_type}: R={reward:.3f}, Valid={details.format_valid}")
        
        # Compare same-length batching quality
        seq_rewards = [r['reward'] for r in sequential_results]
        batch_rewards = [r['reward'] for r in batch_results]
        
        print(f"\n📊 Same-Length Batch Quality:")
        print(f"  Sequential mean: {np.mean(seq_rewards):.3f}")
        print(f"  Batched mean: {np.mean(batch_rewards):.3f}")
        print(f"  Quality delta: {(np.mean(batch_rewards) - np.mean(seq_rewards)) / np.mean(seq_rewards):.1%}")
        
        if abs(np.mean(batch_rewards) - np.mean(seq_rewards)) / np.mean(seq_rewards) < 0.1:
            print(f"✅ Same-length batching preserves quality!")
        else:
            print(f"⚠️ Quality change in same-length batching")
    
    else:
        print(f"  ❌ Lengths not identical: {lengths}")


if __name__ == "__main__":
    print("🚀 Fixed Batch Generation Implementation Test")
    print("=" * 80)
    
    try:
        # Test corrected implementation
        test_corrected_batch_generation()
        
        # Test same-length batching (ideal case)
        test_simple_same_length_batching()
        
        print(f"\n" + "=" * 80)
        print(f"🎯 IMPLEMENTATION FIXES NEEDED:")
        print(f"  1. Use LEFT padding for GPT-2 (not right)")
        print(f"  2. Fix token extraction position calculation") 
        print(f"  3. Study HuggingFace implementation for proper batching")
        print(f"  4. Test same-length batching as safer alternative")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise