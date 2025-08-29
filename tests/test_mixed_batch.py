"""
Test batch generation with mixed P: and A: tasks
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from transformers import GPT2Tokenizer
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.reward_scorer import compute_grpo_rewards
import numpy as np

def test_mixed_batch():
    # Load model and tokenizer
    print("Loading model...")
    model = load_rookworld_model(device='cuda')
    tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load samples - get a mix of both types
    samples = load_and_prepare_samples(n_samples=20, seed=42)
    
    # Ensure we have both types
    p_samples = [s for s in samples if s[0] == 'P'][:8]
    a_samples = [s for s in samples if s[0] == 'A'][:8]
    
    if not p_samples or not a_samples:
        print("ERROR: Not enough samples of each type!")
        return
    
    # Create mixed batch
    mixed_samples = []
    for i in range(4):
        if i < len(p_samples):
            mixed_samples.append(p_samples[i])
        if i < len(a_samples):
            mixed_samples.append(a_samples[i])
    
    print(f"\nMixed batch: {len(mixed_samples)} samples")
    print(f"Order: {[s[0] for s in mixed_samples]}")
    
    # Extract prompts and ground truths
    prompts = [s[1] for s in mixed_samples]
    ground_truths = [s[2] for s in mixed_samples]
    task_types = [s[0] for s in mixed_samples]
    
    print("\n" + "="*80)
    print("TESTING MIXED BATCH GENERATION")
    print("="*80)
    
    # Tokenize all prompts
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_len = max(len(e) for e in encoded)
    
    print(f"\nBatch stats:")
    print(f"  Batch size: {len(prompts)}")
    print(f"  Max prompt length: {max_len} tokens")
    print(f"  P: prompts: {[len(e) for i, e in enumerate(encoded) if task_types[i] == 'P']}")
    print(f"  A: prompts: {[len(e) for i, e in enumerate(encoded) if task_types[i] == 'A']}")
    
    # Create padded batch (left padding for GPT-2)
    batch_ids = torch.full((len(prompts), max_len), tokenizer.eos_token_id, device='cuda', dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), max_len), device='cuda', dtype=torch.long)
    
    for i, tokens in enumerate(encoded):
        batch_ids[i, -len(tokens):] = torch.tensor(tokens, device='cuda')
        attention_mask[i, -len(tokens):] = 1
    
    print("\nGenerating completions for mixed batch...")
    
    # Generate
    with torch.no_grad():
        batch_generated = model.generate(
            batch_ids,
            max_new_tokens=144,  # Full schema
            temperature=0.75,
            top_k=50,
            top_p=0.92,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode completions
    completions = []
    for i in range(len(prompts)):
        full_text = tokenizer.decode(batch_generated[i].cpu().tolist())
        # Remove padding tokens and prompt
        full_text = full_text.replace(tokenizer.eos_token, '')
        completion = full_text[len(prompts[i]):].strip()
        completions.append(completion)
    
    # Show examples
    print("\n" + "="*80)
    print("GENERATION RESULTS")
    print("="*80)
    
    for i in range(len(mixed_samples)):
        print(f"\nSample {i+1} ({task_types[i]} task):")
        print("-" * 40)
        print(f"Prompt: {prompts[i][:60]}...")
        print(f"Generated: {completions[i][:100]}...")
        
        # Quick validation
        if task_types[i] == 'P':
            has_format = 'M:' in completions[i] and 'E:' in completions[i] and 'B:' in completions[i]
            print(f"Format valid: {has_format}")
        else:
            plus_count = completions[i].count('+')
            print(f"Format valid: {plus_count >= 3}")
    
    # Score with reward scorer
    print("\n" + "="*80)
    print("REWARD SCORING")
    print("="*80)
    
    advantages, details = compute_grpo_rewards(
        prompts, completions, verbose=False
    )
    
    # Separate scores by task type
    p_scores = [details[i].shaped_reward for i in range(len(details)) if task_types[i] == 'P']
    a_scores = [details[i].shaped_reward for i in range(len(details)) if task_types[i] == 'A']
    
    print(f"\nP: task rewards: {p_scores}")
    print(f"Mean: {np.mean(p_scores):.3f}")
    
    print(f"\nA: task rewards: {a_scores}")
    print(f"Mean: {np.mean(a_scores):.3f}")
    
    print(f"\nOverall mean reward: {np.mean([d.shaped_reward for d in details]):.3f}")
    
    # Check if mixed batching affects performance
    if np.mean(p_scores) > 0 and np.mean(a_scores) < 0:
        print("\n⚠️ WARNING: A: tasks scoring poorly despite correct format!")
        print("This suggests a scoring bug, not a generation issue.")
    else:
        print("\n✓ Mixed batch generation working correctly!")

if __name__ == "__main__":
    test_mixed_batch()