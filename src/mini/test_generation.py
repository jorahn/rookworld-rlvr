"""
Test generation with RookWorld-LM-124M on dataset samples

This script:
1. Loads 100 samples from the RookWorld dataset
2. Preprocesses them (adds A: prefix where needed)
3. Splits into prompts and ground truth completions
4. Generates completions using the model
5. Scores both generated and ground truth completions
"""

import time
import logging
from typing import List, Tuple, Dict
import torch
import tiktoken
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset import load_and_prepare_samples, preprocess_sample
from reward_scorer import compute_grpo_rewards
from loader import load_rookworld_model
from model import GPT2Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_batch_for_generation(
    samples: List[Tuple[str, str, str, Dict]],
    tokenizer,
    device: str = 'cuda',
    max_length: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], List[str]]:
    """
    Prepare a batch of samples for generation
    
    Returns:
        input_ids: Tokenized prompts [batch_size, seq_len]
        attention_mask: Mask for padding [batch_size, seq_len]
        prompts: List of prompt strings
        ground_truths: List of ground truth completions
        task_types: List of task types ('P' or 'A')
    """
    prompts = []
    ground_truths = []
    task_types = []
    
    for task_type, prompt, completion, _ in samples:
        prompts.append(prompt)
        ground_truths.append(completion)
        task_types.append(task_type)
    
    # Tokenize prompts
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    
    # Find max length for padding
    max_prompt_len = min(max(len(p) for p in encoded_prompts), max_length)
    
    # Create padded tensors
    pad_token_id = 50256  # GPT-2 EOS token
    input_ids = torch.full((len(prompts), max_prompt_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), max_prompt_len), dtype=torch.long)
    
    for i, tokens in enumerate(encoded_prompts):
        # Truncate if necessary
        tokens = tokens[:max_prompt_len]
        # Place tokens at the end (right-aligned for GPT-2)
        input_ids[i, -len(tokens):] = torch.tensor(tokens)
        attention_mask[i, -len(tokens):] = 1
    
    return (
        input_ids.to(device),
        attention_mask.to(device),
        prompts,
        ground_truths,
        task_types
    )


def generate_completions(
    model: GPT2Model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    task_types: List[str],
    tokenizer,
    device: str = 'cuda'
) -> List[str]:
    """
    Generate completions for a batch of prompts
    
    Different generation parameters for P: and A: tasks
    """
    model.eval()
    completions = []
    
    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            # Get single sample
            sample_ids = input_ids[i:i+1]
            sample_mask = attention_mask[i:i+1]
            task_type = task_types[i]
            
            # Remove left padding for generation
            first_non_pad = (sample_mask[0] == 1).nonzero(as_tuple=True)[0][0]
            sample_ids = sample_ids[:, first_non_pad:]
            sample_mask = sample_mask[:, first_non_pad:]
            
            # Different settings for different task types
            if task_type == "P":
                # P: tasks need structured output (moves, evals, best)
                max_new = 144  # CRITICAL: Need at least 144 tokens for full schema
                temperature = 0.7  # More deterministic
                top_p = 0.9
            else:
                # A: tasks need longer output (FEN + reward + flags)
                max_new = 144  # CRITICAL: Need at least 144 tokens for full schema
                temperature = 0.8
                top_p = 0.95
            
            # Generate
            generated = model.generate(
                sample_ids,
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=50,
                top_p=top_p,
                attention_mask=sample_mask
            )
            
            # Decode only the new tokens
            prompt_len = sample_ids.shape[1]
            new_tokens = generated[0, prompt_len:].cpu().tolist()
            completion = tokenizer.decode(new_tokens)
            
            # Clean up completion (remove excess whitespace, etc)
            completion = completion.strip()
            
            # For A: tasks, ensure we don't include the prompt in completion
            if task_type == "A" and completion.startswith("A:"):
                completion = completion[2:].strip()
            
            completions.append(completion)
    
    return completions


def analyze_results(
    prompts: List[str],
    generated: List[str],
    ground_truths: List[str],
    task_types: List[str]
) -> Dict:
    """
    Analyze and compare generated vs ground truth completions
    """
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    # Score both generated and ground truth
    print("\nScoring generated completions...")
    gen_advantages, gen_details = compute_grpo_rewards(
        prompts, generated, 
        reward_shaping="graduated",
        verbose=False
    )
    
    print("Scoring ground truth completions...")
    gt_advantages, gt_details = compute_grpo_rewards(
        prompts, ground_truths,
        reward_shaping="graduated", 
        verbose=False
    )
    
    # Separate by task type
    p_indices = [i for i, t in enumerate(task_types) if t == "P"]
    a_indices = [i for i, t in enumerate(task_types) if t == "A"]
    
    # Compute statistics
    stats = {
        "total_samples": len(prompts),
        "p_samples": len(p_indices),
        "a_samples": len(a_indices),
        
        # Overall statistics
        "generated": {
            "mean_reward": np.mean([d.shaped_reward for d in gen_details]),
            "std_reward": np.std([d.shaped_reward for d in gen_details]),
            "format_valid_rate": np.mean([d.format_valid for d in gen_details]),
            "min_reward": min(d.shaped_reward for d in gen_details),
            "max_reward": max(d.shaped_reward for d in gen_details),
        },
        "ground_truth": {
            "mean_reward": np.mean([d.shaped_reward for d in gt_details]),
            "std_reward": np.std([d.shaped_reward for d in gt_details]),
            "format_valid_rate": np.mean([d.format_valid for d in gt_details]),
            "min_reward": min(d.shaped_reward for d in gt_details),
            "max_reward": max(d.shaped_reward for d in gt_details),
        }
    }
    
    # P: task statistics
    if p_indices:
        p_gen = [gen_details[i] for i in p_indices]
        p_gt = [gt_details[i] for i in p_indices]
        
        stats["p_generated"] = {
            "mean_reward": np.mean([d.shaped_reward for d in p_gen]),
            "format_valid_rate": np.mean([d.format_valid for d in p_gen]),
        }
        stats["p_ground_truth"] = {
            "mean_reward": np.mean([d.shaped_reward for d in p_gt]),
            "format_valid_rate": np.mean([d.format_valid for d in p_gt]),
        }
    
    # A: task statistics
    if a_indices:
        a_gen = [gen_details[i] for i in a_indices]
        a_gt = [gt_details[i] for i in a_indices]
        
        stats["a_generated"] = {
            "mean_reward": np.mean([d.shaped_reward for d in a_gen]),
            "format_valid_rate": np.mean([d.format_valid for d in a_gen]),
        }
        stats["a_ground_truth"] = {
            "mean_reward": np.mean([d.shaped_reward for d in a_gt]),
            "format_valid_rate": np.mean([d.format_valid for d in a_gt]),
        }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nDataset: {stats['total_samples']} samples ({stats['p_samples']} P:, {stats['a_samples']} A:)")
    
    print("\n--- Overall Performance ---")
    print(f"Generated  - Mean Reward: {stats['generated']['mean_reward']:.3f} ± {stats['generated']['std_reward']:.3f}")
    print(f"           - Format Valid: {stats['generated']['format_valid_rate']*100:.1f}%")
    print(f"           - Range: [{stats['generated']['min_reward']:.3f}, {stats['generated']['max_reward']:.3f}]")
    print(f"Ground Truth - Mean Reward: {stats['ground_truth']['mean_reward']:.3f} ± {stats['ground_truth']['std_reward']:.3f}")
    print(f"             - Format Valid: {stats['ground_truth']['format_valid_rate']*100:.1f}%")
    
    if p_indices:
        print("\n--- P: Task Performance ---")
        print(f"Generated  - Mean Reward: {stats['p_generated']['mean_reward']:.3f}")
        print(f"           - Format Valid: {stats['p_generated']['format_valid_rate']*100:.1f}%")
        print(f"Ground Truth - Mean Reward: {stats['p_ground_truth']['mean_reward']:.3f}")
        print(f"             - Format Valid: {stats['p_ground_truth']['format_valid_rate']*100:.1f}%")
    
    if a_indices:
        print("\n--- A: Task Performance ---")
        print(f"Generated  - Mean Reward: {stats['a_generated']['mean_reward']:.3f}")
        print(f"           - Format Valid: {stats['a_generated']['format_valid_rate']*100:.1f}%")
        print(f"Ground Truth - Mean Reward: {stats['a_ground_truth']['mean_reward']:.3f}")
        print(f"             - Format Valid: {stats['a_ground_truth']['format_valid_rate']*100:.1f}%")
    
    # Show some examples
    print("\n" + "="*80)
    print("EXAMPLE OUTPUTS (first 3 of each type)")
    print("="*80)
    
    shown_p = 0
    shown_a = 0
    
    for i, task_type in enumerate(task_types):
        if task_type == "P" and shown_p < 3:
            print(f"\n--- P: Task Example {shown_p + 1} ---")
            print(f"Prompt: {prompts[i]}")
            print(f"Generated: {generated[i][:100]}..." if len(generated[i]) > 100 else f"Generated: {generated[i]}")
            print(f"Ground Truth: {ground_truths[i][:100]}..." if len(ground_truths[i]) > 100 else f"Ground Truth: {ground_truths[i]}")
            print(f"Gen Reward: {gen_details[i].shaped_reward:.3f}, GT Reward: {gt_details[i].shaped_reward:.3f}")
            shown_p += 1
            
        elif task_type == "A" and shown_a < 3:
            print(f"\n--- A: Task Example {shown_a + 1} ---")
            print(f"Prompt: {prompts[i][:80]}...")
            print(f"Generated: {generated[i][:100]}..." if len(generated[i]) > 100 else f"Generated: {generated[i]}")
            print(f"Ground Truth: {ground_truths[i][:100]}..." if len(ground_truths[i]) > 100 else f"Ground Truth: {ground_truths[i]}")
            print(f"Gen Reward: {gen_details[i].shaped_reward:.3f}, GT Reward: {gt_details[i].shaped_reward:.3f}")
            shown_a += 1
        
        if shown_p >= 3 and shown_a >= 3:
            break
    
    return stats


def main():
    """Main testing function"""
    print("="*80)
    print("TESTING ROOKWORLD-LM-124M GENERATION")
    print("="*80)
    
    # Configuration
    n_samples = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load model
    print("\n2. Loading RookWorld-LM-124M model...")
    model = load_rookworld_model(device=device)
    
    # Load dataset samples
    print(f"\n3. Loading {n_samples} samples from dataset...")
    samples = load_and_prepare_samples(n_samples=n_samples, seed=42)
    print(f"   Loaded {len(samples)} samples")
    
    # Count task types
    p_count = sum(1 for s in samples if s[0] == "P")
    a_count = sum(1 for s in samples if s[0] == "A")
    print(f"   Task distribution: {p_count} P: tasks, {a_count} A: tasks")
    
    # Process in batches - can now handle mixed batches with position fix!
    batch_size = 16
    all_prompts = []
    all_generated = []
    all_ground_truths = []
    all_task_types = []
    
    print(f"\n4. Generating completions (batch size: {batch_size})...")
    print("   Note: Using fixed position embeddings for proper mixed batch handling")
    start_time = time.time()
    
    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        
        # Count task types in batch
        p_count = sum(1 for s in batch_samples if s[0] == 'P')
        a_count = sum(1 for s in batch_samples if s[0] == 'A')
        print(f"   Processing batch {batch_start//batch_size + 1}/{(len(samples)-1)//batch_size + 1}... (P:{p_count}, A:{a_count})")
        
        # Prepare batch
        input_ids, attention_mask, prompts, ground_truths, task_types = prepare_batch_for_generation(
            batch_samples, tokenizer, device
        )
        
        # Generate
        generated = generate_completions(
            model, input_ids, attention_mask, task_types, tokenizer, device
        )
        
        # Collect results
        all_prompts.extend(prompts)
        all_generated.extend(generated)
        all_ground_truths.extend(ground_truths)
        all_task_types.extend(task_types)
    
    generation_time = time.time() - start_time
    print(f"   Generation completed in {generation_time:.2f} seconds")
    print(f"   Average time per sample: {generation_time/len(samples):.3f} seconds")
    
    # Analyze results
    print("\n5. Analyzing results...")
    stats = analyze_results(
        all_prompts, all_generated, all_ground_truths, all_task_types
    )
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return stats


if __name__ == "__main__":
    stats = main()