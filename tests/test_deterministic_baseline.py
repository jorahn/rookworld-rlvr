#!/usr/bin/env python3
"""
Deterministic Baseline Test

Establishes a true deterministic baseline by running training with:
- Fixed seeds for all randomness
- Temperature=0 for deterministic generation
- Fixed dataset samples
- Controlled environment for exact reproducibility
"""

import torch
import time
import sys
import json
import tiktoken
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.reward_scorer import RewardScorer


def run_deterministic_baseline(k_samples: int = 4) -> dict:
    """Run fully deterministic training step for baseline."""
    print(f"🎯 Running Deterministic Baseline (k_samples={k_samples})")
    print("=" * 60)
    
    # Set all seeds for complete determinism
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model and data
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load fixed dataset samples
    samples = load_and_prepare_samples(n_samples=20, seed=42)  # Fixed seed
    test_samples = samples[:4]  # Use first 4 samples for testing
    
    print(f"Using {len(test_samples)} samples for deterministic test:")
    for i, (task_type, prompt, ground_truth, data) in enumerate(test_samples):
        print(f"  Sample {i+1} ({task_type}): {prompt[:40]}...")
    
    # Configure for deterministic generation
    config = GRPOConfig(
        k_samples=k_samples,
        temperature=0.01,  # Near-deterministic (0.0 can cause issues)
        top_k=1,           # Most deterministic
        top_p=1.0,         # Disable top-p
        max_new_tokens=144  # Required minimum for schemas
    )
    
    scorer = RewardScorer(
        reward_shaping="graduated",
        continuous_components={"fen_similarity": "exponential", "evaluations": "linear"}
    )
    
    # Process each sample sequentially (current method)
    all_results = []
    total_time = 0
    
    for sample_idx, (task_type, prompt, ground_truth, data) in enumerate(test_samples):
        print(f"\n🔬 Processing Sample {sample_idx+1} ({task_type}):")
        
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        sample_results = {
            'sample_idx': sample_idx,
            'task_type': task_type,
            'prompt': prompt,
            'prompt_length': len(prompt_ids),
            'completions': [],
            'rewards': [],
            'generation_time': 0
        }
        
        # Generate k_samples for this prompt
        gen_start = time.time()
        
        for k in range(k_samples):
            # Set deterministic seed for this specific generation
            torch.manual_seed(42 + sample_idx * 10 + k)
            
            prompt_tensor = torch.tensor(prompt_ids, device="cuda").unsqueeze(0)
            
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_tensor,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    pad_token_id=50256
                )
            
            # Decode and clean
            completion_ids = output_ids[0, len(prompt_ids):].tolist()
            completion = tokenizer.decode(completion_ids).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            # Score completion
            reward, details = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            
            sample_results['completions'].append(completion)
            sample_results['rewards'].append(reward)
            
            print(f"  K={k+1}: Reward={reward:.3f}, Tokens={len(completion_ids)}, Format={details.format_valid}")
            print(f"       Completion: {completion[:60]}...")
        
        sample_results['generation_time'] = time.time() - gen_start
        total_time += sample_results['generation_time']
        
        # Calculate sample statistics
        sample_rewards = sample_results['rewards']
        sample_results['mean_reward'] = np.mean(sample_rewards)
        sample_results['format_valid_count'] = sum(1 for r in sample_rewards if r > -0.1)  # Rough format validity
        
        print(f"  Sample mean reward: {sample_results['mean_reward']:.3f}")
        print(f"  Format valid: {sample_results['format_valid_count']}/{k_samples}")
        print(f"  Generation time: {sample_results['generation_time']:.2f}s")
        
        all_results.append(sample_results)
    
    # Overall statistics
    all_rewards = [r for sample in all_results for r in sample['rewards']]
    all_format_valid = sum(sample['format_valid_count'] for sample in all_results)
    total_samples = len(all_results) * k_samples
    
    baseline_summary = {
        'k_samples': k_samples,
        'total_samples': total_samples,
        'total_generation_time': total_time,
        'avg_generation_time_per_sample': total_time / total_samples,
        'overall_mean_reward': np.mean(all_rewards),
        'reward_std': np.std(all_rewards),
        'min_reward': np.min(all_rewards),
        'max_reward': np.max(all_rewards),
        'format_valid_count': all_format_valid,
        'format_valid_ratio': all_format_valid / total_samples,
        'sample_results': all_results,
        'generation_config': {
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p,
            'max_new_tokens': config.max_new_tokens
        }
    }
    
    print(f"\n📊 Deterministic Baseline Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Mean reward: {baseline_summary['overall_mean_reward']:.3f}")
    print(f"  Reward std: {baseline_summary['reward_std']:.3f}")
    print(f"  Format valid: {all_format_valid}/{total_samples} ({baseline_summary['format_valid_ratio']:.1%})")
    print(f"  Total generation time: {total_time:.2f}s")
    print(f"  Time per generation: {total_time/total_samples:.3f}s")
    
    return baseline_summary


def compare_k_samples_consistency():
    """Test that first 4 samples of k_samples=8 match k_samples=4 exactly."""
    print("\n🔬 Testing K-Samples Consistency")
    print("=" * 60)
    
    # Run with k_samples=4
    baseline_k4 = run_deterministic_baseline(k_samples=4)
    
    # Run with k_samples=8  
    baseline_k8 = run_deterministic_baseline(k_samples=8)
    
    print(f"\n📊 K-Samples Comparison:")
    print(f"  K=4 mean reward: {baseline_k4['overall_mean_reward']:.3f}")
    print(f"  K=8 mean reward: {baseline_k8['overall_mean_reward']:.3f}")
    
    # Check if first 4 generations are identical
    print(f"\nValidating first 4 samples are identical:")
    
    all_match = True
    for sample_idx in range(min(4, len(baseline_k4['sample_results']))):
        k4_sample = baseline_k4['sample_results'][sample_idx]
        k8_sample = baseline_k8['sample_results'][sample_idx]
        
        # Compare first 4 completions
        for k in range(4):
            k4_completion = k4_sample['completions'][k]
            k8_completion = k8_sample['completions'][k]
            k4_reward = k4_sample['rewards'][k] 
            k8_reward = k8_sample['rewards'][k]
            
            match = (k4_completion == k8_completion) and abs(k4_reward - k8_reward) < 1e-6
            
            print(f"  Sample {sample_idx+1}, K={k+1}: {'✅ MATCH' if match else '❌ DIFFER'}")
            
            if not match:
                print(f"    K=4: R={k4_reward:.3f}, '{k4_completion[:30]}...'")
                print(f"    K=8: R={k8_reward:.3f}, '{k8_completion[:30]}...'")
                all_match = False
    
    if all_match:
        print(f"✅ Perfect consistency: First 4 samples identical between k_samples=4 and k_samples=8")
    else:
        print(f"❌ Inconsistency detected: Different results for same samples")
        print(f"This suggests seed management or generation issues")
    
    return all_match, baseline_k4, baseline_k8


def save_deterministic_reference(baseline_data: dict, filename: str):
    """Save deterministic baseline as reference for validation."""
    reference_file = Path(__file__).parent / filename
    
    with open(reference_file, 'w') as f:
        json.dump(baseline_data, f, indent=2, default=str)
    
    print(f"✅ Deterministic reference saved to {reference_file}")


if __name__ == "__main__":
    print("🔬 Deterministic Baseline Establishment")
    print("=" * 70)
    print("Creating reference data for rigorous optimization validation")
    print()
    
    try:
        # Test consistency between k_samples configurations
        consistency_ok, k4_baseline, k8_baseline = compare_k_samples_consistency()
        
        if consistency_ok:
            print(f"\n✅ DETERMINISTIC BASELINE ESTABLISHED")
            print(f"Both k_samples=4 and k_samples=8 show consistent behavior")
            
            # Save both baselines as references
            save_deterministic_reference(k4_baseline, "deterministic_baseline_k4.json")
            save_deterministic_reference(k8_baseline, "deterministic_baseline_k8.json")
            
            # Choose the configuration with better quality for optimization work
            if k4_baseline['overall_mean_reward'] > k8_baseline['overall_mean_reward']:
                print(f"💡 k_samples=4 shows better quality ({k4_baseline['overall_mean_reward']:.3f} vs {k8_baseline['overall_mean_reward']:.3f})")
                print(f"Recommend using k_samples=4 for optimization work")
            else:
                print(f"💡 k_samples=8 shows better quality")
        
        else:
            print(f"\n❌ CONSISTENCY ISSUE DETECTED")
            print(f"Cannot establish reliable baseline until seed/generation issues resolved")
            
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Baseline establishment failed: {e}")
        raise