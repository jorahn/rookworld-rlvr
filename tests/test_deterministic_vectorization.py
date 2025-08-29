#!/usr/bin/env python3
"""
Deterministic Vectorization Validation

Tests that vectorized generation produces IDENTICAL token sequences to sequential
generation when using deterministic settings. This is the gold standard test
before implementing any vectorization in the training pipeline.
"""

import torch
import time
import sys
import tiktoken
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.reward_scorer import RewardScorer


def generate_sequential_deterministic(model, tokenizer, samples, k_samples, config):
    """Generate using current sequential method with deterministic settings."""
    print("🔄 Sequential Generation (Current Method)")
    
    all_results = []
    total_time = 0
    
    for sample_idx, (task_type, prompt, ground_truth, data) in enumerate(samples):
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        sample_results = {
            'sample_idx': sample_idx,
            'task_type': task_type,
            'completions': [],
            'completion_tokens': [],
            'rewards': []
        }
        
        gen_start = time.time()
        
        for k in range(k_samples):
            # Deterministic seed per generation
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
            
            completion_tokens = output_ids[0, len(prompt_ids):].cpu().tolist()
            completion = tokenizer.decode(completion_tokens).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            sample_results['completions'].append(completion)
            sample_results['completion_tokens'].append(completion_tokens)
        
        sample_results['generation_time'] = time.time() - gen_start
        total_time += sample_results['generation_time']
        all_results.append(sample_results)
        
        print(f"  Sample {sample_idx+1} ({task_type}): {sample_results['generation_time']:.3f}s")
    
    print(f"  Total time: {total_time:.3f}s")
    return all_results, total_time


def generate_vectorized_deterministic(model, tokenizer, samples, k_samples, config):
    """Generate using vectorized method with deterministic settings."""
    print("⚡ Vectorized Generation (Test Method)")
    
    all_results = []
    total_time = 0
    
    for sample_idx, (task_type, prompt, ground_truth, data) in enumerate(samples):
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        
        gen_start = time.time()
        
        # Create batch of identical prompts
        prompt_tensor = torch.tensor(prompt_ids, device="cuda").unsqueeze(0)
        batch_prompts = prompt_tensor.repeat(k_samples, 1)
        
        # Use same base seed as sequential method
        torch.manual_seed(42 + sample_idx * 10)
        
        with torch.no_grad():
            batch_outputs = model.generate(
                batch_prompts,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                pad_token_id=50256
            )
        
        sample_results = {
            'sample_idx': sample_idx,
            'task_type': task_type,
            'completions': [],
            'completion_tokens': [],
            'generation_time': time.time() - gen_start
        }
        
        for k in range(k_samples):
            completion_tokens = batch_outputs[k, len(prompt_ids):].cpu().tolist()
            completion = tokenizer.decode(completion_tokens).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            sample_results['completions'].append(completion)
            sample_results['completion_tokens'].append(completion_tokens)
        
        total_time += sample_results['generation_time']
        all_results.append(sample_results)
        
        print(f"  Sample {sample_idx+1} ({task_type}): {sample_results['generation_time']:.3f}s")
    
    print(f"  Total time: {total_time:.3f}s")
    return all_results, total_time


def validate_identical_outputs(sequential_results, vectorized_results):
    """Validate that outputs are identical between methods."""
    print("\n🔍 Validating Identical Outputs")
    print("=" * 50)
    
    assert len(sequential_results) == len(vectorized_results), "Different number of samples"
    
    total_comparisons = 0
    identical_count = 0
    token_mismatches = []
    
    for sample_idx, (seq_sample, vec_sample) in enumerate(zip(sequential_results, vectorized_results)):
        assert seq_sample['task_type'] == vec_sample['task_type'], f"Task type mismatch sample {sample_idx}"
        
        k_samples = len(seq_sample['completions'])
        
        print(f"\nSample {sample_idx+1} ({seq_sample['task_type']}):")
        
        for k in range(k_samples):
            seq_tokens = seq_sample['completion_tokens'][k]
            vec_tokens = vec_sample['completion_tokens'][k]
            seq_completion = seq_sample['completions'][k]
            vec_completion = vec_sample['completions'][k]
            
            tokens_match = seq_tokens == vec_tokens
            text_match = seq_completion == vec_completion
            
            total_comparisons += 1
            if tokens_match and text_match:
                identical_count += 1
                print(f"  K={k+1}: ✅ IDENTICAL")
            else:
                print(f"  K={k+1}: ❌ DIFFERENT")
                print(f"    Sequential tokens: {seq_tokens[:10]}...")
                print(f"    Vectorized tokens: {vec_tokens[:10]}...")
                print(f"    Sequential text: '{seq_completion[:40]}...'")
                print(f"    Vectorized text: '{vec_completion[:40]}...'")
                
                token_mismatches.append({
                    'sample_idx': sample_idx,
                    'k': k,
                    'sequential_tokens': seq_tokens,
                    'vectorized_tokens': vec_tokens
                })
    
    print(f"\n📊 Validation Results:")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Identical outputs: {identical_count}/{total_comparisons}")
    print(f"  Success rate: {identical_count/total_comparisons:.1%}")
    
    if identical_count == total_comparisons:
        print(f"✅ PERFECT: Vectorized generation produces identical results")
        return True, []
    else:
        print(f"❌ MISMATCH: {total_comparisons - identical_count} differences found")
        return False, token_mismatches


def test_deterministic_vectorization():
    """Main test: Compare sequential vs vectorized generation."""
    print("🚀 Deterministic Vectorization Validation")
    print("=" * 70)
    
    # Load model
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load test samples (same as baseline)
    samples = load_and_prepare_samples(n_samples=20, seed=42)
    test_samples = samples[:3]  # Smaller set for detailed validation
    
    # Deterministic configuration
    config = GRPOConfig(
        k_samples=4,  # Start with 4 for easier validation
        temperature=0.01,  # Near-deterministic
        top_k=1,           # Most deterministic
        top_p=1.0,         # Disable top-p sampling
        max_new_tokens=144
    )
    
    print(f"Test configuration:")
    print(f"  Samples: {len(test_samples)}")
    print(f"  K-samples: {config.k_samples}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Top-k: {config.top_k}")
    print(f"  Max tokens: {config.max_new_tokens}")
    
    # Run sequential generation
    sequential_results, seq_time = generate_sequential_deterministic(
        model, tokenizer, test_samples, config.k_samples, config
    )
    
    # Run vectorized generation  
    vectorized_results, vec_time = generate_vectorized_deterministic(
        model, tokenizer, test_samples, config.k_samples, config
    )
    
    # Validate identical outputs
    identical, mismatches = validate_identical_outputs(sequential_results, vectorized_results)
    
    # Performance comparison
    speedup = seq_time / vec_time if vec_time > 0 else 1.0
    
    print(f"\n⚡ Performance Results:")
    print(f"  Sequential time: {seq_time:.3f}s")
    print(f"  Vectorized time: {vec_time:.3f}s")
    print(f"  Speedup: {speedup:.1f}x")
    
    # Save results for investigation if needed
    if not identical:
        mismatch_file = Path(__file__).parent / "vectorization_mismatches.json"
        with open(mismatch_file, 'w') as f:
            json.dump({
                'sequential_results': sequential_results,
                'vectorized_results': vectorized_results, 
                'mismatches': mismatches
            }, f, indent=2, default=str)
        print(f"💾 Mismatch details saved to {mismatch_file}")
    
    return identical, speedup, mismatches


def test_reward_scoring_consistency():
    """Test that reward scoring produces identical results."""
    print("\n🏆 Testing Reward Scoring Consistency")
    print("=" * 50)
    
    # Load baseline data
    baseline_file = Path(__file__).parent / "deterministic_baseline_k4.json"
    if not baseline_file.exists():
        print("⚠️ No baseline file found - create baseline first")
        return False
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    scorer = RewardScorer(
        reward_shaping="graduated",
        continuous_components={"fen_similarity": "exponential", "evaluations": "linear"}
    )
    
    # Test both sequential and batch scoring on same data
    all_prompts = []
    all_completions = []
    baseline_rewards = []
    
    for sample in baseline['sample_results']:
        for k in range(len(sample['completions'])):
            all_prompts.append(sample['prompt'])
            all_completions.append(sample['completions'][k])
            baseline_rewards.append(sample['rewards'][k])
    
    print(f"Testing {len(all_prompts)} prompt-completion pairs")
    
    # Sequential scoring
    seq_rewards = []
    for prompt, completion in zip(all_prompts, all_completions):
        reward, _ = scorer.score_single(prompt, completion, log_details=False)
        seq_rewards.append(reward)
    
    # Batch scoring
    batch_rewards, _ = scorer.score_batch(all_prompts, all_completions, compute_advantages=False)
    
    # Validate consistency
    max_diff = np.abs(np.array(seq_rewards) - batch_rewards).max()
    mean_diff = np.abs(np.array(seq_rewards) - batch_rewards).mean()
    
    print(f"  Sequential vs Batch scoring:")
    print(f"    Max difference: {max_diff:.8f}")
    print(f"    Mean difference: {mean_diff:.8f}")
    
    scoring_identical = max_diff < 1e-6
    
    if scoring_identical:
        print(f"✅ Reward scoring is perfectly consistent")
    else:
        print(f"❌ Reward scoring differs between sequential and batch")
    
    return scoring_identical


if __name__ == "__main__":
    print("🧪 Deterministic Vectorization Validation Suite")
    print("=" * 80)
    print("Testing vectorization with identical token sequence requirements")
    print()
    
    try:
        # Test 1: Deterministic generation comparison
        generation_identical, speedup, mismatches = test_deterministic_vectorization()
        
        # Test 2: Reward scoring consistency
        scoring_identical = test_reward_scoring_consistency()
        
        print(f"\n" + "=" * 80)
        print(f"🎯 VALIDATION SUMMARY")
        print(f"  Generation identical: {'✅' if generation_identical else '❌'}")
        print(f"  Scoring identical: {'✅' if scoring_identical else '❌'}")
        print(f"  Speedup achieved: {speedup:.1f}x")
        
        if generation_identical and scoring_identical:
            print(f"\n🎉 VECTORIZATION READY FOR IMPLEMENTATION")
            print(f"✅ All outputs identical between sequential and vectorized methods")
            print(f"✅ {speedup:.1f}x speedup confirmed")
            print(f"Safe to implement vectorization in training pipeline")
        else:
            print(f"\n❌ VECTORIZATION NOT READY")
            if not generation_identical:
                print(f"❌ Generation produces different outputs ({len(mismatches)} mismatches)")
            if not scoring_identical:
                print(f"❌ Reward scoring produces different results")
            print(f"Must fix issues before implementation")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise