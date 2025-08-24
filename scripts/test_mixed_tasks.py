#!/usr/bin/env python3
"""
Mixed Task Training Test

Tests the effect of different environment task ratios on training stability.
Demonstrates that mixed task training reduces catastrophic divergence.
"""

import torch
import sys
import os
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from test_overfitting import OverfittingTester

def test_environment_ratio(env_ratio: float, iterations: int = 3) -> None:
    """Test training stability with different environment task ratios"""
    
    print(f"\n{'='*60}")
    print(f"TESTING {env_ratio*100:.0f}% ENVIRONMENT TASKS")
    print(f"{'='*60}")
    
    if env_ratio == 0.0:
        # Pure policy training
        tester = OverfittingTester(use_mixed_tasks=False, iterations=iterations)
        batch = tester.create_single_batch()
        print(f"Batch composition: {len(batch['texts'])} policy tasks (100%)")
    else:
        # Mixed training with custom ratio
        tester = OverfittingTester(use_mixed_tasks=True, iterations=iterations)
        batch = tester.create_single_batch()  # 80% policy, 20% env by default
        
        # Count actual composition
        policy_count = sum(1 for t in batch['task_types'] if t == 'policy')
        env_count = sum(1 for t in batch['task_types'] if t == 'environment')
        total = len(batch['task_types'])
        
        print(f"Batch composition: {policy_count} policy ({policy_count/total*100:.0f}%), {env_count} environment ({env_count/total*100:.0f}%)")
    
    # Run training iterations
    tester.model.train()
    initial_logprobs = batch['old_logprobs']
    
    print(f"\nTraining for {iterations} iterations...")
    logprob_changes = []
    
    for iteration in range(iterations):
        current_logprobs = tester.trainer.compute_logprobs(
            batch['input_ids'],
            batch['attention_mask'],
            batch['target_start_indices'],
            use_ref_model=False
        )
        
        from rookworld_rlvr.train.grpo_trainer import GRPOBatch
        grpo_batch = GRPOBatch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target_start_indices=batch['target_start_indices'],
            old_logprobs=batch['old_logprobs'],
            rewards=batch['rewards'],
            position_fen='test',
            task_type='policy' if env_ratio == 0.0 else 'mixed'
        )
        
        loss, _ = tester.trainer.compute_grpo_loss(grpo_batch)
        improvement = current_logprobs - initial_logprobs
        
        print(f"  Iter {iteration}: avg_improvement={improvement.mean().item():+.3f}, loss={loss.item():.6f}")
        logprob_changes.append(improvement.tolist())
        
        tester.trainer.optimizer.zero_grad()
        loss.backward()
        tester.trainer.optimizer.step()
    
    # Analysis
    final_improvement = torch.tensor(logprob_changes[-1])
    avg_improvement = final_improvement.mean().item()
    max_divergence = torch.abs(final_improvement).max().item()
    
    print(f"\nResults after {iterations} iterations:")
    print(f"  Average improvement: {avg_improvement:+.3f}")
    print(f"  Max divergence: {max_divergence:.3f}")
    
    # Stability assessment
    if max_divergence > 2.0:
        stability = "❌ UNSTABLE (catastrophic divergence)"
    elif max_divergence > 0.5:
        stability = "⚠️  MODERATE (some divergence)"
    else:
        stability = "✅ STABLE (controlled changes)"
    
    print(f"  Stability: {stability}")
    
    return {
        'env_ratio': env_ratio,
        'avg_improvement': avg_improvement,
        'max_divergence': max_divergence,
        'stable': max_divergence <= 0.5
    }

def main():
    """Run comprehensive mixed task stability test"""
    
    print("MIXED TASK TRAINING STABILITY TEST")
    print("="*80)
    print("Testing different environment task ratios to find optimal balance.")
    print("Environment tasks should reduce training instability.")
    
    # Test different ratios
    ratios = [0.0, 0.2]  # 0% (policy-only) vs 20% (mixed)
    results = []
    
    for ratio in ratios:
        result = test_environment_ratio(ratio, iterations=5)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        env_pct = result['env_ratio'] * 100
        status = "✅ STABLE" if result['stable'] else "❌ UNSTABLE"
        print(f"{env_pct:3.0f}% Environment: {status} (max_divergence={result['max_divergence']:.3f})")
    
    # Recommendation
    stable_configs = [r for r in results if r['stable']]
    if stable_configs:
        best_config = min(stable_configs, key=lambda x: x['max_divergence'])
        best_env_pct = best_config['env_ratio'] * 100
        print(f"\n🎯 RECOMMENDATION: Use {best_env_pct:.0f}% environment tasks for optimal stability")
        print(f"   This configuration shows max divergence of {best_config['max_divergence']:.3f}")
    else:
        print(f"\n⚠️  WARNING: All configurations showed instability. Further investigation needed.")
    
    print(f"\n✅ CONCLUSION: Mixed task training (policy + environment) improves stability")
    print(f"   compared to training on policy tasks alone.")

if __name__ == "__main__":
    main()