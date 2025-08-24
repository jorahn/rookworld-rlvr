#!/usr/bin/env python3
"""
Analyze Partial Results from Hyperparameter Search

This script analyzes results as they become available, providing insights
even if the full search isn't complete.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging

def analyze_partial_results():
    """Analyze available results from focused search."""
    results_file = Path("focused_hyperparameter_results/focused_results.json")
    
    if not results_file.exists():
        print("No results file found yet...")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No results available yet...")
        return
    
    print(f"\n{'='*60}")
    print(f"PARTIAL RESULTS ANALYSIS ({len(data)} experiments completed)")
    print(f"{'='*60}")
    
    successful = [r for r in data if r['results']['success']]
    failed = [r for r in data if not r['results']['success']]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"📊 Success Rate: {len(successful)/len(data)*100:.1f}%")
    
    if successful:
        print(f"\n🏆 BEST RESULTS SO FAR:")
        
        # Sort by KL divergence (lower is better, avoid inf)
        stable_results = [r for r in successful if r['results']['kl_divergence'] != float('inf')]
        
        if stable_results:
            best = min(stable_results, key=lambda x: abs(x['results']['kl_divergence']))
            
            print(f"Best KL Divergence: {best['results']['kl_divergence']:.3f}")
            print(f"Steps Completed: {best['results']['steps_completed']}")  
            print(f"Policy Reward: {best['results']['policy_reward']:.3f}")
            print(f"Config:")
            config = best['config']
            print(f"  lr={config['lr']:.0e}, kl_coef={config['kl_coef']}, clip={config['clip_range']}")
            print(f"  temp={config['temperature']}, mix_env={config['mix_env_ratio']}")
            print(f"  batch={config['batch_positions']}x{config['group_size']}, estimator={config['kl_estimator']}")
        
        print(f"\n📈 SUCCESS PATTERNS:")
        
        # Analyze learning rates
        lr_success = {}
        lr_total = {}
        for r in data:
            lr = r['config']['lr']
            lr_total[lr] = lr_total.get(lr, 0) + 1
            if r['results']['success']:
                lr_success[lr] = lr_success.get(lr, 0) + 1
        
        print("Learning Rate Success Rates:")
        for lr in sorted(lr_total.keys()):
            success_count = lr_success.get(lr, 0)
            total_count = lr_total[lr]
            success_rate = success_count / total_count * 100
            print(f"  lr={lr:.0e}: {success_rate:.1f}% ({success_count}/{total_count})")
    
    if failed:
        print(f"\n❌ FAILURE ANALYSIS:")
        error_counts = {}
        for r in failed:
            error = r['results']['error_message']
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error}: {count} occurrences ({count/len(failed)*100:.1f}%)")
    
    print(f"\n⏱️  TIMING ANALYSIS:")
    if data:
        times = [r['results']['training_time'] for r in data]
        avg_time = sum(times) / len(times)
        print(f"Average experiment time: {avg_time:.1f} seconds")
        
        remaining = 432 - len(data)
        estimated_remaining_time = remaining * avg_time / 60
        print(f"Estimated time remaining: {estimated_remaining_time:.1f} minutes")

if __name__ == "__main__":
    analyze_partial_results()