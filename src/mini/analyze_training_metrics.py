"""
Analyze GRPO training metrics including continuous rewards.
"""

import re
import numpy as np

def analyze_log(log_file):
    """Extract and analyze metrics from training log."""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract metrics
    steps = []
    losses = []
    pg_losses = []
    kl_divs = []
    kl_forward = []
    kl_reverse = []
    outliers = []
    rewards_mean = []
    rewards_dist = []
    times = []
    
    # Parse step data
    for match in re.finditer(r'Step (\d+) \| Time: ([\d.]+)s.*?Loss: ([-\d.]+).*?PG Loss: ([-\d.]+).*?KL Divergence: ([-\d.]+).*?KL Forward: ([-\d.]+).*?KL Reverse: ([\d.]+).*?Ratio Outliers: ([\d.]+)%', 
                             content, re.DOTALL):
        steps.append(int(match.group(1)))
        times.append(float(match.group(2)))
        losses.append(float(match.group(3)))
        pg_losses.append(float(match.group(4)))
        kl_divs.append(float(match.group(5)))
        kl_forward.append(float(match.group(6)))
        kl_reverse.append(float(match.group(7)))
        outliers.append(float(match.group(8)))
    
    # Parse reward distributions
    for match in re.finditer(r'=== ROLLOUT DETAILS \(Step (\d+)\) ===.*?Rewards: min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+), std=([-\d.]+).*?Reward distribution: ({.*?})', 
                             content, re.DOTALL):
        step = int(match.group(1))
        mean_reward = float(match.group(4))
        dist = match.group(6)
        if step in steps:
            idx = steps.index(step)
            if idx < len(rewards_mean):
                rewards_mean[idx] = mean_reward
            else:
                rewards_mean.append(mean_reward)
                rewards_dist.append(dist)
    
    return {
        'steps': steps,
        'losses': losses,
        'pg_losses': pg_losses,
        'kl_divs': kl_divs,
        'kl_forward': kl_forward,
        'kl_reverse': kl_reverse,
        'outliers': outliers,
        'rewards_mean': rewards_mean,
        'rewards_dist': rewards_dist,
        'times': times
    }

def print_analysis(metrics):
    """Print comprehensive analysis of training metrics."""
    
    print("\n" + "="*80)
    print("GRPO TRAINING ANALYSIS WITH CONTINUOUS REWARDS")
    print("="*80)
    
    n_steps = len(metrics['steps'])
    print(f"\nTotal steps analyzed: {n_steps}")
    print(f"Average time per step: {np.mean(metrics['times']):.2f}s")
    
    # Reward Analysis
    print("\n" + "="*80)
    print("REWARD METRICS")
    print("="*80)
    
    if metrics['rewards_mean']:
        rewards = metrics['rewards_mean']
        print(f"Initial mean reward: {rewards[0]:.3f}")
        print(f"Final mean reward: {rewards[-1]:.3f}")
        print(f"Reward improvement: {rewards[-1] - rewards[0]:+.3f} ({(rewards[-1]/rewards[0] - 1)*100:+.1f}%)")
        print(f"Average reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"Max reward achieved: {max(rewards):.3f} (step {metrics['steps'][rewards.index(max(rewards))]})")
        
        # Trend analysis
        first_half = np.mean(rewards[:len(rewards)//2]) if len(rewards) > 1 else rewards[0]
        second_half = np.mean(rewards[len(rewards)//2:]) if len(rewards) > 1 else rewards[0]
        print(f"\nReward trend:")
        print(f"  First half average: {first_half:.3f}")
        print(f"  Second half average: {second_half:.3f}")
        print(f"  Trend: {second_half - first_half:+.3f}")
    
    # KL Divergence Analysis
    print("\n" + "="*80)
    print("KL DIVERGENCE METRICS")
    print("="*80)
    
    print(f"Average KL divergence: {np.mean(metrics['kl_divs']):.4f}")
    print(f"Max KL divergence: {max(metrics['kl_divs']):.4f} (step {metrics['steps'][metrics['kl_divs'].index(max(metrics['kl_divs']))]})")
    print(f"Min KL divergence: {min(metrics['kl_divs']):.4f} (step {metrics['steps'][metrics['kl_divs'].index(min(metrics['kl_divs']))]})")
    
    # Check for KL explosion
    kl_exploded = any(abs(kl) > 1.0 for kl in metrics['kl_divs'])
    print(f"KL explosion detected: {'YES ⚠️' if kl_exploded else 'NO ✓'}")
    
    print(f"\nKL components:")
    print(f"  Avg Forward KL: {np.mean(metrics['kl_forward']):.4f}")
    print(f"  Avg Reverse KL: {np.mean(metrics['kl_reverse']):.4f}")
    print(f"  Symmetric KL (avg): {np.mean([(f+r)/2 for f,r in zip(metrics['kl_forward'], metrics['kl_reverse'])]):.4f}")
    
    # PPO Clipping Analysis
    print("\n" + "="*80)
    print("PPO CLIPPING METRICS")
    print("="*80)
    
    print(f"Average ratio outliers: {np.mean(metrics['outliers']):.1f}%")
    print(f"Max ratio outliers: {max(metrics['outliers']):.1f}% (step {metrics['steps'][metrics['outliers'].index(max(metrics['outliers']))]})")
    print(f"Steps with >30% clipping: {sum(1 for o in metrics['outliers'] if o > 30)}/{n_steps}")
    print(f"Steps with >50% clipping: {sum(1 for o in metrics['outliers'] if o > 50)}/{n_steps}")
    
    # Loss Analysis
    print("\n" + "="*80)
    print("LOSS METRICS")
    print("="*80)
    
    print(f"Average total loss: {np.mean(metrics['losses']):.4f}")
    print(f"Average PG loss: {np.mean(metrics['pg_losses']):.4f}")
    print(f"Loss trend: {metrics['losses'][-1] - metrics['losses'][0]:+.4f}")
    
    # Detailed Step-by-Step
    print("\n" + "="*80)
    print("STEP-BY-STEP SUMMARY")
    print("="*80)
    print(f"{'Step':<6} {'Reward':<10} {'KL Div':<10} {'Clipping %':<12} {'Loss':<10}")
    print("-"*50)
    
    for i in range(min(20, n_steps)):  # Show first 20 steps
        reward = metrics['rewards_mean'][i] if i < len(metrics['rewards_mean']) else 0.0
        print(f"{metrics['steps'][i]:<6} {reward:<10.3f} {metrics['kl_divs'][i]:<10.4f} {metrics['outliers'][i]:<12.1f} {metrics['losses'][i]:<10.4f}")
    
    # Final Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print("\n✅ POSITIVE INDICATORS:")
    if metrics['rewards_mean'] and metrics['rewards_mean'][-1] > metrics['rewards_mean'][0]:
        print(f"  • Rewards improved by {(metrics['rewards_mean'][-1] - metrics['rewards_mean'][0]):.3f}")
    if not kl_exploded:
        print(f"  • KL divergence stable (no explosion)")
    if np.mean(metrics['outliers']) < 40:
        print(f"  • PPO clipping reasonable ({np.mean(metrics['outliers']):.1f}% average)")
    
    print("\n⚠️ AREAS OF CONCERN:")
    if kl_exploded:
        print(f"  • KL divergence instability detected")
    if metrics['rewards_mean'] and metrics['rewards_mean'][-1] <= metrics['rewards_mean'][0]:
        print(f"  • No reward improvement")
    if np.mean(metrics['outliers']) > 40:
        print(f"  • High clipping rate ({np.mean(metrics['outliers']):.1f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Analyze the latest log
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs_continuous/grpo_training_20250827_183605.log"
    
    print(f"Analyzing: {log_file}")
    metrics = analyze_log(log_file)
    print_analysis(metrics)