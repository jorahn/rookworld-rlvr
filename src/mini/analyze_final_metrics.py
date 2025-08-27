import re
import numpy as np

log_file = "logs_fixed/grpo_training_20250827_175506.log"
with open(log_file, 'r') as f:
    content = f.read()

# Extract metrics
steps = []
rewards = []
kl_divs = []
kl_forward = []
kl_reverse = []
outliers = []

for match in re.finditer(r'=== ROLLOUT DETAILS \(Step (\d+)\) ===.*?Rewards:.*?mean=([\d.]+).*?KL Divergence: ([-\d.]+).*?KL Forward: ([-\d.]+).*?KL Reverse: ([\d.]+).*?Ratio Outliers: ([\d.]+)%', content, re.DOTALL):
    steps.append(int(match.group(1)))
    rewards.append(float(match.group(2)))
    kl_divs.append(float(match.group(3)))
    kl_forward.append(float(match.group(4)))
    kl_reverse.append(float(match.group(5)))
    outliers.append(float(match.group(6)))

print("GRPO Training Analysis (20 Steps)")
print("=" * 80)
print(f"{'Step':<6} {'Reward':<10} {'Δ Reward':<10} {'KL Div':<10} {'KL Fwd':<10} {'KL Rev':<10} {'Clip %':<8}")
print("-" * 80)

for i in range(len(steps)):
    reward_change = ""
    if i > 0:
        diff = rewards[i] - rewards[i-1]
        reward_change = f"{diff:+.3f}"
    else:
        reward_change = "---"
    print(f"{steps[i]:<6} {rewards[i]:<10.3f} {reward_change:<10} {kl_divs[i]:<10.4f} {kl_forward[i]:<10.4f} {kl_reverse[i]:<10.4f} {outliers[i]:<8.1f}")

print("-" * 80)
print("\n📊 Summary Statistics:")
print(f"  Initial reward:     {rewards[0]:.3f}")
print(f"  Final reward:       {rewards[-1]:.3f} (total change: {rewards[-1] - rewards[0]:+.3f})")
print(f"  Best reward:        {max(rewards):.3f} (step {steps[rewards.index(max(rewards))]})")
print(f"  Worst reward:       {min(rewards):.3f} (step {steps[rewards.index(min(rewards))]})")
print(f"  Average reward:     {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")

print(f"\n📈 Reward Trend:")
first_half = np.mean(rewards[:len(rewards)//2])
second_half = np.mean(rewards[len(rewards)//2:])
print(f"  First half avg:     {first_half:.3f}")
print(f"  Second half avg:    {second_half:.3f}")
print(f"  Improvement:        {second_half - first_half:+.3f}")

print(f"\n🎯 KL Divergence:")
print(f"  Max KL forward:     {max(kl_forward):.4f}")
print(f"  Min KL forward:     {min(kl_forward):.4f}")
print(f"  Final KL forward:   {kl_forward[-1]:.4f}")
print(f"  Avg KL reverse:     {np.mean(kl_reverse):.4f}")
print(f"  KL NOT exploding:   {'✓' if max(abs(kl) for kl in kl_forward) < 1.0 else '✗'}")

print(f"\n✂️ PPO Clipping:")
print(f"  Average clipping:   {np.mean(outliers):.1f}%")
print(f"  Max clipping:       {max(outliers):.1f}% (step {steps[outliers.index(max(outliers))]})")
print(f"  Steps >50% clip:    {sum(1 for o in outliers if o > 50)}/{len(outliers)}")
