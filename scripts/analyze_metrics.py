import re

log_file = "logs_fixed/grpo_training_20250827_175506.log"
with open(log_file, 'r') as f:
    content = f.read()

# Extract metrics
steps = []
rewards = []
kl_divs = []
outliers = []

for match in re.finditer(r'=== ROLLOUT DETAILS \(Step (\d+)\) ===.*?Rewards:.*?mean=([\d.]+).*?KL Divergence: ([-\d.]+).*?Ratio Outliers: ([\d.]+)%', content, re.DOTALL):
    steps.append(int(match.group(1)))
    rewards.append(float(match.group(2)))
    kl_divs.append(float(match.group(3)))
    outliers.append(float(match.group(4)))

print("Training Metrics Summary:")
print("=" * 60)
print(f"{'Step':<6} {'Mean Reward':<12} {'KL Divergence':<15} {'Clipping %':<12}")
print("-" * 60)

for i in range(len(steps)):
    reward_change = ""
    if i > 0:
        diff = rewards[i] - rewards[i-1]
        reward_change = f"({diff:+.3f})" if diff != 0 else ""
    print(f"{steps[i]:<6} {rewards[i]:<12.3f} {kl_divs[i]:<15.4f} {outliers[i]:<12.1f}")

print("-" * 60)
print(f"Initial reward: {rewards[0]:.3f}")
print(f"Final reward:   {rewards[-1]:.3f} (change: {rewards[-1] - rewards[0]:+.3f})")
print(f"Average reward: {sum(rewards)/len(rewards):.3f}")
print(f"Max KL div:     {max(kl_divs):.4f}")
print(f"Min KL div:     {min(kl_divs):.4f}")
print(f"Avg clipping:   {sum(outliers)/len(outliers):.1f}%")
