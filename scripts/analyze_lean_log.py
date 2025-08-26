#!/usr/bin/env python3
"""Analyze the 500-step lean training log"""

import re

with open('lean_500step_test.log', 'r') as f:
    lines = f.readlines()

step_data = []
current_step = 0

for line in lines:
    # Extract step number
    if "TRAINING STEP" in line:
        match = re.search(r'STEP (\d+)/(\d+)', line)
        if match:
            current_step = int(match.group(1))
    
    # Extract metrics
    elif "Total Loss:" in line and current_step > 0:
        loss = float(line.split()[-1])
        step_data.append({'step': current_step, 'metric': 'loss', 'value': loss})
    elif "Mean Reward:" in line and current_step > 0:
        reward = float(line.split()[-1])
        step_data.append({'step': current_step, 'metric': 'reward', 'value': reward})
    elif "KL Divergence:" in line and current_step > 0:
        kl = float(line.split()[-1])
        step_data.append({'step': current_step, 'metric': 'kl', 'value': kl})
    elif "GPU 0 memory - allocated:" in line and current_step > 0:
        match = re.search(r'allocated: ([\d.]+)GB, reserved: ([\d.]+)GB', line)
        if match:
            alloc = float(match.group(1))
            resv = float(match.group(2))
            step_data.append({'step': current_step, 'metric': 'gpu0_alloc', 'value': alloc})
            step_data.append({'step': current_step, 'metric': 'gpu0_resv', 'value': resv})

# Organize by step
steps = {}
for item in step_data:
    step = item['step']
    if step not in steps:
        steps[step] = {}
    steps[step][item['metric']] = item['value']

# Print summary at key intervals
print("="*70)
print("LEAN GRPO 500-STEP TRAINING SUMMARY")
print("="*70)
print(f"{'Step':<6} {'Loss':>10} {'Reward':>10} {'KL Div':>10} {'GPU Alloc':>10} {'GPU Resv':>10}")
print("-"*70)

for step_num in [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    if step_num in steps:
        s = steps[step_num]
        loss = s.get('loss', 0)
        reward = s.get('reward', 0)
        kl = s.get('kl', 0)
        gpu_alloc = s.get('gpu0_alloc', 0)
        gpu_resv = s.get('gpu0_resv', 0)
        print(f"{step_num:<6} {loss:>10.4f} {reward:>10.4f} {kl:>10.4f} {gpu_alloc:>10.2f} {gpu_resv:>10.2f}")

# Check for trends
first_10 = [steps[i].get('loss', 0) for i in range(1, 11) if i in steps]
last_10 = [steps[i].get('loss', 0) for i in range(491, 501) if i in steps]

first_gpu = [steps[i].get('gpu0_alloc', 0) for i in range(1, 11) if i in steps]
last_gpu = [steps[i].get('gpu0_alloc', 0) for i in range(491, 501) if i in steps]

print("\n" + "="*70)
print("ANALYSIS:")
print("-"*70)
print(f"Average loss (first 10 steps): {sum(first_10)/len(first_10) if first_10 else 0:.4f}")
print(f"Average loss (last 10 steps):  {sum(last_10)/len(last_10) if last_10 else 0:.4f}")
print(f"GPU memory (start): {first_gpu[0] if first_gpu else 0:.2f}GB")
print(f"GPU memory (end):   {last_gpu[-1] if last_gpu else 0:.2f}GB")

# Check for any errors
error_count = sum(1 for line in lines if 'ERROR' in line)
warning_count = sum(1 for line in lines if 'WARNING' in line)

print(f"\nErrors: {error_count}")
print(f"Warnings: {warning_count}")

# Check if training completed
completed = any("TRAINING COMPLETED SUCCESSFULLY" in line for line in lines)
print(f"\nTraining completed: {'✅ Yes' if completed else '❌ No'}")