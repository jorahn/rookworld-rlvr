#!/usr/bin/env python3
"""Analyze GRPO training metrics and generation quality"""

import re
import sys

def analyze_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract metrics
    metrics = []
    current_step = 0
    rewards_per_step = []
    completions = []
    prompts = []
    
    for i, line in enumerate(lines):
        # Step number
        if "TRAINING STEP" in line:
            match = re.search(r'STEP (\d+)/\d+', line)
            if match:
                current_step = int(match.group(1))
        
        # Loss metrics
        elif "Total Loss:" in line:
            loss = float(line.split()[-1])
            metrics.append({'step': current_step, 'total_loss': loss})
        elif "Policy Loss:" in line:
            if metrics and metrics[-1]['step'] == current_step:
                metrics[-1]['policy_loss'] = float(line.split()[-1])
        elif "KL Penalty:" in line:
            if metrics and metrics[-1]['step'] == current_step:
                metrics[-1]['kl_penalty'] = float(line.split()[-1])
        elif "KL Divergence:" in line:
            if metrics and metrics[-1]['step'] == current_step:
                metrics[-1]['kl_div'] = float(line.split()[-1])
        elif "Mean Reward:" in line:
            if metrics and metrics[-1]['step'] == current_step:
                metrics[-1]['mean_reward'] = float(line.split()[-1])
        
        # Reward distributions
        elif "Rewards - mean:" in line:
            match = re.search(r'mean: ([-\d.]+), std: ([-\d.]+), min: ([-\d.]+), max: ([-\d.]+)', line)
            if match:
                rewards_per_step.append({
                    'step': current_step,
                    'mean': float(match.group(1)),
                    'std': float(match.group(2)),
                    'min': float(match.group(3)),
                    'max': float(match.group(4))
                })
        
        # Sample prompts and completions
        elif "Sample" in line and ("(P)" in line or "(A)" in line):
            task_type = "P" if "(P)" in line else "A"
            prompt_text = line.split(":", 2)[-1].strip()
            prompts.append({'step': current_step, 'type': task_type, 'prompt': prompt_text})
        elif "Completion" in line:
            completion_text = line.split(":", 2)[-1].strip()
            completions.append({'step': current_step, 'text': completion_text})
    
    print("="*70)
    print("GRPO TRAINING METRICS ANALYSIS")
    print("="*70)
    
    # Loss convergence
    print("\n📊 LOSS CONVERGENCE:")
    print("-"*50)
    if metrics:
        print(f"{'Step':<6} {'Total Loss':>12} {'Policy Loss':>12} {'KL Penalty':>12}")
        for m in metrics[:5]:  # First 5 steps
            print(f"{m['step']:<6} {m.get('total_loss', 0):>12.4f} {m.get('policy_loss', 0):>12.4f} {m.get('kl_penalty', 0):>12.4f}")
        print("...")
        for m in metrics[-3:]:  # Last 3 steps
            print(f"{m['step']:<6} {m.get('total_loss', 0):>12.4f} {m.get('policy_loss', 0):>12.4f} {m.get('kl_penalty', 0):>12.4f}")
    
    # KL Divergence trend
    print("\n📈 KL DIVERGENCE TREND:")
    print("-"*50)
    if metrics:
        kl_values = [m.get('kl_div', 0) for m in metrics if 'kl_div' in m]
        if kl_values:
            print(f"Initial KL: {kl_values[0]:.4f}")
            print(f"Final KL: {kl_values[-1]:.4f}")
            print(f"Min KL: {min(kl_values):.4f}")
            print(f"Max KL: {max(kl_values):.4f}")
            print(f"Average KL: {sum(kl_values)/len(kl_values):.4f}")
    
    # Reward distributions
    print("\n🎯 REWARD DISTRIBUTIONS:")
    print("-"*50)
    if rewards_per_step:
        print(f"{'Step':<6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        for r in rewards_per_step[:5]:
            print(f"{r['step']:<6} {r['mean']:>8.3f} {r['std']:>8.3f} {r['min']:>8.3f} {r['max']:>8.3f}")
        if len(rewards_per_step) > 5:
            print("...")
            for r in rewards_per_step[-3:]:
                print(f"{r['step']:<6} {r['mean']:>8.3f} {r['std']:>8.3f} {r['min']:>8.3f} {r['max']:>8.3f}")
    
    # Task distribution
    print("\n📝 TASK DISTRIBUTION:")
    print("-"*50)
    p_tasks = len([p for p in prompts if p['type'] == 'P'])
    a_tasks = len([p for p in prompts if p['type'] == 'A'])
    print(f"Policy tasks (P:): {p_tasks}")
    print(f"Environment tasks (A:): {a_tasks}")
    if p_tasks + a_tasks > 0:
        print(f"P:A ratio: {p_tasks/(p_tasks+a_tasks):.1%} : {a_tasks/(p_tasks+a_tasks):.1%}")
    
    # Sample generation quality
    print("\n🔍 SAMPLE GENERATIONS:")
    print("-"*50)
    
    # Show a few P: task examples
    p_samples = [p for p in prompts if p['type'] == 'P'][:2]
    for i, sample in enumerate(p_samples):
        print(f"\nP: Task Sample {i+1}:")
        print(f"Prompt: {sample['prompt'][:80]}...")
        if i < len(completions):
            print(f"Generated: {completions[i]['text'][:80]}...")
    
    # Show a few A: task examples
    a_samples = [p for p in prompts if p['type'] == 'A'][:2]
    for i, sample in enumerate(a_samples):
        print(f"\nA: Task Sample {i+1}:")
        print(f"Prompt: {sample['prompt'][:80]}...")
        # Find corresponding completion
        comp_idx = len(p_samples) + i
        if comp_idx < len(completions):
            print(f"Generated: {completions[comp_idx]['text'][:80]}...")
    
    # Analysis summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY:")
    print("-"*70)
    
    # Check for healthy training signs
    issues = []
    warnings = []
    good_signs = []
    
    if metrics:
        # Loss behavior
        initial_loss = metrics[0].get('total_loss', 0)
        final_loss = metrics[-1].get('total_loss', 0)
        
        if final_loss < initial_loss:
            good_signs.append("✅ Loss is decreasing")
        else:
            warnings.append("⚠️ Loss not decreasing consistently")
        
        # KL divergence
        if kl_values:
            avg_kl = sum(kl_values)/len(kl_values)
            if avg_kl < -10:
                issues.append("❌ KL divergence too negative (model diverging)")
            elif avg_kl > 10:
                issues.append("❌ KL divergence too high (model not learning)")
            elif -5 < avg_kl < 5:
                good_signs.append("✅ KL divergence in healthy range")
    
    if rewards_per_step:
        # Reward trends
        final_rewards = rewards_per_step[-1] if rewards_per_step else None
        if final_rewards and final_rewards['max'] > 0.2:
            good_signs.append("✅ Some samples achieving positive rewards")
        if final_rewards and final_rewards['std'] > 0.05:
            good_signs.append("✅ Good reward variance (exploration happening)")
    
    # Check generation quality
    if completions:
        # Check for chess-like patterns in completions
        chess_patterns = 0
        for comp in completions[:10]:
            text = comp['text']
            # Look for chess notation patterns
            if any(p in text for p in ['e4', 'Nf3', 'd4', 'Bc4', 'O-O', 'Qd', 'K', 'R', 'B', 'N']):
                chess_patterns += 1
        
        if chess_patterns > len(completions[:10]) * 0.3:
            good_signs.append("✅ Generations contain chess notation")
        else:
            warnings.append("⚠️ Few chess patterns in generations")
    
    # Print summary
    for sign in good_signs:
        print(sign)
    for warning in warnings:
        print(warning)
    for issue in issues:
        print(issue)
    
    print(f"\nOverall: {len(good_signs)} positive signs, {len(warnings)} warnings, {len(issues)} issues")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "lean_bs64_test.log"
    analyze_log(log_file)