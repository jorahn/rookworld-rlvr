"""
Comprehensive permutation testing for reward scorer

Tests that increasing permutations (deviations from ground truth) lead to decreasing rewards.
Uses 50+ samples per task type with 0-20% random permutations.
"""

import numpy as np
import random
import string
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(__file__))

from reward_scorer import RewardScorer, compute_grpo_rewards
from dataset import load_and_prepare_samples

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
logger = logging.getLogger(__name__)


@dataclass
class PermutationResult:
    """Results from a permutation test"""
    task_type: str
    permutation_rate: float
    original_reward: float
    permuted_reward: float
    reward_degradation: float
    original_completion: str
    permuted_completion: str


def apply_character_permutation(text: str, rate: float) -> str:
    """
    Apply random character permutations to text.
    
    Args:
        text: Original text
        rate: Permutation rate (0.0 to 1.0)
        
    Returns:
        Permuted text with approximately rate*len(text) characters modified
    """
    if rate <= 0 or not text:
        return text
    
    chars = list(text)
    n_permutations = int(len(chars) * rate)
    
    # Apply permutations one by one (positions may shift)
    for _ in range(min(n_permutations, len(chars))):
        if not chars:
            break
            
        pos = random.randint(0, len(chars) - 1)
        perm_type = random.choice(['substitute', 'delete', 'insert', 'swap'])
        
        if perm_type == 'substitute' and pos < len(chars):
            # Replace with random character
            if chars[pos].isalpha():
                chars[pos] = random.choice(string.ascii_lowercase)
            elif chars[pos].isdigit():
                chars[pos] = random.choice(string.digits)
            else:
                # Keep special chars mostly intact
                if random.random() < 0.5:
                    chars[pos] = random.choice([' ', '+', ':', '-'])
        
        elif perm_type == 'delete' and len(chars) > 1 and pos < len(chars):
            # Delete character (keep at least 1 char)
            chars.pop(pos)
            
        elif perm_type == 'insert' and pos < len(chars):
            # Insert random character
            if chars[pos].isalpha():
                chars.insert(pos, random.choice(string.ascii_lowercase))
            elif chars[pos].isdigit():
                chars.insert(pos, random.choice(string.digits))
            else:
                chars.insert(pos, random.choice([' ', '+', ':', '-']))
                
        elif perm_type == 'swap' and pos + 1 < len(chars):
            # Swap with next character
            chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
    
    return ''.join(chars)


def apply_structured_permutation(completion: str, task_type: str, rate: float) -> str:
    """
    Apply structured permutations that respect task format.
    
    This is more realistic than random character permutation,
    as it maintains structure while introducing errors.
    """
    if rate <= 0:
        return completion
    
    if task_type == "P":
        # P: task format: "M: [moves] E: [evals] B: [best]"
        
        # Parse sections
        sections = {}
        if "M:" in completion:
            parts = completion.split("M:", 1)
            if len(parts) > 1:
                moves_section = parts[1].split("E:", 1)[0] if "E:" in parts[1] else parts[1]
                sections['moves'] = moves_section.strip()
        
        if "E:" in completion:
            parts = completion.split("E:", 1)
            if len(parts) > 1:
                evals_section = parts[1].split("B:", 1)[0] if "B:" in parts[1] else parts[1]
                sections['evals'] = evals_section.strip()
        
        if "B:" in completion:
            parts = completion.split("B:", 1)
            if len(parts) > 1:
                sections['best'] = parts[1].strip()
        
        # Apply permutations based on rate
        if random.random() < rate * 5:  # Higher chance to affect structure
            # Remove a section
            if random.random() < 0.33 and 'best' in sections:
                del sections['best']
            elif random.random() < 0.5 and 'evals' in sections:
                del sections['evals']
        
        # Permute moves
        if 'moves' in sections and random.random() < rate * 3:
            moves = sections['moves'].split()
            for i in range(len(moves)):
                if random.random() < rate:
                    # Corrupt move notation
                    if len(moves[i]) >= 4:
                        move_chars = list(moves[i])
                        if random.random() < 0.5:
                            # Invalid square
                            move_chars[random.randint(0, 1)] = random.choice('ijklmnoz')
                        else:
                            # Invalid digit
                            move_chars[random.randint(2, 3)] = random.choice('09')
                        moves[i] = ''.join(move_chars)
            sections['moves'] = ' '.join(moves)
        
        # Permute evaluations
        if 'evals' in sections and random.random() < rate * 3:
            evals = sections['evals'].split()
            for i in range(len(evals)):
                if random.random() < rate:
                    try:
                        # Add noise to evaluation
                        val = float(evals[i])
                        val += random.uniform(-10, 10) * rate
                        evals[i] = str(round(val, 2))
                    except:
                        pass
            sections['evals'] = ' '.join(evals)
        
        # Reconstruct completion
        result = ""
        if 'moves' in sections:
            result += f"M: {sections['moves']} "
        if 'evals' in sections:
            result += f"E: {sections['evals']} "
        if 'best' in sections:
            result += f"B: {sections['best']}"
        
        return result.strip() if result else "invalid"
    
    elif task_type == "A":
        # A: task format: "[new_FEN]+[reward]+[terminated]+[truncated]"
        
        parts = completion.split("+")
        
        if len(parts) >= 4:
            # Apply permutations based on rate
            
            # Permute FEN
            if random.random() < rate * 3:
                fen_parts = parts[0].split("/")
                for i in range(len(fen_parts)):
                    if random.random() < rate:
                        # Corrupt rank
                        if fen_parts[i] and fen_parts[i][0].isdigit():
                            # Change number
                            fen_parts[i] = str(random.randint(1, 8)) + fen_parts[i][1:]
                        elif fen_parts[i]:
                            # Change piece
                            pieces = list(fen_parts[i])
                            if pieces:
                                pieces[0] = random.choice('rnbqkpRNBQKP')
                            fen_parts[i] = ''.join(pieces)
                parts[0] = "/".join(fen_parts)
            
            # Permute reward
            if random.random() < rate * 2:
                try:
                    reward = float(parts[1])
                    # Add noise
                    reward += random.uniform(-0.5, 0.5) * rate
                    parts[1] = str(round(reward, 3))
                except:
                    parts[1] = str(random.random())
            
            # Permute flags
            if random.random() < rate * 2:
                parts[2] = random.choice(['true', 'false'])
            if random.random() < rate * 2:
                parts[3] = random.choice(['true', 'false'])
            
            # Remove sections based on rate
            if random.random() < rate * 5:
                # Remove some parts
                parts = parts[:max(1, int(4 * (1 - rate)))]
            
            return "+".join(parts)
        
        return apply_character_permutation(completion, rate)
    
    else:
        return apply_character_permutation(completion, rate)


def generate_test_samples(n_samples: int = 50) -> List[Tuple[str, str, str]]:
    """
    Generate or load test samples with ground truth completions.
    
    Returns:
        List of (task_type, prompt, ground_truth_completion) tuples
    """
    samples = []
    
    # P: task samples with varying quality
    p_prompts = [
        "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "P: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "P: rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3",
        "P: r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "P: 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    
    p_completions = [
        "M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4",
        "M: f1c4 b1c3 d2d4 f1e2 c2c3  E: 0.25 0.22 0.20 0.18 0.15  B: f1c4",
        "M: e4e5 f3e5 d2d4 b1c3 f1e2  E: 0.4 0.38 0.35 0.33 0.30  B: e4e5",
        "M: e5d6 e5f6 e3f4 g2g4 a2a3  E: 1.2 0.9 0.7 0.5 0.3  B: e5d6",
        "M: b5b6 a5a6 g2g3 e2e3 b4b8  E: -2.5 -2.8 -3.0 -3.2 -3.5  B: b5b6",
    ]
    
    # Generate variations for P: tasks
    for _ in range(n_samples // 2):
        idx = random.randint(0, len(p_prompts) - 1)
        samples.append(("P", p_prompts[idx], p_completions[idx]))
    
    # A: task samples
    a_prompts = [
        "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+",
        "A: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+e7e5+e2e4,+",
        "A: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2+g1f3+e2e4,e7e5,+",
        "A: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4+e1g1+e2e4,e7e5,g1f3,b8c6,f1c4,g8f6,+",
        "A: 7k/8/8/8/8/8/8/7K w - - 0 1+h1h2+,+",
    ]
    
    a_completions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2+0.001+false+false",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2+0.001+false+false",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4+0.001+false+false",
        "7k/8/8/8/8/8/7K/8 b - - 1 1+0.001+false+false",
    ]
    
    # Generate variations for A: tasks
    for _ in range(n_samples // 2):
        idx = random.randint(0, len(a_prompts) - 1)
        samples.append(("A", a_prompts[idx], a_completions[idx]))
    
    return samples


def test_permutation_correlation(
    n_samples: int = 50,
    permutation_rates: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
    use_structured: bool = True
) -> Dict[str, List[PermutationResult]]:
    """
    Test correlation between permutation rate and reward degradation.
    
    Args:
        n_samples: Number of samples per task type
        permutation_rates: List of permutation rates to test
        use_structured: Use structured permutations (more realistic) vs random
        
    Returns:
        Dictionary mapping task types to lists of results
    """
    scorer = RewardScorer(reward_shaping="linear", min_reward=-1.0, max_reward=1.0)
    
    # Generate test samples
    samples = generate_test_samples(n_samples * 2)  # Get both P: and A: tasks
    
    results = {"P": [], "A": []}
    
    for task_type, prompt, ground_truth in samples:
        for perm_rate in permutation_rates:
            # Apply permutation
            if use_structured:
                permuted = apply_structured_permutation(ground_truth, task_type, perm_rate)
            else:
                permuted = apply_character_permutation(ground_truth, perm_rate)
            
            # Score original
            orig_reward, orig_details = scorer.score_single(prompt, ground_truth, log_details=False)
            
            # Score permuted
            perm_reward, perm_details = scorer.score_single(prompt, permuted, log_details=False)
            
            # Calculate degradation
            degradation = orig_reward - perm_reward
            
            result = PermutationResult(
                task_type=task_type,
                permutation_rate=perm_rate,
                original_reward=orig_reward,
                permuted_reward=perm_reward,
                reward_degradation=degradation,
                original_completion=ground_truth[:50] + "...",
                permuted_completion=permuted[:50] + "..."
            )
            
            results[task_type].append(result)
    
    return results


def analyze_results(results: Dict[str, List[PermutationResult]]) -> None:
    """
    Analyze and print statistics from permutation test results.
    """
    print("\n" + "="*80)
    print("PERMUTATION TEST RESULTS")
    print("="*80)
    
    for task_type in ["P", "A"]:
        print(f"\n{task_type}: Task Results")
        print("-"*40)
        
        task_results = results[task_type]
        
        # Group by permutation rate
        by_rate = {}
        for result in task_results:
            rate = result.permutation_rate
            if rate not in by_rate:
                by_rate[rate] = []
            by_rate[rate].append(result)
        
        # Calculate statistics per rate
        print(f"{'Perm Rate':<12} {'Avg Reward':<12} {'Std Dev':<12} {'Degradation':<12} {'Samples':<8}")
        print("-"*60)
        
        prev_avg_reward = None
        for rate in sorted(by_rate.keys()):
            rate_results = by_rate[rate]
            rewards = [r.permuted_reward for r in rate_results]
            degradations = [r.reward_degradation for r in rate_results]
            
            avg_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            avg_degradation = np.mean(degradations)
            
            print(f"{rate:>8.1%}     {avg_reward:>8.3f}     {std_reward:>8.3f}     {avg_degradation:>8.3f}     {len(rate_results):>6}")
            
            # Check monotonicity
            if prev_avg_reward is not None and avg_reward > prev_avg_reward + 0.01:
                print("  ⚠️  WARNING: Reward increased with higher permutation!")
            prev_avg_reward = avg_reward
        
        # Correlation analysis
        rates = []
        avg_rewards = []
        for rate in sorted(by_rate.keys()):
            rates.append(rate)
            avg_rewards.append(np.mean([r.permuted_reward for r in by_rate[rate]]))
        
        if len(rates) > 1:
            correlation = np.corrcoef(rates, avg_rewards)[0, 1]
            print(f"\nCorrelation (permutation vs reward): {correlation:.4f}")
            
            if correlation > -0.5:
                print("  ⚠️  WEAK NEGATIVE CORRELATION - Implementation may have issues!")
            else:
                print("  ✓ Strong negative correlation - Higher permutation leads to lower reward")
    
    print("\n" + "="*80)


def plot_results(results: Dict[str, List[PermutationResult]], save_path: str = None) -> None:
    """
    Plot permutation rate vs reward for visual analysis.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, task_type in enumerate(["P", "A"]):
            ax = axes[idx]
            task_results = results[task_type]
            
            # Group by permutation rate
            by_rate = {}
            for result in task_results:
                rate = result.permutation_rate
                if rate not in by_rate:
                    by_rate[rate] = []
                by_rate[rate].append(result.permuted_reward)
            
            # Prepare data for plotting
            rates = sorted(by_rate.keys())
            rewards_by_rate = [by_rate[rate] for rate in rates]
            
            # Box plot
            ax.boxplot(rewards_by_rate, positions=rates, widths=0.015)
            
            # Add mean line
            means = [np.mean(rewards) for rewards in rewards_by_rate]
            ax.plot(rates, means, 'r-', linewidth=2, label='Mean')
            
            # Formatting
            ax.set_xlabel('Permutation Rate')
            ax.set_ylabel('Reward')
            ax.set_title(f'{task_type}: Task - Permutation vs Reward')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add correlation text
            if len(rates) > 1:
                correlation = np.corrcoef(rates, means)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Reward Degradation with Increasing Permutation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("\nMatplotlib not available - skipping visualization")


def run_comprehensive_test():
    """
    Run comprehensive permutation test with detailed analysis.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PERMUTATION TEST FOR REWARD SCORER")
    print("="*80)
    
    # Test parameters
    n_samples = 50
    permutation_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    print(f"\nTest Configuration:")
    print(f"  Samples per task type: {n_samples}")
    print(f"  Permutation rates: {permutation_rates}")
    print(f"  Total test cases: {n_samples * 2 * len(permutation_rates)}")
    
    # Run structured permutation test (more realistic)
    print("\n" + "="*80)
    print("TEST 1: STRUCTURED PERMUTATIONS (Realistic)")
    print("="*80)
    
    structured_results = test_permutation_correlation(
        n_samples=n_samples,
        permutation_rates=permutation_rates,
        use_structured=True
    )
    
    analyze_results(structured_results)
    plot_results(structured_results, "structured_permutation_test.png")
    
    # Run random character permutation test
    print("\n" + "="*80)
    print("TEST 2: RANDOM CHARACTER PERMUTATIONS")
    print("="*80)
    
    random_results = test_permutation_correlation(
        n_samples=n_samples,
        permutation_rates=permutation_rates,
        use_structured=False
    )
    
    analyze_results(random_results)
    plot_results(random_results, "random_permutation_test.png")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, results in [("Structured", structured_results), ("Random", random_results)]:
        print(f"\n{test_name} Permutations:")
        
        for task_type in ["P", "A"]:
            # Calculate overall statistics
            all_results = results[task_type]
            
            # Group by rate for final summary
            by_rate = {}
            for r in all_results:
                if r.permutation_rate not in by_rate:
                    by_rate[r.permutation_rate] = []
                by_rate[r.permutation_rate].append(r.permuted_reward)
            
            # Check monotonicity
            rates = sorted(by_rate.keys())
            means = [np.mean(by_rate[r]) for r in rates]
            
            is_monotonic = all(means[i] >= means[i+1] for i in range(len(means)-1))
            
            print(f"  {task_type}: task - Monotonic decrease: {'✓ YES' if is_monotonic else '✗ NO'}")
            
            if len(rates) > 1:
                correlation = np.corrcoef(rates, means)[0, 1]
                if correlation < -0.7:
                    status = "✓ EXCELLENT"
                elif correlation < -0.5:
                    status = "✓ GOOD"
                elif correlation < -0.3:
                    status = "⚠️  WEAK"
                else:
                    status = "✗ FAILED"
                
                print(f"       Correlation: {correlation:.3f} ({status})")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Run the comprehensive test
    run_comprehensive_test()
    
    # Also run a quick example to show individual permutation effects
    print("\n" + "="*80)
    print("EXAMPLE: Individual Permutation Effects")
    print("="*80)
    
    scorer = RewardScorer()
    
    # P: task example
    p_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    p_ground_truth = "M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4"
    
    print("\nP: Task Example")
    print(f"Original: {p_ground_truth}")
    
    for rate in [0.0, 0.1, 0.2]:
        permuted = apply_structured_permutation(p_ground_truth, "P", rate)
        reward, _ = scorer.score_single(p_prompt, permuted, log_details=False)
        print(f"\n{rate:>3.0%} permuted: {permuted[:60]}...")
        print(f"     Reward: {reward:.3f}")
    
    # A: task example
    a_prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
    a_ground_truth = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
    
    print("\nA: Task Example")
    print(f"Original: {a_ground_truth}")
    
    for rate in [0.0, 0.1, 0.2]:
        permuted = apply_structured_permutation(a_ground_truth, "A", rate)
        reward, _ = scorer.score_single(a_prompt, permuted, log_details=False)
        print(f"\n{rate:>3.0%} permuted: {permuted[:60]}...")
        print(f"     Reward: {reward:.3f}")