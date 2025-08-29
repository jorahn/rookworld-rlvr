#!/usr/bin/env python3
"""
Compare graduated vs continuous reward systems for GRPO training.
Shows how continuous rewards provide better learning signals.
"""

import numpy as np
from reward_scorer import RewardScorer, compute_grpo_rewards

def demonstrate_fen_similarity_improvement():
    """Show how continuous rewards better handle FEN similarity."""
    print("\n" + "="*80)
    print("FEN SIMILARITY: Graduated vs Continuous Rewards")
    print("="*80)
    
    # Test cases with varying FEN accuracy
    test_cases = [
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",
         "Perfect match"),
        
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+false+false",
         "1 char diff (98% similar)"),
        
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1+0.001+false+false",
         "2 char diff (96% similar)"),
        
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+false+false",
         "Multiple diffs (90% similar)"),
        
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1+0.001+false+false",
         "Very different (60% similar)"),
    ]
    
    # Initialize scorers
    graduated_scorer = RewardScorer(
        reward_shaping="graduated",
        continuous_components={}  # No continuous components
    )
    
    continuous_scorer = RewardScorer(
        reward_shaping="graduated",  # Keep base shaping
        continuous_components={
            "fen_similarity": "exponential",  # But use continuous for FEN
        }
    )
    
    print("\n" + "-"*80)
    print(f"{'Case':<25} {'Graduated':<15} {'Continuous':<15} {'Improvement':<15}")
    print("-"*80)
    
    for prompt, completion, description in test_cases:
        grad_reward, grad_details = graduated_scorer.score_single(prompt, completion, log_details=False)
        cont_reward, cont_details = continuous_scorer.score_single(prompt, completion, log_details=False)
        
        improvement = cont_reward - grad_reward
        print(f"{description:<25} {grad_reward:<15.3f} {cont_reward:<15.3f} {improvement:<+15.3f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("- Graduated: All high-similarity FENs (>95%) get same reward (1.0)")
    print("- Continuous: Differentiates between 96%, 98%, and 100% accuracy")
    print("- Learning signal: Model gets rewarded for incremental improvements")
    print("="*80)


def demonstrate_training_impact():
    """Show impact on a batch of training samples."""
    print("\n" + "="*80)
    print("TRAINING BATCH: Impact of Continuous Rewards")
    print("="*80)
    
    # Simulate a batch with varying quality
    np.random.seed(42)
    n_samples = 32
    
    # Generate prompts and completions with varying quality
    prompts = []
    completions = []
    
    for i in range(n_samples):
        # Mix of P: and A: tasks
        if i % 2 == 0:
            # A: task with varying FEN accuracy
            prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+"
            
            # Generate completion with controlled error
            if i < 8:  # Poor (60-70% similarity)
                completion = "rnbqk2r/pppp1ppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+false+false"
            elif i < 16:  # Good (85-95% similarity)
                completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+false+false"
            elif i < 24:  # Very good (95-98% similarity)
                completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
            else:  # Perfect (100% similarity)
                completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
        else:
            # P: task
            prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            completion = "M: e2e4 d2d4 g1f3 E: 0.3 0.3 0.3 B: e2e4"
        
        prompts.append(prompt)
        completions.append(completion)
    
    # Compute rewards with both systems
    grad_advantages, grad_details = compute_grpo_rewards(
        prompts, completions, 
        group_size=8,
        reward_shaping="graduated",
        continuous_components={},
        verbose=False
    )
    
    cont_advantages, cont_details = compute_grpo_rewards(
        prompts, completions,
        group_size=8,
        reward_shaping="graduated",
        continuous_components={"fen_similarity": "exponential"},
        verbose=False
    )
    
    # Analyze distributions
    grad_rewards = np.array([d.shaped_reward for d in grad_details])
    cont_rewards = np.array([d.shaped_reward for d in cont_details])
    
    print("\nReward Distribution Analysis:")
    print("-"*60)
    print(f"{'Metric':<25} {'Graduated':<15} {'Continuous':<15}")
    print("-"*60)
    print(f"{'Mean reward:':<25} {grad_rewards.mean():<15.3f} {cont_rewards.mean():<15.3f}")
    print(f"{'Std deviation:':<25} {grad_rewards.std():<15.3f} {cont_rewards.std():<15.3f}")
    print(f"{'Min reward:':<25} {grad_rewards.min():<15.3f} {cont_rewards.min():<15.3f}")
    print(f"{'Max reward:':<25} {grad_rewards.max():<15.3f} {cont_rewards.max():<15.3f}")
    print(f"{'Unique values:':<25} {len(np.unique(grad_rewards)):<15} {len(np.unique(cont_rewards)):<15}")
    
    print("\nAdvantage Distribution Analysis:")
    print("-"*60)
    print(f"{'Metric':<25} {'Graduated':<15} {'Continuous':<15}")
    print("-"*60)
    print(f"{'Mean advantage:':<25} {grad_advantages.mean():<15.3f} {cont_advantages.mean():<15.3f}")
    print(f"{'Std deviation:':<25} {grad_advantages.std():<15.3f} {cont_advantages.std():<15.3f}")
    print(f"{'Min advantage:':<25} {grad_advantages.min():<15.3f} {cont_advantages.min():<15.3f}")
    print(f"{'Max advantage:':<25} {grad_advantages.max():<15.3f} {cont_advantages.max():<15.3f}")
    
    # Show correlation with actual quality
    print("\n" + "="*80)
    print("LEARNING SIGNAL QUALITY:")
    print("-"*80)
    
    # Group by quality level
    poor_grad = grad_rewards[:8].mean()
    good_grad = grad_rewards[8:16].mean()
    vgood_grad = grad_rewards[16:24].mean()
    perfect_grad = grad_rewards[24:32].mean()
    
    poor_cont = cont_rewards[:8].mean()
    good_cont = cont_rewards[8:16].mean()
    vgood_cont = cont_rewards[16:24].mean()
    perfect_cont = cont_rewards[24:32].mean()
    
    print(f"{'Quality Level':<20} {'Graduated':<15} {'Continuous':<15} {'Delta':<15}")
    print("-"*65)
    print(f"{'Poor (60-70%)':<20} {poor_grad:<15.3f} {poor_cont:<15.3f} {poor_cont-poor_grad:<+15.3f}")
    print(f"{'Good (85-95%)':<20} {good_grad:<15.3f} {good_cont:<15.3f} {good_cont-good_grad:<+15.3f}")
    print(f"{'Very Good (95-98%)':<20} {vgood_grad:<15.3f} {vgood_cont:<15.3f} {vgood_cont-vgood_grad:<+15.3f}")
    print(f"{'Perfect (100%)':<20} {perfect_grad:<15.3f} {perfect_cont:<15.3f} {perfect_cont-perfect_grad:<+15.3f}")
    
    print("\n" + "="*80)
    print("CONCLUSIONS:")
    print("1. Continuous rewards provide finer-grained feedback")
    print("2. Better differentiation between quality levels")
    print("3. Rewards incremental improvements (95% -> 98% -> 100%)")
    print("4. More informative gradients for optimization")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPARING GRADUATED VS CONTINUOUS REWARD SYSTEMS")
    print("="*80)
    
    demonstrate_fen_similarity_improvement()
    demonstrate_training_impact()
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("Use continuous rewards for regression-like tasks (FEN similarity, evaluations)")
    print("Keep discrete rewards for classification tasks (best move selection)")
    print("="*80)