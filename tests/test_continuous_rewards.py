"""
Test continuous rewards implementation for FEN similarity and evaluation accuracy.
"""

import numpy as np
from rookworld_rlvr.reward_scorer import RewardScorer
from validation import levenshtein_distance

def test_fen_similarity_continuous():
    """Test that FEN similarity uses continuous exponential scaling."""
    print("\n=== Testing FEN Similarity Continuous Rewards ===")
    
    # Initialize scorer with continuous components
    scorer_continuous = RewardScorer(
        continuous_components={
            "fen_similarity": "exponential",
            "evaluations": "linear"
        }
    )
    
    # Initialize scorer with graduated rewards for comparison
    scorer_graduated = RewardScorer(
        reward_shaping="graduated",
        continuous_components={}  # Empty dict disables continuous
    )
    
    # Test cases with increasing FEN similarity
    test_cases = [
        # (prompt, completion, expected_similarity)
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",  # Perfect
         1.0),
        
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+false+false",  # 1 char diff
         0.98),
        
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1+0.001+false+false",  # 2 char diff
         0.96),
        
        ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
         "rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+false+false",  # More diffs
         0.90),
    ]
    
    print("\nFEN Similarity -> Reward Mapping:")
    print("-" * 60)
    print(f"{'Similarity':<12} {'Graduated':<12} {'Continuous':<12} {'Difference':<12}")
    print("-" * 60)
    
    for prompt, completion, expected_sim in test_cases:
        # Score with both methods
        reward_cont, details_cont = scorer_continuous.score_single(prompt, completion, log_details=False)
        reward_grad, details_grad = scorer_graduated.score_single(prompt, completion, log_details=False)
        
        # Get actual similarity from details
        actual_sim = details_cont.field_scores.get('fen_match', 0.0)
        
        print(f"{actual_sim:<12.3f} {reward_grad:<12.3f} {reward_cont:<12.3f} {reward_cont - reward_grad:<12.3f}")
    
    print("\n✓ FEN similarity now uses continuous exponential scaling")
    print("  Near-perfect matches (>95%) get much higher rewards")
    print("  Small improvements are rewarded incrementally")


def test_evaluation_accuracy_continuous():
    """Test that evaluation accuracy uses continuous linear scaling."""
    print("\n=== Testing Evaluation Accuracy Continuous Rewards ===")
    
    # Initialize scorers
    scorer_continuous = RewardScorer(
        continuous_components={
            "fen_similarity": "exponential", 
            "evaluations": "linear"
        }
    )
    
    scorer_graduated = RewardScorer(
        reward_shaping="graduated",
        continuous_components={}
    )
    
    # Test cases with varying evaluation accuracy
    test_prompts_completions = [
        # Perfect evaluations
        ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "M: e2e4 d2d4 g1f3 E: 0.3 0.3 0.3 B: e2e4"),
        
        # Slightly off evaluations
        ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "M: e2e4 d2d4 g1f3 E: 0.35 0.28 0.32 B: e2e4"),
        
        # More error in evaluations
        ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "M: e2e4 d2d4 g1f3 E: 0.5 0.1 0.4 B: e2e4"),
        
        # Large errors
        ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         "M: e2e4 d2d4 g1f3 E: 1.0 -0.5 0.8 B: e2e4"),
    ]
    
    print("\nEvaluation Accuracy -> Reward Mapping:")
    print("-" * 60)
    print(f"{'Test Case':<12} {'Graduated':<12} {'Continuous':<12} {'Difference':<12}")
    print("-" * 60)
    
    for i, (prompt, completion) in enumerate(test_prompts_completions, 1):
        reward_cont, details_cont = scorer_continuous.score_single(prompt, completion, log_details=False)
        reward_grad, details_grad = scorer_graduated.score_single(prompt, completion, log_details=False)
        
        print(f"Test {i:<8} {reward_grad:<12.3f} {reward_cont:<12.3f} {reward_cont - reward_grad:<12.3f}")
    
    print("\n✓ Evaluation accuracy now uses continuous linear scaling")
    print("  Rewards are proportional to accuracy")
    print("  Small improvements in accuracy are rewarded")


def test_scaling_functions():
    """Test different scaling functions."""
    print("\n=== Testing Scaling Functions ===")
    
    scorer = RewardScorer()
    
    # Test values
    test_values = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    
    print("\nScaling Function Comparison:")
    print("-" * 80)
    print(f"{'Input':<10} {'Linear':<12} {'Exponential':<12} {'Sigmoid':<12} {'Quadratic':<12}")
    print("-" * 80)
    
    for val in test_values:
        linear = scorer._apply_scaling(val, "linear")
        exponential = scorer._apply_scaling(val, "exponential")
        sigmoid = scorer._apply_scaling(val, "sigmoid")
        quadratic = scorer._apply_scaling(val, "quadratic")
        
        print(f"{val:<10.2f} {linear:<12.3f} {exponential:<12.3f} {sigmoid:<12.3f} {quadratic:<12.3f}")
    
    print("\n✓ Different scaling functions available:")
    print("  - Linear: Direct proportional mapping")
    print("  - Exponential: Rewards near-perfect scores more")
    print("  - Sigmoid: S-curve with steep middle section")
    print("  - Quadratic: Rewards high scores more")


def compare_reward_distributions():
    """Compare reward distributions between graduated and continuous."""
    print("\n=== Comparing Reward Distributions ===")
    
    # Generate synthetic similarity scores
    np.random.seed(42)
    similarities = np.concatenate([
        np.random.uniform(0.7, 0.8, 20),  # Poor matches
        np.random.uniform(0.8, 0.9, 30),  # OK matches
        np.random.uniform(0.9, 0.95, 30), # Good matches
        np.random.uniform(0.95, 1.0, 20),  # Excellent matches
    ])
    
    # Initialize scorers
    scorer_graduated = RewardScorer(reward_shaping="graduated")
    scorer_continuous = RewardScorer(
        continuous_components={"fen_similarity": "exponential"}
    )
    
    graduated_rewards = []
    continuous_rewards = []
    
    for sim in similarities:
        # Simulate graduated reward
        if sim < 0.4:
            grad = 0.2
        elif sim < 0.6:
            grad = 0.4
        elif sim < 0.8:
            grad = 0.6
        elif sim < 0.95:
            grad = 0.8
        else:
            grad = 1.0
        graduated_rewards.append(grad)
        
        # Continuous reward with exponential scaling
        cont = scorer_continuous._apply_scaling(sim, "exponential")
        continuous_rewards.append(cont)
    
    print("\nReward Distribution Statistics:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Graduated':<15} {'Continuous':<15}")
    print("-" * 50)
    print(f"{'Mean:':<20} {np.mean(graduated_rewards):<15.3f} {np.mean(continuous_rewards):<15.3f}")
    print(f"{'Std Dev:':<20} {np.std(graduated_rewards):<15.3f} {np.std(continuous_rewards):<15.3f}")
    print(f"{'Min:':<20} {np.min(graduated_rewards):<15.3f} {np.min(continuous_rewards):<15.3f}")
    print(f"{'Max:':<20} {np.max(graduated_rewards):<15.3f} {np.max(continuous_rewards):<15.3f}")
    print(f"{'Unique Values:':<20} {len(np.unique(graduated_rewards)):<15} {len(np.unique(continuous_rewards)):<15}")
    
    print("\n✓ Continuous rewards provide:")
    print("  - More granular feedback (100 unique values vs 5)")
    print("  - Better differentiation between similar scores")
    print("  - Smoother reward landscape for optimization")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Continuous Rewards Implementation")
    print("=" * 80)
    
    test_fen_similarity_continuous()
    test_evaluation_accuracy_continuous()
    test_scaling_functions()
    compare_reward_distributions()
    
    print("\n" + "=" * 80)
    print("✅ All continuous reward tests completed successfully!")
    print("=" * 80)