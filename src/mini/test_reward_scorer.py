"""
Tests for reward scoring and advantage computation
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from reward_scorer import RewardScorer, RewardDetails, compute_grpo_rewards


class TestRewardScorer:
    """Test reward scoring functionality"""
    
    def test_task_type_identification(self):
        """Test correct identification of task types"""
        scorer = RewardScorer()
        
        assert scorer._identify_task_type("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == "P"
        assert scorer._identify_task_type("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+") == "A"
        assert scorer._identify_task_type("  P: test") == "P"  # With whitespace
        assert scorer._identify_task_type("  A: test") == "A"  # With whitespace
        assert scorer._identify_task_type("invalid prompt") == "unknown"
    
    def test_p_task_perfect_score(self):
        """Test P: task with perfect completion"""
        scorer = RewardScorer(reward_shaping="linear")
        
        prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        completion = "M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4"
        
        reward, details = scorer.score_single(prompt, completion, log_details=False)
        
        assert details.task_type == "P"
        assert details.format_valid == True
        assert details.format_score == 1.0
        assert 'format' in details.field_scores
        # Without Stockfish, content scores will be minimal but format should be perfect
        assert details.field_scores['format'] == 1.0
    
    def test_p_task_invalid_format(self):
        """Test P: task with invalid format"""
        scorer = RewardScorer(min_reward=-0.3)
        
        prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        completion = "random garbage without proper format"
        
        reward, details = scorer.score_single(prompt, completion, log_details=False)
        
        assert details.task_type == "P"
        assert details.format_valid == False
        assert details.format_score == 0.0
        assert reward <= scorer.min_reward + scorer.format_bonus
    
    def test_a_task_perfect_score(self):
        """Test A: task with perfect completion"""
        scorer = RewardScorer(reward_shaping="linear")
        
        prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
        completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
        
        reward, details = scorer.score_single(prompt, completion, log_details=False)
        
        assert details.task_type == "A"
        assert details.format_valid == True
        assert details.format_score == 1.0
        assert 'format' in details.field_scores
        assert 'fen_match' in details.field_scores
        assert 'game_state' in details.field_scores
        assert 'reward_value' in details.field_scores
    
    def test_a_task_partial_score(self):
        """Test A: task with partial correctness"""
        scorer = RewardScorer()
        
        prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
        # Slightly wrong FEN
        completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1+0.001+false+false"
        
        reward, details = scorer.score_single(prompt, completion, log_details=False)
        
        assert details.task_type == "A"
        assert details.format_valid == True
        # FEN should have partial match (not perfect)
        assert 0 < details.field_scores.get('fen_match', 0) < 1.0
    
    def test_reward_shaping_graduated(self):
        """Test graduated reward shaping"""
        scorer = RewardScorer(reward_shaping="graduated")
        
        # Test different raw reward levels
        test_cases = [
            (0.1, -0.3),   # Very bad -> min_reward
            (0.25, 0.2),   # Bad -> 0.2
            (0.45, 0.4),   # OK -> 0.4
            (0.65, 0.6),   # Good -> 0.6
            (0.85, 0.8),   # Very good -> 0.8
            (0.98, 1.0),   # Perfect -> 1.0
        ]
        
        for raw, expected_min in test_cases:
            shaped = scorer._shape_reward(raw, format_valid=True)
            assert shaped >= expected_min - 0.01  # Allow small tolerance
    
    def test_reward_shaping_linear(self):
        """Test linear reward shaping"""
        scorer = RewardScorer(reward_shaping="linear", min_reward=-1.0, max_reward=1.0, format_bonus=0)
        
        # Linear scaling from -1 to 1 (without format bonus)
        assert scorer._shape_reward(0.0, True) == -1.0
        assert scorer._shape_reward(0.5, True) == 0.0
        assert scorer._shape_reward(1.0, True) == 1.0
    
    def test_reward_shaping_binary(self):
        """Test binary reward shaping"""
        scorer = RewardScorer(reward_shaping="binary", min_reward=-1.0, max_reward=1.0, format_bonus=0)
        
        assert scorer._shape_reward(0.4, True) == -1.0  # Below threshold
        assert scorer._shape_reward(0.6, True) == 1.0   # Above threshold
    
    def test_format_bonus(self):
        """Test format bonus application"""
        scorer = RewardScorer(min_reward=-0.5, format_bonus=0.2)
        
        # Bad content but valid format should get bonus
        shaped_with_format = scorer._shape_reward(0.1, format_valid=True)
        shaped_without_format = scorer._shape_reward(0.1, format_valid=False)
        
        assert shaped_with_format > shaped_without_format


class TestBatchScoring:
    """Test batch scoring and advantage computation"""
    
    def test_batch_scoring_consistency(self):
        """Test that batch scoring matches individual scoring"""
        scorer = RewardScorer()
        
        prompts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+",
        ]
        completions = [
            "M: e2e4 d2d4  E: 0.3 0.4  B: e2e4",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",
        ]
        
        # Score individually
        individual_rewards = []
        for prompt, completion in zip(prompts, completions):
            reward, _ = scorer.score_single(prompt, completion, log_details=False)
            individual_rewards.append(reward)
        
        # Score as batch
        batch_rewards, _ = scorer.score_batch(
            prompts, 
            completions, 
            compute_advantages=False
        )
        
        # Should match
        np.testing.assert_array_almost_equal(individual_rewards, batch_rewards)
    
    def test_group_advantages(self):
        """Test group advantage computation"""
        scorer = RewardScorer()
        
        # Create rewards with clear groups
        rewards = np.array([1.0, 0.8, 0.6, 0.4,  # Group 1: mean=0.7
                           0.2, 0.3, 0.1, 0.0])  # Group 2: mean=0.15
        
        advantages = scorer._compute_group_advantages(rewards, group_size=4)
        
        # Group 1 advantages should sum to ~0
        assert abs(advantages[:4].sum()) < 0.01
        
        # Group 2 advantages should sum to ~0  
        assert abs(advantages[4:].sum()) < 0.01
        
        # Higher rewards should have positive advantages
        assert advantages[0] > 0  # 1.0 > 0.7
        assert advantages[4] > 0  # 0.2 > 0.15
        
        # Lower rewards should have negative advantages
        assert advantages[3] < 0  # 0.4 < 0.7
        assert advantages[7] < 0  # 0.0 < 0.15
    
    def test_full_batch_advantages(self):
        """Test advantages with full batch as single group"""
        scorer = RewardScorer()
        
        prompts = ["P: test"] * 8
        completions = ["M: e2e4  E: 0.3  B: e2e4"] * 4 + ["garbage"] * 4
        
        advantages, details = scorer.score_batch(
            prompts,
            completions,
            compute_advantages=True,
            group_size=None  # Full batch
        )
        
        assert len(advantages) == 8
        assert len(details) == 8
        
        # Advantages should sum to ~0
        assert abs(advantages.sum()) < 0.01
        
        # Good completions should have positive advantages
        assert all(advantages[:4] > 0)
        
        # Bad completions should have negative advantages
        assert all(advantages[4:] < 0)


class TestConvenienceFunction:
    """Test the compute_grpo_rewards convenience function"""
    
    def test_compute_grpo_rewards(self):
        """Test the main convenience function"""
        prompts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+",
        ] * 2
        
        completions = [
            "M: e2e4 d2d4  E: 0.3 0.4  B: e2e4",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",
        ] * 2
        
        advantages, details = compute_grpo_rewards(
            prompts,
            completions,
            group_size=2,
            reward_shaping="graduated",
            verbose=False
        )
        
        assert len(advantages) == 4
        assert len(details) == 4
        
        # Check task types identified correctly
        assert details[0].task_type == "P"
        assert details[1].task_type == "A"
        
        # Advantages should be computed
        assert isinstance(advantages, np.ndarray)
        assert advantages.shape == (4,)
    
    def test_verbose_mode(self):
        """Test verbose logging mode"""
        prompts = ["P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        completions = ["M: e2e4  E: 0.3  B: e2e4"]
        
        # Should not raise any errors with verbose=True
        advantages, details = compute_grpo_rewards(
            prompts,
            completions,
            verbose=True
        )
        
        assert len(advantages) == 1
        assert len(details) == 1


class TestRewardDetails:
    """Test RewardDetails dataclass"""
    
    def test_reward_details_structure(self):
        """Test RewardDetails contains expected fields"""
        scorer = RewardScorer()
        
        prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        completion = "M: e2e4 d2d4  E: 0.3 0.4  B: e2e4"
        
        _, details = scorer.score_single(prompt, completion, log_details=False)
        
        assert hasattr(details, 'task_type')
        assert hasattr(details, 'format_valid')
        assert hasattr(details, 'format_score')
        assert hasattr(details, 'field_scores')
        assert hasattr(details, 'weighted_scores')
        assert hasattr(details, 'total_raw_reward')
        assert hasattr(details, 'shaped_reward')
        assert hasattr(details, 'details')
        
        assert isinstance(details.field_scores, dict)
        assert isinstance(details.weighted_scores, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])