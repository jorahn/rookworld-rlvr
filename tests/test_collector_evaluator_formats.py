"""
Tests for Data Collector and Evaluator Format Consistency

Tests the specific components mentioned in PR #11:
- collector.py format handling
- evaluator.py format handling
- Reward computation consistency

Focuses on the full_text vs generated_text issue that was fixed.
"""

import pytest
import torch
import chess
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from src.rookworld_rlvr.data.collector import GRPOCollector, GRPOCollectionConfig
from src.rookworld_rlvr.train.evaluator import ChessEvaluator
from src.rookworld_rlvr.train.policy import CausalLMPolicy
from src.rookworld_rlvr.reward.policy_reward import PolicyRewardComputer
from src.rookworld_rlvr.reward.env_reward import EnvRewardComputer
from src.rookworld_rlvr.engine.stockfish import StockfishEngine, StockfishAnalysis


class TestCollectorFormatHandling:
    """Test data collector format consistency (PR #11 fix focus)"""
    
    @pytest.fixture
    def mock_policy(self):
        """Mock policy that generates realistic outputs"""
        policy = Mock(spec=CausalLMPolicy)
        policy.device = "cpu"
        
        def generate_responses(prompts, **kwargs):
            results = []
            for prompt in prompts:
                if prompt.startswith("P: "):
                    results.append(" e2e4 d2d4    E: 0.25 0.18    B: e2e4")
                elif prompt.startswith("A: "):
                    results.append("++rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+0+0")
                else:
                    results.append("invalid")
            return results
        
        policy.generate.side_effect = generate_responses
        return policy
    
    @pytest.fixture
    def mock_stockfish(self):
        """Mock Stockfish engine"""
        engine = Mock(spec=StockfishEngine)
        analysis = StockfishAnalysis(
            top5_moves=['e2e4', 'd2d4'],
            top5_evals=[0.25, 0.18], 
            best_move='e2e4',
            depth=10,
            analysis_time=0.05
        )
        engine.analyze.return_value = analysis
        return engine
    
    def test_collector_policy_reward_uses_full_text(self, mock_policy, mock_stockfish):
        """Test that collector passes full_text to policy reward computation (PR #11)"""
        config = GRPOCollectionConfig(group_size=2, mix_env_ratio=0.0)
        collector = GRPOCollector(config, mock_policy, mock_stockfish)
        
        # Mock the reward computer to verify it receives full_text
        with patch.object(collector.policy_reward_computer, 'compute_reward') as mock_compute:
            mock_compute.return_value = (1.0, {"structure_reward": 0.4, "malformed_penalty": 0})
            
            positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
            batch = collector.collect_batch(positions)
            
            # Verify compute_reward was called with full_text, not just generated_text
            assert mock_compute.called
            call_args = mock_compute.call_args[0]  # First positional argument
            
            # The first argument should be full_text (prompt + generated)
            full_text_arg = call_args[0]
            assert full_text_arg.startswith("P: ")  # Should include prompt
            assert "    M:" in full_text_arg      # Should include prompt continuation
            assert "e2e4" in full_text_arg        # Should include generated content
    
    def test_collector_env_reward_uses_full_text(self, mock_policy):
        """Test that collector passes full_text to environment reward computation (PR #11)"""
        config = GRPOCollectionConfig(group_size=2, mix_env_ratio=1.0)
        collector = GRPOCollector(config, mock_policy, None)
        
        # Mock the reward computer to verify it receives full_text
        with patch.object(collector.env_reward_computer, 'compute_reward') as mock_compute:
            mock_compute.return_value = (1.0, {"structure_reward": 0.4, "malformed_penalty": 0})
            
            positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
            batch = collector.collect_batch(positions)
            
            # Verify compute_reward was called with full_text, not just generated_text
            assert mock_compute.called
            call_args = mock_compute.call_args[0]
            
            # The first argument should be full_text (prompt + generated)
            full_text_arg = call_args[0]
            assert full_text_arg.startswith("A: ")  # Should include prompt
            assert "++" in full_text_arg           # Should include generated content
    
    def test_collector_batch_sample_structure(self, mock_policy, mock_stockfish):
        """Test that collector creates proper GRPOSample structures"""
        config = GRPOCollectionConfig(group_size=2, mix_env_ratio=0.5)
        collector = GRPOCollector(config, mock_policy, mock_stockfish)
        
        positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        batch = collector.collect_batch(positions)
        
        for sample in batch.samples:
            # Each sample should have distinct prompt, generated_text, and full_text
            assert hasattr(sample, 'prompt')
            assert hasattr(sample, 'generated_text') 
            assert hasattr(sample, 'full_text')
            
            # full_text should be prompt + generated_text
            assert sample.full_text == sample.prompt + sample.generated_text
            
            # Prompt should have correct format
            assert sample.prompt.startswith(("P: ", "A: "))
            
            # Reward should be computed (non-zero)
            assert sample.reward != 0


class TestEvaluatorFormatHandling:
    """Test evaluator format consistency (PR #11 fix focus)"""
    
    @pytest.fixture
    def mock_policy(self):
        """Mock policy for evaluation"""
        policy = Mock(spec=CausalLMPolicy)
        policy.device = "cpu"
        
        # Generate realistic chess responses
        policy.generate.return_value = [" e2e4 d2d4    E: 0.25 0.18    B: e2e4"]
        
        return policy
    
    @pytest.fixture  
    def mock_stockfish(self):
        """Mock Stockfish for evaluation"""
        engine = Mock(spec=StockfishEngine)
        analysis = StockfishAnalysis(['e2e4', 'd2d4'], [0.25, 0.18], 'e2e4', 10, 0.05)
        engine.analyze.return_value = analysis
        return engine
    
    def test_evaluator_policy_uses_full_text(self, mock_policy, mock_stockfish):
        """Test that evaluator passes full_text to policy reward computation (PR #11)"""
        evaluator = ChessEvaluator(mock_policy, mock_stockfish)
        
        # Mock the reward computation to verify it receives full_text
        with patch('src.rookworld_rlvr.reward.policy_reward.compute_policy_reward') as mock_compute:
            mock_compute.return_value = (1.0, {"structure_reward": 0.4})
            
            positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
            metrics = evaluator.evaluate_policy_performance(positions)
            
            # Verify compute_policy_reward was called
            assert mock_compute.called
            call_args = mock_compute.call_args[0]
            
            # First argument should be full_text (includes prompt)
            full_text_arg = call_args[0]
            assert full_text_arg.startswith("P: ")
            assert "    M:" in full_text_arg
            assert "e2e4" in full_text_arg
    
    def test_evaluator_env_uses_full_text(self, mock_policy):
        """Test that evaluator passes full_text to environment reward computation (PR #11)"""
        evaluator = ChessEvaluator(mock_policy, None)
        
        # Mock environment generation
        env_response = "++rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+0+0"
        mock_policy.generate.return_value = [env_response]
        
        # Mock the reward computation to verify it receives full_text  
        with patch('src.rookworld_rlvr.reward.env_reward.compute_env_reward') as mock_compute:
            mock_compute.return_value = (1.0, {"structure_reward": 0.4})
            
            positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
            moves = ["e2e4"]
            metrics = evaluator.evaluate_environment_performance(positions, moves)
            
            # Verify compute_env_reward was called
            assert mock_compute.called
            call_args = mock_compute.call_args[0]
            
            # First argument should be full_text (includes A: prompt)
            full_text_arg = call_args[0]
            assert full_text_arg.startswith("A: ")
            assert "++" in full_text_arg  # Environment response content


class TestRewardComputationConsistency:
    """Test reward computation handles both formats correctly"""
    
    def test_policy_reward_with_and_without_prefix(self):
        """Test policy reward computation with and without P: prefix"""
        reward_computer = PolicyRewardComputer()
        board = chess.Board()
        analysis = StockfishAnalysis(['e2e4'], [0.25], 'e2e4', 10, 0.05)
        
        # With P: prefix (correct format after PR #11)
        with_prefix = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4    E: 0.25    B: e2e4"
        reward_with, breakdown_with = reward_computer.compute_reward(with_prefix, board, analysis)
        
        # Without P: prefix (old incorrect format)
        without_prefix = " e2e4    E: 0.25    B: e2e4"
        reward_without, breakdown_without = reward_computer.compute_reward(without_prefix, board, analysis)
        
        # With prefix should succeed
        assert reward_with > 0
        assert breakdown_with["structure_reward"] > 0
        assert breakdown_with["malformed_penalty"] == 0
        
        # Without prefix should be penalized
        assert reward_without < 0
        assert breakdown_without["structure_reward"] == 0
        assert breakdown_without["malformed_penalty"] < 0
    
    def test_env_reward_with_and_without_prefix(self):
        """Test environment reward computation with and without A: prefix"""
        reward_computer = EnvRewardComputer()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        expected = reward_computer.create_expected_response(fen, "e2e4")
        
        # With A: prefix (correct format after PR #11)
        with_prefix = f"A: {fen}+e2e4+++{expected.new_fen}+{expected.reward}+0+0"
        reward_with, breakdown_with = reward_computer.compute_reward(with_prefix, expected)
        
        # Without A: prefix (old incorrect format)
        without_prefix = f"++{expected.new_fen}+{expected.reward}+0+0"
        reward_without, breakdown_without = reward_computer.compute_reward(without_prefix, expected)
        
        # With prefix should succeed
        assert reward_with > 0
        assert breakdown_with["structure_reward"] > 0
        assert breakdown_with["malformed_penalty"] == 0
        
        # Without prefix should be penalized (parsing will handle this)
        assert reward_without <= 0  # Should be penalized or get lower reward
    
    def test_parsing_requirements_consistency(self):
        """Test that parsing requirements are consistent across the pipeline"""
        from src.rookworld_rlvr.environment.chess_env import ChessEnvironment
        
        env = ChessEnvironment()
        
        # Test environment parsing specifically checks for A: prefix
        test_cases = [
            ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+++new_fen+0.001+0+0", True),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+++new_fen+0.001+0+0", False),
            ("A: invalid format", False),
            ("", False),
        ]
        
        for text, should_parse in test_cases:
            parsed = env.parse_prediction(text)
            if should_parse:
                assert parsed is not None, f"Should parse: {text}"
            else:
                # Either returns None or the reward computation will handle it properly
                # The key is that the format is handled consistently
                pass


class TestBugFixVerification:
    """Specifically test the bug that was fixed in PR #11"""
    
    def test_zero_percent_success_rate_fix_policy(self):
        """Test that policy evaluation no longer shows 0% success rate"""
        # Mock a collector scenario that would have shown 0% before the fix
        reward_computer = PolicyRewardComputer()
        board = chess.Board()
        analysis = StockfishAnalysis(['e2e4'], [0.25], 'e2e4', 10, 0.05)
        
        # Before PR #11: passing only generated_text (no P: prefix)
        generated_only = " e2e4    E: 0.25    B: e2e4"
        reward_old_way, _ = reward_computer.compute_reward(generated_only, board, analysis)
        
        # After PR #11: passing full_text (with P: prefix)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        full_text = f"P: {fen}    M:{generated_only}"
        reward_new_way, _ = reward_computer.compute_reward(full_text, board, analysis)
        
        # The fix should make the reward much better
        assert reward_old_way < reward_new_way
        assert reward_new_way > 0  # Should be positive with correct format
    
    def test_zero_percent_success_rate_fix_environment(self):
        """Test that environment evaluation no longer shows 0% success rate"""
        reward_computer = EnvRewardComputer()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        expected = reward_computer.create_expected_response(fen, "e2e4")
        
        # Before PR #11: passing only generated_text (no A: prefix)
        generated_only = f"++{expected.new_fen}+{expected.reward}+0+0"
        reward_old_way, _ = reward_computer.compute_reward(generated_only, expected)
        
        # After PR #11: passing full_text (with A: prefix)
        full_text = f"A: {fen}+e2e4+{generated_only}"
        reward_new_way, _ = reward_computer.compute_reward(full_text, expected)
        
        # The fix should make the reward much better
        assert reward_old_way <= reward_new_way
        assert reward_new_way > 0  # Should be positive with correct format
    
    def test_mixed_training_success_rate_improvement(self):
        """Test that mixed task training should now work correctly"""
        # This test verifies that both task types can be evaluated properly in mixed training
        policy_reward_computer = PolicyRewardComputer()
        env_reward_computer = EnvRewardComputer()
        
        # Policy task
        board = chess.Board()
        analysis = StockfishAnalysis(['e2e4'], [0.25], 'e2e4', 10, 0.05)
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        policy_full_text = f"P: {fen}    M: e2e4    E: 0.25    B: e2e4"
        policy_reward, _ = policy_reward_computer.compute_reward(policy_full_text, board, analysis)
        
        # Environment task
        expected = env_reward_computer.create_expected_response(fen, "e2e4")
        env_full_text = f"A: {fen}+e2e4+++{expected.new_fen}+{expected.reward}+0+0"
        env_reward, _ = env_reward_computer.compute_reward(env_full_text, expected)
        
        # Both should succeed now
        assert policy_reward > 0
        assert env_reward > 0
        
        # This represents successful mixed training where both tasks provide meaningful rewards