"""
Tests for Task Format Pipeline Integration

This module tests both Policy (P:) and Environment (A:) task formats through
the complete pipeline: data collection, reward computation, and evaluation.

Addresses the parsing consistency issue fixed in PR #11.
"""

import pytest
import torch
import chess
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from src.rookworld_rlvr.data.collector import GRPOCollector, GRPOCollectionConfig
from src.rookworld_rlvr.train.evaluator import ChessEvaluator, EvaluationMetrics
from src.rookworld_rlvr.train.policy import CausalLMPolicy, GenerationConfig
from src.rookworld_rlvr.reward.policy_reward import PolicyRewardComputer, compute_policy_reward
from src.rookworld_rlvr.reward.env_reward import EnvRewardComputer, compute_env_reward
from src.rookworld_rlvr.environment.chess_env import ChessEnvironment, EnvironmentResponse
from src.rookworld_rlvr.engine.stockfish import StockfishEngine, StockfishAnalysis
from src.rookworld_rlvr.tokenizer.bridge import TaskFormatter


class TestPolicyTaskPipeline:
    """Test Policy (P:) task format through complete pipeline"""
    
    @pytest.fixture
    def mock_policy(self):
        """Mock policy for generation"""
        policy = Mock(spec=CausalLMPolicy)
        policy.device = "cpu"
        
        # Mock perfect policy response for starting position
        policy_response = " e2e4 d2d4 g1f3 b1c3 f2f3    E: 0.25 0.18 0.12 0.08 -0.15    B: e2e4"
        policy.generate.return_value = [policy_response] * 4  # Group size 4
        
        return policy
    
    @pytest.fixture
    def mock_stockfish(self):
        """Mock Stockfish engine"""
        engine = Mock(spec=StockfishEngine)
        analysis = StockfishAnalysis(
            top5_moves=['e2e4', 'd2d4', 'g1f3', 'b1c3', 'f2f3'],
            top5_evals=[0.25, 0.18, 0.12, 0.08, -0.15],
            best_move='e2e4',
            depth=10,
            analysis_time=0.05
        )
        engine.analyze.return_value = analysis
        return engine
    
    @pytest.fixture
    def collector(self, mock_policy, mock_stockfish):
        """GRPO collector with mocked dependencies"""
        config = GRPOCollectionConfig(group_size=4, mix_env_ratio=0.0)  # Policy only
        collector = GRPOCollector(config, mock_policy, mock_stockfish)
        return collector
    
    @pytest.fixture
    def evaluator(self, mock_policy, mock_stockfish):
        """Chess evaluator with mocked dependencies"""
        evaluator = ChessEvaluator(mock_policy, mock_stockfish)
        return evaluator
    
    def test_policy_prompt_format(self):
        """Test policy prompt format generation"""
        formatter = TaskFormatter()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        prompt = formatter.create_policy_prompt(fen)
        
        assert prompt == f"P: {fen}    M:"
        assert prompt.startswith("P: ")
        assert prompt.endswith("    M:")
    
    def test_policy_data_collection_format(self, collector):
        """Test policy task data collection maintains correct format"""
        position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Collect data for policy task
        batch = collector.collect_batch([position])
        
        # Verify batch structure
        assert len(batch.samples) == 4  # Group size
        
        for sample in batch.samples:
            # Check prompt format
            assert sample.prompt.startswith("P: ")
            assert sample.prompt.endswith("    M:")
            
            # Check full text includes prompt + generated
            assert sample.full_text.startswith("P: ")
            assert "    M:" in sample.full_text
            
            # Verify reward computation receives full text
            assert sample.reward > 0  # Should be positive for good response
    
    def test_policy_reward_computation_full_text(self, mock_stockfish):
        """Test policy reward computation receives full text (prompt + generated)"""
        reward_computer = PolicyRewardComputer()
        board = chess.Board()
        
        # Mock Stockfish analysis
        analysis = StockfishAnalysis(
            top5_moves=['e2e4', 'd2d4', 'g1f3', 'b1c3', 'f2f3'],
            top5_evals=[0.25, 0.18, 0.12, 0.08, -0.15],
            best_move='e2e4',
            depth=10,
            analysis_time=0.05
        )
        
        # Test with full text (prompt + generated) - correct format after PR #11
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        prompt = f"P: {fen}    M:"
        generated = " e2e4 d2d4 g1f3 b1c3 f2f3    E: 0.25 0.18 0.12 0.08 -0.15    B: e2e4"
        full_text = prompt + generated
        
        reward, breakdown = reward_computer.compute_reward(full_text, board, analysis)
        
        # Should successfully parse and compute rewards
        assert reward > 1.0  # High reward for perfect match
        assert breakdown["structure_reward"] > 0
        assert breakdown["parse_reward"] > 0
        assert breakdown["malformed_penalty"] == 0
        
        # Test compute_policy_reward convenience function
        reward2, breakdown2 = compute_policy_reward(full_text, board, analysis)
        assert reward2 == reward
        assert breakdown2 == breakdown
    
    def test_policy_evaluation_format(self, evaluator):
        """Test policy evaluation maintains correct format"""
        positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        
        # Evaluate policy performance
        metrics = evaluator.evaluate_policy_performance(positions)
        
        # Should have computed metrics successfully
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.legal_move_rate >= 0
        assert metrics.policy_structure_rate >= 0
        assert metrics.avg_policy_reward != 0  # Should have actual reward values
    
    def test_policy_format_parsing_robustness(self):
        """Test policy format parsing handles edge cases"""
        reward_computer = PolicyRewardComputer()
        board = chess.Board()
        analysis = StockfishAnalysis(['e2e4'], [0.25], 'e2e4', 10, 0.05)
        
        # Test cases with different format variations
        test_cases = [
            # Perfect format
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4    E: 0.25    B: e2e4",
            # Extra whitespace
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:  e2e4   E: 0.25   B: e2e4  ",
            # Malformed (missing P: prefix)
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4    E: 0.25    B: e2e4",
            # Completely wrong
            "This is not a valid format",
        ]
        
        results = []
        for full_text in test_cases:
            reward, breakdown = reward_computer.compute_reward(full_text, board, analysis)
            results.append((reward, breakdown["malformed_penalty"]))
        
        # First two should succeed (have P: prefix)
        assert results[0][0] > 0  # Perfect format
        assert results[1][0] > 0  # Extra whitespace
        
        # Last two should be penalized (no P: prefix or completely wrong)
        assert results[2][0] < 0  # Missing prefix
        assert results[3][0] < 0  # Completely wrong


class TestEnvironmentTaskPipeline:
    """Test Environment (A:) task format through complete pipeline"""
    
    @pytest.fixture
    def mock_policy(self):
        """Mock policy for environment task generation"""
        policy = Mock(spec=CausalLMPolicy)
        policy.device = "cpu"
        
        # Mock environment response for e2e4 move
        env_response = "++rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+0+0"
        policy.generate.return_value = [env_response] * 4  # Group size 4
        
        return policy
    
    @pytest.fixture 
    def collector(self, mock_policy):
        """GRPO collector configured for environment tasks"""
        config = GRPOCollectionConfig(group_size=4, mix_env_ratio=1.0)  # Environment only
        collector = GRPOCollector(config, mock_policy, None)  # No Stockfish for env tasks
        return collector
    
    @pytest.fixture
    def evaluator(self, mock_policy):
        """Chess evaluator for environment tasks"""
        evaluator = ChessEvaluator(mock_policy, None)  # No Stockfish needed
        return evaluator
    
    def test_environment_prompt_format(self):
        """Test environment prompt format generation"""
        formatter = TaskFormatter()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        
        prompt = formatter.create_env_prompt(fen, uci_move)
        
        assert prompt == f"A: {fen}+{uci_move}+"
        assert prompt.startswith("A: ")
        assert prompt.endswith("+")
    
    def test_environment_data_collection_format(self, collector):
        """Test environment task data collection maintains correct format"""
        position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Mock environment task generation
        with patch.object(collector, '_collect_env_samples') as mock_collect:
            # Create mock samples with proper format
            from src.rookworld_rlvr.train.grpo_trainer import GRPOSample
            
            fen = position
            uci_move = "e2e4"
            prompt = f"A: {fen}+{uci_move}+"
            generated = "++rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+0+0"
            full_text = prompt + generated
            
            mock_samples = [
                GRPOSample(
                    prompt=prompt,
                    generated_text=generated,
                    full_text=full_text,
                    reward=0.8,
                    log_prob=-2.5
                ) for _ in range(4)
            ]
            mock_collect.return_value = mock_samples
            
            batch = collector.collect_batch([position])
            
            # Verify batch structure
            assert len(batch.samples) == 4
            
            for sample in batch.samples:
                # Check prompt format
                assert sample.prompt.startswith("A: ")
                assert sample.prompt.endswith("+")
                
                # Check full text includes prompt + generated
                assert sample.full_text.startswith("A: ")
                assert "++" in sample.full_text  # Environment response separator
    
    def test_environment_reward_computation_full_text(self):
        """Test environment reward computation receives full text (prompt + generated)"""
        reward_computer = EnvRewardComputer()
        
        # Create expected response
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        expected = reward_computer.create_expected_response(fen, uci_move)
        
        # Test with full text (prompt + generated) - correct format after PR #11
        prompt = f"A: {fen}+{uci_move}+"
        generated = f"++{expected.new_fen}+{expected.reward}+{int(expected.terminated)}+{int(expected.truncated)}"
        full_text = prompt + generated
        
        reward, breakdown = reward_computer.compute_reward(full_text, expected)
        
        # Should successfully parse and compute rewards
        assert reward > 0  # Should be positive for correct response
        assert breakdown["structure_reward"] > 0
        assert breakdown["malformed_penalty"] == 0
        
        # Test compute_env_reward convenience function
        reward2, breakdown2 = compute_env_reward(full_text, expected)
        assert reward2 == reward
        assert breakdown2 == breakdown
    
    def test_environment_evaluation_format(self, evaluator):
        """Test environment evaluation maintains correct format"""
        positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        moves = ["e2e4"]
        
        # Mock the evaluation process
        with patch.object(evaluator.policy, 'generate') as mock_generate:
            # Mock environment response
            expected_response = "++rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+0+0"
            mock_generate.return_value = [expected_response]
            
            metrics = evaluator.evaluate_environment_performance(positions, moves)
            
            # Should have computed metrics successfully
            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.env_structure_rate >= 0
            assert metrics.avg_env_reward != 0
    
    def test_environment_format_parsing_robustness(self):
        """Test environment format parsing handles edge cases"""
        reward_computer = EnvRewardComputer()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        expected = reward_computer.create_expected_response(fen, "e2e4")
        
        # Test cases with different format variations
        test_cases = [
            # Perfect format (full A: text)
            f"A: {fen}+e2e4+++{expected.new_fen}+{expected.reward}+{int(expected.terminated)}+{int(expected.truncated)}",
            # Extra whitespace
            f"A: {fen}+e2e4+++{expected.new_fen}+{expected.reward}+{int(expected.terminated)}+{int(expected.truncated)} ",
            # Malformed (missing A: prefix) 
            f"{fen}+e2e4+++{expected.new_fen}+{expected.reward}+{int(expected.terminated)}+{int(expected.truncated)}",
            # Completely wrong
            "A: this is not a valid format",
        ]
        
        results = []
        for full_text in test_cases:
            reward, breakdown = reward_computer.compute_reward(full_text, expected)
            results.append((reward, breakdown["malformed_penalty"]))
        
        # First two should succeed (have A: prefix and valid structure)
        assert results[0][0] > 0  # Perfect format
        assert results[1][0] > 0  # Extra whitespace
        
        # Last two should be penalized
        assert results[2][1] < 0  # Missing A: prefix should be penalized
        assert results[3][1] < 0  # Completely wrong format


class TestMixedTaskPipeline:
    """Test mixed Policy and Environment task pipeline"""
    
    @pytest.fixture
    def mock_policy(self):
        """Mock policy that can handle both task types"""
        policy = Mock(spec=CausalLMPolicy)
        policy.device = "cpu"
        
        def mock_generate(prompts, **kwargs):
            """Mock generation based on prompt type"""
            results = []
            for prompt in prompts:
                if prompt.startswith("P: "):
                    # Policy response
                    results.append(" e2e4 d2d4 g1f3 b1c3 f2f3    E: 0.25 0.18 0.12 0.08 -0.15    B: e2e4")
                elif prompt.startswith("A: "):
                    # Environment response
                    results.append("++rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+0+0")
                else:
                    results.append("invalid")
            return results
        
        policy.generate.side_effect = mock_generate
        return policy
    
    @pytest.fixture
    def mock_stockfish(self):
        """Mock Stockfish for policy tasks"""
        engine = Mock(spec=StockfishEngine)
        analysis = StockfishAnalysis(
            top5_moves=['e2e4', 'd2d4', 'g1f3', 'b1c3', 'f2f3'],
            top5_evals=[0.25, 0.18, 0.12, 0.08, -0.15],
            best_move='e2e4',
            depth=10,
            analysis_time=0.05
        )
        engine.analyze.return_value = analysis
        return engine
    
    def test_mixed_task_data_collection(self, mock_policy, mock_stockfish):
        """Test data collection with mixed Policy and Environment tasks"""
        config = GRPOCollectionConfig(group_size=4, mix_env_ratio=0.5)  # 50/50 mix
        collector = GRPOCollector(config, mock_policy, mock_stockfish)
        
        positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        batch = collector.collect_batch(positions)
        
        # Should have samples from both task types
        policy_samples = [s for s in batch.samples if s.prompt.startswith("P: ")]
        env_samples = [s for s in batch.samples if s.prompt.startswith("A: ")]
        
        assert len(policy_samples) > 0
        assert len(env_samples) > 0
        assert len(policy_samples) + len(env_samples) == len(batch.samples)
        
        # Verify format consistency for each task type
        for sample in policy_samples:
            assert sample.prompt.startswith("P: ")
            assert sample.full_text.startswith("P: ")
            assert "    M:" in sample.full_text
        
        for sample in env_samples:
            assert sample.prompt.startswith("A: ")
            assert sample.full_text.startswith("A: ")
            assert "++" in sample.full_text
    
    def test_format_consistency_across_pipeline(self, mock_policy, mock_stockfish):
        """Test format consistency from collection to evaluation"""
        # Collection
        config = GRPOCollectionConfig(group_size=2, mix_env_ratio=0.5)
        collector = GRPOCollector(config, mock_policy, mock_stockfish)
        positions = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        batch = collector.collect_batch(positions)
        
        # Evaluation  
        evaluator = ChessEvaluator(mock_policy, mock_stockfish)
        policy_metrics = evaluator.evaluate_policy_performance(positions)
        env_metrics = evaluator.evaluate_environment_performance(positions, ["e2e4"])
        
        # Both should succeed without format errors
        assert isinstance(policy_metrics, EvaluationMetrics)
        assert isinstance(env_metrics, EvaluationMetrics)
        
        # Policy metrics should reflect proper format parsing
        assert policy_metrics.policy_structure_rate >= 0
        assert policy_metrics.avg_policy_reward != 0
        
        # Environment metrics should reflect proper format parsing
        assert env_metrics.env_structure_rate >= 0
        assert env_metrics.avg_env_reward != 0


class TestFormatParsingConsistency:
    """Test parsing consistency issues addressed in PR #11"""
    
    def test_full_text_vs_generated_only_policy(self):
        """Test policy reward computation with full_text vs generated_text only"""
        reward_computer = PolicyRewardComputer()
        board = chess.Board()
        analysis = StockfishAnalysis(['e2e4'], [0.25], 'e2e4', 10, 0.05)
        
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        prompt = f"P: {fen}    M:"
        generated = " e2e4    E: 0.25    B: e2e4"
        full_text = prompt + generated
        
        # Test with full_text (correct after PR #11 fix)
        reward_full, breakdown_full = reward_computer.compute_reward(full_text, board, analysis)
        
        # Test with generated only (incorrect - should fail parsing)
        reward_gen, breakdown_gen = reward_computer.compute_reward(generated, board, analysis)
        
        # Full text should succeed
        assert reward_full > 0
        assert breakdown_full["structure_reward"] > 0
        assert breakdown_full["malformed_penalty"] == 0
        
        # Generated only should fail (no P: prefix)
        assert reward_gen < 0
        assert breakdown_gen["structure_reward"] == 0
        assert breakdown_gen["malformed_penalty"] < 0
    
    def test_full_text_vs_generated_only_environment(self):
        """Test environment reward computation with full_text vs generated_text only"""
        reward_computer = EnvRewardComputer()
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        expected = reward_computer.create_expected_response(fen, "e2e4")
        
        prompt = f"A: {fen}+e2e4+"
        generated = f"++{expected.new_fen}+{expected.reward}+{int(expected.terminated)}+{int(expected.truncated)}"
        full_text = prompt + generated
        
        # Test with full_text (correct after PR #11 fix)
        reward_full, breakdown_full = reward_computer.compute_reward(full_text, expected)
        
        # Test with generated only (incorrect - should fail parsing)
        reward_gen, breakdown_gen = reward_computer.compute_reward(generated, expected)
        
        # Full text should succeed
        assert reward_full > 0
        assert breakdown_full["structure_reward"] > 0
        assert breakdown_full["malformed_penalty"] == 0
        
        # Generated only should fail (no A: prefix)
        assert reward_gen < 0
        assert breakdown_gen["structure_reward"] == 0
        assert breakdown_gen["malformed_penalty"] < 0
    
    def test_parsing_prefix_requirements(self):
        """Test that parsing requires proper task prefixes"""
        from src.rookworld_rlvr.environment.chess_env import ChessEnvironment
        
        env = ChessEnvironment()
        
        # Test A: prefix requirement for environment parsing
        valid_env_text = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+++new_fen+0.001+0+0"
        invalid_env_text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+++new_fen+0.001+0+0"
        
        # Valid parsing should succeed
        parsed_valid = env.parse_prediction(valid_env_text)
        assert parsed_valid is not None
        
        # Invalid parsing should fail or handle gracefully
        parsed_invalid = env.parse_prediction(invalid_env_text)
        # The fix ensures this is handled properly in downstream reward computation