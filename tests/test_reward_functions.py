"""
Tests for Reward Functions

Verify policy and environment reward computation for RookWorld GRPO training.
"""

import pytest
import chess
from src.rookworld_rlvr.reward.policy_reward import (
    PolicyRewardComputer, PolicyRewardConfig, ParsedPolicyOutput
)
from src.rookworld_rlvr.engine.stockfish import StockfishAnalysis
from src.rookworld_rlvr.reward.env_reward import (
    EnvRewardComputer, EnvRewardConfig
)
from src.rookworld_rlvr.environment.chess_env import EnvironmentResponse


class TestPolicyRewardComputer:
    """Test policy reward computation"""
    
    @pytest.fixture
    def reward_computer(self):
        """Create policy reward computer with default config"""
        return PolicyRewardComputer()
    
    @pytest.fixture
    def board(self):
        """Create starting chess board"""
        return chess.Board()
    
    @pytest.fixture
    def stockfish_analysis(self):
        """Mock Stockfish analysis"""
        return StockfishAnalysis(
            top5_moves=['e2e4', 'd2d4', 'g1f3', 'b1c3', 'f2f3'],
            top5_evals=[0.25, 0.18, 0.12, 0.08, -0.15],
            best_move='e2e4',
            depth=10,
            analysis_time=0.05
        )
    
    def test_parse_valid_output(self, reward_computer):
        """Test parsing valid policy output"""
        output = " e2e4 d2d4 g1f3 b1c3 f2f3    E: 0.25 0.18 0.12 0.08 -0.15    B: e2e4"
        
        parsed = reward_computer.parse_policy_output(output)
        
        assert parsed.is_valid_format
        assert parsed.moves == ['e2e4', 'd2d4', 'g1f3', 'b1c3', 'f2f3']
        assert parsed.evaluations == [0.25, 0.18, 0.12, 0.08, -0.15]
        assert parsed.best_move == 'e2e4'
        assert len(parsed.parsing_errors) == 0
    
    def test_parse_malformed_output(self, reward_computer):
        """Test parsing malformed outputs"""
        malformed_outputs = [
            "just some random text",
            "e2e4 d2d4    but no E: or B: markers",
            "e2e4    E: not_a_number    B: e2e4",
            "invalid_move    E: 0.25    B: e2e4",
        ]
        
        for output in malformed_outputs:
            parsed = reward_computer.parse_policy_output(output)
            # With flexible parsing, malformed outputs should have parsing errors
            assert len(parsed.parsing_errors) > 0 or not parsed.is_valid_format
    
    def test_compute_reward_perfect_match(self, reward_computer, board, stockfish_analysis):
        """Test reward computation for perfect match"""
        # Perfect output matching Stockfish analysis
        output = " e2e4 d2d4 g1f3 b1c3 f2f3    E: 0.25 0.18 0.12 0.08 -0.15    B: e2e4"
        
        reward, breakdown = reward_computer.compute_reward(output, board, stockfish_analysis)
        
        # Should get maximum rewards for all components
        assert breakdown["structure_reward"] > 0
        assert breakdown["parse_reward"] > 0  
        assert breakdown["move_match_reward"] > 0
        assert breakdown["eval_accuracy_reward"] > 0
        assert breakdown["best_move_reward"] > 0
        assert breakdown["malformed_penalty"] == 0
        
        # Total reward should be positive and substantial (graduated system gives max 1.0)
        assert reward >= 1.0  # Expect high reward for perfect match
    
    def test_compute_reward_partial_match(self, reward_computer, board, stockfish_analysis):
        """Test reward computation for partial match"""
        # Some correct moves, some wrong evals, correct best move
        output = " e2e4 d2d4 a2a4 h2h4 g2g4    E: 0.5 0.3 0.1 -0.2 -0.5    B: e2e4"
        
        reward, breakdown = reward_computer.compute_reward(output, board, stockfish_analysis)
        
        # Should get structure and parse rewards
        assert breakdown["structure_reward"] > 0
        assert breakdown["parse_reward"] > 0
        
        # Partial move match (2/5 moves correct, graduated max 0.2)
        assert 0 < breakdown["move_match_reward"] <= 0.2
        
        # Some eval accuracy (not perfect due to differences, graduated max 0.2)
        assert 0 <= breakdown["eval_accuracy_reward"] <= 0.2
        
        # Perfect best move (graduated reward system gives 0.2 max)
        assert breakdown["best_move_reward"] == 0.2
        
        assert breakdown["malformed_penalty"] == 0
        assert reward > 0  # Should still be positive
    
    def test_compute_reward_malformed(self, reward_computer, board, stockfish_analysis):
        """Test reward computation for malformed output"""
        malformed_output = "this is completely wrong format"
        
        reward, breakdown = reward_computer.compute_reward(malformed_output, board, stockfish_analysis)
        
        # Should only get malformed penalty
        assert breakdown["structure_reward"] == 0
        assert breakdown["parse_reward"] == 0
        assert breakdown["move_match_reward"] == 0
        assert breakdown["eval_accuracy_reward"] == 0
        assert breakdown["best_move_reward"] == 0
        # Graduated penalty system gives -0.1 instead of config value -1.0
        assert breakdown["malformed_penalty"] == -0.1
        
        assert reward == -0.1  # Should be negative (graduated penalty)
    
    def test_stockfish_analysis_structure(self, stockfish_analysis):
        """Test Stockfish analysis structure"""
        assert isinstance(stockfish_analysis.top5_moves, list)
        assert isinstance(stockfish_analysis.top5_evals, list)  
        assert isinstance(stockfish_analysis.best_move, str)
        
        assert len(stockfish_analysis.top5_moves) > 0
        assert len(stockfish_analysis.top5_evals) == len(stockfish_analysis.top5_moves)
        assert stockfish_analysis.best_move in stockfish_analysis.top5_moves
        
        # All moves should be valid UCI format
        for move in stockfish_analysis.top5_moves:
            try:
                chess.Move.from_uci(move)
            except ValueError:
                pytest.fail(f"Invalid UCI move in analysis: {move}")


class TestEnvRewardComputer:
    """Test environment reward computation"""
    
    @pytest.fixture
    def reward_computer(self):
        """Create environment reward computer with default config"""
        return EnvRewardComputer()
    
    @pytest.fixture
    def expected_response(self):
        """Create expected environment response"""
        return EnvironmentResponse(
            previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            uci_move="e2e4",
            move_history="",
            new_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            reward=0.001,
            terminated=False,
            truncated=False
        )
    
    def test_compute_reward_exact_match(self, reward_computer, expected_response):
        """Test reward computation for exact match"""
        # Perfect A: format output
        perfect_output = expected_response.to_structured_string()
        
        reward, breakdown = reward_computer.compute_reward(perfect_output, expected_response)
        
        # Should get all positive rewards
        assert breakdown["structure_reward"] > 0
        assert breakdown["fen_exact_reward"] > 0
        assert breakdown["reward_accuracy_reward"] > 0
        assert breakdown["flags_accuracy_reward"] > 0
        assert breakdown["malformed_penalty"] == 0
        
        # Should get exact match, not similarity
        assert breakdown["fen_similarity_reward"] == 0  # Exact match takes precedence
        
        assert reward > 1.0  # Should be substantial positive reward
    
    def test_compute_reward_partial_match(self, reward_computer, expected_response):
        """Test reward computation for partial match"""
        # Similar but not exact output
        partial_output = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+"
            "e2e4++"
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1+"  # Wrong turn
            "0.002+"  # Wrong reward
            "1+"      # Wrong terminated flag
            "0"
        )
        
        reward, breakdown = reward_computer.compute_reward(partial_output, expected_response)
        
        # Should get structure reward
        assert breakdown["structure_reward"] > 0
        
        # Should get similarity reward (not exact)
        assert breakdown["fen_exact_reward"] == 0
        assert breakdown["fen_similarity_reward"] > 0
        
        # Partial reward and flag accuracy
        assert 0 < breakdown["reward_accuracy_reward"] < reward_computer.config.r_env_reward_accuracy
        assert 0 < breakdown["flags_accuracy_reward"] < reward_computer.config.r_env_flags_accuracy
        
        assert reward > 0  # Should still be positive overall
    
    def test_compute_reward_malformed(self, reward_computer, expected_response):
        """Test reward computation for malformed output"""
        malformed_output = "A: this is not a valid format"
        
        reward, breakdown = reward_computer.compute_reward(malformed_output, expected_response)
        
        # Should only get malformed penalty
        assert breakdown["structure_reward"] == 0
        assert breakdown["fen_exact_reward"] == 0
        assert breakdown["fen_similarity_reward"] == 0
        assert breakdown["reward_accuracy_reward"] == 0
        assert breakdown["flags_accuracy_reward"] == 0
        assert breakdown["malformed_penalty"] == reward_computer.config.r_env_malformed
        
        assert reward == reward_computer.config.r_env_malformed  # Should be negative
    
    def test_create_expected_response(self, reward_computer):
        """Test creating expected responses"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        
        response = reward_computer.create_expected_response(fen, uci_move)
        
        assert response.previous_fen == fen
        assert response.uci_move == uci_move
        assert response.reward == 0.001  # Default reward
        assert not response.terminated  # Game not over
        assert not response.truncated   # Not max moves
        
        # New FEN should be different
        assert response.new_fen != fen
        assert "4P3" in response.new_fen  # Pawn should have moved
    
    def test_compute_reward_from_components(self, reward_computer):
        """Test convenience method for computing reward from components"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        
        # Create expected response first
        expected = reward_computer.create_expected_response(fen, uci_move)
        perfect_output = expected.to_structured_string()
        
        # Test the convenience method
        reward, breakdown = reward_computer.compute_reward_from_components(
            fen, uci_move, perfect_output
        )
        
        assert reward > 1.0  # Should be positive for perfect match
        assert breakdown["structure_reward"] > 0
        assert breakdown["fen_exact_reward"] > 0
    
    def test_invalid_move_handling(self, reward_computer):
        """Test handling of invalid moves"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        invalid_move = "e2e5"  # Invalid: pawn can't move 3 squares
        
        reward, breakdown = reward_computer.compute_reward_from_components(
            fen, invalid_move, "any output"
        )
        
        assert reward == reward_computer.config.r_env_malformed
        assert "error" in breakdown


class TestRewardConfig:
    """Test reward configuration classes"""
    
    def test_policy_reward_config_defaults(self):
        """Test policy reward config defaults"""
        config = PolicyRewardConfig()
        
        assert config.r_policy_structure > 0
        assert config.r_policy_parse > 0
        assert config.r_policy_move_match > 0
        assert config.r_policy_eval_accuracy > 0
        assert config.r_policy_best_move > 0
        assert config.r_policy_malformed < 0
        assert config.require_exact_count
    
    def test_env_reward_config_defaults(self):
        """Test environment reward config defaults"""
        config = EnvRewardConfig()
        
        assert config.r_env_structure > 0
        assert config.r_env_fen_exact > 0
        assert config.r_env_fen_similarity > 0
        assert config.r_env_reward_accuracy > 0
        assert config.r_env_flags_accuracy > 0
        assert config.r_env_malformed < 0
    
    def test_custom_config(self):
        """Test custom configuration"""
        custom_policy_config = PolicyRewardConfig(
            r_policy_structure=0.5,
            r_policy_malformed=-2.0
        )
        
        reward_computer = PolicyRewardComputer(custom_policy_config)
        assert reward_computer.config.r_policy_structure == 0.5
        assert reward_computer.config.r_policy_malformed == -2.0