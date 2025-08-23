"""
Tests for ChessEnvironment utility

Verify chess state management and A: task ground truth generation.
"""

import pytest
import chess
from src.rookworld_rlvr.environment.chess_env import ChessEnvironment, EnvironmentResponse


class TestChessEnvironment:
    """Test ChessEnvironment utility functionality"""
    
    @pytest.fixture
    def chess_env(self):
        """Create chess environment instance"""
        return ChessEnvironment()
    
    def test_initialization(self, chess_env):
        """Test environment initialization"""
        assert chess_env.default_reward == 0.001
    
    def test_apply_move_valid(self, chess_env):
        """Test applying valid moves"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        
        response = chess_env.apply_move(fen, uci_move)
        
        assert isinstance(response, EnvironmentResponse)
        assert response.previous_fen == fen
        assert response.uci_move == uci_move
        # Check key components of the new FEN (en passant handling can vary)
        assert "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq" in response.new_fen
        assert response.new_fen.endswith("0 1")
        assert response.reward == 0.001
        assert not response.terminated  # Game not over
        assert not response.truncated   # Not max moves
    
    def test_apply_move_invalid(self, chess_env):
        """Test applying invalid moves"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        invalid_move = "e2e5"  # Can't move pawn 3 squares
        
        with pytest.raises(ValueError):
            chess_env.apply_move(fen, invalid_move)
    
    def test_apply_move_game_over(self, chess_env):
        """Test move that ends the game"""
        # Simplified mate in 1 position (white to move)
        fen = "k7/8/1K6/8/8/8/8/7Q w - - 0 1"
        mate_move = "h1h8"  # Checkmate
        
        response = chess_env.apply_move(fen, mate_move)
        
        assert response.terminated  # Game should be over
        assert not response.truncated
    
    def test_structured_string_format(self, chess_env):
        """Test A: format string generation"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        
        response = chess_env.apply_move(fen, uci_move, move_history="")
        structured = response.to_structured_string()
        
        # Check that the structured string contains the key components
        assert structured.startswith(f"{fen}+{uci_move}++")
        assert "+0.001+0+0" in structured
        assert "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq" in structured
    
    def test_parse_prediction_valid(self, chess_env):
        """Test parsing valid A: task predictions"""
        prediction = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+"
                     "e2e4++"
                     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+"
                     "0.001+0+0")
        
        parsed = chess_env.parse_prediction(prediction)
        
        assert parsed is not None
        assert parsed.previous_fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert parsed.uci_move == "e2e4"
        assert parsed.reward == 0.001
        assert not parsed.terminated
        assert not parsed.truncated
    
    def test_parse_prediction_with_prefix(self, chess_env):
        """Test parsing prediction with A: prefix"""
        prediction = ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+"
                     "e2e4++"
                     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+"
                     "0.001+0+0")
        
        parsed = chess_env.parse_prediction(prediction)
        
        assert parsed is not None
        assert parsed.previous_fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def test_parse_prediction_invalid(self, chess_env):
        """Test parsing invalid predictions"""
        invalid_predictions = [
            "invalid format",
            "not+enough+parts",
            "too+many+parts+but+not+enough+still+here",
            "fen+move+history+fen+invalid_reward+0+0",
            "fen+move+history+fen+0.001+invalid_flag+0",
        ]
        
        for prediction in invalid_predictions:
            parsed = chess_env.parse_prediction(prediction)
            assert parsed is None
    
    def test_validate_prediction_exact_match(self, chess_env):
        """Test validation with exact FEN match"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        
        expected = chess_env.apply_move(fen, uci_move)
        prediction = expected.to_structured_string()
        
        validation = chess_env.validate_prediction(prediction, expected)
        
        assert validation["is_valid_format"]
        assert validation["fen_exact_match"]
        assert validation["fen_similarity_score"] == 1.0
        assert validation["reward_accuracy"] == 1.0
        assert validation["flag_accuracy"] == 1.0
    
    def test_validate_prediction_partial_match(self, chess_env):
        """Test validation with partial match"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci_move = "e2e4"
        
        expected = chess_env.apply_move(fen, uci_move)
        
        # Create prediction with slight differences
        wrong_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"  # Wrong turn
        prediction = (f"{fen}+{uci_move}++"
                     f"{wrong_fen}+0.002+1+0")  # Wrong FEN, reward, terminated flag
        
        validation = chess_env.validate_prediction(prediction, expected)
        
        assert validation["is_valid_format"]
        assert not validation["fen_exact_match"]
        assert validation["fen_similarity_score"] > 0.8  # Should be quite similar
        assert validation["reward_accuracy"] > 0.9       # Small reward difference
        assert validation["flag_accuracy"] == 0.5        # One flag wrong
    
    def test_legal_moves(self, chess_env):
        """Test legal move generation"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        legal_moves = chess_env.get_legal_moves(fen)
        
        assert isinstance(legal_moves, list)
        assert len(legal_moves) == 20  # 20 legal moves in starting position
        assert "e2e4" in legal_moves
        assert "e2e3" in legal_moves
        assert "g1f3" in legal_moves
        
        # Test invalid FEN
        invalid_legal = chess_env.get_legal_moves("invalid fen")
        assert invalid_legal == []
    
    def test_is_valid_move(self, chess_env):
        """Test move validation"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        assert chess_env.is_valid_move(fen, "e2e4")  # Valid
        assert chess_env.is_valid_move(fen, "g1f3")  # Valid
        assert not chess_env.is_valid_move(fen, "e2e5")  # Invalid (3 squares)
        assert not chess_env.is_valid_move(fen, "a1a2")  # Invalid (rook blocked)
        
        # Test with invalid FEN
        assert not chess_env.is_valid_move("invalid", "e2e4")
    
    def test_sample_positions(self, chess_env):
        """Test sample position generation"""
        positions = chess_env.create_sample_positions(5)
        
        assert len(positions) == 5
        
        # Check all positions are valid FENs
        for fen in positions:
            try:
                board = chess.Board(fen)
                assert not board.is_game_over()  # Should be playable positions
            except ValueError:
                pytest.fail(f"Invalid FEN generated: {fen}")
        
        # Starting position should be included
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert starting_fen in positions
    
    def test_fen_similarity(self, chess_env):
        """Test FEN similarity computation"""
        fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen2 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Identical
        fen3 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"  # Different
        fen4 = "completely different string"  # Very different
        
        # Test identical
        similarity_same = chess_env._compute_fen_similarity(fen1, fen2)
        assert similarity_same == 1.0
        
        # Test similar
        similarity_close = chess_env._compute_fen_similarity(fen1, fen3)
        assert 0.5 < similarity_close < 1.0
        
        # Test very different
        similarity_diff = chess_env._compute_fen_similarity(fen1, fen4)
        assert 0.0 <= similarity_diff < 0.5


class TestEnvironmentResponse:
    """Test EnvironmentResponse dataclass"""
    
    def test_to_structured_string(self):
        """Test structured string conversion"""
        response = EnvironmentResponse(
            previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            uci_move="e2e4",
            move_history="",
            new_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            reward=0.001,
            terminated=False,
            truncated=False
        )
        
        structured = response.to_structured_string()
        expected = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+"
                   "e2e4++"
                   "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+"
                   "0.001+0+0")
        
        assert structured == expected
    
    def test_with_move_history(self):
        """Test with move history"""
        response = EnvironmentResponse(
            previous_fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5",
            uci_move="c4c5",
            move_history="e2e4 e7e5 g1f3 b8c6 d2d4 g8f6 c2c4",
            new_fen="r1bqkb1r/pppp1ppp/2n2n2/2P1p3/3P4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 5",
            reward=0.001,
            terminated=False,
            truncated=False
        )
        
        structured = response.to_structured_string()
        assert "e2e4 e7e5 g1f3 b8c6 d2d4 g8f6 c2c4" in structured