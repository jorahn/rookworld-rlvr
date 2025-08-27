"""
Comprehensive tests for dataset processing and validation

Tests cover:
1. Preprocessing (A: prefix addition)
2. P: task parsing
3. A: task parsing
4. Format validation
5. Content validation
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset import preprocess_sample, parse_p_task, parse_a_task
from validation import (
    validate_p_format, validate_a_format,
    validate_a_fen, validate_a_flags, validate_a_reward,
    levenshtein_distance, P_WEIGHTS, A_WEIGHTS
)


class TestPreprocessing:
    """Test sample preprocessing"""
    
    def test_preserves_p_prefix(self):
        """P: tasks should remain unchanged"""
        text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = preprocess_sample(text)
        assert result == text
    
    def test_preserves_a_prefix(self):
        """A: tasks should remain unchanged"""
        text = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
        result = preprocess_sample(text)
        assert result == text
    
    def test_adds_a_prefix(self):
        """Samples without prefix should get A: added"""
        text = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+result"
        result = preprocess_sample(text)
        assert result.startswith("A: ")
        assert "rnbqkbnr/pppppppp" in result
    
    def test_handles_whitespace(self):
        """Should handle leading/trailing whitespace"""
        text = "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+  "
        result = preprocess_sample(text)
        assert result.startswith("A: ")
        assert not result.startswith("A:  ")  # No extra spaces


class TestPTaskParsing:
    """Test P: (Policy) task parsing"""
    
    def test_parse_complete_p_task(self):
        """Test parsing a complete P: task with all sections"""
        text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1                                  M: e2e4 d2d4 g1f3 c2c4 b1c3      E: 0.3 0.35 0.28 0.32 0.29         B: e2e4"
        
        prompt, completion, data = parse_p_task(text)
        
        # Check prompt
        assert prompt == "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # Check completion
        assert completion.startswith("M:")
        assert "E:" in completion
        assert "B:" in completion
        
        # Check parsed data
        assert data['fen'] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert data['moves'] == ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'b1c3']
        assert data['evals'] == [0.3, 0.35, 0.28, 0.32, 0.29]
        assert data['best_move'] == 'e2e4'
    
    def test_parse_p_task_prompt_only(self):
        """Test parsing P: task with prompt only (no completion)"""
        text = "P: 8/8/8/8/8/8/8/K1k5 w - - 0 1"
        
        prompt, completion, data = parse_p_task(text)
        
        assert prompt == text
        assert completion == ""
        assert data['fen'] == "8/8/8/8/8/8/8/K1k5 w - - 0 1"
        assert 'moves' not in data
        assert 'evals' not in data
        assert 'best_move' not in data
    
    def test_parse_p_task_partial_completion(self):
        """Test parsing P: task with partial completion"""
        text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1  M: e2e4 d2d4"
        
        prompt, completion, data = parse_p_task(text)
        
        assert prompt == "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert completion.startswith("M:")  # Just check it starts with M:
        assert "e2e4" in completion
        assert "d2d4" in completion
        assert data['moves'] == ['e2e4', 'd2d4']
        assert 'evals' not in data
        assert 'best_move' not in data
    
    def test_parse_invalid_p_task(self):
        """Test that non-P: task raises error"""
        text = "A: some other task"
        
        with pytest.raises(ValueError, match="Not a P: task"):
            parse_p_task(text)


class TestATaskParsing:
    """Test A: (Environment) task parsing"""
    
    def test_parse_complete_a_task(self):
        """Test parsing a complete A: task with all sections"""
        text = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
        
        prompt, completion, data = parse_a_task(text)
        
        # Check prompt
        assert prompt == "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
        
        # Check completion
        assert completion == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
        
        # Check parsed data
        assert data['fen'] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert data['move'] == "e2e4"
        assert data['history'] == ","  # Empty history with comma
        assert data['new_fen'] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        assert data['reward'] == 0.001
        assert data['terminated'] == False
        assert data['truncated'] == False
    
    def test_parse_a_task_with_history(self):
        """Test parsing A: task with move history"""
        text = "A: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3+d2d4+e7e5,b1c3,g8f6+r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3+0.001+false+false"
        
        prompt, completion, data = parse_a_task(text)
        
        assert data['history'] == "e7e5,b1c3,g8f6"
        assert data['move'] == "d2d4"
    
    def test_parse_a_task_prompt_only(self):
        """Test parsing A: task with prompt only"""
        text = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
        
        prompt, completion, data = parse_a_task(text)
        
        assert prompt == text
        assert completion == ""
        assert data['fen'] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert data['move'] == "e2e4"
        assert data['history'] == ","
        assert 'new_fen' not in data
    
    def test_parse_invalid_a_task(self):
        """Test that non-A: task raises error"""
        text = "P: some other task"
        
        with pytest.raises(ValueError, match="Not an A: task"):
            parse_a_task(text)


class TestPFormatValidation:
    """Test P: task format validation"""
    
    def test_valid_p_format(self):
        """Test valid P: format with all sections"""
        completion = "M: e2e4 d2d4 g1f3  E: 0.3 0.35 0.28  B: e2e4"
        score, details = validate_p_format(completion)
        
        assert score == 1.0
        assert details['has_moves'] == True
        assert details['has_evals'] == True
        assert details['has_best'] == True
        assert details['moves'] == ['e2e4', 'd2d4', 'g1f3']
        assert details['evals'] == [0.3, 0.35, 0.28]
        assert details['best_move'] == 'e2e4'
    
    def test_missing_sections(self):
        """Test P: format with missing sections"""
        completion = "M: e2e4 d2d4"  # Missing E: and B:
        score, details = validate_p_format(completion)
        
        assert score == 0.0  # Not all sections present
        assert details['has_moves'] == True
        assert details['has_evals'] == False
        assert details['has_best'] == False
    
    def test_empty_completion(self):
        """Test empty completion"""
        score, details = validate_p_format("")
        
        assert score == 0.0
        assert details['has_moves'] == False
        assert details['has_evals'] == False
        assert details['has_best'] == False


class TestAFormatValidation:
    """Test A: task format validation"""
    
    def test_valid_a_format(self):
        """Test valid A: format with all sections"""
        completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
        score, details = validate_a_format(completion)
        
        assert score == 1.0
        assert details['num_sections'] == 4
        assert details['has_fen'] == True
        assert details['has_reward'] == True
        assert details['has_terminated'] == True
        assert details['has_truncated'] == True
        assert details['reward'] == 0.001
        assert details['terminated'] == False
        assert details['truncated'] == False
    
    def test_missing_sections(self):
        """Test A: format with missing sections"""
        completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        score, details = validate_a_format(completion)
        
        assert score == 0.0  # Not all sections present
        assert details['num_sections'] == 1
        assert details['has_fen'] == False  # No + delimiter, so not recognized
    
    def test_invalid_values(self):
        """Test A: format with invalid values"""
        completion = "not_a_fen+not_a_number+maybe+perhaps"
        score, details = validate_a_format(completion)
        
        assert score == 0.0
        assert details['has_fen'] == False  # No / in first section
        assert details['has_reward'] == False  # Not numeric


class TestContentValidation:
    """Test content validation functions"""
    
    def test_levenshtein_distance(self):
        """Test edit distance calculation"""
        assert levenshtein_distance("abc", "abc") == 0
        assert levenshtein_distance("abc", "adc") == 1  # One substitution
        assert levenshtein_distance("abc", "abcd") == 1  # One insertion
        assert levenshtein_distance("abcd", "abc") == 1  # One deletion
        assert levenshtein_distance("", "abc") == 3
    
    def test_validate_a_fen(self):
        """Test FEN validation with edit distance"""
        fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        
        # Identical FENs
        assert validate_a_fen(fen1, fen1) == 1.0
        
        # Different FENs
        score = validate_a_fen(fen1, fen2)
        assert 0 < score < 1
    
    def test_validate_a_flags(self):
        """Test game state flag validation"""
        # Legal move, game continues
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = "e2e4"
        score = validate_a_flags(fen, move, "false", "false")
        assert score == 1.0  # Both flags correct
        
        # Illegal move
        score = validate_a_flags(fen, "e2e5", "false", "true")
        assert score == 1.0  # Both flags correct (truncated=true for illegal)
        
        # Wrong flags
        score = validate_a_flags(fen, move, "true", "true")
        assert score == 0.0  # Both flags wrong
    
    def test_validate_a_reward(self):
        """Test reward value validation"""
        # Normal move
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = "e2e4"
        
        # Correct reward
        assert validate_a_reward(fen, move, 0.001) == 1.0
        
        # Wrong reward
        assert validate_a_reward(fen, move, 1.0) < 0.5
        
        # Illegal move should have 0 reward
        assert validate_a_reward(fen, "e2e5", 0.0) == 1.0


class TestWeightedScoring:
    """Test weighted scoring system"""
    
    def test_p_weights(self):
        """Test P: task weight priorities"""
        assert P_WEIGHTS['best_move'] > P_WEIGHTS['format']
        assert P_WEIGHTS['format'] > P_WEIGHTS['candidates']
        assert P_WEIGHTS['candidates'] > P_WEIGHTS['evaluations']
    
    def test_a_weights(self):
        """Test A: task weight priorities"""
        assert A_WEIGHTS['format'] > A_WEIGHTS['fen_match']
        assert A_WEIGHTS['fen_match'] > A_WEIGHTS['game_state']
        assert A_WEIGHTS['game_state'] > A_WEIGHTS['reward_value']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-x"])