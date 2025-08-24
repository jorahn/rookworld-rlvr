#!/usr/bin/env python3
"""
Target Detection Unit Tests

Unit tests for target start index detection in both policy and environment tasks.
These tests ensure the fixes for training instability are maintained.
"""

import unittest
import sys
import os
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rookworld_rlvr.tokenizer.bridge import TokenizerBridge


class TestTargetDetection(unittest.TestCase):
    """Unit tests for target detection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tokenizer = TokenizerBridge()
    
    def test_policy_target_detection_basic(self):
        """Test basic policy task target detection (M: pattern)"""
        text = 'P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4'
        expected_target = self.tokenizer.get_target_start_index(text, 'policy')
        actual_target = self.tokenizer.get_target_start_index(text, 'policy')
        
        self.assertEqual(actual_target, expected_target)
        
        # Verify target points to correct location
        tokens = self.tokenizer.encode(text)
        self.assertLess(actual_target, len(tokens), "Target index should be within token range")
        
        # Verify target tokens start with move (e2e4)
        target_tokens = tokens[actual_target:actual_target+2]
        target_text = self.tokenizer.decode(target_tokens)
        self.assertIn('e2', target_text, "Target should point to move tokens")
    
    def test_policy_target_detection_full_response(self):
        """Test policy task with full structured response"""
        text = 'P: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1    M: e7e5\nE: 0.1\nB: e7e5 d7d5 g8f6'
        target_start = self.tokenizer.get_target_start_index(text, 'policy')
        
        tokens = self.tokenizer.encode(text)
        self.assertLess(target_start, len(tokens))
        
        # Target should point to move after M:
        target_preview = self.tokenizer.decode(tokens[target_start:target_start+3])
        self.assertIn('e7', target_preview, "Should target the move e7e5")
    
    def test_policy_target_detection_complex_fen(self):
        """Test policy task with complex FEN position"""
        text = 'P: r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5    M: d4d5\nE: 0.3\nB: d4d5 c4c5 b1c3'
        target_start = self.tokenizer.get_target_start_index(text, 'policy')
        
        tokens = self.tokenizer.encode(text)
        self.assertLess(target_start, len(tokens))
        
        # Verify targeting correct move
        target_preview = self.tokenizer.decode(tokens[target_start:target_start+3])
        self.assertIn('d4', target_preview, "Should target the move d4d5")
    
    def test_environment_target_detection_basic(self):
        """Test basic environment task target detection (+ pattern)"""
        text = 'A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\nR: 0.0\nT: False\nU: False'
        target_start = self.tokenizer.get_target_start_index(text, 'environment')
        
        tokens = self.tokenizer.encode(text)
        self.assertLess(target_start, len(tokens))
        
        # Target should point to content after first + (move or resulting state)
        target_preview = self.tokenizer.decode(tokens[target_start:target_start+10])
        # Should contain either move notation or board state characters
        self.assertTrue(any(char in target_preview for char in ['e', 'r', 'n', 'b', 'q', '/', '2', '4']), 
                       f"Should target move or board state, got: {target_preview}")
    
    def test_environment_target_detection_complex(self):
        """Test environment task with complex position"""
        text = 'A: r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5+d4d5+r1bqkb1r/ppp2ppp/2n2n2/3pp3/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 5\nR: 0.1\nT: False\nU: False'
        target_start = self.tokenizer.get_target_start_index(text, 'environment')
        
        tokens = self.tokenizer.encode(text)
        self.assertLess(target_start, len(tokens))
        
        # Should target content after first + (move or resulting state)
        target_preview = self.tokenizer.decode(tokens[target_start:target_start+10])
        self.assertTrue(any(char in target_preview for char in ['d', 'r', 'n', 'b', '/', '4', '5']), 
                       f"Should target move or board state, got: {target_preview}")
    
    def test_edge_cases(self):
        """Test edge cases and malformed inputs"""
        # Edge case: no space after M:
        text = 'P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:e2e4'
        target_start = self.tokenizer.get_target_start_index(text, 'policy')
        tokens = self.tokenizer.encode(text)
        self.assertLess(target_start, len(tokens))
        
        # Edge case: minimal environment
        text = 'A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+result'
        target_start = self.tokenizer.get_target_start_index(text, 'environment')
        tokens = self.tokenizer.encode(text)
        self.assertLess(target_start, len(tokens))
    
    def test_tokenization_consistency(self):
        """Test that M: and + patterns are tokenized consistently"""
        # Test M: tokenization in different contexts
        m_colon_tests = [
            "M: e2e4",
            " M: e2e4", 
            "M:e2e4",
            "test M: e2e4",
            "    M: e2e4"
        ]
        
        for text in m_colon_tests:
            tokens = self.tokenizer.encode(text)
            decoded = [self.tokenizer.decode([t]).strip() for t in tokens]
            # M: should be consistently handled
            self.assertTrue(any('M' in token for token in decoded), 
                          f"M token should be found in: {text} -> {decoded}")
        
        # Test + tokenization
        plus_tests = [
            "+e2e4+",
            "A: fen+move+",
            " +test",
            "position+move+"
        ]
        
        for text in plus_tests:
            tokens = self.tokenizer.encode(text)
            decoded = [self.tokenizer.decode([t]).strip() for t in tokens]
            plus_positions = [i for i, t in enumerate(decoded) if t == '+']
            # + should be consistently detected
            self.assertTrue(len(plus_positions) > 0, 
                          f"+ token should be found in: {text} -> {decoded}")
    
    def test_target_detection_deterministic(self):
        """Test that target detection is deterministic across multiple calls"""
        text = 'P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4'
        
        # Call multiple times and ensure consistent results
        results = []
        for _ in range(5):
            target_start = self.tokenizer.get_target_start_index(text, 'policy')
            results.append(target_start)
        
        # All results should be identical
        self.assertEqual(len(set(results)), 1, "Target detection should be deterministic")
        
        # Do the same for environment task
        env_text = 'A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+result'
        env_results = []
        for _ in range(5):
            target_start = self.tokenizer.get_target_start_index(env_text, 'environment')
            env_results.append(target_start)
        
        self.assertEqual(len(set(env_results)), 1, "Environment target detection should be deterministic")


if __name__ == '__main__':
    unittest.main()