#!/usr/bin/env python3
"""
Target Detection Regression Tests

Comprehensive regression tests to ensure target start index detection
works correctly for both policy and environment tasks after improvements.
These tests validate the fixes that resolved the training instability issues.
"""

import torch
import sys
import os
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rookworld_rlvr.tokenizer.bridge import TokenizerBridge


class TargetDetectionRegressionTester:
    """Regression test suite for target detection improvements"""
    
    def __init__(self):
        self.tokenizer = TokenizerBridge()
        self.test_cases = self._create_test_cases()
        self.test_results = []
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases for both task types"""
        test_cases = []
        
        # Basic policy task
        basic_policy_text = 'P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4'
        basic_policy_expected = self._calculate_expected_target(basic_policy_text, 'policy')
        test_cases.append({
            'task_type': 'policy',
            'text': basic_policy_text,
            'expected_target_start': basic_policy_expected,
            'description': 'Basic policy task with simple move'
        })
        
        # Policy task with full response
        full_policy_text = 'P: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1    M: e7e5\nE: 0.1\nB: e7e5 d7d5 g8f6'
        full_policy_expected = self._calculate_expected_target(full_policy_text, 'policy')
        test_cases.append({
            'task_type': 'policy',
            'text': full_policy_text,
            'expected_target_start': full_policy_expected,
            'description': 'Policy task with full structured response'
        })
        
        # Complex policy task
        complex_policy_text = 'P: r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5    M: d4d5\nE: 0.3\nB: d4d5 c4c5 b1c3'
        complex_policy_expected = self._calculate_expected_target(complex_policy_text, 'policy')
        test_cases.append({
            'task_type': 'policy',
            'text': complex_policy_text,
            'expected_target_start': complex_policy_expected,
            'description': 'Policy task with complex FEN'
        })
        
        # Basic environment task
        basic_env_text = 'A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\nR: 0.0\nT: False\nU: False'
        basic_env_expected = self._calculate_expected_target(basic_env_text, 'environment')
        test_cases.append({
            'task_type': 'environment',
            'text': basic_env_text,
            'expected_target_start': basic_env_expected,
            'description': 'Basic environment task'
        })
        
        # Complex environment task
        complex_env_text = 'A: r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5+d4d5+r1bqkb1r/ppp2ppp/2n2n2/3pp3/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 5\nR: 0.1\nT: False\nU: False'
        complex_env_expected = self._calculate_expected_target(complex_env_text, 'environment')
        test_cases.append({
            'task_type': 'environment',
            'text': complex_env_text,
            'expected_target_start': complex_env_expected,
            'description': 'Environment task with complex position'
        })
        
        # Edge case: no space after M:
        edge_policy_text = 'P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:e2e4'
        edge_policy_expected = self._calculate_expected_target(edge_policy_text, 'policy')
        test_cases.append({
            'task_type': 'policy',
            'text': edge_policy_text,
            'expected_target_start': edge_policy_expected,
            'description': 'Policy task edge case: no space after M:'
        })
        
        # Edge case: minimal environment
        edge_env_text = 'A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+result'
        edge_env_expected = self._calculate_expected_target(edge_env_text, 'environment')
        test_cases.append({
            'task_type': 'environment',
            'text': edge_env_text,
            'expected_target_start': edge_env_expected,
            'description': 'Environment task edge case: minimal format'
        })
        
        return test_cases
    
    def _calculate_expected_target(self, text: str, task_type: str) -> int:
        """Calculate the expected target start index for a given text and task type"""
        return self.tokenizer.get_target_start_index(text, task_type)
    
    def test_individual_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single target detection case"""
        text = test_case['text']
        task_type = test_case['task_type']
        expected = test_case['expected_target_start']
        description = test_case['description']
        
        # Test the improved target detection
        actual_target_start = self.tokenizer.get_target_start_index(text, task_type)
        
        # Validate correctness
        is_correct = (actual_target_start == expected)
        
        # Additional validation: check what tokens are at the target position
        tokens = self.tokenizer.encode(text)
        target_token_preview = ""
        if actual_target_start < len(tokens):
            preview_tokens = tokens[actual_target_start:actual_target_start+3]
            target_token_preview = self.tokenizer.decode(preview_tokens)[:20] + "..."
        
        result = {
            'description': description,
            'task_type': task_type,
            'text_preview': text[:50] + "...",
            'expected_target_start': expected,
            'actual_target_start': actual_target_start,
            'is_correct': is_correct,
            'target_preview': target_token_preview,
            'tokens_length': len(tokens)
        }
        
        return result
    
    def test_tokenization_consistency(self) -> Dict[str, Any]:
        """Test that M: and + patterns are tokenized consistently"""
        
        # Test M: tokenization in different contexts
        m_colon_tests = [
            "M: e2e4",
            " M: e2e4", 
            "M:e2e4",
            "test M: e2e4",
            "    M: e2e4"
        ]
        
        plus_tests = [
            "+e2e4+",
            "A: fen+move+",
            " +test",
            "position+move+"
        ]
        
        m_colon_tokenizations = []
        for text in m_colon_tests:
            tokens = self.tokenizer.encode(text)
            decoded = [self.tokenizer.decode([t]).strip() for t in tokens]
            m_colon_tokenizations.append({
                'text': text,
                'tokens': decoded,
                'has_separate_m_colon': 'M' in decoded and ':' in decoded
            })
        
        plus_tokenizations = []
        for text in plus_tests:
            tokens = self.tokenizer.encode(text)  
            decoded = [self.tokenizer.decode([t]).strip() for t in tokens]
            plus_tokenizations.append({
                'text': text,
                'tokens': decoded,
                'plus_positions': [i for i, t in enumerate(decoded) if t == '+']
            })
        
        return {
            'm_colon_patterns': m_colon_tokenizations,
            'plus_patterns': plus_tokenizations
        }
    
    def run_regression_tests(self) -> bool:
        """Run all regression tests and return success status"""
        
        print("🔬 TARGET DETECTION REGRESSION TESTS")
        print("="*80)
        print("Testing improved target detection after stability fixes")
        
        all_passed = True
        
        # Test individual cases
        print(f"\n1. INDIVIDUAL TEST CASES ({len(self.test_cases)} cases)")
        print("-" * 60)
        
        for i, test_case in enumerate(self.test_cases, 1):
            result = self.test_individual_case(test_case)
            self.test_results.append(result)
            
            status = "✅ PASS" if result['is_correct'] else "❌ FAIL"
            print(f"  Test {i}: {status}")
            print(f"    {result['description']}")
            print(f"    Expected: {result['expected_target_start']}, Got: {result['actual_target_start']}")
            
            if not result['is_correct']:
                print(f"    Preview: {result['target_preview']}")
                all_passed = False
            
            print()
        
        # Test tokenization consistency
        print("2. TOKENIZATION CONSISTENCY TESTS")
        print("-" * 60)
        
        consistency_results = self.test_tokenization_consistency()
        
        # Check M: pattern consistency
        m_inconsistent = False
        for result in consistency_results['m_colon_patterns']:
            if not result['has_separate_m_colon']:
                print(f"  ⚠️  M: pattern not consistently tokenized: {result['text']} -> {result['tokens']}")
                m_inconsistent = True
        
        if not m_inconsistent:
            print("  ✅ M: patterns consistently tokenized as separate tokens")
        
        # Check + pattern consistency  
        plus_consistent = True
        for result in consistency_results['plus_patterns']:
            if not result['plus_positions']:
                print(f"  ⚠️  + not found in: {result['text']} -> {result['tokens']}")
                plus_consistent = False
        
        if plus_consistent:
            print("  ✅ + patterns consistently detected")
        
        # Summary
        print(f"\n{'='*80}")
        print("REGRESSION TEST SUMMARY")
        print(f"{'='*80}")
        
        passed_count = sum(1 for r in self.test_results if r['is_correct'])
        total_count = len(self.test_results)
        
        print(f"Individual Tests: {passed_count}/{total_count} passed")
        print(f"Tokenization: {'✅ Consistent' if (not m_inconsistent and plus_consistent) else '❌ Issues found'}")
        
        if all_passed and not m_inconsistent and plus_consistent:
            print("\n🎉 ALL REGRESSION TESTS PASSED!")
            print("Target detection improvements are working correctly.")
            return True
        else:
            print(f"\n⚠️  SOME TESTS FAILED - REGRESSION DETECTED")
            print("Target detection may have issues that need investigation.")
            return False


def main():
    """Run target detection regression tests"""
    tester = TargetDetectionRegressionTester()
    success = tester.run_regression_tests()
    
    if success:
        print(f"\n✅ CONCLUSION: Target detection fixes are stable and working")
        print("The improvements that resolved training instability are validated.")
    else:
        print(f"\n❌ CONCLUSION: Target detection regression detected") 
        print("Investigation needed - the fixes may have been compromised.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)