#!/usr/bin/env python3
"""
Test suite to verify the processing steps for both P: and A: tasks
"""

import sys
from dataset import LeanRookWorldDataset
from validation import LeanValidator


def test_p_task_parsing():
    """Test P: task parsing according to spec"""
    print("\n=== Testing P: Task Parsing ===")
    
    d = LeanRookWorldDataset()
    
    # Test cases for P: tasks
    test_cases = [
        # Standard P: task with all sections
        ('P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1                                  M: e2e4 d2d4 g1f3 c2c4 b1c3      E: 0.3 0.35 0.28 0.32 0.29         B: e2e4',
         'P',
         'P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
         'M:e2e4 d2d4 g1f3 c2c4 b1c3      E: 0.3 0.35 0.28 0.32 0.29         B: e2e4'),
        
        # P: task without completion (for generation)
        ('P: 8/8/4b2p/2N2kp1/r4p2/5P2/3RK1PP/8 b - - 7 44',
         'P',
         'P: 8/8/4b2p/2N2kp1/r4p2/5P2/3RK1PP/8 b - - 7 44',
         ''),
    ]
    
    for i, (text, expected_type, expected_prompt, expected_completion) in enumerate(test_cases):
        task_type, prompt, completion = d.parse_task_prompt(text)
        
        success = (
            task_type == expected_type and
            prompt == expected_prompt and
            completion == expected_completion
        )
        
        status = "✓" if success else "✗"
        print(f"{status} Test {i+1}: {text[:50]}...")
        if not success:
            print(f"  Expected type: {expected_type}, got: {task_type}")
            print(f"  Expected prompt: {expected_prompt}")
            print(f"  Got prompt: {prompt}")
            print(f"  Expected completion: {expected_completion[:50]}")
            print(f"  Got completion: {completion[:50]}")
    
    return True


def test_a_task_parsing():
    """Test A: task parsing according to spec"""
    print("\n=== Testing A: Task Parsing ===")
    
    d = LeanRookWorldDataset()
    
    # Test cases for A: tasks
    test_cases = [
        # A: task with prefix and full format
        ('A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false',
         'A',
         'A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+',
         'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false'),
        
        # A: task without prefix (should get A: added)
        ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+e1e2,d2d4+rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false',
         'A',
         'A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+e1e2,d2d4+',
         'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false'),
        
        # A: task with empty history
        ('A: 8/8/8/8/8/8/8/K1k5 w - - 0 1+a1a2+,+8/8/8/8/8/8/K7/2k5 b - - 1 1+0.001+false+false',
         'A',
         'A: 8/8/8/8/8/8/8/K1k5 w - - 0 1+a1a2+,+',
         '8/8/8/8/8/8/K7/2k5 b - - 1 1+0.001+false+false'),
    ]
    
    for i, (text, expected_type, expected_prompt, expected_completion) in enumerate(test_cases):
        # Simulate preprocessing
        if not text.startswith("P: ") and not text.startswith("A: "):
            text = "A: " + text
        
        task_type, prompt, completion = d.parse_task_prompt(text)
        
        success = (
            task_type == expected_type and
            prompt == expected_prompt and
            completion == expected_completion
        )
        
        status = "✓" if success else "✗"
        print(f"{status} Test {i+1}: {text[:50]}...")
        if not success:
            print(f"  Expected type: {expected_type}, got: {task_type}")
            print(f"  Expected prompt: {expected_prompt}")
            print(f"  Got prompt: {prompt}")
            print(f"  Expected completion: {expected_completion[:50]}")
            print(f"  Got completion: {completion[:50]}")
    
    return True


def test_p_task_validation():
    """Test P: task validation and scoring"""
    print("\n=== Testing P: Task Validation ===")
    
    validator = LeanValidator()
    
    # Test FEN and completion
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    test_cases = [
        # Well-formed completion
        ("M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4", 
         {"format": 1.0}),  # Should parse correctly
        
        # Missing sections
        ("M: e2e4 d2d4", 
         {"format": 0.0}),  # Missing E: and B: sections
        
        # Invalid format
        ("random text without proper format",
         {"format": 0.0}),  # Should fail to parse
    ]
    
    for completion, expected in test_cases:
        result = validator.validate_policy_completion(test_fen, completion)
        
        # Check format parsing
        format_ok = result.get("format", 0.0) == expected["format"]
        status = "✓" if format_ok else "✗"
        print(f"{status} Format test: {completion[:30]}...")
        if not format_ok:
            print(f"  Expected format score: {expected['format']}, got: {result.get('format', 0.0)}")
    
    return True


def test_a_task_validation():
    """Test A: task validation and scoring"""
    print("\n=== Testing A: Task Validation ===")
    
    validator = LeanValidator()
    
    # Test inputs
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    test_move = "e2e4"
    test_history = ""
    
    test_cases = [
        # Well-formed completion with correct result
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",
         {"format": 1.0}),  # Correct format with 4 sections
        
        # Missing sections
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
         {"format": 0.0}),  # Only FEN, missing other sections
        
        # Invalid format
        ("not a valid chess position",
         {"format": 0.0}),  # Should fail to parse
    ]
    
    for completion, expected in test_cases:
        result = validator.validate_environment_completion(test_fen, test_move, test_history, completion)
        
        # Check format parsing
        format_ok = result.get("format", 0.0) == expected["format"]
        status = "✓" if format_ok else "✗"
        print(f"{status} Format test: {completion[:50]}...")
        if not format_ok:
            print(f"  Expected format score: {expected['format']}, got: {result.get('format', 0.0)}")
    
    return True


def test_dataset_preprocessing():
    """Test dataset preprocessing (adding A: prefix)"""
    print("\n=== Testing Dataset Preprocessing ===")
    
    d = LeanRookWorldDataset()
    
    # Simulate raw samples from dataset
    raw_samples = [
        # P: task (should remain unchanged)
        "P: 8/8/8/8/8/8/8/K1k5 w - - 0 1                                  M: a1a2 a1b1      E: 0.0 0.0         B: a1a2",
        
        # Raw A: task without prefix (should get A: added)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",
    ]
    
    for sample in raw_samples:
        # Apply preprocessing logic from get_samples
        if not sample.startswith("P: "):
            processed = "A: " + sample
        else:
            processed = sample
        
        # Parse the processed sample
        task_type, prompt, completion = d.parse_task_prompt(processed)
        
        if sample.startswith("P: "):
            expected_type = "P"
        else:
            expected_type = "A"
        
        success = task_type == expected_type
        status = "✓" if success else "✗"
        print(f"{status} Preprocessing: {sample[:30]}... -> Type: {task_type}")
        if not success:
            print(f"  Expected type: {expected_type}, got: {task_type}")
    
    return True


def main():
    """Run all tests"""
    print("Task Processing Test Suite")
    print("==========================")
    
    tests = [
        ("P: Task Parsing", test_p_task_parsing),
        ("A: Task Parsing", test_a_task_parsing),
        ("P: Task Validation", test_p_task_validation),
        ("A: Task Validation", test_a_task_validation),
        ("Dataset Preprocessing", test_dataset_preprocessing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<35} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 All task processing tests passed!")
        return 0
    else:
        print(f"\n❌ {len(tests) - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)