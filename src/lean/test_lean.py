#!/usr/bin/env python3
"""
Test script for lean GRPO implementation

Quick validation that all components work together.
"""

import sys
import logging
import torch
from transformers import GPT2Tokenizer

# Import lean components
from model import LeanRookWorldModel
from dataset import LeanRookWorldDataset
from validation import LeanValidator
from grpo import LeanGRPOTrainer


def test_model_loading():
    """Test model loading and GPU placement"""
    print("\\n=== Testing Model Loading ===")
    
    try:
        # Load models
        tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
        
        if torch.cuda.is_available():
            model.to_device("cuda:0")
            print(f"✓ Model loaded on cuda:0")
        else:
            print("⚠ CUDA not available, using CPU")
        
        # Test generation
        test_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = model.generate_tokens(inputs["input_ids"], max_new_tokens=10)
        
        print(f"✓ Generation test passed - generated shape: {generated.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("\\n=== Testing Dataset ===")
    
    try:
        dataset = LeanRookWorldDataset()
        dataset.load()
        
        # Get a small batch
        batch = dataset.get_training_batch(2)
        
        print(f"✓ Dataset loaded - batch size: {len(batch)}")
        
        for i, (task_type, prompt, completion) in enumerate(batch):
            print(f"  Sample {i+1} ({task_type}): {prompt[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_validator():
    """Test validation system"""
    print("\\n=== Testing Validator ===")
    
    try:
        validator = LeanValidator()
        
        # Test P: task validation
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        test_completion = "M: e2e4 d2d4 Nf3 c2c4 Nb1c3  E: 0.2 0.1 0.3 0.15 0.25  B: e2e4"
        
        rewards = validator.validate_policy_completion(test_fen, test_completion)
        print(f"✓ P: task validation - rewards: {rewards}")
        
        # Test A: task validation  
        test_move = "e2e4"
        test_history = ""  # Empty history for initial moves
        test_env_completion = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
        
        env_rewards = validator.validate_environment_completion(test_fen, test_move, test_history, test_env_completion)
        print(f"✓ A: task validation - rewards: {env_rewards}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validator test failed: {e}")
        return False


def test_grpo_setup():
    """Test GRPO trainer setup"""
    print("\\n=== Testing GRPO Setup ===")
    
    try:
        # Setup models
        tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
        ref_model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
        
        if torch.cuda.is_available():
            train_model.to_device("cuda:0")
            ref_model.to_device("cuda:0")  # Use same GPU for test
        
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # Setup trainer
        trainer = LeanGRPOTrainer(
            model=train_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            group_size=2,
            learning_rate=1e-5
        )
        
        print(f"✓ GRPO trainer setup successful")
        print(f"  Group size: {trainer.group_size}")
        print(f"  Clip range: {trainer.clip_range}")
        print(f"  KL coef: {trainer.kl_coef}")
        
        return True
        
    except Exception as e:
        print(f"✗ GRPO setup test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Lean GRPO Implementation Test Suite")
    print("==================================")
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Dataset", test_dataset),
        ("Validator", test_validator),
        ("GRPO Setup", test_grpo_setup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\\nPassed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\\n🎉 All tests passed! Lean implementation is ready.")
        return 0
    else:
        print(f"\\n❌ {len(tests) - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)