#!/usr/bin/env python3
"""
Padding Generation Issue Investigation

Investigates why padding affects generation quality even with attention masks.
This is the core issue preventing effective batch generation.
"""

import torch
import sys
import tiktoken
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.loader import load_rookworld_model


def investigate_padding_effects():
    """Investigate exactly how padding affects generation."""
    print("🔍 Investigating Padding Effects on Generation")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    test_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    prompt_ids = tokenizer.encode(test_prompt, disallowed_special=())
    prompt_len = len(prompt_ids)
    
    print(f"Test prompt: {test_prompt}")
    print(f"Prompt length: {prompt_len} tokens")
    
    generation_params = {
        'max_new_tokens': 50,  # Shorter for detailed analysis
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    # Test different padding scenarios
    test_cases = [
        ("No padding", prompt_ids, None),
        ("Left padding +10", [50256] * 10 + prompt_ids, [0] * 10 + [1] * prompt_len),
        ("Right padding +10", prompt_ids + [50256] * 10, [1] * prompt_len + [0] * 10),
        ("Left padding +50", [50256] * 50 + prompt_ids, [0] * 50 + [1] * prompt_len),
    ]
    
    results = {}
    
    for case_name, token_ids, attention_pattern in test_cases:
        print(f"\n📋 {case_name}:")
        
        # Create tensor
        input_tensor = torch.tensor([token_ids], device="cuda")
        attention_mask = None
        if attention_pattern is not None:
            attention_mask = torch.tensor([attention_pattern], device="cuda")
        
        print(f"  Input shape: {input_tensor.shape}")
        if attention_mask is not None:
            print(f"  Attention mask: {attention_mask.shape}, active tokens: {attention_mask.sum().item()}")
        
        # Generate with fixed seed
        torch.manual_seed(42)
        
        try:
            with torch.no_grad():
                if attention_mask is not None:
                    output = model.generate(input_tensor, attention_mask=attention_mask, **generation_params)
                else:
                    output = model.generate(input_tensor, **generation_params)
            
            # Extract completion (skip original prompt + any padding)
            if case_name == "No padding":
                completion_tokens = output[0, prompt_len:].cpu().tolist()
            elif case_name.startswith("Left padding"):
                padding_len = len(token_ids) - prompt_len
                completion_tokens = output[0, len(token_ids):].cpu().tolist()
            else:  # Right padding
                completion_tokens = output[0, prompt_len:prompt_len+generation_params['max_new_tokens']].cpu().tolist()
            
            completion = tokenizer.decode(completion_tokens).strip()
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            results[case_name] = {
                'completion': completion,
                'completion_tokens': completion_tokens,
                'length': len(completion_tokens)
            }
            
            print(f"  Generated: '{completion[:50]}...'")
            print(f"  Length: {len(completion_tokens)} tokens")
            
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            results[case_name] = None
    
    # Compare results
    print(f"\n🔍 Comparing Generation Consistency:")
    baseline = results.get("No padding")
    if baseline is None:
        print("  ❌ No baseline to compare against")
        return False
    
    all_consistent = True
    for case_name, result in results.items():
        if case_name == "No padding" or result is None:
            continue
        
        # Compare with baseline
        same_completion = result['completion'] == baseline['completion']
        same_tokens = result['completion_tokens'] == baseline['completion_tokens']
        
        print(f"  {case_name}:")
        print(f"    Same completion text: {same_completion}")
        print(f"    Same tokens: {same_tokens}")
        
        if not same_completion:
            print(f"    Baseline: '{baseline['completion'][:40]}...'")
            print(f"    This case: '{result['completion'][:40]}...'")
            all_consistent = False
    
    if all_consistent:
        print(f"✅ Padding doesn't affect generation (attention masks work correctly)")
    else:
        print(f"❌ Padding affects generation (this is the root cause!)")
    
    return all_consistent


def test_batch_size_effects():
    """Test if batch size itself affects individual generation quality."""
    print(f"\n🔢 Testing Batch Size Effects")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    test_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    prompt_ids = tokenizer.encode(test_prompt, disallowed_special=())
    prompt_tensor = torch.tensor([prompt_ids], device="cuda")
    
    generation_params = {
        'max_new_tokens': 50,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    batch_sizes = [1, 2, 4, 8]
    
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Each batch contains identical prompts")
    
    for batch_size in batch_sizes:
        print(f"\n📦 Batch size {batch_size}:")
        
        # Create batch of identical prompts
        batch_prompts = prompt_tensor.repeat(batch_size, 1)
        
        torch.manual_seed(42)  # Same seed for all tests
        
        try:
            with torch.no_grad():
                outputs = model.generate(batch_prompts, **generation_params)
            
            # Check if all outputs in batch are identical (they should be with same prompt/seed)
            first_completion = tokenizer.decode(outputs[0, len(prompt_ids):].cpu().tolist()).strip()
            
            all_identical = True
            for i in range(1, batch_size):
                completion_i = tokenizer.decode(outputs[i, len(prompt_ids):].cpu().tolist()).strip()
                if completion_i != first_completion:
                    all_identical = False
                    break
            
            print(f"  All outputs identical: {all_identical}")
            print(f"  Output: '{first_completion[:50]}...'")
            
            if not all_identical:
                print(f"  ⚠️ Batch size {batch_size} produces different outputs for identical prompts")
                # Show differences
                for i in range(min(batch_size, 3)):
                    comp = tokenizer.decode(outputs[i, len(prompt_ids):].cpu().tolist()).strip()
                    print(f"    Position {i}: '{comp[:30]}...'")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n💡 Batch size effects analysis complete")


def test_generation_method_alternatives():
    """Test alternative approaches to batch generation."""
    print(f"\n🛠️ Testing Alternative Generation Approaches")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    from rookworld_rlvr.dataset import load_and_prepare_samples
    samples = load_and_prepare_samples(n_samples=10, seed=42)
    
    # Get one P: and one A: task for testing
    p_sample = next(s for s in samples if s[0] == 'P')
    a_sample = next(s for s in samples if s[0] == 'A')
    
    test_cases = [
        ("P: task", p_sample[1]),
        ("A: task", a_sample[1])
    ]
    
    generation_params = {
        'max_new_tokens': 100,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.95,
        'pad_token_id': 50256
    }
    
    for task_name, prompt in test_cases:
        print(f"\n📋 {task_name}: {prompt[:40]}...")
        
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        
        # Method 1: Sequential (baseline)
        torch.manual_seed(42)
        sequential_completions = []
        
        for k in range(4):
            torch.manual_seed(42 + k)  # Different seeds for diversity
            prompt_tensor = torch.tensor([prompt_ids], device="cuda")
            
            with torch.no_grad():
                output = model.generate(prompt_tensor, **generation_params)
            
            completion = tokenizer.decode(output[0, len(prompt_ids):].cpu().tolist()).strip()
            sequential_completions.append(completion)
        
        # Method 2: Batch with same-length prompts only (avoid padding)
        if len(prompt_ids) < 100:  # Only test if reasonable length
            torch.manual_seed(42)
            batch_prompts = torch.tensor([prompt_ids], device="cuda").repeat(4, 1)
            
            with torch.no_grad():
                batch_outputs = model.generate(batch_prompts, **generation_params)
            
            batch_completions = []
            for k in range(4):
                completion = tokenizer.decode(batch_outputs[k, len(prompt_ids):].cpu().tolist()).strip()
                batch_completions.append(completion)
            
            print(f"  Sequential samples (diverse seeds):")
            for k, comp in enumerate(sequential_completions):
                print(f"    K={k+1}: '{comp[:40]}...'")
            
            print(f"  Batched samples (single seed):")
            for k, comp in enumerate(batch_completions):
                print(f"    K={k+1}: '{comp[:40]}...'")
            
            # Check if batched samples are more similar (expected with single seed)
            batch_unique = len(set(batch_completions))
            seq_unique = len(set(sequential_completions))
            
            print(f"  Diversity comparison:")
            print(f"    Sequential unique: {seq_unique}/4")
            print(f"    Batched unique: {batch_unique}/4")
    
    print(f"\n💡 Alternative approaches analysis complete")


if __name__ == "__main__":
    print("🚀 Generation Padding Issue Investigation")
    print("=" * 80)
    
    try:
        # Test 1: Understand padding effects
        padding_consistent = investigate_padding_effects()
        
        # Test 2: Batch size effects
        test_batch_size_effects()
        
        # Test 3: Alternative approaches
        test_generation_method_alternatives()
        
        print(f"\n" + "=" * 80)
        print(f"🎯 KEY FINDINGS")
        print(f"  Padding affects generation: {'❌ YES' if not padding_consistent else '✅ NO'}")
        print(f"\n💡 IMPLICATIONS:")
        if not padding_consistent:
            print(f"  - Naive batch generation with mixed lengths will fail")
            print(f"  - Need alternative approach: same-length batching or better masking")
            print(f"  - Attention masks alone are insufficient for generation isolation")
        else:
            print(f"  - Padding issues may be elsewhere in implementation")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        raise