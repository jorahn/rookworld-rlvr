#!/usr/bin/env python3
"""
Debug Token Extraction from Padded Sequences

Investigates the exact token extraction bug that's causing broken completions
like 'f6 0' instead of proper chess analysis.
"""

import torch
import sys
import tiktoken
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.loader import load_rookworld_model


def debug_token_extraction():
    """Debug exactly what tokens are being extracted."""
    print("🔍 Debugging Token Extraction from Padded Sequences")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Simple test prompt
    test_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    prompt_ids = tokenizer.encode(test_prompt, disallowed_special=())
    prompt_len = len(prompt_ids)
    
    print(f"Test prompt: {test_prompt}")
    print(f"Prompt tokens: {prompt_ids}")
    print(f"Prompt length: {prompt_len}")
    print(f"Decoded back: '{tokenizer.decode(prompt_ids)}'")
    
    # Test different padding strategies and extraction methods
    padding_cases = [
        ("No padding", prompt_ids, None, prompt_len),
        ("Left pad +10", [50256] * 10 + prompt_ids, [0] * 10 + [1] * prompt_len, 10 + prompt_len),
        ("Right pad +10", prompt_ids + [50256] * 10, [1] * prompt_len + [0] * 10, prompt_len),
    ]
    
    generation_params = {
        'max_new_tokens': 20,  # Short for debugging
        'temperature': 0.01,
        'top_k': 1,
        'top_p': 1.0,
        'pad_token_id': 50256
    }
    
    for case_name, input_tokens, attention_pattern, extraction_start in padding_cases:
        print(f"\n📋 Case: {case_name}")
        print(f"  Input tokens: {input_tokens[:5]}...{input_tokens[-5:]}")
        print(f"  Input length: {len(input_tokens)}")
        print(f"  Extraction start position: {extraction_start}")
        
        # Create tensors
        input_tensor = torch.tensor([input_tokens], device="cuda")
        attention_mask = None
        if attention_pattern is not None:
            attention_mask = torch.tensor([attention_pattern], device="cuda") 
            print(f"  Attention mask: {attention_pattern[:5]}...{attention_pattern[-5:]}")
        
        # Generate
        torch.manual_seed(42)
        
        try:
            with torch.no_grad():
                if attention_mask is not None:
                    output = model.generate(input_tensor, attention_mask=attention_mask, **generation_params)
                else:
                    output = model.generate(input_tensor, **generation_params)
            
            print(f"  Output shape: {output.shape}")
            print(f"  Output tokens: {output[0].cpu().tolist()}")
            
            # Extract completion using different strategies
            print(f"\n  Token extraction strategies:")
            
            # Strategy 1: From extraction_start position
            completion_tokens_1 = output[0, extraction_start:].cpu().tolist()
            completion_1 = tokenizer.decode(completion_tokens_1)
            print(f"    Strategy 1 (from pos {extraction_start}): {completion_tokens_1[:10]}")
            print(f"    Decoded: '{completion_1[:50]}...'")
            
            # Strategy 2: From original prompt length (ignoring padding)
            completion_tokens_2 = output[0, prompt_len:].cpu().tolist() 
            completion_2 = tokenizer.decode(completion_tokens_2)
            print(f"    Strategy 2 (from pos {prompt_len}): {completion_tokens_2[:10]}")
            print(f"    Decoded: '{completion_2[:50]}...'")
            
            # Strategy 3: From end of input (input_len to output_len)
            completion_tokens_3 = output[0, len(input_tokens):].cpu().tolist()
            completion_3 = tokenizer.decode(completion_tokens_3)
            print(f"    Strategy 3 (from pos {len(input_tokens)}): {completion_tokens_3[:10]}")
            print(f"    Decoded: '{completion_3[:50]}...'")
            
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
    
    print(f"\n💡 Key Questions:")
    print(f"  1. Which extraction strategy gives valid completions?")
    print(f"  2. Are we using the wrong position calculation?")
    print(f"  3. Does padding affect the output sequence structure?")


def test_proper_padding_token():
    """Test different padding tokens to see which works best."""
    print(f"\n🔧 Testing Different Padding Tokens")
    print("=" * 60)
    
    model = load_rookworld_model(device="cuda")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    test_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    prompt_ids = tokenizer.encode(test_prompt, disallowed_special=())
    
    # Test different padding tokens
    padding_tokens = [
        (50256, "EOS token (current)"),
        (0, "Token 0"),
        (50257 - 1, "Last vocab token"),
    ]
    
    generation_params = {
        'max_new_tokens': 20,
        'temperature': 0.01,
        'top_k': 1,
        'top_p': 1.0
    }
    
    # Baseline: no padding
    torch.manual_seed(42)
    baseline_tensor = torch.tensor([prompt_ids], device="cuda")
    with torch.no_grad():
        baseline_output = model.generate(baseline_tensor, pad_token_id=50256, **generation_params)
    baseline_completion = tokenizer.decode(baseline_output[0, len(prompt_ids):].cpu().tolist()).strip()
    
    print(f"Baseline (no padding): '{baseline_completion[:50]}...'")
    
    for pad_token, pad_name in padding_tokens:
        print(f"\n📋 Testing {pad_name} ({pad_token}):")
        
        # Create left-padded version
        padded_ids = [pad_token] * 10 + prompt_ids
        padded_tensor = torch.tensor([padded_ids], device="cuda")
        attention_mask = torch.tensor([[0] * 10 + [1] * len(prompt_ids)], device="cuda")
        
        torch.manual_seed(42)  # Same seed
        
        try:
            generation_params['pad_token_id'] = pad_token
            
            with torch.no_grad():
                padded_output = model.generate(
                    padded_tensor, 
                    attention_mask=attention_mask, 
                    **generation_params
                )
            
            # Extract completion from correct position
            completion_tokens = padded_output[0, len(padded_ids):].cpu().tolist()
            completion = tokenizer.decode(completion_tokens).strip()
            
            print(f"  Generated: '{completion[:50]}...'")
            print(f"  Same as baseline: {completion == baseline_completion}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def debug_huggingface_approach():
    """Check how HuggingFace handles batched generation."""
    print(f"\n📚 Investigating Proper Batched Generation Approach")
    print("=" * 60)
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        # Load HuggingFace version for comparison
        hf_model = GPT2LMHeadModel.from_pretrained("jrahn/RookWorld-LM-124M")
        hf_tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
        hf_model = hf_model.to("cuda")
        hf_model.eval()
        
        # Set pad token
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        
        test_prompts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+"
        ]
        
        print(f"Testing HuggingFace approach with {len(test_prompts)} prompts:")
        
        # Tokenize with padding
        encoded = hf_tokenizer(
            test_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=False
        )
        
        input_ids = encoded['input_ids'].to("cuda")
        attention_mask = encoded['attention_mask'].to("cuda")
        
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Attention shape: {attention_mask.shape}")
        print(f"  Pad token ID: {hf_tokenizer.pad_token_id}")
        
        # Generate with HuggingFace
        torch.manual_seed(42)
        
        with torch.no_grad():
            hf_outputs = hf_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=hf_tokenizer.pad_token_id
            )
        
        print(f"  Output shape: {hf_outputs.shape}")
        
        # Decode outputs
        for i in range(len(test_prompts)):
            # Extract only the new tokens (after original prompt)
            original_length = attention_mask[i].sum().item()  # Length of non-padded prompt
            generated_tokens = hf_outputs[i, original_length:].cpu().tolist()
            completion = hf_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"  Prompt {i+1}: '{completion[:50]}...'")
        
        print(f"\n💡 HuggingFace approach works - study their implementation")
        return True
        
    except ImportError:
        print(f"  HuggingFace transformers not available for comparison")
        return False
    except Exception as e:
        print(f"  ❌ HuggingFace test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Debug Batched Generation Implementation Issues")
    print("=" * 80)
    
    try:
        # Debug 1: Token extraction issues
        debug_token_extraction()
        
        # Debug 2: Padding token issues  
        test_proper_padding_token()
        
        # Debug 3: Reference implementation
        hf_works = debug_huggingface_approach()
        
        print(f"\n" + "=" * 80)
        print(f"🎯 DEBUGGING SUMMARY")
        print(f"The dramatic quality loss indicates implementation bugs:")
        print(f"  1. Token extraction from wrong positions")
        print(f"  2. Incorrect padding strategy") 
        print(f"  3. Attention mask not working as expected")
        print(f"\nFix these implementation issues to achieve proper batched generation.")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        raise