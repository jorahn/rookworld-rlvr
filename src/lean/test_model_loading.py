#!/usr/bin/env python3
"""
Test that the model loads correctly and generates reasonable output
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_direct_load():
    print("Testing direct model load from HuggingFace...")
    
    # Load tokenizer and model directly
    tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    model = GPT2LMHeadModel.from_pretrained("jrahn/RookWorld-LM-124M")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Test P: task
    prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print(f"\nPrompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"First 20 tokens: {input_ids[0, :20].tolist()}")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated}")
    
    # Extract just the completion
    completion = generated[len(prompt):].strip()
    print(f"\nCompletion only: {completion}")
    
    # Check if it looks reasonable (should start with M: for moves)
    if completion.startswith("M:"):
        print("✓ Completion starts with M: as expected for P: task")
    else:
        print("✗ Completion doesn't start with M:")

def test_lean_model():
    print("\n" + "="*60)
    print("Testing lean model wrapper...")
    
    from model import LeanRookWorldModel
    from transformers import GPT2Tokenizer
    
    # Load via lean wrapper
    model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
    tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to_device(device)
    model.eval()
    
    # Test same prompt
    prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print(f"\nPrompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"First 20 tokens: {input_ids[0, :20].tolist()}")
    
    # Generate using lean model's method
    with torch.no_grad():
        generated = model.generate_tokens(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
    
    print(f"Generated shape: {generated.shape}")
    print(f"First 20 generated tokens: {generated[0, :20].tolist()}")
    
    # Decode
    completion = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nCompletion: {completion}")
    
    # Check if reasonable
    if "M:" in completion or any(c in completion for c in "abcdefgh12345678"):
        print("✓ Completion contains chess-like content")
    else:
        print("✗ Completion looks like garbage")

if __name__ == "__main__":
    test_direct_load()
    test_lean_model()