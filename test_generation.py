#!/usr/bin/env python3
"""
Quick test script to see what RookWorld-LM generates
"""

import torch
from src.rookworld_rlvr.model.loader import load_pretrained_model
from src.rookworld_rlvr.train.policy import CausalLMPolicy
from src.rookworld_rlvr.train.config import GRPOConfig

def test_generation():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = GRPOConfig(device=device)
    
    # Load model
    model = load_pretrained_model(
        'jrahn/RookWorld-LM-124M',
        device=device
    )
    model = model.to(device)  # Ensure on correct device
    
    # Create policy (no reference model needed for generation)
    policy = CausalLMPolicy(model, None, config, device=device)
    
    # Test policy task
    print("=== Policy (P:) Task Test ===")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
    prompt = f"P: {fen}    M:"
    
    print(f"Prompt: {prompt}")
    
    # Generate 3 samples
    for i in range(3):
        generated = policy.generate(prompt, max_new_tokens=50)
        print(f"Sample {i+1}: '{generated}'")
    
    print("\n=== Environment (A:) Task Test ===")
    prompt_env = f"A: {fen}+e2e4+"
    print(f"Prompt: {prompt_env}")
    
    # Generate 3 samples  
    for i in range(3):
        generated = policy.generate(prompt_env, max_new_tokens=32)
        print(f"Sample {i+1}: '{generated}'")

if __name__ == "__main__":
    test_generation()