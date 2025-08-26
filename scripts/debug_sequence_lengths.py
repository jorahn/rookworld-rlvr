#!/usr/bin/env python3
"""
Debug actual sequence lengths being used during training
"""

import torch
import logging
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_memory(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"{stage}: {allocated:.2f}GB")

def main():
    print("=== Sequence Length Analysis ===")
    
    # Clean start
    torch.cuda.empty_cache()
    print_memory("Start")
    
    # Import modules
    from rookworld_rlvr.train.config import GRPOConfig
    from rookworld_rlvr.model.loader import load_rookworld_model
    from rookworld_rlvr.train.policy import CausalLMPolicy
    
    # Create config with suspected problematic values
    config = GRPOConfig(
        batch_positions=2,
        group_size=4,
        steps=5,
        max_new_tokens=144,
        max_new_tokens_env=150,
        max_positions=200
    )
    
    print(f"Config: max_positions={config.max_positions}, max_new_tokens={config.max_new_tokens}")
    
    # Load models
    training_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
    training_model = training_model.to("cuda:0")
    print_memory("Training model loaded")
    
    if torch.cuda.device_count() > 1:
        reference_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
        reference_model = reference_model.to("cuda:1")
        print_memory("Reference model loaded")
    else:
        reference_model = training_model
    
    # Create policy
    policy = CausalLMPolicy(training_model, reference_model, config)
    print_memory("Policy created")
    
    # Test actual chess prompt lengths
    from rookworld_rlvr.tokenization.chess_tokenizer import ChessTokenizer
    tokenizer = ChessTokenizer()
    
    # Test typical chess positions
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5",  # Middlegame
    ]
    
    for i, fen in enumerate(test_positions):
        # Test Policy prompt
        policy_prompt = tokenizer.create_chess_prompts([fen], "policy")[0]
        policy_tokens = tokenizer.encode_batch([policy_prompt])
        policy_length = policy_tokens["lengths"][0]
        print(f"Position {i+1} Policy prompt: {policy_length} tokens")
        print(f"  Text: {policy_prompt[:100]}...")
        
        # Test Environment prompt  
        env_prompt = tokenizer.create_chess_prompts([fen + " e2e4"], "environment")[0]
        env_tokens = tokenizer.encode_batch([env_prompt])
        env_length = env_tokens["lengths"][0]
        print(f"Position {i+1} Env prompt: {env_length} tokens")
        print(f"  Text: {env_prompt[:100]}...")
        
        # Calculate total sequence length with generation
        max_total_policy = policy_length + config.max_new_tokens
        max_total_env = env_length + config.max_new_tokens_env
        print(f"  Max total policy sequence: {max_total_policy} tokens")
        print(f"  Max total env sequence: {max_total_env} tokens")
        
        # Calculate memory for logits tensor
        batch_size = config.batch_positions * config.group_size  # 2 * 4 = 8
        vocab_size = training_model.config.vocab_size  # 50257
        
        # Memory for logits in FP32 (4 bytes per element)
        policy_logits_memory = batch_size * max_total_policy * vocab_size * 4 / 1024**3
        env_logits_memory = batch_size * max_total_env * vocab_size * 4 / 1024**3
        
        print(f"  Logits memory (policy): {policy_logits_memory:.2f}GB")
        print(f"  Logits memory (env): {env_logits_memory:.2f}GB")
        print()
    
    # Test what happens with a realistic forward pass
    print("Testing forward pass memory...")
    batch_size = config.batch_positions * config.group_size  # 8
    realistic_seq_len = 50 + 100  # 50 token prompt + 100 generated tokens
    
    input_ids = torch.randint(0, 1000, (batch_size, realistic_seq_len), device="cuda:0")
    
    print_memory("Before forward pass")
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = training_model.forward(input_ids)
        logits = outputs["logits"]
        print(f"Logits shape: {logits.shape}")
        logits_memory = logits.numel() * 4 / 1024**3  # FP32
        print(f"Logits memory: {logits_memory:.2f}GB")
    
    print_memory("After forward pass")
    
    # Test with max possible sequence length
    print("\nTesting max sequence length...")
    max_input_ids = torch.randint(0, 1000, (batch_size, config.max_positions), device="cuda:0")
    
    try:
        with torch.no_grad():
            max_outputs = training_model.forward(max_input_ids)
            max_logits = max_outputs["logits"]
            max_logits_memory = max_logits.numel() * 4 / 1024**3
            print(f"Max logits shape: {max_logits.shape}")
            print(f"Max logits memory: {max_logits_memory:.2f}GB")
    except Exception as e:
        print(f"Max sequence failed: {e}")
    
    print_memory("Final")

if __name__ == "__main__":
    main()