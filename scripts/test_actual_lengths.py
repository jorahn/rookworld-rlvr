#!/usr/bin/env python3
"""
Test actual sequence lengths during generation
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("=== Actual Sequence Length Test ===")
    
    from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
    
    tokenizer = TokenizerBridge()
    
    # Test typical chess prompts
    prompts = [
        "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:",
        "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 + e2e4    ",
        "P: r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5    M:",
    ]
    
    print("Prompt lengths:")
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        print(f"  Prompt {i+1}: {len(tokens)} tokens")
        print(f"    Text: {prompt[:60]}...")
    
    print()
    
    # Test what happens with max_positions=200 vs longer sequences
    max_positions = 200
    max_new_tokens = 144
    
    print(f"Configuration: max_positions={max_positions}, max_new_tokens={max_new_tokens}")
    
    # Simulate sequences that grow during generation
    typical_prompt_len = 30
    max_generated_len = max_new_tokens
    total_expected = typical_prompt_len + max_generated_len
    print(f"Expected total length: {typical_prompt_len} + {max_generated_len} = {total_expected}")
    
    if total_expected > max_positions:
        print(f"⚠️ PROBLEM: Expected length {total_expected} > max_positions {max_positions}")
        print("This could cause sequences to exceed configured limits!")
    
    # Test batch padding behavior
    print("\nBatch padding test:")
    # Simulate different sequence lengths in a batch
    seq_lengths = [50, 120, 180, 174, 200]  # Various lengths
    batch_size = 4
    vocab_size = 50257
    
    for max_len in seq_lengths:
        # Calculate memory for logits if all sequences padded to this length
        logits_memory = batch_size * max_len * vocab_size * 4 / 1024**3
        print(f"  Max length {max_len}: logits would be {logits_memory:.2f}GB")
        
        if logits_memory > 20:
            print(f"    ⚠️ This would cause 20GB+ memory usage!")
    
    # Test what happens with model's n_positions vs config max_positions
    print(f"\nModel config check:")
    print(f"  Training max_positions: {max_positions}")
    print(f"  Model n_positions: 1024 (from model config)")
    
    if max_positions > 200:
        print(f"⚠️ If sequences are padded to model n_positions (1024):")
        huge_memory = batch_size * 1024 * vocab_size * 4 / 1024**3
        print(f"   Logits memory would be: {huge_memory:.2f}GB")
        print(f"   This explains the 22GB memory usage!")

if __name__ == "__main__":
    main()