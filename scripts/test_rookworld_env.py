#!/usr/bin/env python3

"""Test RookWorld model understanding of environment tasks"""

import sys
sys.path.insert(0, 'src')

from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.loader import load_rookworld_model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
import torch
import chess

def main():
    # Initialize model
    print("Loading RookWorld-LM model...")
    model = load_rookworld_model("jrahn/RookWorld-LM-124M")
    model.cpu()
    model.eval()
    
    tokenizer = TokenizerBridge("gpt2")
    
    # Test multiple environment task examples
    test_cases = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5"),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "g1f3"),
    ]
    
    for i, (fen, move) in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}: {move} in position")
        print(f"{'='*60}")
        
        # Create proper environment prompt
        prompt = f"A: {fen}+{move}+"
        print(f"Testing prompt: {prompt}")
        
        # Tokenize
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])
        
        # Generate with more tokens and check for EOT
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=128,  # Increased to allow full completion
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Check if generation was truncated or ended naturally
        generated_tokens = outputs[0, len(tokens):].tolist()
        generated_text = tokenizer.decode(generated_tokens)
        
        # Check for EOT token
        eos_token_id = tokenizer.eos_token_id
        ended_with_eos = len(generated_tokens) > 0 and generated_tokens[-1] == eos_token_id
        was_truncated = len(generated_tokens) == 128  # Hit max_new_tokens limit
        
        full_response = prompt + generated_text
        
        print(f"Generated: '{generated_text}'")
        print(f"Generated tokens: {len(generated_tokens)}")
        print(f"Ended with EOS token: {ended_with_eos}")
        print(f"Was truncated (hit max tokens): {was_truncated}")
        print(f"Full response: '{full_response}'")
        
        # Test what expected format should be
        from rookworld_rlvr.environment.chess_env import ChessEnvironment
        chess_env = ChessEnvironment()
        expected = chess_env.apply_move(fen, move)
        print(f"Expected format: {expected.to_structured_string()}")
        
        # Try parsing the model's response
        parsed = chess_env.parse_prediction(full_response)
        print(f"Parsed successfully: {parsed is not None}")
        if parsed:
            print(f"Parsed result: {parsed}")
        else:
            print("Failed to parse - likely incomplete generation")

if __name__ == "__main__":
    main()