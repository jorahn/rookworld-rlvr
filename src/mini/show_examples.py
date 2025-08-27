"""
Show clear examples of model generation
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from transformers import GPT2Tokenizer
from dataset import load_and_prepare_samples
from loader import load_rookworld_model

def show_examples():
    # Load model and tokenizer
    print("Loading model...")
    model = load_rookworld_model(device='cuda')
    tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load samples
    samples = load_and_prepare_samples(n_samples=20, seed=42)
    
    # Separate P and A tasks
    p_samples = [s for s in samples if s[0] == 'P'][:3]
    a_samples = [s for s in samples if s[0] == 'A'][:3]
    
    print("\n" + "="*80)
    print("P: TASK EXAMPLES (Policy/Chain-of-Thought)")
    print("="*80)
    
    for i, (task_type, prompt, ground_truth, _) in enumerate(p_samples, 1):
        print(f"\nExample {i}:")
        print("-" * 40)
        print(f"PROMPT:\n{prompt}\n")
        print(f"GROUND TRUTH COMPLETION:\n{ground_truth}\n")
        
        # Generate
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device='cuda')
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=144,  # Full schema
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full_text = tokenizer.decode(generated[0].cpu().tolist())
        completion = full_text[len(prompt):].strip()
        
        # Clean up endoftext tokens for display
        completion = completion.replace('<|endoftext|>', '[END]')
        
        print(f"GENERATED COMPLETION:\n{completion}\n")
        
        # Analyze structure
        has_m = "M:" in completion
        has_e = "E:" in completion
        has_b = "B:" in completion
        print(f"Structure check: M:{has_m} E:{has_e} B:{has_b}")
    
    print("\n" + "="*80)
    print("A: TASK EXAMPLES (Environment/State Transition)")
    print("="*80)
    
    for i, (task_type, prompt, ground_truth, _) in enumerate(a_samples, 1):
        print(f"\nExample {i}:")
        print("-" * 40)
        print(f"PROMPT:\n{prompt}\n")
        print(f"GROUND TRUTH COMPLETION:\n{ground_truth}\n")
        
        # Generate
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device='cuda')
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=144,  # Full schema
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full_text = tokenizer.decode(generated[0].cpu().tolist())
        completion = full_text[len(prompt):].strip()
        
        # Clean up for display
        completion = completion.replace('<|endoftext|>', '[END]')
        
        print(f"GENERATED COMPLETION:\n{completion}\n")
        
        # Analyze structure (should be FEN+reward+terminated+truncated)
        plus_count = completion.count('+')
        print(f"Structure check: Found {plus_count} '+' delimiters (need 3)")
        
        # Try to parse the fields
        if '+' in completion:
            parts = completion.split('+')
            print(f"Fields detected: {len(parts)} parts")
            if len(parts) >= 1:
                print(f"  Field 1 (FEN): {parts[0][:50]}...")
            if len(parts) >= 2:
                print(f"  Field 2 (reward): {parts[1]}")
            if len(parts) >= 3:
                print(f"  Field 3 (terminated): {parts[2]}")
            if len(parts) >= 4:
                print(f"  Field 4 (truncated): {parts[3][:20]}")

if __name__ == "__main__":
    show_examples()