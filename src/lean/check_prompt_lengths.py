#!/usr/bin/env python3
"""
Check prompt lengths in the dataset
"""

from dataset import LeanRookWorldDataset
from transformers import GPT2Tokenizer

def main():
    # Load dataset and tokenizer
    dataset = LeanRookWorldDataset()
    dataset.load()
    
    tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get a batch of samples
    batch = dataset.get_training_batch(20)
    
    print("Analyzing prompt lengths...")
    print("="*60)
    
    p_lengths = []
    a_lengths = []
    
    for task_type, prompt, _ in batch:
        tokens = tokenizer.encode(prompt)
        length = len(tokens)
        
        if task_type == "P":
            p_lengths.append(length)
            if length < 50:  # Short P: task
                print(f"Short P: task ({length} tokens): {prompt[:100]}")
        else:
            a_lengths.append(length)
            if length > 100:  # Long A: task
                print(f"Long A: task ({length} tokens): {prompt[:100]}")
    
    print("\n" + "="*60)
    print("Statistics:")
    
    if p_lengths:
        print(f"\nP: tasks ({len(p_lengths)} samples):")
        print(f"  Min length: {min(p_lengths)} tokens")
        print(f"  Max length: {max(p_lengths)} tokens")
        print(f"  Avg length: {sum(p_lengths)/len(p_lengths):.1f} tokens")
    
    if a_lengths:
        print(f"\nA: tasks ({len(a_lengths)} samples):")
        print(f"  Min length: {min(a_lengths)} tokens")
        print(f"  Max length: {max(a_lengths)} tokens") 
        print(f"  Avg length: {sum(a_lengths)/len(a_lengths):.1f} tokens")
    
    # Check what happens with batching
    print("\n" + "="*60)
    print("Batch padding analysis:")
    
    prompts = [prompt for _, prompt, _ in batch[:4]]
    print(f"\nTokenizing batch of {len(prompts)} prompts...")
    
    # Individual lengths
    individual_lengths = [len(tokenizer.encode(p)) for p in prompts]
    print(f"Individual lengths: {individual_lengths}")
    print(f"Max length in batch: {max(individual_lengths)}")
    
    # Batched with padding
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    print(f"Padded batch shape: {inputs['input_ids'].shape}")
    
    # Check padding per sequence
    for i in range(len(prompts)):
        pad_count = (inputs['input_ids'][i] == tokenizer.pad_token_id).sum().item()
        real_tokens = inputs['attention_mask'][i].sum().item()
        print(f"Seq {i}: {real_tokens} real tokens, {pad_count} padding tokens")

if __name__ == "__main__":
    main()