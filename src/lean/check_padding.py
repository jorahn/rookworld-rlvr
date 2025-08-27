#!/usr/bin/env python3
"""
Check padding tokens and batch generation
"""

import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")

print(f"Vocab size: {len(tokenizer)}")
print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print(f"\nAfter setting pad_token = eos_token:")
print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

# Check token 50256
print(f"\nToken 50256: '{tokenizer.decode([50256])}'")
print(f"Token 220: '{tokenizer.decode([220])}'")

# Test left padding with batch
tokenizer.padding_side = "left"
prompts = [
    "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "P: 8/8/8/8/8/8/8/K1k5 w - - 0 1"
]

print("\n=== Testing batch tokenization with left padding ===")
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")

print("\nFirst sequence (longer):")
print(f"Tokens: {inputs['input_ids'][0].tolist()}")
print(f"Mask:   {inputs['attention_mask'][0].tolist()}")

print("\nSecond sequence (shorter, should have padding on left):")
print(f"Tokens: {inputs['input_ids'][1].tolist()}")
print(f"Mask:   {inputs['attention_mask'][1].tolist()}")

# Check if first tokens are padding
first_tokens = inputs['input_ids'][:, 0:5]
print(f"\nFirst 5 tokens of each sequence:")
for i, tokens in enumerate(first_tokens):
    print(f"Seq {i}: {tokens.tolist()} -> '{tokenizer.decode(tokens)}'")
    
# Count padding tokens
for i in range(len(prompts)):
    pad_count = (inputs['input_ids'][i] == tokenizer.pad_token_id).sum().item()
    print(f"Sequence {i} has {pad_count} padding tokens")