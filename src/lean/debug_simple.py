#!/usr/bin/env python3
"""Simple debug of generation issue"""

import torch
from transformers import GPT2Tokenizer
from model import LeanRookWorldModel

# Setup
tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
if torch.cuda.is_available():
    model.to_device("cuda:0")

# Simulate what GRPO does
prompts = [
    "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,"
]

# Tokenize with padding (as GRPO does)
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
if torch.cuda.is_available():
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

print("Input shapes:")
print(f"  input_ids: {inputs['input_ids'].shape}")
print(f"  attention_mask: {inputs['attention_mask'].shape}")

# Generate
with torch.no_grad():
    generated = model.generate_tokens(
        inputs["input_ids"],
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        attention_mask=inputs["attention_mask"]
    )

print(f"\nGenerated shape: {generated.shape}")

# Decode as GRPO does
full_sequences = torch.cat([inputs["input_ids"], generated], dim=1)
full_texts = tokenizer.batch_decode(full_sequences, skip_special_tokens=True)
prompt_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

print("\nPrompt texts:")
for i, p in enumerate(prompt_texts):
    print(f"{i}: '{p}'")

print("\nFull texts:")
for i, f in enumerate(full_texts):
    print(f"{i}: '{f[:100]}...'")

# Extract completions
completions = []
for full_text, prompt_text in zip(full_texts, prompt_texts):
    if full_text.startswith(prompt_text):
        completion = full_text[len(prompt_text):].strip()
    else:
        print(f"WARNING: Full text doesn't start with prompt!")
        print(f"  Prompt: '{prompt_text[:50]}'")
        print(f"  Full: '{full_text[:50]}'")
        completion = "EXTRACTION FAILED"
    completions.append(completion)

print("\nExtracted completions:")
for i, c in enumerate(completions):
    print(f"{i}: '{c[:80]}...'")