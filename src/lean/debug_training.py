#!/usr/bin/env python3
"""Debug training generation issue"""

import torch
from transformers import GPT2Tokenizer
from model import LeanRookWorldModel
from dataset import LeanRookWorldDataset
from grpo import LeanGRPOTrainer
from validation import LeanValidator
import logging

logging.basicConfig(level=logging.INFO)

# Setup
tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
ref_model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")

if torch.cuda.is_available():
    model.to_device("cuda:0")
    ref_model.to_device("cuda:1")

trainer = LeanGRPOTrainer(model, ref_model, tokenizer, group_size=2)

# Get real prompts from dataset
dataset = LeanRookWorldDataset()
batch_data = dataset.get_training_batch(2)
prompts = [prompt for _, prompt, _ in batch_data]
task_types = [task_type for task_type, _, _ in batch_data]

print("\nPrompts from dataset:")
for i, p in enumerate(prompts):
    print(f"{i}: {p[:80]}...")

# Generate completions using trainer's method
completions, logprobs = trainer._generate_group(prompts, model, "cuda:0")

print("\nGenerated completions:")
for i, c in enumerate(completions):
    print(f"{i}: {c[:80]}...")

# Also test with validator
validator = LeanValidator()
validator.start_engine()

batch = trainer.collect_rollouts(prompts, task_types, validator)

print("\nRollout completions:")
for i, c in enumerate(batch.completions):
    print(f"{i}: {c[:80]}...")
    
print("\nRewards:", batch.rewards)