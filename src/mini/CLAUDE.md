# CLAUDE.md - Mini Implementation

This file provides guidance for working with the self-contained mini implementation of GRPO training for RookWorld-LM.

## Overview

The `src/mini` directory contains a **minimal, self-sufficient implementation** of GRPO (Group Relative Policy Optimization) for fine-tuning RookWorld-LM-124M on chess tasks. This implementation is:

- **Pure PyTorch**: No transformers library dependency
- **Self-contained**: All code within src/mini, no external dependencies
- **Minimalist**: Only essential components, ~1000 lines total
- **Clean**: Clear separation of concerns, well-documented

## Key Components

### Core Modules (Existing)
- `model.py` - Pure PyTorch GPT-2 implementation (124M params)
- `loader.py` - HuggingFace weight loader for RookWorld-LM
- `dataset.py` - Data loading and preprocessing
- `reward_scorer.py` - Reward computation with validation
- `validation.py` - Format and content validation

### GRPO Training (New)
- `grpo.py` - Core GRPO algorithm and loss computation
- `train.py` - Training loop and data collection
- `config.py` - Configuration dataclass
- `test_grpo.py` - Testing and validation

## Quick Start

```bash
# Navigate to mini directory
cd src/mini

# Run training with default settings
uv run python train.py

# Run with custom settings
uv run python train.py --steps 100 --k_samples 4 --lr 1e-5

# Test implementation
uv run python test_grpo.py
```

## Task Specifications

### P: Tasks (Policy/Analysis)
- **Input**: `P: [FEN]`
- **Output**: `M: [moves] E: [evals] B: [best]`
- **Tokens**: 40-60 typically

### A: Tasks (Environment/State)
- **Input**: `A: [FEN]+[move]+[history]+`
- **Output**: `[new_FEN]+[reward]+[terminated]+[truncated]`
- **Tokens**: 80-105 typically

## GRPO Algorithm

1. **Sample Generation**: Generate K completions per prompt
2. **Reward Scoring**: Score using graduated rewards (0.2 → 1.0)
3. **Advantage Computation**: Group-relative baseline (mean of K samples)
4. **Policy Update**: Maximize advantage-weighted log probs with KL penalty
5. **KL Regularization**: Prevent divergence from reference model

## Key Design Decisions

- **No external dependencies**: Everything in src/mini
- **Position embedding fix**: Handles left-padding correctly
- **Attention mask fix**: Replaces -inf with -1e9 for stability
- **Token cleaning**: Removes <|endoftext|> before scoring
- **Graduated rewards**: Partial credit for format/content validity

## Testing

```bash
# Run all tests
uv run python test_dataset.py
uv run python test_reward_scorer.py
uv run python test_generation.py
uv run python test_grpo.py

# Quick functionality check
uv run python test_mixed_batch.py
```

## Performance

- **Generation**: ~0.2-0.3s per sample on GPU
- **P: tasks**: 93.2% format validity
- **A: tasks**: 100% format validity
- **Training**: ~1-2 samples/second with K=4

## Notes

- This is a research implementation optimized for clarity
- Focus is on correctness over performance
- Suitable for experiments and understanding GRPO
- Not intended for production use