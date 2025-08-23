# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for training RookWorld-LM using Group Relative Policy Optimization (GRPO) on chess tasks. The project implements GRPO fine-tuning for RookWorld-LM (GPT-2 124M) to perform dual tasks:
1. Playing legal chess moves as a policy agent
2. Acting as an environment by predicting board states after moves

## Common Development Commands

### Environment Setup
```bash
uv sync                    # Install dependencies and sync lock file
uv run python --version    # Verify Python environment
```

### Code Quality
```bash
uv run black .             # Format code
uv run isort .             # Sort imports  
uv run flake8 .            # Lint code
```

### Testing
```bash
uv run pytest             # Run all tests
uv run pytest tests/      # Run tests in tests directory
uv run pytest -v          # Verbose test output
```

### Training
Based on the README.md specifications:
```bash
# Install training dependencies first
uv add torch>=2.0 transformers>=4.41 accelerate chess safetensors

# Basic GRPO training
uv run python train_rookworld_grpo.py --steps 1000 --group-size 8

# Policy-only training
uv run python train_rookworld_grpo.py --mix-env-ratio 0.0 --steps 2000

# High-performance settings
uv run python train_rookworld_grpo.py \
    --steps 5000 \
    --batch-positions 16 \
    --group-size 16 \
    --n-parallel-games 8 \
    --lr 5e-6 \
    --temperature 0.5
```

## Project Architecture

### Package Structure
- `src/rookworld_rlvr/`: Main package containing the GRPO implementation
- Uses `hatchling` build backend with source layout under `src/`

### Key Components (from README specification)
- **GRPO Algorithm**: Group-relative baseline with PPO-style clipped policy gradient
- **Dual Task Framework**: Policy task (FEN → UCI move) and Environment task (FEN + UCI → next FEN)  
- **Verification System**: Uses python-chess for move legality and exact FEN comparison
- **Self-Play Management**: Parallel games with position buffer for diverse training data

### Configuration Management
The project uses a comprehensive `GRPOConfig` dataclass covering:
- Model parameters (RookWorld-LM-124M by default)
- GRPO hyperparameters (group_size=8, clip_range=0.2, kl_coef=0.02)
- Training schedule and sampling parameters
- Reward shaping for both policy and environment tasks
- Self-play and evaluation settings

### Key Dependencies
- `torch>=2.0`: Deep learning framework
- `transformers>=4.41`: HuggingFace models and tokenizers  
- `chess`: Move validation and board representation
- `accelerate`: Training optimization

## Development Notes

### Model Details
- Base model: `jrahn/RookWorld-LM-124M` (GPT-2 architecture trained on chess data)
- Encoding: FEN notation for positions, UCI for moves
- Tokenizer: Standard GPT-2 BPE

### Training Approach
- Uses verifiable rewards via python-chess instead of learned value functions
- Implements token-mean logprob aggregation for stability
- Supports mixed training on policy and environment tasks
- Includes curriculum learning through opening positions and self-play

### Code Quality Standards
- Black code formatting (line length 88)
- isort import sorting with black profile
- flake8 linting
- pytest for testing with test discovery in `tests/` directory