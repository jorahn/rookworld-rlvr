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
uv add torch>=2.0 chess tiktoken safetensors

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

### ✅ Implemented Components (Phase 1 Complete)
- **Pure PyTorch GPT-2**: Complete 124M parameter implementation numerically identical to HuggingFace
  - `src/rookworld_rlvr/model/config.py` - Configuration dataclass with RookWorld-LM specs
  - `src/rookworld_rlvr/model/gpt2.py` - Full transformer architecture with attention, MLP, generation
  - `src/rookworld_rlvr/model/loader.py` - HuggingFace safetensors weight loading with tensor transposition
- **Numerical Parity**: Verified ≤1e-4 tolerance vs HuggingFace transformers on chess prompts
- **Chess Behavior**: Successfully generates valid moves (g1f3, e2e4, c2-c3) from starting positions
- **Comprehensive Testing**: 16/16 tests passing with architecture, parity, and robustness validation

### 🚧 Next Phase Components (In Progress)
- **GRPO Algorithm**: Group-relative baseline with PPO-style clipped policy gradient
- **Dual Task Framework**: Policy task (structured Stockfish analysis) and Environment task (structured state prediction)
- **Structured Output Learning**: Multi-task learning with classification (move matching) and regression (evaluation accuracy)
- **Verification System**: Two-tier validation with structure parsing and content verification using Stockfish/python-chess
- **Self-Play Management**: Parallel games with position buffer for diverse training data

### Configuration Management
The project uses a comprehensive `GRPOConfig` dataclass covering:
- Model parameters (RookWorld-LM-124M by default)
- GRPO hyperparameters (group_size=8, clip_range=0.2, kl_coef=0.02)
- Training schedule and sampling parameters
- Structured reward system for output format validation and content accuracy
- Self-play and evaluation settings

### Key Dependencies
- `torch>=2.0`: Deep learning framework (pure PyTorch implementation)
- `chess`: Move validation, board representation, and Stockfish integration
- `tiktoken`: GPT-2 BPE tokenization
- `safetensors`: Model weight loading (optional)

## Development Notes

### Model Details
- Base model: `jrahn/RookWorld-LM-124M` (GPT-2 architecture trained on chess data)
- Implementation: Pure PyTorch (no transformers library dependency)
- Encoding: FEN notation for positions, UCI for moves  
- Tokenizer: tiktoken GPT-2 BPE

### Training Approach
- **Structured Output Learning**: Trains model to generate well-formed Stockfish-quality analysis, not just play chess
- **Two-tier Verification**: Structure validation (correct format parsing) + content verification (Stockfish/python-chess)
- **Multi-task Learning**: Classification (move matching) and regression (evaluation accuracy) combined
- **Verifiable Rewards**: Uses Stockfish analysis and python-chess validation for ground truth
- **Token-mean Logprob**: Aggregation for training stability across variable-length outputs
- **Mixed Training**: Policy task (analysis generation) and environment task (state prediction)
- **Curriculum Learning**: Opening positions and self-play for diverse training scenarios

### Code Quality Standards
- Black code formatting (line length 88)
- isort import sorting with black profile
- flake8 linting
- pytest for testing with test discovery in `tests/` directory