# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for **post-training (fine-tuning) RookWorld-LM** using Group Relative Policy Optimization (GRPO) on chess tasks. The project implements GRPO fine-tuning to improve the pre-trained RookWorld-LM model (GPT-2 124M) on dual tasks:
1. **Policy Task (P:)**: Generating structured chess analysis (moves, evaluations, best lines) 
2. **Environment Task (A:)**: Predicting board states and game outcomes after moves

**Key Point**: This is **fine-tuning/post-training** of the existing pre-trained `jrahn/RookWorld-LM-124M` model from HuggingFace, not training from scratch.

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

### Performance Testing
```bash
uv run python test_performance.py  # Benchmark PyTorch optimizations (BF16, torch.compile, etc.)
```

### Testing
```bash
uv run pytest             # Run all tests
uv run pytest tests/      # Run tests in tests directory
uv run pytest -v          # Verbose test output
```

### Fine-tuning (Post-training)
Based on the README.md specifications:
```bash
# Install fine-tuning dependencies first
uv add torch>=2.0 chess tiktoken safetensors

# Basic GRPO fine-tuning of pre-trained RookWorld-LM (STABLE)
./train.sh

# Short test run
STEPS=5 ./train.sh

# Policy-only fine-tuning (structured analysis generation)
MIX_ENV_RATIO=0.0 STEPS=2000 ./train.sh

# High-performance fine-tuning settings (RTX 4090 optimized)
STEPS=5000 BATCH_POSITIONS=16 GROUP_SIZE=16 LR=1e-5 USE_TORCH_COMPILE=true ./train.sh

# Resume fine-tuning (automatic detection)
./train.sh  # Auto-resumes if checkpoint exists
```

## Project Architecture

### Package Structure
- `src/rookworld_rlvr/`: Main package containing the GRPO implementation
- Uses `hatchling` build backend with source layout under `src/`

### ✅ Implemented Components (Phases 1-3 Complete)

#### **Phase 1: Pure PyTorch Foundation** ✅
- **Pure PyTorch GPT-2**: Complete 124M parameter implementation numerically identical to HuggingFace
  - `src/rookworld_rlvr/model/config.py` - Configuration dataclass with RookWorld-LM specs
  - `src/rookworld_rlvr/model/gpt2.py` - Full transformer architecture with attention, MLP, generation
  - `src/rookworld_rlvr/model/loader.py` - HuggingFace safetensors weight loading with tensor transposition
- **Numerical Parity**: Verified ≤1e-4 tolerance vs HuggingFace transformers on chess prompts
- **Chess Behavior**: Successfully generates valid moves (g1f3, e2e4, c2-c3) from starting positions
- **Comprehensive Testing**: 16/16 tests passing with architecture, parity, and robustness validation

#### **Phase 2: GRPO Training Infrastructure** ✅
- **Complete GRPO Implementation**: Group-relative baselines with PPO-style clipped policy gradients
  - `src/rookworld_rlvr/train/grpo_trainer.py` - Full GRPO algorithm with adaptive KL control
  - `src/rookworld_rlvr/train/policy.py` - Unified policy wrapper for both tasks
  - `src/rookworld_rlvr/data/collector.py` - GRPO data collection for P: and A: tasks
- **Task Multiplexing**: Unified `P:<FEN> M:` and `A:<FEN>+<UCI>+` format support
- **Reward Systems**: Two-tier verification (structure + content) for both policy and environment tasks
  - `src/rookworld_rlvr/reward/policy_reward.py` - Stockfish-verified policy rewards
  - `src/rookworld_rlvr/reward/env_reward.py` - Chess-rules verified environment rewards

#### **Phase 3: Training Features** ✅
- **Resume & Recovery System**: Complete checkpoint management with automatic recovery
  - `src/rookworld_rlvr/train/checkpoint_manager.py` - Advanced checkpoint management
  - CLI support: `--resume-from-checkpoint`, `--auto-resume`, `--recovery-mode`
  - Automatic recovery from NaN losses with learning rate reduction
  - Run identity preservation across interruptions
- **RTX 4090 Optimizations**: Verified performance gains
  - BF16 mixed precision (1.5x speedup)
  - torch.compile optimization (1.29x speedup) 
  - Tensor Core utilization (`torch.set_float32_matmul_precision('high')`)
  - TF32 acceleration for Ampere GPUs
- **Stability Improvements (COMPLETED)**: **Full training stability achieved**
  - Graduated 5-level reward system (0.2→0.4→0.6→0.8→1.0)
  - KL warmup with configurable factor (fully implemented)
  - Conservative reward penalties (-0.3 vs -1.0) prevent gradient explosion
  - Higher KL divergence threshold (50.0) for appropriate tolerance
  - **VERIFIED**: Stable training with KL warmup disabled and real KL regularization active
- **Evaluation & Monitoring**: Chess-specific evaluators with tactical position testing

### ✅ Phase Components COMPLETED

#### **Phase 4: Advanced Training Features** ✅
- **Enhanced Learning Rate Schedules**: KL warmup **fully implemented**, cosine annealing active
- **Self-Play Management**: Parallel games with position buffer for diverse training data
  - `src/rookworld_rlvr/train/self_play.py` - Self-play game management
- **Comprehensive Evaluation**: Tactical position testing and benchmarking
  - `src/rookworld_rlvr/train/evaluator.py` - Chess-specific evaluation metrics

#### **Phase 5: Future Optimizations** (Documented)
- **Flash Attention Integration**: 2-3x attention speedup potential (see `docs/performance_optimizations.md`)
- **vLLM Integration**: 5x speedup for GRPO multi-completion sampling
- **Advanced Memory Optimizations**: CPU optimizer offloading for larger models

### Configuration Management
The project uses a comprehensive `GRPOConfig` dataclass covering:
- **Model parameters**: RookWorld-LM-124M by default with pure PyTorch implementation
- **GRPO hyperparameters**: group_size=8, clip_range=0.2, kl_coef=0.02, adaptive KL control
- **Training schedule**: Steps, learning rate, cosine annealing, **KL warmup (implemented)**
- **Performance optimizations**: BF16 mixed precision, torch.compile, RTX 4090 optimizations
- **Resume & Recovery**: Checkpoint management, automatic recovery, run identity tracking
- **Reward systems**: **Graduated 5-level rewards** with exponential moving average normalization
- **Self-play and evaluation**: Position generation and chess-specific metrics

### Key Dependencies
- `torch>=2.0`: Deep learning framework (pure PyTorch implementation)
- `chess`: Move validation, board representation, and Stockfish integration
- `tiktoken`: GPT-2 BPE tokenization
- `safetensors`: Model weight loading (optional)

### Performance Optimizations (RTX 4090)
- **BFloat16 Mixed Precision**: Optimized for RTX 4090 with better numerical stability than FP16
- **Tensor Core Utilization**: `torch.set_float32_matmul_precision('high')` for maximum Tensor Core usage
- **PyTorch 2.x Compile**: `torch.compile()` with `reduce-overhead` mode for 3-5x speedup
- **TF32 Acceleration**: Enabled for Ampere+ GPUs for additional performance gains
- **GRPO Memory Efficiency**: No critic network required, 50% memory reduction vs PPO
- **Future Optimizations**: Flash Attention and vLLM integration documented in `docs/performance_optimizations.md`

## Development Notes

### Model Details
- **Base model**: `jrahn/RookWorld-LM-124M` (pre-trained GPT-2 124M specialized for chess)
- **Implementation**: Pure PyTorch (no transformers library dependency)
- **Encoding**: FEN notation for positions, UCI for moves  
- **Tokenizer**: tiktoken GPT-2 BPE
- **Task**: Post-training/fine-tuning to improve performance on structured chess analysis

### Fine-tuning Approach
- **Structured Output Learning**: Trains model to generate well-formed Stockfish-quality analysis, not just play chess
- **Graduated Reward System**: 5-level progression (0.2→0.4→0.6→0.8→1.0) with partial credit
- **Multi-task Learning**: Classification (move matching) and regression (evaluation accuracy) combined
- **Verifiable Rewards**: Uses Stockfish analysis and python-chess validation for ground truth
- **Token-mean Logprob**: Aggregation for training stability across variable-length outputs
- **Mixed Training**: Policy task (analysis generation) and environment task (state prediction)
- **KL Warmup**: Configurable warmup period with 0.0 factor to prevent early divergence
- **Reward Normalization**: Exponential moving average for stable training dynamics
- **Curriculum Learning**: Opening positions and self-play for diverse training scenarios

### Code Quality Standards
- Black code formatting (line length 88)
- isort import sorting with black profile
- flake8 linting
- pytest for testing with test discovery in `tests/` directory