# RookWorld-RLVR: GRPO Training System

**Fine-tune RookWorld-LM with Group Relative Policy Optimization on chess tasks**

## 🎯 Mini Implementation (Mainline)

The **mini implementation** in `src/mini/` is now the mainline code. This clean, self-contained implementation provides:

- **Pure PyTorch**: No transformers library dependency
- **Memory efficient**: Fixed memory leaks, stable VRAM usage (~4.8GB)
- **Enhanced GRPO**: Advanced KL divergence types, adaptive control, GAE
- **Stable training**: Verified over 100+ steps with consistent performance
- **Complete system**: Model, training, evaluation, and testing in ~1500 lines

See [`src/mini/README.md`](src/mini/README.md) for detailed documentation.

## Quick Start

```bash
# Navigate to mini implementation
cd src/mini

# Install dependencies
uv sync

# Run training with default settings
uv run python train.py

# Run with custom parameters
uv run python train.py --steps 100 --batch_size 8 --k_samples 8 --lr 1e-5

# Test enhanced features
uv run python test_enhanced_grpo.py
```

## Executive Summary

Clean, self-contained implementation of Group Relative Policy Optimization (GRPO) for fine-tuning RookWorld-LM (GPT-2 124M) on chess tasks. The system trains on dual tasks:

1. **Policy Task (P:)**: Generating structured chess analysis with moves, evaluations, and best lines
2. **Environment Task (A:)**: Predicting board states and game outcomes after moves

## Repository Structure

This repository contains only the essential components for the mini GRPO implementation:

```
rookworld-rlvr/
├── src/mini/              # Complete GRPO implementation (~1500 lines)
│   ├── model.py           # Pure PyTorch GPT-2 implementation
│   ├── loader.py          # HuggingFace weight loading
│   ├── dataset.py         # Data loading and preprocessing
│   ├── reward_scorer.py   # Reward computation with validation
│   ├── validation.py      # Format and content validation
│   ├── grpo.py           # Enhanced GRPO algorithm
│   ├── train.py          # Training loop
│   ├── config.py         # Configuration dataclass
│   └── test_*.py         # Comprehensive test suite
├── docs/
│   ├── CRITICAL_INSIGHTS.md       # Important architectural lessons
│   └── performance_optimizations.md # Future optimization plans
├── README.md             # This file
├── CLAUDE.md            # Development guidelines
└── pyproject.toml       # Dependencies and project config
```

## Key Features ✅

- **Pure PyTorch GPT-2** (124M parameters, no transformers dependency)
- **Enhanced GRPO** with advanced KL divergence types and adaptive control
- **Memory efficient** (stable VRAM ~4.8GB, fixed memory leaks)
- **Graduated reward system** (0.2 → 1.0 based on quality)
- **Mixed batch handling** with position embeddings fix
- **Comprehensive testing** and validation suite
- **Verified training stability** over 100+ steps

## Verified Performance

- **Ground truth scoring**: Uses dataset targets for meaningful rewards
- **Reward distribution**: Proper variation (0.1-1.0)
- **KL divergence**: Stable with adaptive control
- **Memory usage**: Constant 4.8GB VRAM (fixed leaks)
- **Format validity**: P: tasks 93%, A: tasks 100%

## Training Configuration

Default configuration in `src/mini/config.py`:
```python
# Model
model_path = "jrahn/RookWorld-LM-124M"
device = "cuda"

# GRPO hyperparameters
k_samples = 8          # Completions per prompt
clip_range = 0.2       # PPO clipping
kl_coef = 0.02        # KL penalty

# Enhanced features
kl_type = "forward"    # forward, reverse, or symmetric
adaptive_kl = True     # Adaptive KL control
use_gae = True        # Generalized Advantage Estimation
value_loss_coef = 0.1  # Value function coefficient

# Training
learning_rate = 1e-5
batch_size = 8
max_steps = 1000
```

## Performance Benchmarks

### Training Efficiency
- **Generation**: ~0.2-0.3s per sample on RTX 4090
- **Training step**: ~18-20s for batch_size=8, k_samples=8
- **Memory usage**: Stable 4.8GB VRAM (fixed memory leaks)
- **Convergence**: Improvement visible within 10-20 steps

### Task Performance (with Ground Truth Scoring)
- **P: tasks** (Policy/Analysis):
  - Format validity: 93.2%
  - Reward range: 0.098-0.993 (reflecting analysis quality)
  - Best move accuracy: Improving with training

- **A: tasks** (Environment/State):
  - Format validity: 100%
  - State prediction accuracy: Often perfect (1.0)
  - Deterministic transitions: Correctly learned

## Development

### Testing
```bash
cd src/mini

# Run all tests
uv run python test_dataset.py
uv run python test_reward_scorer.py
uv run python test_generation.py
uv run python test_grpo.py
uv run python test_enhanced_grpo.py

# Test mixed batch handling
uv run python test_mixed_batch.py
```

### Monitoring Training
```bash
# Run with detailed logging
uv run python train_logged.py --steps 100 --batch_size 8 --log_dir logs

# Analyze metrics
tail -f logs/grpo_training_*.log | grep "Step"

# Check GPU memory
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1
```

## Key Insights

1. **Memory leak prevention**: Critical to detach tensors and clear caches
2. **Ground truth scoring**: Uses dataset targets for meaningful rewards (not just format validation)
3. **Position embeddings**: Must handle left-padding correctly for mixed batches
4. **Token cleaning**: Remove `<|endoftext|>` before scoring
5. **Continuous rewards**: Better gradients for FEN similarity and evaluation accuracy
6. **Mixed training**: Balance of P: and A: tasks maintains stability

## Contributing

This is a research implementation focused on clarity and correctness. The mini implementation serves as both a reference and a working training system.

## License

MIT License - See LICENSE file for details

## Citation

If you use this code in your research, please cite:
```bibtex
@software{rookworld_rlvr,
  title = {RookWorld-RLVR: GRPO Training for Chess Tasks},
  author = {RookWorld Team},
  year = {2024},
  url = {https://github.com/jrahn/rookworld-rlvr}
}
```