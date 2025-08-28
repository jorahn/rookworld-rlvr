# RookWorld-RLVR: GRPO Training System

**Fine-tune RookWorld-LM with Group Relative Policy Optimization on chess tasks**

## 🎯 Mini Implementation (Clean & Complete)

This repository contains a **streamlined, production-ready** implementation of GRPO training for RookWorld-LM:

- **Self-contained**: Complete system in `src/mini/` (~1500 lines)
- **Pure PyTorch**: No transformers library dependency  
- **Memory optimized**: Stable VRAM usage (~4.8GB)
- **Enhanced GRPO**: Advanced KL control, GAE, value functions
- **Comprehensive**: Model, training, testing, and analysis

See [`src/mini/README.md`](src/mini/README.md) for detailed documentation.

## Quick Start

```bash
# Install dependencies
uv sync

# Run training with optimized settings
./train.sh

# Or run mini implementation directly
cd src/mini && uv run python train.py

# Test enhanced GRPO features
cd src/mini && uv run python test_enhanced_grpo.py
```

## Executive Summary

Production implementation of Group Relative Policy Optimization (GRPO) for fine-tuning RookWorld-LM (GPT-2 124M) on chess tasks. The system trains on dual tasks:

1. **Policy Task (P:)**: Generating structured chess analysis with moves, evaluations, and best lines
2. **Environment Task (A:)**: Predicting board states and game outcomes after moves

## 🏗️ Repository Structure

```
rookworld-rlvr/
├── src/mini/           # Complete GRPO implementation
│   ├── model.py        # Pure PyTorch GPT-2 (124M)
│   ├── grpo.py         # Enhanced GRPO algorithm  
│   ├── train.py        # Training loop
│   ├── dataset.py      # Data loading
│   ├── reward_scorer.py # Graduated rewards
│   └── test_*.py       # Comprehensive tests
├── train.sh            # Main training script
├── docs/               # Critical insights and optimizations  
└── README.md           # This file
```

## ✅ Complete Features

- **Enhanced GRPO**: Forward/reverse/symmetric KL, adaptive control, GAE
- **Memory optimized**: Fixed leaks, stable 4.8GB VRAM usage
- **Graduated rewards**: 5-level system (0.2 → 0.4 → 0.6 → 0.8 → 1.0)
- **Mixed batch handling**: Proper position embeddings for P:/A: tasks
- **Ground truth scoring**: Uses dataset targets for meaningful rewards
- **Comprehensive testing**: Full test coverage of all components

#### **Training Metrics (Verified with Ground Truth)**
- **Ground truth scoring**: Uses dataset targets for meaningful rewards
- **Reward distribution**: Proper variation (0.1-1.0) instead of constant 0.212
- **KL divergence stable**: Adaptive control prevents explosion
- **PPO clipping healthy**: 6-32% average across runs
- **Memory stable**: Constant 4.8GB VRAM (fixed memory leaks)
- **Format validity**: P: tasks 93%, A: tasks 100%

## Technical Architecture

### Mini Implementation Structure
```
src/mini/
├── model.py           # Pure PyTorch GPT-2 implementation
├── loader.py          # HuggingFace weight loading
├── dataset.py         # Data loading and preprocessing
├── reward_scorer.py   # Reward computation with validation
├── validation.py      # Format and content validation
├── grpo.py           # Enhanced GRPO algorithm
├── train.py          # Training loop
├── config.py         # Configuration dataclass
└── test_*.py         # Comprehensive tests
```

### Key Innovations

1. **Position Embedding Fix**: Handles left-padding correctly in mixed batches
2. **Memory Management**: Detaches tensors, clears cache, explicit GPU cleanup
3. **Enhanced GRPO**:
   - Forward, reverse, and symmetric KL divergence
   - Adaptive KL coefficient with target tracking
   - Multiple baseline methods (group mean, EMA, adaptive)
   - GAE and value function support

4. **Graduated Rewards**:
   - 0.2: Format valid only
   - 0.4: Some correct fields
   - 0.6: Most fields correct
   - 0.8: Near perfect
   - 1.0: Perfect execution

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