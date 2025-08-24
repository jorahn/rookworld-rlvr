# Full GRPO Training Script

This directory contains the complete GRPO training implementation with automatic batch size optimization.

## Files

- **`train_full_grpo.py`** - Main training script with full implementation
- **`test_full_grpo.py`** - Validation script for testing the pipeline
- **`README_train_full_grpo.md`** - This documentation

## Features

### 🎯 Core Implementation
- **HuggingFace Weight Loading**: Automatically loads `jrahn/RookWorld-LM-124M` weights
- **256 Diverse Positions**: Generates opening, mid-game, and endgame positions
- **80/20 Task Distribution**: 80% Policy tasks (P: → M:), 20% Environment tasks (A: → result)
- **Group Size 4**: Creates 4 rollouts per prompt as requested
- **Memory Optimization**: Automatically finds maximum efficient batch size

### 🚀 Automatic Batch Size Optimization
The script automatically determines the optimal batch size by:
1. Testing configurations from conservative (2,4) to aggressive (128,4)  
2. Measuring peak GPU memory after 2-3 training steps
3. Including all memory components:
   - Model parameters (~496MB for 124M)
   - Forward pass activations
   - Backward pass gradients  
   - Adam optimizer states (2x parameters)
4. Selecting configuration that uses ~85% of GPU memory

### 🎲 Position Generation Strategy
- **20% Opening positions**: Starting position, e4, d4, e4-e5, Sicilian, etc.
- **60% Mid-game positions**: Generated through 8-20 moves with weighted selection favoring:
  - Center control (d4, e4, d5, e5 squares)
  - Piece development (knights, bishops)
- **20% Endgame positions**: Positions with <14 pieces remaining

## Usage

### Basic Training
```bash
# Test the pipeline (recommended first)
python scripts/test_full_grpo.py

# Run full training with defaults
python scripts/train_full_grpo.py

# Custom training
python scripts/train_full_grpo.py --steps 500 --max-positions 128
```

### Advanced Options
```bash
python scripts/train_full_grpo.py \
  --steps 1000 \
  --max-positions 256 \
  --output-dir "outputs/my_training" \
  --seed 42 \
  --stockfish-path "/usr/local/bin/stockfish" \
  --device "cuda"
```

## Expected Performance

### RTX 4090 (24GB)
- **Optimal batch size**: ~64 positions × 4 rollouts = 256 effective batch
- **Memory usage**: ~20-22GB (85-90% utilization)
- **Training speed**: ~2-3 steps/second with BF16 + torch.compile
- **Total samples per step**: 256 (64 positions × 4 rollouts each)

### RTX 3080 (10GB)  
- **Optimal batch size**: ~24 positions × 4 rollouts = 96 effective batch
- **Memory usage**: ~8-9GB (80-90% utilization)
- **Training speed**: ~1-2 steps/second

### Configuration Details

```python
config = GRPOConfig(
    # Model
    model_name_or_path="jrahn/RookWorld-LM-124M",  # HF weights
    
    # Training (auto-optimized)
    steps=1000,
    batch_positions=4,  # Will be scaled up automatically
    group_size=4,       # 4 rollouts per position
    
    # Hyperparameters (stability-tested)
    lr=1e-5,
    kl_coef=0.01,
    clip_range=0.2,
    temperature=0.7,
    
    # Task distribution
    mix_env_ratio=0.2,  # 80% Policy, 20% Environment
    
    # Performance optimizations  
    use_mixed_precision=True,    # BF16 for RTX 4090
    use_torch_compile=True,      # 3-5x speedup
    torch_compile_mode="reduce-overhead",
)
```

## Output Structure

```
outputs/full_grpo/
├── train_full_grpo.log          # Comprehensive training logs
├── checkpoint-100/              # Regular checkpoints
│   ├── model.pt                # Model state
│   ├── trainer.pt              # Trainer state  
│   └── memory_profiles.json    # Memory usage data
├── checkpoint-200/
└── final_checkpoint/           # Final trained model
    ├── model.pt
    └── memory_profiles.json
```

## Memory Profiling Output

The script outputs detailed memory analysis:

```
Finding optimal batch size (target: 85.0% GPU utilization)
Profiling memory: batch_positions=2, group_size=4
  Peak memory: 3.42GB (14.2% utilization)
Profiling memory: batch_positions=4, group_size=4  
  Peak memory: 5.18GB (21.6% utilization)
...
Profiling memory: batch_positions=64, group_size=4
  Peak memory: 20.84GB (86.8% utilization)

Optimal batch configuration found:
  Batch positions: 48
  Group size: 4
  Effective batch size: 192
  Memory usage: 18.56GB (77.3%)
```

## Training Logs

Real-time training progress:

```
Step   10 | Loss: 2.3456 | Reward: 0.234 (best: 0.234) | P/E: 8/2 | Time: 1.23s
Step   20 | Loss: 2.1234 | Reward: 0.267 (best: 0.267) | P/E: 7/3 | Time: 1.18s
Step   30 | Loss: 1.9876 | Reward: 0.298 (best: 0.298) | P/E: 8/2 | Time: 1.21s
```

Where:
- **P/E**: Policy/Environment task counts (should approximate 80/20)
- **Reward**: Average reward across all rollouts  
- **best**: Best reward seen so far

## Troubleshooting

### Out of Memory
```bash
# Reduce initial batch size in the script:
# batch_positions=2,  # Start even more conservative
```

### Slow Performance  
```bash
# Disable optimizations for debugging:
python scripts/train_full_grpo.py --steps 50
# Then edit script to set:
# use_torch_compile=False
# use_mixed_precision=False
```

### Model Loading Issues
```bash
# Test model loading separately:
python -c "
from src.rookworld_rlvr.model.loader import load_pretrained_model
model = load_pretrained_model('jrahn/RookWorld-LM-124M')
print('Model loaded successfully')
"
```

## Implementation Notes

### Key Classes
- **`FullGRPOTrainer`**: Main orchestrator class
- **`MemoryProfile`**: Memory usage measurement dataclass  
- **Position generation**: Creates diverse chess scenarios
- **Batch optimization**: Automatic memory-based scaling

### Integration with Existing Code
The script uses the complete existing implementation:
- `GRPOTrainer` for core training logic
- `GRPODataCollector` for batch collection  
- `CausalLMPolicy` for model inference
- `StockfishEngine` for reward computation
- All existing config and utilities

This provides a production-ready training pipeline that automatically optimizes for your hardware while maintaining full compatibility with the existing GRPO framework.