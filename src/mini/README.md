# Mini Implementation - GRPO Training for RookWorld-LM

**Clean, self-contained implementation of Group Relative Policy Optimization**

## Overview

The mini implementation is a pure PyTorch training system for fine-tuning RookWorld-LM (124M parameters) on chess tasks using GRPO. This is now the **mainline implementation** with verified training stability and performance.

### Key Features
- Pure PyTorch GPT-2 (no transformers dependency)
- Enhanced GRPO with advanced KL control
- Memory efficient (~4.8GB VRAM stable)
- Comprehensive testing suite
- Clean architecture (~1500 lines total)

## Quick Start

```bash
# Default training
uv run python train.py

# Custom parameters
uv run python train.py --steps 100 --batch_size 8 --k_samples 8 --lr 1e-5

# With detailed logging
uv run python train_logged.py --steps 100 --log_dir logs

# Run tests
uv run python test_enhanced_grpo.py
```

## Architecture

```
src/mini/
├── Core Model
│   ├── model.py           # Pure PyTorch GPT-2 (124M)
│   └── loader.py          # HuggingFace weight loading
│
├── Data & Rewards
│   ├── dataset.py         # Data loading and preprocessing
│   ├── reward_scorer.py   # Graduated reward computation
│   └── validation.py      # Format and content validation
│
├── GRPO Training
│   ├── grpo.py           # Enhanced GRPO algorithm
│   ├── train.py          # Basic training loop
│   ├── train_logged.py   # Training with detailed logging
│   └── config.py         # Configuration dataclass
│
├── Testing
│   ├── test_dataset.py   # Dataset tests
│   ├── test_reward_scorer.py
│   ├── test_generation.py
│   ├── test_grpo.py
│   ├── test_enhanced_grpo.py
│   └── test_mixed_batch.py
│
└── Analysis
    ├── analyze_metrics.py
    └── analyze_final_metrics.py
```

## Training Tasks

### P: Tasks (Policy/Analysis)
Generate structured chess analysis with moves, evaluations, and best lines.

**Format:**
```
Input:  P: [FEN]
Output: M: [moves] E: [evals] B: [best]
```

**Example:**
```
P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
M: e2e4,d2d4,g1f3,c2c4,b1c3 E: 0.3,0.2,0.1,0.2,0.1 B: e2e4
```

### A: Tasks (Environment/State)
Predict board state and game outcome after moves.

**Format:**
```
Input:  A: [FEN]+[move]+[history]+
Output: [new_FEN]+[reward]+[terminated]+[truncated]
```

## Enhanced GRPO Features

### 1. Advanced KL Divergence
```python
# Forward KL: D_KL(π||π_ref)
kl_forward = (policy_probs * (policy_logprobs - ref_logprobs)).mean()

# Reverse KL: D_KL(π_ref||π)
kl_reverse = (ref_probs * (ref_logprobs - policy_logprobs)).mean()

# Symmetric KL: Average of both
kl_symmetric = (kl_forward + kl_reverse) / 2
```

### 2. Adaptive KL Control
Automatically adjusts KL coefficient to maintain target divergence:
```python
if kl_divergence > target_kl * 1.5:
    kl_coef *= 1.5  # Increase penalty
elif kl_divergence < target_kl / 1.5:
    kl_coef /= 1.5  # Decrease penalty
```

### 3. Baseline Methods
- **group_mean**: Mean reward of K samples per prompt
- **ema**: Exponential moving average baseline
- **adaptive**: Dynamically adjusted baseline

### 4. Generalized Advantage Estimation (GAE)
Optional value function for variance reduction:
```python
advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)
```

## Graduated Reward System

Rewards are computed based on task completion quality with ground truth comparison:

| Score | Description | Criteria |
|-------|-------------|----------|
| -0.3  | Invalid format | Parse failure |
| 0.2   | Format valid | Structure correct only |
| 0.4   | Some fields correct | Partial content match |
| 0.6   | Most fields correct | Good accuracy vs ground truth |
| 0.8   | Near perfect | Minor deviations from target |
| 1.0   | Perfect | Exact match with ground truth |

### Ground Truth Scoring (NEW)
- **P: tasks**: Compare moves, evaluations, best move with dataset targets
- **A: tasks**: Compare FEN, reward, terminated/truncated flags with expected values
- **Continuous components**: FEN similarity (exponential), evaluation accuracy (linear)

## Memory Management

Critical optimizations to prevent memory leaks:

```python
# 1. Detach tensors during collection
sequences.append(output_ids.detach().cpu())

# 2. Clear reference model cache periodically
if step % 5 == 0:
    ref_model.clear_cache()

# 3. Explicit GPU cleanup
del large_tensors
torch.cuda.empty_cache()

# 4. Move to CPU during rollout collection
rollout_data = {
    'sequences': sequences.to('cpu'),
    'rewards': rewards.to('cpu')
}
```

## Verified Performance

### Training Stability (Latest)
- **Memory usage**: Stable at 4.8GB VRAM (fixed memory leaks)
- **Ground truth scoring**: Properly uses dataset targets for meaningful rewards
- **Reward distribution**: Realistic variation (0.1-1.0) instead of constant 0.212
- **KL divergence**: Stable with adaptive control (no explosion)
- **PPO clipping**: Healthy 6-32% average

### Task-Specific Performance
- **P: tasks**: Variable rewards (0.098-0.993) reflecting analysis quality
- **A: tasks**: High accuracy (often 1.0) for deterministic state transitions
- **Format validity**: P: 93.2%, A: 100%
- **Training speed**: ~18-20s/step (BS=8, K=8)

## Configuration

Key hyperparameters in `config.py`:

```python
@dataclass
class GRPOConfig:
    # Model
    model_path = "jrahn/RookWorld-LM-124M"
    device = "cuda"
    
    # GRPO Core
    k_samples = 8           # Samples per prompt
    clip_range = 0.2        # PPO clipping
    kl_coef = 0.02         # KL penalty
    
    # Enhanced Features
    kl_type = "forward"     # KL divergence type
    adaptive_kl = True      # Adaptive control
    baseline_type = "group_mean"
    use_gae = True         # Use GAE
    
    # Training
    learning_rate = 1e-5
    batch_size = 8
    max_steps = 1000
```

## Running Experiments

### Basic Training
```bash
# Quick test (10 steps)
uv run python train.py --steps 10

# Standard training (100 steps)
uv run python train.py --steps 100

# Large batch training
uv run python train.py --batch_size 32 --k_samples 8
```

### Advanced Features
```bash
# Test symmetric KL divergence
uv run python train.py --kl_type symmetric

# Test adaptive baseline
uv run python train.py --baseline_type adaptive

# Disable GAE
uv run python train.py --no_gae
```

### Monitoring
```bash
# Watch training progress
tail -f logs/grpo_training_*.log | grep "Step"

# Monitor GPU memory
watch -n 1 nvidia-smi

# Analyze metrics
uv run python analyze_final_metrics.py
```

## Testing

```bash
# Full test suite
uv run python test_dataset.py
uv run python test_reward_scorer.py
uv run python test_generation.py
uv run python test_grpo.py
uv run python test_enhanced_grpo.py

# Quick validation
uv run python test_mixed_batch.py
```

## Debugging Tips

### Memory Issues
1. Check for tensor accumulation in loops
2. Ensure sequences are detached and moved to CPU
3. Monitor with `nvidia-smi` for VRAM spikes
4. Use `torch.cuda.memory_summary()` for details

### KL Explosion
1. Enable adaptive KL control
2. Reduce initial `kl_coef`
3. Use KL warmup if needed
4. Check for numerical instabilities

### Poor Rewards
1. Verify dataset quality
2. Check reward scorer logic
3. Ensure token cleaning (remove `<|endoftext|>`)
4. Review graduated reward thresholds

## Implementation Notes

### Position Embeddings Fix
Handles left-padding correctly in mixed batches:
```python
# Compute actual positions for left-padded sequences
positions = torch.arange(seq_len, device=x.device)
positions = positions.unsqueeze(0).expand(batch_size, -1)
# Adjust for padding
positions = positions * attention_mask
```

### Token Cleaning
Removes special tokens before scoring:
```python
# Clean <|endoftext|> tokens
completion = completion.replace("<|endoftext|>", "").strip()
```

### Mixed Batch Handling
Properly handles both P: and A: tasks in same batch with different max lengths and validation requirements.

## Citation

If using this implementation in research:
```bibtex
@software{rookworld_mini,
  title = {Mini GRPO Implementation for RookWorld-LM},
  author = {RookWorld Team},
  year = {2024},
  url = {https://github.com/jrahn/rookworld-rlvr}
}
```