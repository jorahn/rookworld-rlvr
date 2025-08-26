# Lean GRPO Implementation for RookWorld-LM

This directory contains a minimal, focused implementation of GRPO training for RookWorld-LM without the complexity and potential memory leaks of the original codebase.

## Key Features

- **Minimal Dependencies**: Only essential components, no dead code
- **Clear GPU Placement**: Training model on `cuda:0`, frozen reference model on `cuda:1`
- **Extensive Logging**: Detailed logging of data flow, tensor shapes, memory usage
- **Simple Validation**: Stockfish validation for "P:" tasks, python-chess for "A:" tasks
- **No Mixed Precision**: Keep it simple, no complex optimizations
- **Dataset Focus**: Uses prepared RookWorld dataset only, no self-play

## File Structure

```
src/lean/
├── __init__.py          # Package initialization
├── model.py            # Minimal RookWorld model wrapper
├── dataset.py          # Simple dataset loading from jrahn/rookworld_7m
├── validation.py       # Stockfish and chess validation
├── grpo.py            # Core GRPO algorithm implementation
├── train_lean.py      # Main training script
├── test_lean.py       # Test suite
└── README.md          # This file
```

## Usage

### Prerequisites

```bash
# Install dependencies
pip install torch transformers datasets chess python-chess

# Ensure Stockfish is available
sudo apt-get install stockfish  # or install from stockfishchess.org
```

### Quick Test

```bash
cd src/lean
python test_lean.py
```

### Training

```bash
cd src/lean
python train_lean.py --steps 10 --batch-size 4 --log-level DEBUG
```

### Training Options

- `--steps`: Number of training steps (default: 100)
- `--batch-size`: Batch size for training (default: 8)
- `--group-size`: GRPO group size for baseline computation (default: 8)
- `--learning-rate`: Learning rate for AdamW optimizer (default: 1e-5)
- `--clip-range`: PPO clipping range (default: 0.2)
- `--kl-coef`: KL divergence penalty coefficient (default: 0.02)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--stockfish-path`: Custom path to Stockfish executable

## Architecture

### Model Setup
- **Training Model**: Loaded on `cuda:0`, parameters require gradients
- **Reference Model**: Loaded on `cuda:1`, frozen (no gradients)
- Both models are identical copies of `jrahn/RookWorld-LM-124M`

### Training Process

1. **Batch Loading**: Get prompts from RookWorld dataset
2. **Task Parsing**: Identify "P:" (policy) and "A:" (environment) tasks
3. **Generation**: Generate ~144 token completions using training model
4. **Reference Logprobs**: Compute logprobs using frozen reference model
5. **Validation**: Validate completions using Stockfish/chess rules
6. **Reward Computation**: Convert validation results to rewards
7. **GRPO Step**: Apply group-relative baselines and PPO-style update

### Logging

Extensive logging covers:
- Model loading and GPU placement
- Dataset batch composition  
- Tensor shapes and device placement
- Memory usage on both GPUs
- Generation and validation results
- Training metrics (loss, KL divergence, rewards)
- Timing information

## Validation

### P: Task (Policy)
Expected format: `M: move1 move2 ... E: eval1 eval2 ... B: best_move`

Validation rewards:
- **Structure** (0.2): Correct parsing
- **Moves** (0.3): Legal moves + Stockfish agreement bonus
- **Best Move** (0.3): Legal + Stockfish agreement bonus

### A: Task (Environment)  
Expected format: Resulting board state after move application

Validation rewards:
- **Structure** (0.2): Valid move applied
- **Position** (0.5): Correct resulting FEN
- **Game State** (0.3): Correct check/mate indicators

## Memory Management

- Explicit tensor device placement with logging
- CPU detachment of intermediate results
- Regular `torch.cuda.empty_cache()` calls
- No gradient accumulation across batches
- Clear separation of training vs reference computation

## Differences from Original

### Removed Complexity
- No complex config classes for every component
- No mixed precision training
- No checkpoint/resume system  
- No self-play components
- No multiple reward systems
- No advanced optimizations

### Added Focus
- Extensive logging at every step
- Clear GPU placement strategy
- Simple reward computation
- Direct dataset usage
- Minimal dependencies

## Troubleshooting

### Memory Issues
- Reduce `--batch-size` and `--group-size`
- Check GPU memory with logging output
- Ensure proper tensor device placement

### Validation Issues
- Verify Stockfish installation: `stockfish --help`
- Check dataset format parsing in logs
- Enable DEBUG logging for detailed validation info

### Training Issues
- Monitor KL divergence (should be reasonable, not exploding)
- Check reward distributions in logs
- Verify both models are on correct GPUs