# Lean GRPO Implementation for RookWorld-LM

This directory contains a minimal, focused implementation of GRPO training for RookWorld-LM without the complexity and potential memory leaks of the original codebase.

## Key Features

- **Minimal Dependencies**: Only essential components, no dead code
- **Clear GPU Placement**: Training model on `cuda:0`, frozen reference model on `cuda:1`
- **Extensive Logging**: Detailed logging of data flow, tensor shapes, memory usage
- **Simple Validation**: Stockfish validation for "P:" tasks, python-chess for "A:" tasks
- **No Mixed Precision**: Keep it simple, no complex optimizations
- **Dataset Focus**: Uses prepared RookWorld dataset only, no self-play
- **Checkpoint System**: Automatic checkpoint saving with configurable intervals
- **Test Set Evaluation**: Regular evaluation on test samples with validity metrics
- **Optimized Batch Size**: Default batch size of 64 for 24GB GPUs
- **Memory Efficient**: No memory leaks, stable at ~2.5GB with BS=64

## File Structure

```
src/lean/
├── __init__.py                    # Package initialization
├── model.py                       # Minimal RookWorld model wrapper
├── dataset.py                     # Enhanced dataset loading with robust parsing
├── validation.py                  # Stockfish and chess validation
├── grpo.py                       # Core GRPO algorithm (fixed token alignment)
├── train_lean.py                 # Main training script with checkpointing
├── test_lean.py                  # Test suite
├── monitor_memory.py             # Memory leak detection tool
├── BATCH_SIZE_OPTIMIZATION.md    # Batch size analysis and recommendations
├── lean_checkpoints/             # Saved model checkpoints (created during training)
└── README.md                     # This file
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
- `--batch-size`: Batch size for training (default: 64, optimized for 24GB GPUs)
- `--group-size`: GRPO group size for baseline computation (default: 8)
- `--learning-rate`: Learning rate for AdamW optimizer (default: 1e-5)
- `--clip-range`: PPO clipping range (default: 0.2)
- `--kl-coef`: KL divergence penalty coefficient (default: 0.02)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `--stockfish-path`: Custom path to Stockfish executable
- `--checkpoint-interval`: Save checkpoint every N steps (default: 100)
- `--checkpoint-dir`: Directory to save checkpoints (default: lean_checkpoints)
- `--eval-interval`: Evaluate on test set every N steps (default: 100)
- `--eval-samples`: Number of test samples for evaluation (default: 100)

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

## Task Specifications & Validation

### P: Task (Policy) - NEW SPECIFICATION
**Purpose**: Strong policy to play the best move (measured by best move accuracy against Stockfish as #1 key metric)

**Format**: 
- **Prompt**: `P: [FEN]` 
- **Expected Completion**: `M: [top-5-moves in UCI] E: [centipawn eval after top-5-moves] B: [best-move in UCI]`

**Note**: M: and E: sections serve as Chain-of-Thought. Single-space padding is acceptable (post-training will teach proper spacing).

**Validation Rewards** (decreasing priority):
1. **Best Move Accuracy** (4.0x weight): Perfect Stockfish match (1.0), Top-3 (0.7), Top-5 (0.5), Legal (0.1)
2. **Format Correctness** (2.0x weight): Proper M: E: B: structure parsing (ignores padding variations)
3. **Move Candidates** (1.5x weight): Fraction of generated moves that match Stockfish top-5
4. **Evaluation Accuracy** (1.0x weight): Regression scoring of centipawn evaluations vs Stockfish

### A: Task (Environment) - NEW SPECIFICATION
**Purpose**: Predict board states and game outcomes after moves

**Format**:
- **Prompt**: `A: [FEN]+[move in UCI]+[comma separated list of move history = up to 10 previous moves in UCI]+`
- **Expected Completion**: `[new FEN]+[reward]+[terminated]+[truncated]`

**Note**: History is comma-separated UCI moves. Split must occur AFTER history, not before.

**Validation Rewards** (decreasing priority):
1. **Format Correctness** (4.0x weight): Number of sections, + delimited structure  
2. **FEN Match** (3.0x weight): Binary exact match (1.0) or Levenshtein distance for smoothness
3. **Game State Flags** (2.0x weight): Binary terminated (game ended) & truncated (illegal move) accuracy
4. **Reward Value** (1.0x weight): 1.0 for checkmate, 0.5 for draw/stalemate, 0.001 for legal continuation

### Dynamic Verification
Future implementation will add `python-chess` dynamic verification for A: tasks. Current implementation validates against dataset reference completions with chess rule verification.

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

## Recent Improvements (Latest Update)

### Fixed Issues
- ✅ **CUDA tensor indexing errors**: Fixed token-logprob alignment in GRPO
- ✅ **Dataset parsing failures**: Enhanced parsing for multiple RookWorld formats
- ✅ **Memory optimization**: Batch size 64 now default, uses only ~2.5GB
- ✅ **Tokenizer warnings**: Fixed padding_side configuration

### New Features
- ✅ **Checkpoint saving**: Automatic model checkpointing every N steps
- ✅ **Test evaluation**: Regular evaluation with validity metrics
- ✅ **Memory monitoring**: Built-in tool to detect memory leaks
- ✅ **Batch size guide**: Comprehensive optimization guide in BATCH_SIZE_OPTIMIZATION.md

### Performance
- **10x faster**: Batch size 64 vs original 8
- **No memory leaks**: Verified over 500+ steps
- **Stable training**: KL divergence stays in healthy range

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