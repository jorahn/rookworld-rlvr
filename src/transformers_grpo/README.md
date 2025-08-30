# Transformers/TRL GRPO Reference Implementation

Lean reference implementation using HuggingFace transformers and TRL for GRPO training on chess tasks.

## Features

- **HuggingFace Integration**: Uses transformers and TRL libraries
- **Verifiable Rewards**: Format + content scoring with optional Stockfish validation
- **Simple Architecture**: Minimal code for easy understanding and modification
- **Hardware Optimized**: BF16 precision and torch.compile support

## Usage

```bash
# Basic training
python train_transformers_grpo.py

# With custom parameters
python train_transformers_grpo.py --batch_size 8 --learning_rate 2e-5 --bf16

# With Stockfish validation (if available)
python train_transformers_grpo.py --stockfish_path /usr/local/bin/stockfish
```

## Key Components

- `rewards.py`: Verifiable reward system with format and content scoring
- `train.py`: Main GRPO training script using TRL's OnlineDPOTrainer
- `train_transformers_grpo.py`: Simple runner script

## Dependencies

- transformers >= 4.46.3
- trl >= 0.10.0  
- accelerate >= 0.30.0
- torch >= 2.0
- chess (for move validation)

This implementation serves as a clean reference for comparing against the pure PyTorch implementation in `src/rookworld_rlvr/`.