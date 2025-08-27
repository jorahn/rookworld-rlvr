# Mini Implementation - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Documentation](#module-documentation)
4. [Task Specifications](#task-specifications)
5. [Reward System](#reward-system)
6. [Usage Guide](#usage-guide)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

## Overview

The Mini implementation is a clean, minimal dataset processing and reward scoring system for GRPO (Group Relative Policy Optimization) training on chess tasks. It addresses critical issues discovered in previous implementations, particularly around dataset preprocessing and reward computation.

### Key Problems Solved

1. **Dataset Format Issue**: Raw samples without "P: " prefix are A: tasks and need "A: " prepended
2. **Token Length Mismatch**: P: tasks (40-60 tokens) vs A: tasks (80-105 tokens) cause padding issues when mixed
3. **Reward Transparency**: Previous implementations had opaque reward calculations
4. **Validation Priorities**: Not all errors are equally important

## Architecture

```
src/mini/
├── dataset.py          # Dataset loading and preprocessing
├── validation.py       # Format and content validation
├── reward_scorer.py    # GRPO reward computation
├── __init__.py        # Package exports
├── test_dataset.py    # Dataset processing tests
├── test_reward_scorer.py      # Reward scoring tests
├── test_permutation_scoring.py # Permutation correlation tests
├── README.md          # Quick start guide
└── DOCUMENTATION.md   # This file
```

### Data Flow

```
Raw Dataset Sample
       ↓
[Preprocessing] → Add "A: " if needed
       ↓
[Parsing] → Split into (prompt, completion)
       ↓
[Generation] → Model produces completion
       ↓
[Validation] → Format + Content checks
       ↓
[Scoring] → Weighted rewards
       ↓
[Shaping] → Graduated/Linear/Binary
       ↓
[Advantages] → Group-relative baseline
       ↓
GRPO Training
```

## Module Documentation

### dataset.py

Main functions for data processing:

#### `preprocess_sample(text: str) -> str`
Adds "A: " prefix to samples not starting with "P: " or "A: ".

**Critical Logic:**
```python
if not text.startswith("P: ") and not text.startswith("A: "):
    return "A: " + text
```

#### `parse_p_task(text: str) -> Tuple[str, str, Dict]`
Parses Policy tasks into components.

**Input Format:**
```
P: [FEN] M: [moves] E: [evals] B: [best]
```

**Returns:**
- `prompt`: "P: [FEN]"
- `completion`: "M: ... E: ... B: ..."
- `parsed_data`: Dict with extracted components

#### `parse_a_task(text: str) -> Tuple[str, str, Dict]`
Parses Environment tasks into components.

**Input Format:**
```
A: [FEN]+[move]+[history]+[new_FEN]+[reward]+[terminated]+[truncated]
```

**Returns:**
- `prompt`: "A: [FEN]+[move]+[history]+"
- `completion`: "[new_FEN]+[reward]+[terminated]+[truncated]"
- `parsed_data`: Dict with extracted components

#### `get_batch_by_type(samples, task_type, batch_size)`
Returns homogeneous batches to avoid padding issues.

### validation.py

Validation functions with weighted priorities:

#### Weight Systems

```python
P_WEIGHTS = {
    'best_move': 4.0,    # Most important - correct best move
    'format': 2.0,       # Important - proper structure
    'candidates': 1.5,   # Useful - good move suggestions
    'evaluations': 1.0   # Nice to have - accurate evals
}

A_WEIGHTS = {
    'format': 4.0,       # Most important - correct structure
    'fen_match': 3.0,    # Important - correct board state
    'game_state': 2.0,   # Useful - correct flags
    'reward_value': 1.0  # Nice to have - accurate reward
}
```

#### Validation Functions

**Format Validators (Binary):**
- `validate_p_format(completion)` - Checks M:, E:, B: sections
- `validate_a_format(completion)` - Checks 4 '+' delimited sections

**Content Validators for P: tasks:**
- `validate_p_best_move(fen, best_move, stockfish)` - Classification
- `validate_p_candidates(fen, moves, stockfish)` - Classification
- `validate_p_evaluations(fen, evals, stockfish)` - Regression

**Content Validators for A: tasks:**
- `validate_a_fen(expected, generated)` - Edit distance
- `validate_a_flags(fen, move, terminated, truncated)` - Classification
- `validate_a_reward(fen, move, reward)` - Regression

### reward_scorer.py

GRPO reward computation with detailed logging:

#### `RewardScorer` Class

**Initialization Parameters:**
- `stockfish_path`: Path to Stockfish engine (optional)
- `reward_shaping`: "graduated", "linear", or "binary"
- `min_reward`: Minimum reward value (default: -0.3)
- `max_reward`: Maximum reward value (default: 1.0)
- `format_bonus`: Bonus for valid format (default: 0.1)

#### Key Methods

##### `score_single(prompt, completion, log_details=True)`
Scores a single prompt+completion pair.

**Process:**
1. Identify task type (P: or A:)
2. Validate format
3. Validate content (if format valid)
4. Apply weighted scoring
5. Apply reward shaping
6. Log details (if requested)

##### `score_batch(prompts, completions, compute_advantages=True)`
Scores multiple pairs efficiently.

**Returns:**
- `advantages`: NumPy array of group-relative advantages
- `details`: List of RewardDetails objects

##### `compute_grpo_rewards()` (Convenience Function)
Simple interface for training:

```python
advantages, details = compute_grpo_rewards(
    prompts, 
    completions,
    group_size=8,
    reward_shaping="graduated",
    verbose=True
)
```

#### Reward Shaping Strategies

**Graduated (Recommended):**
```python
if raw_reward < 0.2:  → -0.3 (or 0.2 with valid format)
elif raw_reward < 0.4: → 0.2
elif raw_reward < 0.6: → 0.4
elif raw_reward < 0.8: → 0.6
elif raw_reward < 0.95: → 0.8
else:                   → 1.0
```

**Linear:**
```python
shaped = min_reward + (max_reward - min_reward) * raw_reward
```

**Binary:**
```python
shaped = max_reward if raw_reward > 0.5 else min_reward
```

## Task Specifications

### P: Tasks (Policy/Chain-of-Thought)

**Purpose:** Train the model to generate structured chess analysis.

**Format:**
```
Prompt:     P: [FEN]
Completion: M: [move1 move2 move3 move4 move5] E: [eval1 eval2 eval3 eval4 eval5] B: [best_move]
```

**Example:**
```
P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4
```

**Validation Priorities:**
1. Best move accuracy (4.0x weight)
2. Format correctness (2.0x weight)
3. Candidate move quality (1.5x weight)
4. Evaluation accuracy (1.0x weight)

### A: Tasks (Environment/State Transition)

**Purpose:** Train the model to predict chess environment dynamics.

**Format:**
```
Prompt:     A: [FEN]+[move]+[history]+
Completion: [new_FEN]+[reward]+[terminated]+[truncated]
```

**Example:**
```
A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false
```

**Fields:**
- `FEN`: Current board position
- `move`: Move to make (UCI format)
- `history`: Comma-separated previous moves
- `new_FEN`: Resulting position
- `reward`: Numeric reward (1.0=checkmate, 0.5=draw, 0.001=continue, 0.0=illegal)
- `terminated`: Game ended (true/false)
- `truncated`: Illegal move (true/false)

**Validation Priorities:**
1. Format correctness (4.0x weight)
2. FEN accuracy (3.0x weight)
3. Game state flags (2.0x weight)
4. Reward value (1.0x weight)

## Reward System

### Weighted Scoring

Each validation component has a weight reflecting its importance:

```python
total_weighted = Σ(component_score × component_weight) / Σ(weights)
```

### Group-Relative Advantages

GRPO requires advantages computed relative to a group baseline:

```python
group_baseline = mean(group_rewards)
advantages = rewards - group_baseline

# Optional normalization
if std(group_rewards) > 0.01:
    advantages = advantages / std(group_rewards)
```

### Detailed Logging

The reward scorer provides comprehensive logging:

```
Task Type: P
Format: VALID (score: 1.000)
Field Scores:
  format         : 1.000 (weight: 2.0, weighted: 2.000)
  best_move      : 0.800 (weight: 4.0, weighted: 3.200)
  candidates     : 0.600 (weight: 1.5, weighted: 0.900)
  evaluations    : 0.400 (weight: 1.0, weighted: 0.400)
Total Raw Reward: 0.650
Shaped Reward: 0.600
```

## Usage Guide

### Basic Usage

```python
from mini import preprocess_sample, parse_p_task, compute_grpo_rewards

# Preprocess raw dataset sample
raw = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
processed = preprocess_sample(raw)  # Adds "A: " prefix
print(processed[:50])  # "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/..."

# Parse task
if processed.startswith("P:"):
    prompt, completion, data = parse_p_task(processed)
else:
    prompt, completion, data = parse_a_task(processed)
```

### Training Integration

```python
from mini import load_and_prepare_samples, get_batch_by_type, compute_grpo_rewards

# Load dataset
samples = load_and_prepare_samples(n_samples=1000)

# Get separate batches for P: and A: tasks
p_batch = get_batch_by_type(samples, "P", batch_size=32)
a_batch = get_batch_by_type(samples, "A", batch_size=32)

# Process P: batch (avoids padding issues)
p_prompts = [prompt for _, prompt, _, _ in p_batch]
p_completions = model.generate(p_prompts)  # Your model

# Compute GRPO rewards
advantages, details = compute_grpo_rewards(
    p_prompts,
    p_completions,
    group_size=8,
    reward_shaping="graduated"
)

# Use advantages in GRPO loss
loss = compute_grpo_loss(advantages, ...)  # Your training code
```

### Custom Reward Scoring

```python
from mini import RewardScorer

# Initialize with custom settings
scorer = RewardScorer(
    stockfish_path="/usr/bin/stockfish",
    reward_shaping="graduated",
    min_reward=-0.5,
    max_reward=1.0,
    format_bonus=0.2
)

# Score individual samples
reward, details = scorer.score_single(
    prompt="P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    completion="M: e2e4 d2d4  E: 0.3 0.4  B: e2e4",
    log_details=True
)

print(f"Reward: {reward:.3f}")
print(f"Format valid: {details.format_valid}")
print(f"Field scores: {details.field_scores}")
```

## Testing

### Running Tests

```bash
# All tests
uv run pytest src/mini/ -v

# Specific test files
uv run pytest src/mini/test_dataset.py -v
uv run pytest src/mini/test_reward_scorer.py -v
uv run pytest src/mini/test_permutation_scoring.py -v

# Quick validation
uv run python src/mini/dataset.py
uv run python src/mini/validation.py
uv run python src/mini/reward_scorer.py
```

### Test Coverage

- **test_dataset.py**: 24 tests
  - Preprocessing logic
  - P: and A: task parsing
  - Format validation
  - Content validation

- **test_reward_scorer.py**: 15 tests
  - Task identification
  - Scoring accuracy
  - Reward shaping
  - Batch processing
  - Advantage computation

- **test_permutation_scoring.py**: 500+ test cases
  - Correlation between permutation and reward
  - Monotonic decrease validation
  - Structured vs random permutations

### Validation Metrics

Permutation tests show excellent correlation:
- **P: tasks**: -0.98 correlation (structured), -0.93 (random)
- **A: tasks**: -0.99 correlation (structured), -0.92 (random)

Higher permutation rates consistently lead to lower rewards, confirming correct implementation.

## Troubleshooting

### Common Issues

#### 1. Dataset Not Loading
```python
# Error: Dataset not found
# Solution: Ensure you have access to jrahn/rookworld_7m on HuggingFace
from datasets import load_dataset
dataset = load_dataset("jrahn/rookworld_7m", streaming=True)
```

#### 2. Stockfish Not Found
```python
# Warning: Stockfish not found, P: task validation will be limited
# Solution: Install Stockfish
# Ubuntu: sudo apt-get install stockfish
# Mac: brew install stockfish
# Or specify path:
scorer = RewardScorer(stockfish_path="/path/to/stockfish")
```

#### 3. Memory Issues with Large Batches
```python
# Solution: Process in smaller groups
for i in range(0, len(samples), 100):
    batch = samples[i:i+100]
    # Process batch
```

#### 4. Padding Issues
```python
# Problem: Mixing P: and A: tasks causes padding problems
# Solution: Always use get_batch_by_type()
p_batch = get_batch_by_type(samples, "P", batch_size)
a_batch = get_batch_by_type(samples, "A", batch_size)
```

### Debug Logging

Enable detailed logging to diagnose issues:

```python
import logging

# Set log level
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logging.getLogger('mini.reward_scorer').setLevel(logging.DEBUG)
logging.getLogger('mini.dataset').setLevel(logging.DEBUG)
```

### Performance Tips

1. **Batch Processing**: Always process multiple samples together
2. **Separate Task Types**: Process P: and A: tasks separately
3. **Cache Stockfish Results**: Reuse engine instance
4. **Use Streaming**: For large datasets, use streaming=True
5. **Precompute Validations**: Cache validation results when possible

## Model Inference

### Architecture Details

The mini implementation includes a complete GPT-2 model for inference:

#### GPT2Config
```python
class GPT2Config:
    vocab_size = 50257      # GPT-2 BPE vocabulary
    n_positions = 1024      # Max sequence length
    n_embd = 768           # Embedding dimension
    n_layer = 12           # Number of transformer blocks
    n_head = 12            # Number of attention heads
    n_inner = 3072         # FFN dimension (4 * n_embd)
```

#### Key Components

1. **Attention Module**
   - Multi-head self-attention with causal masking
   - Fixed NaN issue in softmax with -1e9 replacement
   - Proper handling of padded tokens

2. **MLP Module**
   - Two-layer feedforward with GELU activation
   - Projects from n_embd → n_inner → n_embd

3. **GPT2Model**
   - Token + position embeddings
   - 12 transformer blocks
   - Tied input/output embeddings

### Weight Loading Process

```python
from mini.loader import load_rookworld_model

# Load model with HuggingFace weights
model = load_rookworld_model(
    model_name="jrahn/RookWorld-LM-124M",
    device="cuda"
)
```

Weight conversion handles:
- Transposition of linear layers (HF stores as [in, out], PyTorch expects [out, in])
- Mapping from HF naming to our minimal naming
- Support for both safetensors and pytorch_model.bin

### Generation Best Practices

#### Critical Parameters
```python
# Must generate enough tokens for full schema
max_new_tokens = 144  # Minimum for complete outputs

# Task-specific settings
if task_type == "P":
    temperature = 0.7  # More deterministic for structured output
    top_p = 0.9
else:  # A: task
    temperature = 0.8
    top_p = 0.95
```

#### Tokenization
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
tokenizer.pad_token = tokenizer.eos_token  # Critical!
```

#### Batch Generation with Padding
```python
# Left-padding for GPT-2
batch_ids = torch.full((batch_size, max_len), tokenizer.eos_token_id)
attention_mask = torch.zeros((batch_size, max_len))

for i, tokens in enumerate(encoded):
    batch_ids[i, -len(tokens):] = torch.tensor(tokens)
    attention_mask[i, -len(tokens):] = 1

# Generate with attention mask
generated = model.generate(
    batch_ids,
    attention_mask=attention_mask,
    max_new_tokens=144,
    pad_token_id=tokenizer.eos_token_id
)
```

### Known Issues and Solutions

1. **NaN in Attention Scores**
   - Issue: All -inf values in softmax produce NaN
   - Solution: Replace -inf with -1e9 before softmax

2. **Position Embedding Misalignment (FIXED)**
   - Issue: With left-padding, tokens at different positions get wrong position embeddings
   - Root cause: P: task tokens start at position 59, A: task at position 52 (for 100 max length)
   - Solution: Adjust position IDs to start from 0 where real tokens begin
   - Result: Mixed batches now achieve >90% format validity

3. **Incomplete Outputs**
   - Issue: Model truncates outputs
   - Solution: Generate at least 144 tokens

4. **Padding Token Not Set**
   - Issue: HF tokenizer has no default pad_token
   - Solution: Set pad_token = eos_token

### Critical Implementation Detail: Position Embeddings

The model uses a critical fix for position embeddings with padding:

```python
# When using left-padding, positions start from 0 for real tokens
if attention_mask is not None and past_length == 0:
    position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    
    for i in range(batch_size):
        valid_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            first_valid = valid_positions[0].item()
            num_valid = len(valid_positions)
            # Real tokens get positions 0, 1, 2, ...
            position_ids[i, first_valid:first_valid+num_valid] = torch.arange(num_valid, device=device)
```

This ensures that regardless of padding amount, tokens always see the correct position embeddings as learned during training.

## Advanced Topics

### Custom Validation

Extend validation with your own functions:

```python
def my_custom_validator(completion: str) -> float:
    # Your validation logic
    return score

# Add to validation pipeline
custom_score = my_custom_validator(completion)
weighted_score = custom_score * my_weight
```

### Alternative Reward Shaping

Implement custom shaping functions:

```python
class CustomScorer(RewardScorer):
    def _shape_reward(self, raw_reward: float, format_valid: bool) -> float:
        # Your shaping logic
        if raw_reward < 0.3:
            return -1.0
        elif raw_reward < 0.7:
            return raw_reward * 2 - 0.6
        else:
            return 1.0
```

### Integration with Training Frameworks

Example with PyTorch:

```python
import torch
from mini import compute_grpo_rewards

class GRPOTrainer:
    def compute_loss(self, prompts, completions):
        # Get advantages
        advantages, details = compute_grpo_rewards(
            prompts, completions,
            group_size=self.group_size
        )
        
        # Convert to tensor
        advantages = torch.tensor(advantages, device=self.device)
        
        # Your GRPO loss computation
        loss = self.grpo_loss(advantages, ...)
        
        return loss, details
```

## References

- GRPO Paper: [Group Relative Policy Optimization](https://arxiv.org/abs/...)
- Chess Notation: [UCI Protocol](https://www.chessprogramming.org/UCI)
- FEN Format: [Forsyth-Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
- Stockfish: [Official Documentation](https://stockfishchess.org/)