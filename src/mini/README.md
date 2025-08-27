# Mini Implementation - Data Preparation and Verification

Clean implementation of dataset processing and validation for RookWorld GRPO training.

## Key Learnings Applied

### 1. Dataset Format Issues Solved
- **Critical Fix**: Samples without "P: " prefix are A: tasks and get "A: " prepended automatically
- **Token Length Differences**: P: tasks (40-60 tokens) vs A: tasks (80-105 tokens)
- **Solution**: Separate batch processing for P: and A: tasks to avoid padding issues

### 2. Task Specifications

#### P: Tasks (Policy/Chain-of-Thought)
```
Prompt:     "P: [FEN]"
Completion: "M: [top-5-moves] E: [centipawn-evals] B: [best-move]"

Example:
P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4
```

#### A: Tasks (Environment/State Transition)  
```
Prompt:     "A: [FEN]+[move]+[history]+"
Completion: "[new_FEN]+[reward]+[terminated]+[truncated]"

Example:
A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false
```

## Module Structure

### `dataset.py` - Data Processing
- `preprocess_sample()` - Adds "A: " prefix to non-P: samples
- `parse_p_task()` - Parses P: tasks into prompt/completion/data
- `parse_a_task()` - Parses A: tasks into prompt/completion/data
- `load_and_prepare_samples()` - Loads and processes dataset samples
- `get_batch_by_type()` - Returns batches of single task type (avoids padding)

### `validation.py` - Format and Content Validation

#### Weighted Priorities
```python
P_WEIGHTS = {
    'best_move': 4.0,    # Most important
    'format': 2.0,       
    'candidates': 1.5,   
    'evaluations': 1.0   # Least important
}

A_WEIGHTS = {
    'format': 4.0,       # Most important
    'fen_match': 3.0,    
    'game_state': 2.0,   
    'reward_value': 1.0  # Least important
}
```

#### Format Validators
- `validate_p_format()` - Checks M:, E:, B: sections
- `validate_a_format()` - Checks 4 '+' delimited sections

#### Content Validators
- P: tasks: best move accuracy, candidate quality, eval accuracy
- A: tasks: FEN edit distance, game flags, reward value

### `test_dataset.py` - Comprehensive Tests
- 24 tests covering all functionality
- Tests preprocessing, parsing, format validation, content validation

### `reward_scorer.py` - GRPO Reward Computation
- `RewardScorer` class for detailed validation and scoring
- Supports multiple reward shaping strategies (graduated, linear, binary)
- Computes group-relative advantages for GRPO
- Detailed logging of validation results
- `compute_grpo_rewards()` convenience function for training

## Usage Examples

### Basic Data Processing
```python
from mini import preprocess_sample, parse_p_task, validate_p_task

# Preprocess raw sample
raw = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
processed = preprocess_sample(raw)  # Adds "A: " prefix

# Parse P: task
p_text = "P: [FEN] M: e2e4 d2d4 E: 0.3 0.4 B: e2e4"
prompt, completion, data = parse_p_task(p_text)

# Validate with weighted scoring
scores = validate_p_task(data['fen'], completion)
print(f"Total weighted score: {scores['total_weighted']:.3f}")
```

### GRPO Reward Scoring
```python
from mini import compute_grpo_rewards

# Batch of prompts and generated completions
prompts = [
    "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
]
completions = [
    "M: e2e4 d2d4  E: 0.3 0.4  B: e2e4",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
]

# Compute advantages for GRPO training
advantages, details = compute_grpo_rewards(
    prompts, 
    completions,
    group_size=8,
    reward_shaping="graduated",
    verbose=True  # Enable detailed logging
)

# Use advantages in GRPO loss computation
for i, (adv, detail) in enumerate(zip(advantages, details)):
    print(f"Sample {i}: {detail.task_type} task, "
          f"shaped_reward={detail.shaped_reward:.3f}, "
          f"advantage={adv:.3f}")
```

## Running Tests

```bash
# Run all tests
uv run pytest src/mini/test_dataset.py -v

# Test individual modules
uv run python src/mini/dataset.py
uv run python src/mini/validation.py
```

## Key Improvements Over Previous Implementations

1. **Correct Preprocessing**: Automatic "A: " prefix for non-P: samples
2. **Clear Parsing**: Separate logic for P: and A: tasks with proper delimiters
3. **Prioritized Validation**: Weighted scoring based on importance
4. **Batch Separation**: Avoid mixing task types to prevent padding issues
5. **Comprehensive Tests**: 100% test coverage of critical functions

## Next Steps

When ready to train:
1. Use `get_batch_by_type()` to get homogeneous batches
2. Apply validation to compute weighted rewards
3. Process P: and A: tasks separately in training loop
4. Monitor weighted scores to track model improvement