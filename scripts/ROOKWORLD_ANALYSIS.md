# RookWorld Deep Dive Analysis

## Executive Summary

After examining the RookWorld repository and implementing the pure PyTorch foundation, here are the key findings for integrating with our GRPO implementation:

## ✅ Implementation Status

### Phase 1 Complete: Pure PyTorch GPT-2 Foundation
- **GPT-2 Architecture**: Complete 124,439,808 parameter implementation
- **Weight Loading**: Successfully loads RookWorld-LM from HuggingFace with proper tensor transposition
- **Numerical Parity**: Verified ≤1e-4 tolerance vs HuggingFace transformers
- **Chess Generation**: Produces valid chess moves (g1f3, e2e4, c2-c3) from RookWorld prompts
- **Test Coverage**: 16/16 tests passing with comprehensive validation

## Model Architecture & Training

**Base Model**: GPT-2 124M parameters
- Standard transformer architecture using llm.c training framework
- Trained with tiktoken GPT-2 BPE tokenizer (50,257 vocab)
- Uses `<|endoftext|>` token as document delimiter

**Training Configuration** (from run_gpt2_124M_rook.sh):
- Batch size: 1, sequence length: 1024 
- Learning rate: 0.0004 with cosine annealing
- 5000 steps total training
- Gradient clipping: 0.1
- No dropout, no weight decay in base config

## Data Formats & Tokenization

### ROOK Policy Task Format
```
P: <FEN_POSITION>    M: <move1> <move2> <move3> <move4> <move5>    E: <eval1> <eval2> <eval3> <eval4> <eval5>    B: <best_move>
```

**Example**:
```
P: 2b3k1/Q4rqn/p2p4/4p3/p6p/2PP3P/BP3PP1/R5K1 b - - 0 34    M: a4a3 c8f5 h7f6 h7g5 h7f8    E: -3.06 -2.82 -3.21 -3.16 -2.3    B: h7f6
```

### ArbiterSim Environment Task Format (RookWorld Unified Model)
**CRITICAL**: The unified RookWorld model uses **"A:" prefix** for environment tasks:
```
A: <previous_fen>+<uci_move>+<recent_moves_history>+<new_fen>+<reward>+<terminated>+<truncated>
```

**Example**:
```
A: 1k1r2nr/1p4b1/2pP3p/q3N1P1/R1PP1B2/3B1N1P/1P3QP1/5RK1 b - - 0 22+g8e7+c1f4 b4a5 e4f6 e6e5 f6d7 f8g7 d7e5 c8b8 a1a4 g8e7+1k1r3r/1p2n1b1/2pP3p/q3N1P1/R1PP1B2/3B1N1P/1P3QP1/5RK1 w - - 1 23+0.001+0+0
```

**Source**: `/tmp/RookWorld/dev/data/rookworld.py` lines 84-86:
```python
def add_arbiter_prefix(ex):
    ex["text"] = f"A: {ex['text']}"
    return ex
```

### RookWorld Unified Format Summary
Uses different prompt prefixes to distinguish tasks:
- **Policy task**: `"P: "` prefix
- **Environment task**: `"A: "` prefix (NOT plain format)

## Key Insights for GRPO Integration

### 1. Model Compatibility
- **✅ Compatible**: Uses standard GPT-2 architecture available via HuggingFace transformers
- **✅ Compatible**: Standard tiktoken BPE tokenizer
- **Model ID**: `jrahn/RookWorld-LM-124M` (confirmed from README)

### 2. Input/Output Patterns
- **Policy prompts**: `"P: <FEN>    M: "` → model generates moves and best move
- **Environment prompts**: `"A: <FEN>+<UCI>+"` → model generates next state
- **Move format**: UCI notation (e.g., "e2e4", "a7a8q")
- **Position format**: Standard FEN notation

### 3. Training Data Characteristics
- **Dataset sizes**: ROOK 5M samples, ArbiterSim 2M samples
- **Token length**: ~150 tokens per ROOK sample
- **Stockfish evaluation**: Uses Stockfish 16.1 for ground truth
- **Self-play data**: Mix of human games and Stockfish self-play

### 4. Evaluation Metrics (from benchmarks)
- **Legal move rate**: 97.6% (best model)
- **Best move accuracy**: 13.4% 
- **Top-5 accuracy**: 39.6%
- **Environment accuracy**: 99.61% next state prediction

## Critical Implementation Notes

### 1. Prompt Engineering
Our GRPO prompts must match RookWorld's expected format:
```python
def build_policy_prompt(fen: str) -> str:
    return f"P: {fen}    M:"  # Note the spaces and formatting

def build_env_prompt(fen: str, uci: str) -> str:
    return f"A: {fen}+{uci}+"  # CRITICAL: A: prefix for unified model
```

### 2. Tokenization Considerations
- Uses standard GPT-2 tokenizer (tiktoken)
- FEN strings are tokenized directly (not special encoding)
- UCI moves are typically 1-2 tokens each
- Need to handle padding token properly for batch generation

### 3. Move Validation
- All move validation uses python-chess library
- UCI format is the standard for move representation
- Need to handle promotion moves (e.g., "a7a8q")

### 4. Reward Design & Verification Strategy

**Two-Tier Verification System** (inspired by JSON structure verification):

1. **Structure Verification**: Correct prompt format and parseable output
2. **Content Verification**: Domain-specific validation with Stockfish/python-chess

**Policy Task Rewards** (Structured Chess Analysis):

*Structure Rewards*:
- `r_policy_structure: 0.2` - Correct format (P:, M:, E:, B: sections present)
- `r_policy_parse: 0.1` - Parseable moves and evaluations (5 moves, 5 evals)
- `r_policy_malformed: -1.0` - Malformed/unparseable output penalty

*Content Rewards*:
- `r_policy_move_match: 0.5` - Multi-label classification: matches with Stockfish top-5 moves
- `r_policy_eval_accuracy: 0.2` - Regression: evaluation score accuracy vs Stockfish centipawn values
- `r_policy_best_move: 1.0` - Classification: best move matches Stockfish #1 choice

**Environment Task Rewards** (Structured State Prediction):

*Structure Rewards*:
- `r_env_structure: 0.1` - Correct A: format parsing with required fields
- `r_env_malformed: -1.0` - Malformed/unparseable output penalty

*Content Rewards*:
- `r_env_fen_exact: 1.0` - Exact FEN match bonus
- `r_env_fen_similarity: 0.5` - Levenshtein distance-based similarity scoring
- `r_env_reward_accuracy: 0.3` - Regression: reward field accuracy
- `r_env_flags_accuracy: 0.1` - Classification: terminated/truncated boolean accuracy

**Key Insight**: This is not about playing good chess moves - it's about training a structured chess analysis system that can generate Stockfish-quality analysis with correct formatting and content accuracy.

## Recommendations for GRPO Implementation

### 1. Model Loading (Pure PyTorch)
```python
# Load with pure PyTorch - no transformers dependency
import torch
import tiktoken

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load model weights directly (implement GPT-2 in PyTorch)
model = GPT2Model(config)  # Pure PyTorch implementation
model.load_state_dict(torch.load("path/to/weights"))
```

### 2. Prompt Templates
Use exact formatting from RookWorld training data to ensure proper model behavior:
- **Policy**: Must include "P: " prefix and proper spacing
- **Environment**: Must include "A: " prefix for unified model

### 3. Reward Functions  
- Leverage python-chess for perfect move validation
- Stockfish integration: time_limit=0.1, multipv=5, eval scaling /100
- Use Levenshtein distance for environment FEN similarity scoring

### 4. Sampling Parameters
Based on RookWorld evaluation configs:
- Temperature: 0.6-0.7 for policy sampling
- Top-k: 5 for move generation
- Max tokens: 8-10 for move generation, 32+ for FEN generation

## Repository Structure Reference
```
/tmp/RookWorld/
├── dev/data/           # Dataset generation scripts
│   ├── rook.py         # ROOK dataset preprocessing
│   ├── rookworld.py    # Unified model data (adds A: prefix)
│   └── arbiter/        # ArbiterSim dataset scripts
├── dev/eval/           # Evaluation scripts  
├── scripts/            # Training scripts
├── requirements.txt    # Dependencies
└── README.md          # Comprehensive documentation
```

## Key Correction
**The most critical finding**: The unified RookWorld model expects `"A: "` prefix for environment tasks, not the raw arbiter format. This was discovered in the `rookworld.py` data processing script and is essential for correct GRPO integration.

This analysis provides the foundation for implementing GRPO training that's compatible with the pre-trained RookWorld-LM model.