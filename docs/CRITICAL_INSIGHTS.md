# Critical Architecture Insights - RookWorld GRPO Implementation

## ⚠️ CRITICAL CORRECTIONS TO AVOID FUTURE MISTAKES

### 1. RookWorld-LM is a UNIFIED MODEL, Not Separate Models

**❌ WRONG ASSUMPTION:**
- Policy task = one model that generates moves
- Environment task = separate model/wrapper for state prediction

**✅ CORRECT UNDERSTANDING:**
- RookWorld-LM is ONE model that handles BOTH tasks via prompt prefixes
- `P: <FEN>    M:` → Generates structured Stockfish analysis
- `A: <FEN>+<UCI>+` → Generates structured environment response
- Same model, same weights, different prompt formats

### 2. Task Output Formats Are Structured, Not Simple

**❌ WRONG: Policy Task**
```
Input:  P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:
Output: e2e4  # Simple move
```

**✅ CORRECT: Policy Task** 
```
Input:  P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:
Output: e2e4 d2d4 g1f3 b1c3 f2f4    E: 0.25 0.18 0.12 0.08 0.15    B: e2e4
```

**❌ WRONG: Environment Task**
```
Input:  A: <FEN>+<UCI>+
Output: <new_fen>  # Simple state
```

**✅ CORRECT: Environment Task**
```
Input:  A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+
Output: +rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+0+0
```

### 3. Reward System is Structure + Content Verification

**❌ WRONG:** Simple binary rewards for correct moves/states

**✅ CORRECT:** Two-tier reward system:
1. **Structure Verification**: Can the output be parsed correctly?
2. **Content Verification**: Is the parsed content accurate vs ground truth?

**Policy Task Rewards:**
- Structure: Correct `M:`, `E:`, `B:` sections present and parseable
- Content: Move matching (classification), eval accuracy (regression), best move (classification)

**Environment Task Rewards:**
- Structure: Correct `A:` format with all required fields
- Content: FEN accuracy (exact + similarity), reward accuracy, flag accuracy

### 4. GRPO Training Uses Same Pipeline for Both Tasks

**❌ WRONG:** Different training loops for policy vs environment

**✅ CORRECT:** Unified GRPO pipeline:
- Sample positions from buffer/self-play
- Mix policy and environment prompts based on `mix_env_ratio`
- Generate G samples for each prompt
- Compute task-specific rewards
- Apply same GRPO loss (group advantage, PPO clipping, KL penalty)

### 5. Architecture Components Needed

**❌ WRONG PLAN:**
- Separate policy wrapper
- Separate environment wrapper
- Different data collection pipelines

**✅ CORRECT ARCHITECTURE:**

1. **Unified Model Wrapper** (`CausalLMPolicy`):
   - Loads single RookWorld-LM model
   - Handles both `P:` and `A:` prompt generation
   - Tracks logprobs for GRPO training
   
2. **Chess Environment Utility** (`ChessEnvironment`):
   - Creates ground truth for environment task validation
   - NOT a separate model - just state management
   
3. **Unified Data Collection**:
   - `collect_policy_group()`: Creates `P:` prompts, rewards structured analysis
   - `collect_env_group()`: Creates `A:` prompts, rewards structured state prediction
   - Same model, same generation pipeline

4. **Task-Specific Reward Functions**:
   - `compute_policy_reward()`: Structure + content verification for analysis
   - `compute_env_reward()`: Structure + content verification for state prediction

### 6. Implementation Priority Order (CORRECTED)

1. **Tokenization Bridge** - Essential for everything
2. **Unified Model Wrapper** - ONE wrapper for both tasks
3. **Chess Environment Utility** - Ground truth generation for A: task
4. **Structured Reward Functions** - Parse and verify both task outputs
5. **Data Collection Pipeline** - Unified collection using task-specific prompts
6. **GRPO Trainer** - Same training loop for both tasks

## Key Architectural Principles

1. **One Model, Two Tasks**: RookWorld-LM handles both via prompt prefixes
2. **Structured Learning**: Train on structured output formats, not simple responses
3. **Unified Pipeline**: Same generation, same GRPO, different rewards
4. **Verification Focus**: Structure parsing + content accuracy, not game playing
5. **Mixed Training**: Blend P: and A: tasks in same training batches

## References to Source Truth

- **Task Formats**: `ROOKWORLD_ANALYSIS.md` lines 32-50 (ROOK vs ArbiterSim formats)
- **Reward Design**: `README.md` lines 133-143 (structured reward coefficients)
- **Data Collection**: `README.md` lines 553-649 (collect_policy_group vs collect_env_group)
- **Unified Model**: `ROOKWORLD_ANALYSIS.md` lines 72-76 (single model, prompt prefixes)

---

**This document serves as a critical reference to prevent architectural misunderstandings that could derail the implementation.**