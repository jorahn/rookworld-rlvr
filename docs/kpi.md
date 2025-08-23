# KPI Specification

## Implementation Status

### ✅ Foundation Model Verification (Phase 1 Complete)
- **GPT-2 Architecture**: 124,439,808 parameters exactly matching RookWorld-LM-124M
- **Numerical Parity**: ≤1e-4 tolerance verified vs HuggingFace transformers 
- **Chess Behavior**: Successfully generates valid chess moves (g1f3, e2e4, c2-c3)
- **Weight Loading**: HuggingFace → PyTorch conversion with proper tensor transposition
- **Test Coverage**: 16/16 tests passing including architecture, parity, and robustness

### 🚧 Next Critical Path
- Pure PyTorch tokenization wrapper (removing transformers dependency)
- GRPO training implementation
- Stockfish integration for reward computation

## Primary Success Metrics

### Policy Best-Move Accuracy
- **Target**: Policy best-move accuracy vs Stockfish @ pinned configuration
- **Measurement**: Top-1 accuracy on curated evaluation sets
- **Baseline**: ≥30% accuracy on `policy_core_2k.jsonl` evaluation set
- **Continuous Monitoring**: 7-epoch rolling slope must be > 0 for training continuation
- **Current Status**: Foundation ready for evaluation harness implementation

### Environment Simulation Fidelity (Constraint)
- **Target**: ≥99.9% exact board equality on evaluation suite
- **Measurement**: Full board state comparison using python-chess after UCI application
- **Components**: Pieces, side-to-move, castling rights, en-passant, half/full move counters
- **Evaluation Set**: `env_core_1k.jsonl` diverse transition pairs

## Secondary Metrics (Monitoring)

### Legal Move Rate
- **Target**: ≥99.0% legal move generation
- **Fallback Trigger**: <99.0% triggers fallback data mix

### Training Stability
- **KL Divergence**: Monitor vs SFT reference model
- **Entropy**: Track to detect policy collapse
- **Reward Components**: Structure validation, content accuracy, evaluation proximity

### Performance Metrics
- **Tokens/sec**: Training throughput monitoring
- **Engine Queue Latency**: Stockfish evaluation bottleneck detection
- **Top-k Overlap**: k=5 move overlap with Stockfish recommendations

## Reporting Requirements

### Real-time Monitoring
- Time-series export to JSONL format
- Optional WandB integration (disabled by default)
- Sample trajectory storage every N steps (32 samples)

### Evaluation Cadence
- Run `scripts/eval_all.py` at regular intervals
- Generate `eval/run_{step}.json` + CSV outputs
- Exit non-zero if KPI thresholds not met

## Success Criteria for 1-Week Run

### Continue Conditions
- Policy best-move accuracy 7-epoch rolling slope > 0
- No metric regression > 1σ from baseline
- Environment accuracy ≥ 99.9%

### Stop/Escalate Conditions
- Best-move accuracy stagnation
- Legal move rate < 99.0%
- Environment accuracy < 99.9%
- Loss NaN or training instability

## Stockfish Configuration (Pinned)

### Engine Settings
- **Version**: Stockfish 16.1 (fixed)
- **Skill Level**: 20
- **MultiPV**: 5 (for top-k evaluation)
- **Threads**: Configurable per hardware
- **Hash**: Configurable per hardware
- **Evaluation Mode**: Fixed depth OR fixed node time (choose one)
- **Deterministic**: All UCI options logged for reproducibility

### Process Management
- Process pool with concurrency limits
- Deterministic seeding for reproducible analysis
- Timeout/retry handling for robustness