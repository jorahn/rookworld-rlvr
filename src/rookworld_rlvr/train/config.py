"""
GRPO Training Configuration for RookWorld-LM

This module provides comprehensive configuration management for Group Relative Policy
Optimization (GRPO) training of RookWorld-LM on chess tasks.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class GRPOConfig:
    """Comprehensive configuration for GRPO training of RookWorld-LM.
    
    This configuration covers all aspects of the GRPO training pipeline:
    - Model parameters and paths
    - GRPO-specific hyperparameters  
    - Training schedule and optimization
    - Reward system configuration
    - Self-play and evaluation settings
    - System and logging configuration
    """
    
    # =============================================================================
    # Model Configuration
    # =============================================================================
    model_name_or_path: str = "jrahn/RookWorld-LM-124M"
    """HuggingFace model identifier or local path to RookWorld-LM weights."""
    
    # =============================================================================
    # Optimization Parameters
    # =============================================================================
    lr: float = 1e-5
    """Learning rate for policy optimization. Validated for stability with mixed tasks."""
    
    weight_decay: float = 0.01
    """L2 regularization weight for AdamW optimizer."""
    
    grad_clip_norm: float = 1.0
    """Gradient clipping norm to prevent training instability."""
    
    warmup_steps: int = 100
    """Number of learning rate warmup steps."""
    
    # =============================================================================
    # GRPO Algorithm Parameters
    # =============================================================================
    group_size: int = 8
    """Number of samples per position for group-relative baseline (G parameter)."""
    
    clip_range: float = 0.2
    """PPO-style policy gradient clipping range."""
    
    kl_coef: float = 0.01
    """KL divergence penalty coefficient for policy regularization. Reduced for stability."""
    
    kl_target: Optional[float] = None
    """Target KL divergence for adaptive KL control. If None, uses fixed kl_coef."""
    
    kl_estimator: str = "kl3"
    """KL estimator type: 'kl1' (simple diff), 'kl2' (exp-based), 'kl3' (quadratic)"""
    
    # =============================================================================
    # Sampling Configuration
    # =============================================================================
    temperature: float = 0.7
    """Sampling temperature for generation. Lower = more deterministic."""
    
    top_k: int = 0
    """Top-k filtering for generation. 0 = disabled."""
    
    top_p: float = 0.95
    """Nucleus (top-p) sampling threshold."""
    
    max_new_tokens: int = 64
    """Maximum tokens to generate for both P: and A: tasks."""
    
    # =============================================================================
    # Training Schedule
    # =============================================================================
    steps: int = 1000
    """Total number of training steps to run."""
    
    batch_positions: int = 8
    """Number of chess positions per training batch."""
    
    mix_env_ratio: float = 0.2
    """Fraction of training samples that are A: (environment) tasks vs P: (policy). 
    Validated: 20% environment tasks provide 36.9% stability improvement."""
    
    # =============================================================================
    # Self-Play Configuration
    # =============================================================================
    n_parallel_games: int = 4
    """Number of parallel self-play games for position generation."""
    
    max_game_len: int = 150
    """Maximum moves per self-play game before reset."""
    
    position_buffer_size: int = 1000
    """Size of position buffer for training diversity."""
    
    sample_opening_frac: float = 0.3
    """Fraction of positions sampled from common openings vs game positions."""
    
    # =============================================================================
    # Policy Task Rewards (P: structured analysis generation)
    # =============================================================================
    r_policy_structure: float = 0.2
    """Reward for correctly formatted P: output with M:, E:, B: sections."""
    
    r_policy_parse: float = 0.1
    """Reward for parseable moves and evaluation scores."""
    
    r_policy_move_match: float = 0.5
    """Per-move reward for matching Stockfish top-5 moves (classification)."""
    
    r_policy_eval_accuracy: float = 0.2
    """Reward based on evaluation score accuracy vs Stockfish (regression)."""
    
    r_policy_best_move: float = 1.0
    """Bonus reward when best move matches Stockfish #1 recommendation."""
    
    r_policy_malformed: float = -1.0
    """Penalty for malformed or unparseable P: task output."""
    
    # =============================================================================
    # Environment Task Rewards (A: state transition prediction)
    # =============================================================================
    r_env_structure: float = 0.1
    """Reward for correctly formatted A: output structure."""
    
    r_env_fen_exact: float = 1.0
    """Bonus for exact FEN string match in state prediction."""
    
    r_env_fen_similarity: float = 0.5
    """Partial reward based on FEN string similarity (Levenshtein distance)."""
    
    r_env_reward_accuracy: float = 0.3
    """Reward for accurate reward field prediction (regression)."""
    
    r_env_flags_accuracy: float = 0.1
    """Reward for accurate terminated/truncated flag prediction (classification)."""
    
    r_env_malformed: float = -1.0
    """Penalty for malformed or unparseable A: task output."""
    
    # =============================================================================
    # Evaluation Configuration
    # =============================================================================
    eval_every: int = 50
    """Run evaluation every N training steps."""
    
    eval_positions: int = 100
    """Number of positions to use for evaluation."""
    
    save_every: int = 100
    """Save model checkpoint every N training steps."""
    
    # =============================================================================
    # Resume and Recovery Configuration
    # =============================================================================
    resume_from_checkpoint: Optional[str] = None
    """Path to checkpoint to resume from. If None, starts fresh training."""
    
    auto_resume: bool = False
    """Automatically resume from latest checkpoint if available in output_dir."""
    
    enable_recovery: bool = True
    """Enable automatic recovery on training instability (NaN losses)."""
    
    force_new_run: bool = False
    """Force new run even if checkpoint exists (for experiments)."""
    
    max_checkpoint_keep: int = 3
    """Maximum number of regular checkpoints to keep (rotating deletion)."""
    
    recovery_lr_factor: float = 0.5
    """Factor to reduce learning rate by after recovery from instability."""
    
    recovery_checkpoint_interval: int = 500
    """Steps between creating special recovery checkpoints."""
    
    run_id: Optional[str] = None
    """Unique run identifier. Auto-generated if None."""
    
    append_logs_on_resume: bool = True
    """Append to existing log files on resume instead of overwriting."""
    
    # =============================================================================
    # Stockfish Engine Configuration
    # =============================================================================
    stockfish_time_limit: float = 0.1
    """Time limit in seconds for Stockfish analysis per position."""
    
    stockfish_multipv: int = 5
    """Number of top moves to analyze with Stockfish (for P: task ground truth)."""
    
    stockfish_path: Optional[str] = None
    """Path to Stockfish binary. If None, uses system PATH."""
    
    # =============================================================================
    # System Configuration
    # =============================================================================
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    """Device for model inference and training ('cuda' or 'cpu')."""
    
    seed: int = 42
    """Random seed for reproducible training."""
    
    output_dir: str = "rookworld_grpo_checkpoints"
    """Directory for saving checkpoints and training artifacts."""
    
    log_file: str = "training.log"
    """File for detailed training logs."""
    
    # =============================================================================
    # Performance Optimizations
    # =============================================================================
    use_mixed_precision: bool = field(default_factory=lambda: torch.cuda.is_available())
    """Enable mixed precision training with BF16 autocast for RTX 4090 optimization. Auto-enabled for CUDA."""
    
    use_torch_compile: bool = True
    """Enable torch.compile optimization for faster model execution."""
    
    torch_compile_mode: str = "reduce-overhead"
    """Torch compile mode: 'default', 'reduce-overhead', 'max-autotune'."""
    
    torch_compile_backend: str = "inductor"
    """Torch compile backend. Use 'inductor' for best performance."""
    
    use_gradient_checkpointing: bool = False
    """Enable gradient checkpointing to trade compute for memory."""
    
    use_cuda_graphs: bool = False
    """Enable CUDA graphs for static computation patterns (experimental)."""
    
    enable_cudnn_benchmark: bool = field(default_factory=lambda: torch.cuda.is_available())
    """Enable cudnn benchmark for dynamic kernel selection. Auto-enabled for CUDA."""
    
    pin_memory: bool = field(default_factory=lambda: torch.cuda.is_available())
    """Use pinned memory for faster CPU-GPU transfers. Auto-enabled for CUDA."""
    
    non_blocking_transfer: bool = True
    """Use non-blocking transfers for async CPU-GPU operations."""
    
    # =============================================================================
    # Advanced Training Options  
    # =============================================================================
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients before optimizer step."""
    
    dataloader_num_workers: int = 0
    """Number of worker processes for data loading. 0 = main process only."""
    
    prefetch_factor: int = 2
    """Number of batches to prefetch when num_workers > 0."""
    
    # =============================================================================
    # Logging and Monitoring
    # =============================================================================
    log_interval: int = 10
    """Print training metrics every N steps."""
    
    wandb_project: Optional[str] = None
    """Weights & Biases project name. If None, W&B logging is disabled."""
    
    wandb_run_name: Optional[str] = None
    """Weights & Biases run name. If None, auto-generates from config."""
    
    # =============================================================================
    # Test Positions for Evaluation
    # =============================================================================
    test_positions: List[str] = field(default_factory=lambda: [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Common openings
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # d4
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # e4 e5
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian
        # Middlegame positions
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5",
        "r1bqk2r/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7",
        # Tactical positions
        "r2q1rk1/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQ - 0 7",
        "rnbqk2r/pp1p1ppp/4pn2/2p5/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq c6 0 5",
        # Endgame positions
        "8/8/8/8/8/8/k7/K7 w - - 0 1",  # King and pawn endgame
    ])
    """Standard chess positions used for evaluation benchmarks."""
    
    def __post_init__(self):
        """Validate configuration parameters and compute derived values."""
        # Validation checks
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {self.group_size}")
        
        if not (0.0 <= self.mix_env_ratio <= 1.0):
            raise ValueError(f"mix_env_ratio must be in [0,1], got {self.mix_env_ratio}")
        
        if self.clip_range <= 0:
            raise ValueError(f"clip_range must be > 0, got {self.clip_range}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        
        # Ensure device is valid
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Performance optimization validation
        if not torch.cuda.is_available():
            if self.use_mixed_precision:
                print("Warning: Mixed precision disabled - CUDA not available")
                self.use_mixed_precision = False
            if self.enable_cudnn_benchmark:
                print("Warning: cudnn benchmark disabled - CUDA not available")
                self.enable_cudnn_benchmark = False
            if self.pin_memory:
                print("Warning: Pin memory disabled - CUDA not available")
                self.pin_memory = False
            if self.use_cuda_graphs:
                print("Warning: CUDA graphs disabled - CUDA not available")
                self.use_cuda_graphs = False
        
        # Validate torch compile options
        if self.torch_compile_mode not in ['default', 'reduce-overhead', 'max-autotune']:
            raise ValueError(f"Invalid torch_compile_mode: {self.torch_compile_mode}")
        
        # Validate gradient accumulation
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size accounting for gradient accumulation."""
        return self.batch_positions * self.group_size * self.gradient_accumulation_steps
    
    def get_steps_per_epoch(self) -> int:
        """Estimate steps per epoch based on position buffer cycling."""
        return max(1, self.position_buffer_size // self.batch_positions)
    
    def summary(self) -> str:
        """Generate a human-readable summary of key configuration parameters."""
        return f"""GRPO Training Configuration Summary:
        
Model: {self.model_name_or_path}
Training Steps: {self.steps}
Effective Batch Size: {self.get_effective_batch_size()} (positions={self.batch_positions}, group_size={self.group_size})
Learning Rate: {self.lr} (warmup_steps={self.warmup_steps})

GRPO Parameters:
- Group Size: {self.group_size}
- Clip Range: {self.clip_range}  
- KL Coefficient: {self.kl_coef} {'(adaptive)' if self.kl_target else '(fixed)'}

Task Mix:
- Policy (P:) Tasks: {100*(1-self.mix_env_ratio):.1f}%
- Environment (A:) Tasks: {100*self.mix_env_ratio:.1f}%

Sampling:
- Temperature: {self.temperature}
- Top-p: {self.top_p}
- Max Tokens: {self.max_new_tokens}

Self-Play:
- Parallel Games: {self.n_parallel_games}
- Position Buffer: {self.position_buffer_size}
- Opening Fraction: {100*self.sample_opening_frac:.1f}%

Performance Optimizations:
- Mixed Precision (BF16): {self.use_mixed_precision}
- Torch Compile: {self.use_torch_compile} ({self.torch_compile_mode})
- Gradient Checkpointing: {self.use_gradient_checkpointing}
- CUDA/Tensor Core Optimizations: {self.enable_cudnn_benchmark}

Device: {self.device}
Output Directory: {self.output_dir}
"""