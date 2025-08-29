"""
Configuration for GRPO training in mini implementation

Simple dataclass with essential hyperparameters only.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class GRPOConfig:
    """Configuration for minimalist GRPO training"""
    
    # Model
    model_path: str = "jrahn/RookWorld-LM-124M"
    device: str = "cuda"
    
    # GRPO hyperparameters
    k_samples: int = 4  # Number of completions per prompt
    clip_range: float = 0.2  # PPO-style clipping
    kl_coef: float = 0.02  # KL penalty coefficient
    
    # Advanced baseline computation
    baseline_type: str = "group_mean"  # group_mean, ema, learned, adaptive
    ema_alpha: float = 0.1  # EMA smoothing factor
    baseline_update_freq: int = 10  # Update learned baseline every N steps
    
    # KL divergence variants
    kl_type: str = "forward"  # forward (KL1), reverse (KL2), symmetric (KL3)
    adaptive_kl: bool = True  # Enable adaptive KL control
    kl_target: float = 0.01  # Target KL for adaptive control
    kl_horizon: int = 10000  # Horizon for KL adaptation
    
    # Enhanced PPO features
    use_gae: bool = True  # Use Generalized Advantage Estimation
    gae_lambda: float = 0.95  # GAE lambda parameter
    value_loss_coef: float = 0.1  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy regularization coefficient
    
    # Training
    learning_rate: float = 1e-5
    batch_size: int = 8  # Number of prompts per batch
    max_steps: int = 1000
    grad_clip: float = 1.0
    
    # Learning rate schedule
    lr_schedule_type: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 50  # Number of warmup steps (10% of training)
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of initial LR (1e-6 final)
    cosine_restart_steps: Optional[int] = None  # Optional cosine restart period
    
    # Performance optimizations
    use_bf16: bool = False  # BFloat16 mixed precision training
    use_torch_compile: bool = False  # PyTorch 2.x compilation
    compile_mode: str = "reduce-overhead"  # Compilation mode
    enable_tf32: bool = True  # TF32 acceleration for Ampere+ GPUs
    tensor_core_precision: str = "high"  # "highest", "high", or "medium" for matmul precision
    
    # Batch generation optimizations
    use_batch_generation: bool = False  # Enable batch generation for 4x speedup
    batch_generation_mode: str = "mixed"  # "mixed", "task_specific" batch generation strategy
    batch_generation_size: int = 8  # Batch size for generation
    
    # Generation
    max_new_tokens: int = 144  # Must be >=144 for complete schemas
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    
    # Data
    n_train_samples: int = 100
    n_eval_samples: int = 20
    data_seed: int = 42
    reward_shaping: str = "graduated"  # graduated, linear, binary, or continuous
    continuous_components: Optional[Dict[str, str]] = None  # Components to use continuous rewards
    
    # Logging
    log_freq: int = 1  # Log every N steps
    eval_freq: int = 10  # Evaluate every N steps
    save_freq: int = 100  # Save checkpoint every N steps
    
    # Memory management
    ref_model_cache_size: int = 0  # Disable caching by default to prevent leaks
    log_memory_every: int = 10  # Log memory every N steps
    memory_warning_gb: float = 18.0  # Warning threshold for GPU memory
    history_buffer_size: int = 10  # Keep only last N entries in RAM
    emergency_cleanup_interval: int = 50  # Force GPU cache clear every N steps
    log_prob_chunk_size: int = 16  # Process log_probs in chunks of this size to save memory
    
    # Paths
    checkpoint_dir: str = "checkpoints/mini_grpo"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.k_samples > 0, "k_samples must be positive"
        assert 0 < self.clip_range < 1, "clip_range must be in (0, 1)"
        assert self.kl_coef >= 0, "kl_coef must be non-negative"
        assert self.max_new_tokens >= 144, "Need at least 144 tokens for schemas"
        assert self.reward_shaping in ["graduated", "linear", "binary", "continuous"]
        assert self.baseline_type in ["group_mean", "ema", "learned", "adaptive"]
        assert self.kl_type in ["forward", "reverse", "symmetric"]
        assert 0 < self.ema_alpha < 1, "ema_alpha must be in (0, 1)"
        assert 0 <= self.gae_lambda <= 1, "gae_lambda must be in [0, 1]"
        
        # Set default continuous components if not specified
        if self.continuous_components is None:
            self.continuous_components = {
                "fen_similarity": "exponential",  # Rewards near-perfect FEN matches
                "evaluations": "linear",  # Direct proportional to accuracy
            }