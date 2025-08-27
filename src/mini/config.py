"""
Configuration for GRPO training in mini implementation

Simple dataclass with essential hyperparameters only.
"""

from dataclasses import dataclass
from typing import Optional


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
    
    # Generation
    max_new_tokens: int = 144  # Must be >=144 for complete schemas
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    
    # Data
    n_train_samples: int = 100
    n_eval_samples: int = 20
    data_seed: int = 42
    reward_shaping: str = "graduated"  # graduated, linear, or binary
    
    # Logging
    log_freq: int = 1  # Log every N steps
    eval_freq: int = 10  # Evaluate every N steps
    save_freq: int = 100  # Save checkpoint every N steps
    
    # Paths
    checkpoint_dir: str = "checkpoints/mini_grpo"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.k_samples > 0, "k_samples must be positive"
        assert 0 < self.clip_range < 1, "clip_range must be in (0, 1)"
        assert self.kl_coef >= 0, "kl_coef must be non-negative"
        assert self.max_new_tokens >= 144, "Need at least 144 tokens for schemas"
        assert self.reward_shaping in ["graduated", "linear", "binary"]
        assert self.baseline_type in ["group_mean", "ema", "learned", "adaptive"]
        assert self.kl_type in ["forward", "reverse", "symmetric"]
        assert 0 < self.ema_alpha < 1, "ema_alpha must be in (0, 1)"
        assert 0 <= self.gae_lambda <= 1, "gae_lambda must be in [0, 1]"