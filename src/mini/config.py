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