"""
GPT-2 Model Configuration for RookWorld GRPO Implementation

This module provides the configuration dataclass for the pure PyTorch GPT-2 implementation,
designed to be compatible with HuggingFace GPT-2 configurations while enabling
pure PyTorch training without transformers library dependency.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPT2Config:
    """
    Configuration class for GPT-2 model compatible with RookWorld-LM-124M
    
    Parameters match HuggingFace GPT-2 configuration for weight compatibility.
    """
    # Model architecture
    vocab_size: int = 50257      # GPT-2 BPE vocabulary size
    n_positions: int = 1024      # Maximum sequence length (context window)
    n_embd: int = 768            # Embedding dimension
    n_layer: int = 12            # Number of transformer blocks
    n_head: int = 12             # Number of attention heads
    n_inner: Optional[int] = None  # Inner dimension for feed-forward (default: 4 * n_embd)
    
    # Activation function
    activation_function: str = "gelu_new"  # GELU activation (matches HF default)
    
    # Regularization
    resid_pdrop: float = 0.1     # Residual connection dropout
    embd_pdrop: float = 0.1      # Embedding dropout  
    attn_pdrop: float = 0.1      # Attention dropout
    
    # Layer normalization
    layer_norm_epsilon: float = 1e-5  # Layer norm epsilon
    
    # Initialization
    initializer_range: float = 0.02  # Weight initialization standard deviation
    
    # Special tokens
    bos_token_id: int = 50256    # Beginning of sequence token
    eos_token_id: int = 50256    # End of sequence token  
    
    # Additional settings
    use_cache: bool = True       # Whether to use past key values for efficiency
    tie_word_embeddings: bool = True  # Tie input/output embeddings
    
    # Performance optimizations
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing
    
    def __post_init__(self):
        """Post-initialization validation and defaults"""
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
            
        # Validate architecture constraints
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        
        # Calculate head dimension
        self.head_dim = self.n_embd // self.n_head
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPT2Config":
        """Create config from dictionary (for loading from JSON/HF config)"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


# RookWorld-LM-124M specific configuration
ROOKWORLD_CONFIG = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12, 
    n_head=12,
    n_inner=3072,  # 4 * 768
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=50256,
    eos_token_id=50256,
    use_cache=True,
    tie_word_embeddings=True
)