"""
Pure PyTorch GPT-2 Implementation for RookWorld GRPO Training

This module implements a GPT-2 architecture compatible with HuggingFace weights
but using only PyTorch (no transformers library dependency). Designed specifically
for the RookWorld-LM-124M model used in GRPO training.
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPT2Config


class GPT2Attention(nn.Module):
    """Multi-head causal self-attention module"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        
        assert config.n_embd % config.n_head == 0
        
        # Combined query, key, value projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.size()
        
        # Combined QKV projection
        qkv = self.c_attn(x)  # [batch_size, seq_len, 3 * n_embd]
        
        # Split into Q, K, V and reshape for multi-head attention
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle past key-value cache
        present_key_value = None
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        if use_cache:
            present_key_value = (k, v)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        _, _, q_len, k_len = attn_weights.size()
        causal_mask = self.causal_mask[:, :, k_len - q_len:k_len, :k_len]
        attn_weights = torch.where(
            causal_mask.bool(),
            attn_weights,
            torch.finfo(attn_weights.dtype).min
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights += attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_embd)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present_key_value


class GPT2MLP(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self._gelu_new(x)  # Use GELU activation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    def _gelu_new(self, x: torch.Tensor) -> torch.Tensor:
        """GELU activation function (Gaussian Error Linear Unit)"""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPT2Block(nn.Module):
    """Transformer block with pre-layer normalization"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-layer norm attention
        residual = x
        x = self.ln_1(x)
        attn_output, present_key_value = self.attn(
            x, 
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        x = residual + attn_output
        
        # Pre-layer norm MLP
        residual = x
        x = self.ln_2(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output
        
        return x, present_key_value


class GPT2Model(nn.Module):
    """GPT-2 Model for causal language modeling"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head (tied with input embeddings if specified)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will be handled in forward pass
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 initialization scheme"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.size()
        
        # Handle past key values for generation
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
        
        # Position IDs
        position_ids = torch.arange(
            past_length, seq_len + past_length, 
            dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        
        # Token embeddings + position embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.drop(hidden_states)
        
        # Process through transformer blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            if use_cache:
                presents.append(present)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        if self.config.tie_word_embeddings:
            # Tie weights with input embeddings
            logits = F.linear(hidden_states, self.wte.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": presents,
                "hidden_states": hidden_states
            }
        else:
            return (logits, presents, hidden_states)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Simple generation method for inference"""
        self.eval()
        
        with torch.no_grad():
            past_key_values = None
            generated = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated[:, -1:] if past_key_values is not None else generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs["logits"][:, -1, :]  # Last token logits
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for early stopping
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
        
        return generated
    
    def get_num_params(self) -> int:
        """Count the number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 50256) -> torch.Tensor:
    """Create attention mask from input ids"""
    return (input_ids != pad_token_id).float()


# Factory function for easy model creation
def create_rookworld_model() -> GPT2Model:
    """Create a GPT2Model configured for RookWorld-LM-124M"""
    from .config import ROOKWORLD_CONFIG
    return GPT2Model(ROOKWORLD_CONFIG)