"""
Minimal GPT-2 implementation for RookWorld-LM-124M inference

This is a stripped-down, self-contained GPT-2 implementation optimized
for the mini codebase. Only includes necessary components for inference.
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2Config:
    """Minimal config for RookWorld-LM-124M"""
    def __init__(self):
        self.vocab_size = 50257
        self.n_positions = 1024
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.n_inner = 3072  # 4 * n_embd
        self.activation_function = "gelu"
        self.resid_pdrop = 0.0  # No dropout for inference
        self.embd_pdrop = 0.0
        self.attn_pdrop = 0.0
        self.layer_norm_epsilon = 1e-5
        self.use_cache = True


class Attention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()  # batch, sequence, embedding
        
        # QKV projection and split
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle past key-values for generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_kv = (k, v) if use_cache else None
        
        # Attention scores
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        _, _, q_len, k_len = att.shape
        causal_mask = self.mask[:, :, k_len - q_len:k_len, :k_len]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: [B, T] where 1=keep, 0=mask
            # We only mask keys, not queries, to avoid NaN in softmax
            # Padded positions won't contribute to the output anyway
            
            # Mask padded key positions for all queries
            # This ensures padded tokens don't influence attention
            key_mask = attention_mask[:, :k_len].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k_len]
            att = att.masked_fill(key_mask == 0, float('-inf'))
        
        # Softmax and apply values
        # Handle NaN case: if all values are -inf, softmax will produce NaN
        # Replace -inf with a large negative number for stability
        att = torch.where(torch.isinf(att), torch.full_like(att, -1e9), att)
        att = F.softmax(att, dim=-1)
        
        # Set attention weights to 0 for padded query positions
        if attention_mask is not None:
            query_mask = attention_mask[:, k_len - q_len:k_len]  # [B, q_len]
            # Expand for heads and key dimension
            query_mask = query_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, q_len, 1]
            att = att * query_mask  # Zero out attention for padded queries
        
        y = torch.matmul(att, v)
        
        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, q_len, C)
        y = self.c_proj(y)
        
        return y, present_kv


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with residual
        attn_out, present_kv = self.attn(
            self.ln_1(x), 
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_kv=past_kv
        )
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        
        return x, present_kv


class GPT2Model(nn.Module):
    """Minimal GPT-2 model for inference"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head (tied with embeddings)
        # No separate lm_head - we'll use wte.weight in forward()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Input shape
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        inputs_embeds = self.wte(input_ids)
        
        # Position embeddings
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(2)
        
        position_ids = torch.arange(
            past_length, past_length + seq_len, 
            dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.wpe(position_ids)
        
        # Combine embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Pass through transformer blocks
        presents = []
        for i, block in enumerate(self.h):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, present_kv = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_kv=past_kv
            )
            if use_cache:
                presents.append(present_kv)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # LM head (tied with input embeddings)
        logits = F.linear(hidden_states, self.wte.weight)
        
        return {
            "logits": logits,
            "past_key_values": presents if use_cache else None
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        pad_token_id: int = 50256,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Simple generation for inference"""
        self.eval()
        
        with torch.no_grad():
            # Initialize
            batch_size = input_ids.shape[0]
            generated = input_ids
            past_key_values = None
            
            # Create initial attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            for _ in range(max_new_tokens):
                # Get next token logits
                if past_key_values is not None:
                    # Only pass the last token when using cache
                    outputs = self(
                        input_ids=generated[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                else:
                    outputs = self(
                        input_ids=generated,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                
                past_key_values = outputs["past_key_values"]
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_tokens], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device)
                ], dim=1)
                
                # Check if all sequences have generated EOS
                if (next_tokens == pad_token_id).all():
                    break
            
            return generated