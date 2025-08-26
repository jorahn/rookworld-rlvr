"""
Lean Model Implementation for RookWorld GRPO Training

Simple PyTorch GPT-2 model loading without complex config classes.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Config
import logging

logger = logging.getLogger(__name__)


class LeanRookWorldModel(nn.Module):
    """Minimal wrapper for RookWorld-LM model with dual GPU placement"""
    
    def __init__(self, model_name: str = "jrahn/RookWorld-LM-124M"):
        super().__init__()
        
        logger.info(f"Loading RookWorld model: {model_name}")
        
        # Load the pre-trained model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.model.config
        
        # Key model info for logging
        logger.info(f"Model loaded - vocab: {self.config.vocab_size}, "
                   f"layers: {self.config.n_layer}, "
                   f"embd: {self.config.n_embd}")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional hidden states return"""
        
        logger.debug(f"Forward pass - input_ids shape: {input_ids.shape}, "
                    f"device: {input_ids.device}")
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        logger.debug(f"Forward pass output - logits shape: {logits.shape}")
        
        if return_logits:
            return logits, None
        else:
            return outputs.logits, outputs.hidden_states
    
    def to_device(self, device: str):
        """Move model to specified device with logging"""
        logger.info(f"Moving model to device: {device}")
        
        # Count parameters for memory logging
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"Model parameters: {param_count:,}")
        
        self.to(device)
        
        # Log memory usage if CUDA
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_cached = torch.cuda.memory_reserved(device) / 1024**3
                logger.info(f"GPU {device} memory - allocated: {memory_allocated:.2f}GB, "
                           f"cached: {memory_cached:.2f}GB")
        
        return self
    
    def generate_tokens(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 144,
        temperature: float = 1.0,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate tokens for completions"""
        
        logger.debug(f"Generating {max_new_tokens} tokens with temp={temperature}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.config.eos_token_id,
                attention_mask=torch.ones_like(input_ids)
            )
        
        # Extract only the newly generated tokens
        generated = outputs[:, input_ids.shape[1]:]
        
        logger.debug(f"Generated tokens shape: {generated.shape}")
        
        return generated