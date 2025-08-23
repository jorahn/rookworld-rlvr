"""
Unified Model Wrapper for RookWorld GRPO Training

This module provides a unified policy wrapper that handles both Policy (P:) and 
Environment (A:) tasks using the same RookWorld-LM model with different prompt prefixes.

Key insights:
- RookWorld-LM is ONE model that handles both tasks via prompt prefixes
- P: prompts generate structured Stockfish analysis
- A: prompts generate structured environment responses
- Same model, same GRPO training, different reward functions
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from ..model.gpt2 import GPT2Model
from ..model.loader import load_pretrained_model
from ..tokenizer.bridge import TokenizerBridge


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    pad_token_id: Optional[int] = None


class CausalLMPolicy:
    """
    Unified policy wrapper for RookWorld-LM GRPO training
    
    Handles both Policy (P:) and Environment (A:) tasks using the same model
    with different prompt prefixes. Provides generation with logprob tracking
    for GRPO training.
    """
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: Union[str, torch.device] = "cpu",
        torch_dtype: torch.dtype = torch.float32
    ):
        """
        Initialize unified policy wrapper
        
        Args:
            model_name_or_path: Path to RookWorld-LM model
            device: Device to load models on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Initialize tokenizer
        self.tokenizer = TokenizerBridge()
        
        # Load trainable model
        print(f"Loading trainable model from {model_name_or_path}")
        self.model = load_pretrained_model(
            model_name_or_path, 
            device=device, 
            torch_dtype=torch_dtype
        )
        self.model.train()
        
        # Load reference model (frozen copy for GRPO baseline)
        print(f"Loading reference model from {model_name_or_path}")
        self.ref_model = load_pretrained_model(
            model_name_or_path, 
            device=device, 
            torch_dtype=torch_dtype
        )
        self.ref_model.eval()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def generate_batch(
        self, 
        prompts: List[str], 
        generation_config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Generate batch of structured outputs with logprob tracking
        
        Args:
            prompts: List of prompts (P: or A: format)
            generation_config: Generation configuration
            
        Returns:
            Dictionary with:
                - sequences: Full sequences including prompt [batch_size, total_seq_len]
                - generated_ids: Generated tokens only [batch_size, max_new_tokens]
                - seq_logprob: Sequence logprobs for generated tokens [batch_size]
                - texts: List of generated text strings (without prompts)
        """
        self.model.eval()
        
        batch_size = len(prompts)
        
        # Tokenize prompts
        prompt_encoding = self.tokenizer.encode_batch(
            prompts, 
            padding=True, 
            device=self.device
        )
        input_ids = prompt_encoding["input_ids"]
        attention_mask = prompt_encoding["attention_mask"] 
        prompt_lengths = prompt_encoding["lengths"]
        
        # Generate sequences
        generated_sequences = []
        generated_logprobs = []
        
        for i in range(batch_size):
            prompt_ids = input_ids[i:i+1, :prompt_lengths[i]]
            prompt_mask = attention_mask[i:i+1, :prompt_lengths[i]]
            
            # Generate with logprob tracking
            sequence, logprobs = self._generate_with_logprobs(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config
            )
            
            generated_sequences.append(sequence[0])  # Remove batch dimension
            generated_logprobs.append(logprobs[0])   # Remove batch dimension
        
        # Pad generated sequences to same length
        max_total_len = max(seq.size(0) for seq in generated_sequences)
        padded_sequences = torch.full(
            (batch_size, max_total_len), 
            self.tokenizer.pad_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        # Extract generated tokens only (without prompts)
        max_new_tokens = generation_config.max_new_tokens
        generated_ids = torch.full(
            (batch_size, max_new_tokens),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        seq_logprobs = torch.zeros(batch_size, device=self.device)
        generated_texts = []
        
        for i, (seq, logprob, prompt_len) in enumerate(zip(generated_sequences, generated_logprobs, prompt_lengths)):
            # Store full sequence
            seq_len = seq.size(0)
            padded_sequences[i, :seq_len] = seq
            
            # Extract generated part
            gen_start = prompt_len.item()
            gen_tokens = seq[gen_start:]
            gen_len = min(gen_tokens.size(0), max_new_tokens)
            
            generated_ids[i, :gen_len] = gen_tokens[:gen_len]
            seq_logprobs[i] = logprob
            
            # Decode generated text
            gen_text = self.tokenizer.decode(gen_tokens, skip_special=True)
            generated_texts.append(gen_text)
        
        return {
            "sequences": padded_sequences,
            "generated_ids": generated_ids,
            "seq_logprob": seq_logprobs,
            "texts": generated_texts
        }
    
    def _generate_with_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, 
        generation_config: GenerationConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text with logprob tracking
        
        Args:
            input_ids: Input token IDs [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            generation_config: Generation configuration
            
        Returns:
            Tuple of (full_sequence, sequence_logprob)
        """
        sequence = input_ids.clone()
        total_logprob = torch.tensor(0.0, device=self.device)
        
        for _ in range(generation_config.max_new_tokens):
            # Forward pass
            outputs = self.model(sequence, attention_mask=None)
            logits = outputs["logits"]
            next_token_logits = logits[0, -1, :] # Last position logits
            
            # Apply temperature
            if generation_config.temperature != 1.0:
                next_token_logits = next_token_logits / generation_config.temperature
            
            # Apply top-k filtering
            if generation_config.top_k is not None:
                top_k = min(generation_config.top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering  
            if generation_config.top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            
            if generation_config.do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Track logprob of chosen token
            token_logprob = torch.log(probs[next_token.item()] + 1e-10)
            total_logprob += token_logprob
            
            # Append to sequence
            sequence = torch.cat([sequence, next_token.unsqueeze(0)], dim=1)
            
            # Check for early stopping
            if (generation_config.pad_token_id is not None and 
                next_token.item() == generation_config.pad_token_id):
                break
        
        return sequence, total_logprob
    
    def compute_logprobs(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        target_start_indices: torch.Tensor,
        use_ref: bool = False
    ) -> torch.Tensor:
        """
        Compute token-mean logprobs for GRPO training
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            target_start_indices: Start indices for target tokens [batch_size]
            use_ref: Whether to use reference model
            
        Returns:
            Token-mean logprobs for target sequences [batch_size]
        """
        model = self.ref_model if use_ref else self.model
        
        with torch.set_grad_enabled(not use_ref):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"][:, :-1, :]  # Shift for next-token prediction
            targets = input_ids[:, 1:]  # Target tokens
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather target token probabilities
            token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            
            # Create mask for target tokens only (not prompt tokens)
            batch_size, seq_len = targets.shape
            target_mask = torch.zeros_like(targets, dtype=torch.bool)
            
            for b in range(batch_size):
                start_idx = target_start_indices[b].item()
                if start_idx < seq_len:
                    target_mask[b, start_idx:] = attention_mask[b, start_idx+1:seq_len+1] == 1
            
            # Apply mask and compute mean
            masked_logprobs = token_logprobs.masked_fill(~target_mask, 0.0)
            n_target_tokens = target_mask.sum(dim=1).clamp(min=1)
            
            return masked_logprobs.sum(dim=1) / n_target_tokens
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters"""
        return self.model.get_num_params()
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""  
        self.model.eval()
        
    def to(self, device: Union[str, torch.device]):
        """Move models to device"""
        self.device = device
        self.model = self.model.to(device)
        self.ref_model = self.ref_model.to(device)
        return self