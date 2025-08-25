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
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        config,
        device: Union[str, torch.device] = "cpu",
        torch_dtype: torch.dtype = torch.float32
    ):
        """
        Initialize unified policy wrapper with pre-loaded models
        
        Args:
            model: Pre-loaded trainable model
            ref_model: Pre-loaded reference model (frozen)
            config: Training configuration
            device: Device models are on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = TokenizerBridge()
        
        # Use pre-loaded models
        self.model = model
        self.ref_model = ref_model
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_new_tokens: int = 64,
        top_p: float = 0.95
    ) -> str:
        """
        Generate single response for evaluation purposes
        
        Args:
            prompt: Single prompt string
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling threshold
            
        Returns:
            Generated text string (without prompt)
        """
        config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        result = self.generate_batch([prompt], config)
        return result['texts'][0]
    
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
            generated_logprobs.append(logprobs.item() if logprobs.dim() == 0 else logprobs[0])   # Handle scalar or tensor
        
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
        
        # Detach all tensors to prevent memory leaks
        return {
            "sequences": padded_sequences.detach(),
            "generated_ids": generated_ids.detach(),
            "seq_logprob": seq_logprobs.detach(),
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
        # Ensure input tensors are on correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        
        sequence = input_ids.clone()
        total_logprob = torch.tensor(0.0, device=self.device)
        
        # Create initial attention mask
        current_attention_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
        
        for _ in range(generation_config.max_new_tokens):
            # Forward pass
            outputs = self.model(sequence, attention_mask=current_attention_mask)
            logits = outputs["logits"]
            next_token_logits = logits[0, -1, :] # Last position logits
            
            # Apply temperature
            if generation_config.temperature != 1.0:
                next_token_logits = next_token_logits / generation_config.temperature
            
            # Apply top-k filtering
            if generation_config.top_k is not None and generation_config.top_k > 0:
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
            
            # Append to sequence (ensure tensor is on correct device)
            next_token = next_token.to(self.device)
            sequence = torch.cat([sequence, next_token.unsqueeze(0)], dim=1)
            
            # Expand attention mask for new token
            new_attention = torch.ones((1, 1), device=self.device, dtype=current_attention_mask.dtype)
            current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
            
            # Check for early stopping
            if (generation_config.pad_token_id is not None and 
                next_token.item() == generation_config.pad_token_id):
                break
        
        # Detach tensors and cleanup intermediate variables\n        sequence = sequence.detach()\n        total_logprob = total_logprob.detach()\n        \n        # Cleanup intermediate tensors\n        del current_attention_mask, next_token_logits\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n        \n        return sequence, total_logprob
    
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
        
        # Ensure tensors are on the correct device
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device) if attention_mask is not None else None
        target_start_indices = target_start_indices.to(model_device)
        
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
    
    @torch.no_grad()
    def score_legal_moves(
        self, 
        fen: str, 
        legal_moves: List[str]
    ) -> torch.Tensor:
        """
        Efficiently score all legal moves for a position using batch processing
        
        Args:
            fen: Chess position in FEN notation
            legal_moves: List of legal moves in UCI format
            
        Returns:
            Logprob scores for each legal move [len(legal_moves)]
        """
        if not legal_moves:
            return torch.tensor([])
        
        # Create prompts for all legal moves
        base_prompt = f"P: {fen}    M:"
        move_prompts = [f"{base_prompt} {move}" for move in legal_moves]
        
        # Tokenize all prompts in batch
        encoded = self.tokenizer.encode_batch(
            move_prompts,
            max_length=self.config.max_positions,
            padding=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Calculate where the move tokens start for each prompt
        # Need to find target start index by searching for 'M:' pattern
        base_encoded = self.tokenizer.encode_batch([base_prompt])
        base_tokens = base_encoded['input_ids'][0]  # Get first sequence from batch
        
        # Find the actual target start position by looking for 'M:' pattern
        target_start_idx = None
        for j in range(len(base_tokens) - 1):
            current_decoded = self.tokenizer.decode([base_tokens[j].item()]).strip()
            next_decoded = self.tokenizer.decode([base_tokens[j + 1].item()]).strip()
            if current_decoded == 'M' and next_decoded == ':':
                target_start_idx = j + 2  # Start after both 'M' and ':'
                break
            elif current_decoded.endswith('M') and next_decoded == ':':
                target_start_idx = j + 2
                break
            elif current_decoded == 'M:':
                target_start_idx = j + 1
                break
        
        # Fallback to base_length if no pattern found (shouldn't happen with our format)
        if target_start_idx is None:
            target_start_idx = len(base_tokens)
            
        target_start_indices = torch.full((len(legal_moves),), target_start_idx, dtype=torch.long)
        
        # Forward pass through model (batch all moves at once)
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision, dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"][:, :-1, :]  # Shift for next token prediction
            targets = input_ids[:, 1:]
            
            # Compute log probabilities for move tokens only
            log_probs = F.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            
            # Create masks for move tokens (exclude base prompt)
            batch_size, seq_len = targets.shape  
            move_masks = torch.zeros_like(targets, dtype=torch.bool)
            
            for i in range(batch_size):
                # Mask tokens starting from the move (after base prompt)
                start_idx = max(0, target_start_idx - 1)  # -1 for shift
                end_idx = attention_mask[i, 1:].sum().item()  # Actual sequence length
                if start_idx < end_idx:
                    move_masks[i, start_idx:end_idx] = True
            
            # Apply mask and compute mean logprob per move
            masked_logprobs = token_logprobs.masked_fill(~move_masks, 0.0)
            n_move_tokens = move_masks.sum(dim=1).clamp(min=1)
            move_scores = masked_logprobs.sum(dim=1) / n_move_tokens
            
        return move_scores
    
    def score_legal_moves_batch(
        self, 
        positions_and_moves: List[Tuple[str, List[str]]]
    ) -> List[torch.Tensor]:
        """
        Score legal moves for multiple positions efficiently
        
        Args:
            positions_and_moves: List of (fen, legal_moves) tuples
            
        Returns:
            List of score tensors, one per position
        """
        if not positions_and_moves:
            return []
        
        # Flatten all position-move combinations
        all_prompts = []
        position_indices = []
        move_counts = []
        
        for pos_idx, (fen, legal_moves) in enumerate(positions_and_moves):
            if not legal_moves:
                move_counts.append(0)
                continue
                
            base_prompt = f"P: {fen}    M:"
            for move in legal_moves:
                all_prompts.append(f"{base_prompt} {move}")
                position_indices.append(pos_idx)
            move_counts.append(len(legal_moves))
        
        if not all_prompts:
            return [torch.tensor([]) for _ in positions_and_moves]
        
        # Process all prompts in large batch
        encoded = self.tokenizer.encode_batch(
            all_prompts,
            max_length=self.config.max_positions,
            padding=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get scores for all prompts
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision, dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute scores efficiently (simplified version)
            logits = outputs["logits"][:, -1, :]  # Use last token logits as proxy
            scores = F.log_softmax(logits, dim=-1).max(dim=-1)[0]
        
        # Split scores back by position
        result_scores = []
        start_idx = 0
        
        for count in move_counts:
            if count == 0:
                result_scores.append(torch.tensor([]))
            else:
                end_idx = start_idx + count
                result_scores.append(scores[start_idx:end_idx])
                start_idx = end_idx
        
        return result_scores
    
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