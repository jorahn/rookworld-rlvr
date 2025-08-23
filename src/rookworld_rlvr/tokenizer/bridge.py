"""
Pure PyTorch Tokenization Bridge

This module provides a tokenization wrapper using tiktoken for GPT-2 BPE tokenization
without any transformers library dependency. Designed specifically for RookWorld-LM
GRPO training with chess-specific prompt formats.
"""

from typing import List, Dict, Any, Optional, Union
import torch
import tiktoken


class TokenizerBridge:
    """
    Pure PyTorch tokenization bridge using tiktoken
    
    Handles batch encoding/decoding for RookWorld-LM training without
    transformers library dependency.
    """
    
    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tokenizer bridge
        
        Args:
            encoding_name: tiktoken encoding name (default: "gpt2")
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.tokenizer.n_vocab
        
        # GPT-2 special tokens
        self.eos_token_id = 50256  # <|endoftext|>
        self.pad_token_id = 50256  # Use EOS as pad token (standard for GPT-2)
    
    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """
        Encode single text to token IDs
        
        Args:
            text: Input text to encode
            add_eos: Whether to add EOS token at the end
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.encode(text)
        
        if add_eos:
            tokens.append(self.eos_token_id)
            
        return tokens
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token IDs to decode
            skip_special: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Filter out pad tokens if requested
        if skip_special:
            token_ids = [t for t in token_ids if t != self.pad_token_id]
            
        return self.tokenizer.decode(token_ids)
    
    def encode_batch(
        self, 
        texts: List[str], 
        max_length: Optional[int] = None,
        padding: bool = True,
        add_eos: bool = False,
        device: Union[str, torch.device] = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batch of texts to tensors
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length (auto-detect if None)
            padding: Whether to pad sequences to same length
            add_eos: Whether to add EOS token to each sequence
            device: Device to place tensors on
            
        Returns:
            Dictionary with:
                - input_ids: Token ID tensor [batch_size, seq_len]
                - attention_mask: Attention mask [batch_size, seq_len]
                - lengths: Original sequence lengths [batch_size]
        """
        # Encode all texts
        encoded = []
        for text in texts:
            tokens = self.encode(text, add_eos=add_eos)
            encoded.append(tokens)
        
        # Determine max length
        lengths = [len(tokens) for tokens in encoded]
        if max_length is None:
            max_length = max(lengths) if lengths else 1
        else:
            # Truncate sequences that are too long
            encoded = [tokens[:max_length] for tokens in encoded]
            lengths = [min(length, max_length) for length in lengths]
        
        # Create padded tensors
        batch_size = len(texts)
        input_ids = torch.full(
            (batch_size, max_length), 
            self.pad_token_id, 
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_length), 
            dtype=torch.long,
            device=device
        )
        
        # Fill in actual tokens and attention mask
        for i, (tokens, length) in enumerate(zip(encoded, lengths)):
            input_ids[i, :length] = torch.tensor(tokens, dtype=torch.long, device=device)
            attention_mask[i, :length] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long, device=device)
        }
    
    def decode_batch(self, input_ids: torch.Tensor, skip_special: bool = True) -> List[str]:
        """
        Decode batch of token tensors to texts
        
        Args:
            input_ids: Token ID tensor [batch_size, seq_len]
            skip_special: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        texts = []
        for i in range(input_ids.size(0)):
            tokens = input_ids[i]
            text = self.decode(tokens, skip_special=skip_special)
            texts.append(text)
        
        return texts
    
    def get_prompt_length(self, text: str) -> int:
        """
        Get token length of prompt text
        
        Args:
            text: Prompt text
            
        Returns:
            Number of tokens in prompt
        """
        return len(self.encode(text))
    
    def create_chess_prompts(self, fens: List[str], task_type: str = "policy") -> List[str]:
        """
        Create RookWorld-style chess prompts
        
        Args:
            fens: List of FEN position strings
            task_type: Either "policy" or "environment"
            
        Returns:
            List of formatted prompts
        """
        prompts = []
        
        for fen in fens:
            if task_type == "policy":
                # Policy task: P: <FEN>    M:
                prompt = f"P: {fen}    M:"
            elif task_type == "environment":
                # Environment task: A: <FEN>+ (UCI move will be added later)
                prompt = f"A: {fen}+"
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            
            prompts.append(prompt)
        
        return prompts
    
    def create_env_prompts(self, fens: List[str], uci_moves: List[str]) -> List[str]:
        """
        Create environment task prompts with UCI moves
        
        Args:
            fens: List of FEN position strings  
            uci_moves: List of UCI move strings
            
        Returns:
            List of formatted A: prompts
        """
        if len(fens) != len(uci_moves):
            raise ValueError("fens and uci_moves must have same length")
        
        prompts = []
        for fen, uci in zip(fens, uci_moves):
            # Environment task format: A: <FEN>+<UCI>+
            prompt = f"A: {fen}+{uci}+"
            prompts.append(prompt)
        
        return prompts
    
    def split_generated_text(self, full_text: str, prompt: str) -> str:
        """
        Extract generated text from full sequence
        
        Args:
            full_text: Full generated sequence including prompt
            prompt: Original prompt text
            
        Returns:
            Generated text only (without prompt)
        """
        if full_text.startswith(prompt):
            return full_text[len(prompt):]
        else:
            # Fallback if prompt doesn't match exactly
            return full_text


# Convenience function for easy import
def create_tokenizer_bridge(encoding_name: str = "gpt2") -> TokenizerBridge:
    """Create a TokenizerBridge instance"""
    return TokenizerBridge(encoding_name)