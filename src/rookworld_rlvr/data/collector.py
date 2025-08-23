"""
Unified Data Collection Pipeline for RookWorld GRPO Training

This module provides unified data collection for both Policy (P:) and Environment (A:)
tasks using the same model with different prompt formats. Implements the GRPO 
group-based data collection pattern.

Key insights:
- Uses ONE model for both P: and A: tasks
- Collects G samples per position for group-relative advantages
- Mixed training: Policy and Environment tasks in same batches
- Structured reward computation for both task types
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import random
import chess
import torch

from ..train.policy import CausalLMPolicy, GenerationConfig
from ..environment.chess_env import ChessEnvironment, EnvironmentResponse
from ..reward.policy_reward import PolicyRewardComputer, get_stockfish_analysis_stub
from ..reward.env_reward import EnvRewardComputer


@dataclass
class GRPOCollectionConfig:
    """Configuration for GRPO data collection"""
    # Group settings
    group_size: int = 8              # G samples per position for GRPO
    
    # Generation settings
    max_new_tokens_policy: int = 50  # Enough for full P: structured output
    max_new_tokens_env: int = 64     # Enough for full A: structured output
    temperature: float = 0.7         # Sampling temperature
    top_k: Optional[int] = None      # Top-k filtering
    top_p: Optional[float] = 0.95    # Nucleus sampling
    
    # Task mixing
    mix_env_ratio: float = 0.25      # Fraction of environment tasks
    
    # Device
    device: str = "cpu"


class PositionBuffer:
    """Buffer for managing diverse chess positions"""
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize position buffer
        
        Args:
            capacity: Maximum number of positions to store
        """
        from collections import deque
        self.positions = deque(maxlen=capacity)
        self.opening_positions = self._load_opening_positions()
    
    def _load_opening_positions(self) -> List[str]:
        """Load common opening positions"""
        return [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Initial
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # d4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # e4 e5
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",  # Indian
        ]
    
    def add(self, fen: str):
        """Add position to buffer"""
        if fen not in self.positions:  # Avoid duplicates
            self.positions.append(fen)
    
    def sample(self, n: int = 1, prefer_opening: float = 0.3) -> List[str]:
        """
        Sample positions with optional opening bias
        
        Args:
            n: Number of positions to sample
            prefer_opening: Probability of sampling from openings vs buffer
            
        Returns:
            List of FEN position strings
        """
        samples = []
        for _ in range(n):
            if random.random() < prefer_opening or len(self.positions) < 10:
                samples.append(random.choice(self.opening_positions))
            else:
                samples.append(random.choice(self.positions))
        return samples
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.positions)


class GRPODataCollector:
    """Unified data collector for both P: and A: tasks"""
    
    def __init__(
        self,
        policy: CausalLMPolicy,
        config: GRPOCollectionConfig = None
    ):
        """
        Initialize GRPO data collector
        
        Args:
            policy: Unified policy wrapper
            config: Collection configuration
        """
        self.policy = policy
        self.config = config or GRPOCollectionConfig()
        
        # Initialize components
        self.chess_env = ChessEnvironment()
        self.policy_reward_computer = PolicyRewardComputer()
        self.env_reward_computer = EnvRewardComputer()
        self.position_buffer = PositionBuffer()
    
    def collect_policy_group(self, board: chess.Board) -> Dict[str, Any]:
        """
        Collect GRPO group for Policy (P:) task
        
        Args:
            board: Chess board position
            
        Returns:
            Dictionary with GRPO group data:
                - input_ids: Token sequences [group_size, seq_len]
                - attention_mask: Attention masks [group_size, seq_len] 
                - target_start: Start indices for generated tokens [group_size]
                - old_logprobs: Logprobs from generation [group_size]
                - rewards: Computed rewards [group_size]
                - meta: Metadata dictionary
        """
        fen = board.fen()
        
        # Create P: prompts
        prompts = self.policy.tokenizer.create_chess_prompts([fen] * self.config.group_size, "policy")
        
        # Get Stockfish analysis for reward computation
        stockfish_analysis = get_stockfish_analysis_stub(board)  # Will be replaced with real Stockfish
        
        # Generate structured outputs
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens_policy,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=True
        )
        
        outputs = self.policy.generate_batch(prompts, generation_config)
        
        # Compute rewards for each generated output
        rewards = []
        reward_breakdowns = []
        
        for generated_text in outputs["texts"]:
            reward, breakdown = self.policy_reward_computer.compute_reward(
                generated_text, board, stockfish_analysis
            )
            rewards.append(reward)
            reward_breakdowns.append(breakdown)
        
        # Prepare sequences for training
        # Combine prompts + generated text for full sequences
        full_texts = []
        for prompt, generated_text in zip(prompts, outputs["texts"]):
            full_text = prompt + generated_text
            full_texts.append(full_text)
        
        # Tokenize full sequences
        encoding = self.policy.tokenizer.encode_batch(
            full_texts,
            padding=True,
            device=self.config.device
        )
        
        # Get prompt lengths for target start calculation
        prompt_lengths = []
        for prompt in prompts:
            prompt_lengths.append(self.policy.tokenizer.get_prompt_length(prompt))
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "target_start": torch.tensor(prompt_lengths, device=self.config.device),
            "old_logprobs": outputs["seq_logprob"],
            "rewards": torch.tensor(rewards, device=self.config.device),
            "meta": {
                "task_type": "policy",
                "fen": fen,
                "stockfish_analysis": stockfish_analysis,
                "reward_breakdowns": reward_breakdowns,
                "generated_texts": outputs["texts"]
            }
        }
    
    def collect_env_group(self, board: chess.Board) -> Optional[Dict[str, Any]]:
        """
        Collect GRPO group for Environment (A:) task
        
        Args:
            board: Chess board position
            
        Returns:
            Dictionary with GRPO group data or None if no legal moves
        """
        fen = board.fen()
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        # Sample a random legal move
        move = random.choice(legal_moves)
        uci_move = move.uci()
        
        # Create expected response for reward computation
        expected_response = self.chess_env.apply_move(fen, uci_move)
        
        # Create A: prompts
        prompts = self.policy.tokenizer.create_env_prompts([fen] * self.config.group_size, [uci_move] * self.config.group_size)
        
        # Generate structured outputs
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens_env,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=True
        )
        
        outputs = self.policy.generate_batch(prompts, generation_config)
        
        # Compute rewards for each generated output
        rewards = []
        reward_breakdowns = []
        
        for generated_text in outputs["texts"]:
            reward, breakdown = self.env_reward_computer.compute_reward(
                generated_text, expected_response
            )
            rewards.append(reward)
            reward_breakdowns.append(breakdown)
        
        # Prepare sequences for training
        full_texts = []
        for prompt, generated_text in zip(prompts, outputs["texts"]):
            full_text = prompt + generated_text
            full_texts.append(full_text)
        
        # Tokenize full sequences
        encoding = self.policy.tokenizer.encode_batch(
            full_texts,
            padding=True,
            device=self.config.device
        )
        
        # Get prompt lengths for target start calculation
        prompt_lengths = []
        for prompt in prompts:
            prompt_lengths.append(self.policy.tokenizer.get_prompt_length(prompt))
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "target_start": torch.tensor(prompt_lengths, device=self.config.device),
            "old_logprobs": outputs["seq_logprob"],
            "rewards": torch.tensor(rewards, device=self.config.device),
            "meta": {
                "task_type": "environment",
                "fen": fen,
                "uci_move": uci_move,
                "expected_response": expected_response,
                "reward_breakdowns": reward_breakdowns,
                "generated_texts": outputs["texts"]
            }
        }
    
    def collect_mixed_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Collect mixed batch of Policy and Environment tasks
        
        Args:
            batch_size: Number of groups to collect
            
        Returns:
            List of GRPO group dictionaries
        """
        batch_groups = []
        
        # Sample positions for the batch
        positions = self.position_buffer.sample(batch_size)
        
        for fen in positions:
            try:
                board = chess.Board(fen)
                
                # Decide task type based on mix ratio
                if random.random() < self.config.mix_env_ratio:
                    # Environment task
                    group = self.collect_env_group(board)
                    if group is not None:
                        batch_groups.append(group)
                        # Add position back to buffer for diversity
                        self.position_buffer.add(fen)
                else:
                    # Policy task
                    group = self.collect_policy_group(board)
                    batch_groups.append(group)
                    # Add position back to buffer for diversity
                    self.position_buffer.add(fen)
                    
            except Exception as e:
                print(f"Warning: Failed to collect data for position {fen}: {e}")
                continue
        
        return batch_groups
    
    def add_positions_to_buffer(self, positions: List[str]):
        """
        Add positions to the buffer
        
        Args:
            positions: List of FEN position strings
        """
        for fen in positions:
            self.position_buffer.add(fen)
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get information about the position buffer"""
        return {
            "buffer_size": self.position_buffer.size(),
            "opening_positions": len(self.position_buffer.opening_positions),
            "total_available": self.position_buffer.size() + len(self.position_buffer.opening_positions)
        }


# Utility functions for easy import
def create_data_collector(
    policy: CausalLMPolicy,
    config: GRPOCollectionConfig = None
) -> GRPODataCollector:
    """Create a GRPODataCollector instance"""
    return GRPODataCollector(policy, config)


def sample_chess_positions(n_positions: int = 100) -> List[str]:
    """
    Sample diverse chess positions for training
    
    Args:
        n_positions: Number of positions to generate
        
    Returns:
        List of FEN position strings
    """
    chess_env = ChessEnvironment()
    return chess_env.create_sample_positions(n_positions)