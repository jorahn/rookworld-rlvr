"""
Environment Task Reward Computation for RookWorld GRPO Training

This module computes structured rewards for the Environment (A:) task, which expects
the model to generate structured environment responses in the format:
A: <prev_fen>+<uci_move>+<history>+<new_fen>+<reward>+<terminated>+<truncated>

Two-tier reward system:
1. Structure Verification: Can the output be parsed correctly?
2. Content Verification: Is the parsed content accurate vs chess rules?
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass

from ..environment.chess_env import ChessEnvironment, EnvironmentResponse


@dataclass
class EnvRewardConfig:
    """Configuration for environment task rewards"""
    # Structure rewards
    r_env_structure: float = 0.1         # Correct A: format parsing
    r_env_malformed: float = -1.0        # Malformed/unparseable output penalty
    
    # Content rewards
    r_env_fen_exact: float = 1.0         # Exact FEN match bonus
    r_env_fen_similarity: float = 0.5    # Levenshtein distance-based similarity
    r_env_reward_accuracy: float = 0.3   # Reward field accuracy (regression)
    r_env_flags_accuracy: float = 0.1    # Terminated/truncated classification
    
    # Content validation thresholds
    similarity_threshold: float = 0.5    # Minimum similarity for partial credit
    reward_tolerance: float = 1.0        # Absolute tolerance for reward accuracy


class EnvRewardComputer:
    """Computes structured rewards for Environment (A:) task outputs"""
    
    def __init__(self, config: EnvRewardConfig = None):
        """
        Initialize environment reward computer
        
        Args:
            config: Reward configuration (uses defaults if None)
        """
        self.config = config or EnvRewardConfig()
        self.chess_env = ChessEnvironment()
    
    def compute_reward(
        self,
        generated_text: str,
        expected_response: EnvironmentResponse
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute total reward for environment task output
        
        Args:
            generated_text: Model's generated output (should be A: format)
            expected_response: Ground truth environment response
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Validate the prediction against expected response
        validation = self.chess_env.validate_prediction(generated_text, expected_response)
        
        reward_breakdown = {
            "structure_reward": 0.0,
            "fen_exact_reward": 0.0,
            "fen_similarity_reward": 0.0, 
            "reward_accuracy_reward": 0.0,
            "flags_accuracy_reward": 0.0,
            "malformed_penalty": 0.0,
            "total_reward": 0.0,
            # Additional info
            "is_valid_format": validation["is_valid_format"],
            "fen_exact_match": validation["fen_exact_match"],
            "fen_similarity_score": validation["fen_similarity_score"],
            "reward_accuracy": validation["reward_accuracy"],
            "flag_accuracy": validation["flag_accuracy"]
        }
        
        if validation["is_valid_format"]:
            # Structure verification passed
            reward_breakdown["structure_reward"] = self.config.r_env_structure
            
            # Content verification
            content_rewards = self._compute_content_rewards(validation)
            reward_breakdown.update(content_rewards)
        else:
            # Complete parsing failure
            reward_breakdown["malformed_penalty"] = self.config.r_env_malformed
        
        # Calculate total reward
        total_reward = (
            reward_breakdown["structure_reward"] +
            reward_breakdown["fen_exact_reward"] +
            reward_breakdown["fen_similarity_reward"] +
            reward_breakdown["reward_accuracy_reward"] +
            reward_breakdown["flags_accuracy_reward"] +
            reward_breakdown["malformed_penalty"]
        )
        reward_breakdown["total_reward"] = total_reward
        
        return total_reward, reward_breakdown
    
    def _compute_content_rewards(self, validation: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute content-based rewards from validation results
        
        Args:
            validation: Validation results from ChessEnvironment.validate_prediction
            
        Returns:
            Dictionary with content reward components
        """
        rewards = {
            "fen_exact_reward": 0.0,
            "fen_similarity_reward": 0.0,
            "reward_accuracy_reward": 0.0,
            "flags_accuracy_reward": 0.0
        }
        
        # FEN accuracy rewards
        if validation["fen_exact_match"]:
            # Exact match gets full reward
            rewards["fen_exact_reward"] = self.config.r_env_fen_exact
        else:
            # Partial credit based on similarity
            similarity = validation["fen_similarity_score"]
            if similarity >= self.config.similarity_threshold:
                rewards["fen_similarity_reward"] = similarity * self.config.r_env_fen_similarity
        
        # Reward field accuracy (regression)
        reward_accuracy = validation["reward_accuracy"]
        rewards["reward_accuracy_reward"] = reward_accuracy * self.config.r_env_reward_accuracy
        
        # Flag accuracy (classification)
        flag_accuracy = validation["flag_accuracy"]
        rewards["flags_accuracy_reward"] = flag_accuracy * self.config.r_env_flags_accuracy
        
        return rewards
    
    def create_expected_response(
        self,
        fen: str,
        uci_move: str,
        move_history: str = "",
        max_moves: int = 150
    ) -> EnvironmentResponse:
        """
        Create expected response for environment task
        
        Args:
            fen: Current board position
            uci_move: UCI move to apply
            move_history: Recent moves for context
            max_moves: Maximum moves before truncation
            
        Returns:
            Expected EnvironmentResponse
        """
        return self.chess_env.apply_move(fen, uci_move, move_history, max_moves)
    
    def compute_reward_from_components(
        self,
        fen: str,
        uci_move: str,
        generated_text: str,
        move_history: str = ""
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Convenience method to compute reward from individual components
        
        Args:
            fen: Original board position
            uci_move: UCI move that was applied
            generated_text: Model's generated A: format response
            move_history: Recent moves for context
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        try:
            # Create expected response
            expected = self.create_expected_response(fen, uci_move, move_history)
            
            # Compute reward
            return self.compute_reward(generated_text, expected)
            
        except ValueError as e:
            # Invalid move or position - return malformed penalty
            reward_breakdown = {
                "structure_reward": 0.0,
                "fen_exact_reward": 0.0,
                "fen_similarity_reward": 0.0,
                "reward_accuracy_reward": 0.0,
                "flags_accuracy_reward": 0.0,
                "malformed_penalty": self.config.r_env_malformed,
                "total_reward": self.config.r_env_malformed,
                "error": str(e)
            }
            return self.config.r_env_malformed, reward_breakdown


# Utility functions for easy import
def compute_policy_reward(
    generated_text: str,
    board,  # chess.Board - import will be handled by caller
    stockfish_analysis: Dict[str, Any],
    config = None  # PolicyRewardConfig - import will be handled by caller
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function for computing policy rewards
    
    Args:
        generated_text: Model's generated policy output
        board: Chess board position
        stockfish_analysis: Stockfish analysis results
        config: Reward configuration
        
    Returns:
        Tuple of (total_reward, reward_breakdown)
    """
    from .policy_reward import PolicyRewardComputer
    computer = PolicyRewardComputer(config)
    return computer.compute_reward(generated_text, board, stockfish_analysis)


def compute_env_reward(
    generated_text: str,
    expected_response: EnvironmentResponse,
    config: EnvRewardConfig = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function for computing environment rewards
    
    Args:
        generated_text: Model's generated environment output
        expected_response: Expected environment response
        config: Reward configuration
        
    Returns:
        Tuple of (total_reward, reward_breakdown)
    """
    computer = EnvRewardComputer(config)
    return computer.compute_reward(generated_text, expected_response)