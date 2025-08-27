"""
Reward/Advantage Scorer for GRPO Training

Processes prompt+completion tuples and returns shaped rewards with detailed logging.
Implements advantage calculation for group-relative baselines.
"""

import logging
from typing import Tuple, Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from dataset import parse_p_task, parse_a_task
from validation import (
    validate_p_format, validate_a_format,
    validate_p_best_move, validate_p_candidates, validate_p_evaluations,
    validate_a_fen, validate_a_flags, validate_a_reward,
    P_WEIGHTS, A_WEIGHTS
)

# Configure detailed logging
logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class RewardDetails:
    """Container for detailed reward information"""
    task_type: str
    format_valid: bool
    format_score: float
    field_scores: Dict[str, float]
    weighted_scores: Dict[str, float]
    total_raw_reward: float
    shaped_reward: float
    details: Dict


class RewardScorer:
    """
    Computes rewards for prompt+completion pairs with detailed validation.
    
    Features:
    - Task-specific validation (P: vs A:)
    - Weighted scoring based on field importance
    - Reward shaping for better learning
    - Detailed logging of all components
    - Group advantage calculation for GRPO
    """
    
    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        reward_shaping: str = "graduated",
        min_reward: float = -0.3,
        max_reward: float = 1.0,
        format_bonus: float = 0.1
    ):
        """
        Initialize reward scorer.
        
        Args:
            stockfish_path: Path to Stockfish for P: task validation
            reward_shaping: Type of shaping ("graduated", "linear", "binary")
            min_reward: Minimum reward (for invalid completions)
            max_reward: Maximum reward (for perfect completions)
            format_bonus: Bonus for correct format even with wrong content
        """
        self.stockfish_path = stockfish_path
        self.reward_shaping = reward_shaping
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.format_bonus = format_bonus
        
        logger.info(f"RewardScorer initialized - shaping: {reward_shaping}, "
                   f"range: [{min_reward}, {max_reward}]")
    
    def score_single(
        self,
        prompt: str,
        completion: str,
        log_details: bool = True
    ) -> Tuple[float, RewardDetails]:
        """
        Score a single prompt+completion pair.
        
        Args:
            prompt: The input prompt (P: or A: task)
            completion: The generated completion
            log_details: Whether to log detailed validation info
            
        Returns:
            (shaped_reward, RewardDetails) tuple
        """
        # Determine task type from prompt
        task_type = self._identify_task_type(prompt)
        
        if task_type == "P":
            reward, details = self._score_p_task(prompt, completion)
        elif task_type == "A":
            reward, details = self._score_a_task(prompt, completion)
        else:
            # Unknown task type
            logger.warning(f"Unknown task type for prompt: {prompt[:50]}...")
            details = RewardDetails(
                task_type="unknown",
                format_valid=False,
                format_score=0.0,
                field_scores={},
                weighted_scores={},
                total_raw_reward=0.0,
                shaped_reward=self.min_reward,
                details={"error": "Unknown task type"}
            )
            reward = self.min_reward
        
        if log_details:
            self._log_reward_details(prompt, completion, details)
        
        return reward, details
    
    def score_batch(
        self,
        prompts: List[str],
        completions: List[str],
        compute_advantages: bool = True,
        group_size: Optional[int] = None
    ) -> Tuple[np.ndarray, List[RewardDetails]]:
        """
        Score a batch of prompt+completion pairs.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            compute_advantages: Whether to compute group-relative advantages
            group_size: Size of groups for advantage calculation (None = full batch)
            
        Returns:
            (rewards_or_advantages, details_list) tuple
        """
        assert len(prompts) == len(completions)
        
        rewards = []
        details_list = []
        
        # Score each pair
        for prompt, completion in zip(prompts, completions):
            reward, details = self.score_single(prompt, completion, log_details=False)
            rewards.append(reward)
            details_list.append(details)
        
        rewards = np.array(rewards)
        
        # Log batch statistics
        logger.info(f"Batch scoring - samples: {len(prompts)}, "
                   f"mean reward: {rewards.mean():.3f}, "
                   f"std: {rewards.std():.3f}, "
                   f"min: {rewards.min():.3f}, max: {rewards.max():.3f}")
        
        # Compute advantages if requested
        if compute_advantages:
            if group_size is None:
                group_size = len(prompts)
            
            advantages = self._compute_group_advantages(rewards, group_size)
            
            logger.info(f"Advantages computed - mean: {advantages.mean():.3f}, "
                       f"std: {advantages.std():.3f}")
            
            return advantages, details_list
        else:
            return rewards, details_list
    
    def _identify_task_type(self, prompt: str) -> str:
        """Identify whether prompt is P: or A: task"""
        if prompt.strip().startswith("P:"):
            return "P"
        elif prompt.strip().startswith("A:"):
            return "A"
        else:
            return "unknown"
    
    def _score_p_task(self, prompt: str, completion: str) -> Tuple[float, RewardDetails]:
        """Score a P: (Policy) task"""
        
        # Parse prompt to get FEN
        try:
            _, _, prompt_data = parse_p_task(prompt)
            fen = prompt_data.get('fen', '')
        except:
            fen = prompt[3:].strip() if prompt.startswith("P:") else ""
        
        # Validate format
        format_score, format_details = validate_p_format(completion)
        format_valid = format_score > 0.5
        
        field_scores = {}
        weighted_scores = {}
        
        if format_valid:
            # Content validation
            
            # Best move (highest priority)
            if 'best_move' in format_details:
                best_score = validate_p_best_move(fen, format_details['best_move'], self.stockfish_path)
                field_scores['best_move'] = best_score
                weighted_scores['best_move'] = best_score * P_WEIGHTS['best_move']
            
            # Candidate moves
            if 'moves' in format_details:
                candidates_score = validate_p_candidates(fen, format_details['moves'], self.stockfish_path)
                field_scores['candidates'] = candidates_score
                weighted_scores['candidates'] = candidates_score * P_WEIGHTS['candidates']
            
            # Evaluations
            if 'evals' in format_details:
                evals_score = validate_p_evaluations(fen, format_details['evals'], self.stockfish_path)
                field_scores['evaluations'] = evals_score
                weighted_scores['evaluations'] = evals_score * P_WEIGHTS['evaluations']
        
        # Add format score
        field_scores['format'] = format_score
        weighted_scores['format'] = format_score * P_WEIGHTS['format']
        
        # Calculate total weighted reward
        total_weight = sum(P_WEIGHTS.values())
        total_weighted = sum(weighted_scores.values()) / total_weight
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(total_weighted, format_valid)
        
        details = RewardDetails(
            task_type="P",
            format_valid=format_valid,
            format_score=format_score,
            field_scores=field_scores,
            weighted_scores=weighted_scores,
            total_raw_reward=total_weighted,
            shaped_reward=shaped_reward,
            details=format_details
        )
        
        return shaped_reward, details
    
    def _score_a_task(self, prompt: str, completion: str) -> Tuple[float, RewardDetails]:
        """Score an A: (Environment) task"""
        
        # Parse prompt to get FEN, move, history
        try:
            _, _, prompt_data = parse_a_task(prompt)
            fen = prompt_data.get('fen', '')
            move = prompt_data.get('move', '')
            history = prompt_data.get('history', '')
        except:
            # Fallback parsing
            if "+" in prompt:
                parts = prompt[3:].split("+") if prompt.startswith("A:") else prompt.split("+")
                fen = parts[0].strip() if len(parts) > 0 else ""
                move = parts[1].strip() if len(parts) > 1 else ""
                history = parts[2].strip() if len(parts) > 2 else ""
            else:
                fen, move, history = "", "", ""
        
        # Validate format
        format_score, format_details = validate_a_format(completion)
        format_valid = format_score > 0.5
        
        field_scores = {}
        weighted_scores = {}
        
        if format_valid:
            # Content validation
            
            # FEN match (edit distance)
            if 'new_fen' in format_details:
                # Calculate expected FEN
                import chess
                try:
                    board = chess.Board(fen)
                    move_obj = chess.Move.from_uci(move)
                    if move_obj in board.legal_moves:
                        board.push(move_obj)
                        expected_fen = board.fen()
                    else:
                        expected_fen = fen
                except:
                    expected_fen = fen
                
                fen_score = validate_a_fen(expected_fen, format_details['new_fen'])
                field_scores['fen_match'] = fen_score
                weighted_scores['fen_match'] = fen_score * A_WEIGHTS['fen_match']
            
            # Game state flags
            if 'terminated' in format_details and 'truncated' in format_details:
                terminated = 'true' if format_details['terminated'] else 'false'
                truncated = 'true' if format_details['truncated'] else 'false'
                flags_score = validate_a_flags(fen, move, terminated, truncated)
                field_scores['game_state'] = flags_score
                weighted_scores['game_state'] = flags_score * A_WEIGHTS['game_state']
            
            # Reward value
            if 'reward' in format_details:
                reward_score = validate_a_reward(fen, move, format_details['reward'])
                field_scores['reward_value'] = reward_score
                weighted_scores['reward_value'] = reward_score * A_WEIGHTS['reward_value']
        
        # Add format score
        field_scores['format'] = format_score
        weighted_scores['format'] = format_score * A_WEIGHTS['format']
        
        # Calculate total weighted reward
        total_weight = sum(A_WEIGHTS.values())
        total_weighted = sum(weighted_scores.values()) / total_weight
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(total_weighted, format_valid)
        
        details = RewardDetails(
            task_type="A",
            format_valid=format_valid,
            format_score=format_score,
            field_scores=field_scores,
            weighted_scores=weighted_scores,
            total_raw_reward=total_weighted,
            shaped_reward=shaped_reward,
            details=format_details
        )
        
        return shaped_reward, details
    
    def _shape_reward(self, raw_reward: float, format_valid: bool) -> float:
        """
        Apply reward shaping to raw scores.
        
        Args:
            raw_reward: Raw weighted reward (0 to 1)
            format_valid: Whether format was valid
            
        Returns:
            Shaped reward value
        """
        if self.reward_shaping == "graduated":
            # Graduated rewards: 0.2, 0.4, 0.6, 0.8, 1.0
            if raw_reward < 0.2:
                shaped = self.min_reward if not format_valid else 0.2
            elif raw_reward < 0.4:
                shaped = 0.2
            elif raw_reward < 0.6:
                shaped = 0.4
            elif raw_reward < 0.8:
                shaped = 0.6
            elif raw_reward < 0.95:
                shaped = 0.8
            else:
                shaped = 1.0
                
        elif self.reward_shaping == "linear":
            # Linear scaling from min to max
            shaped = self.min_reward + (self.max_reward - self.min_reward) * raw_reward
            
        elif self.reward_shaping == "binary":
            # Binary: good (>0.5) or bad
            threshold = 0.5
            shaped = self.max_reward if raw_reward > threshold else self.min_reward
            
        else:
            # No shaping, use raw reward
            shaped = raw_reward
        
        # Add format bonus if applicable
        if format_valid and shaped < 0:
            shaped += self.format_bonus
        
        return shaped
    
    def _compute_group_advantages(self, rewards: np.ndarray, group_size: int) -> np.ndarray:
        """
        Compute group-relative advantages for GRPO.
        
        Args:
            rewards: Array of rewards
            group_size: Size of groups for baseline calculation
            
        Returns:
            Array of advantages (rewards - group_baseline)
        """
        advantages = np.zeros_like(rewards)
        
        # Process in groups
        for i in range(0, len(rewards), group_size):
            group_rewards = rewards[i:i+group_size]
            
            # Compute baseline as group mean
            baseline = group_rewards.mean()
            
            # Compute advantages
            group_advantages = group_rewards - baseline
            
            # Normalize by std if significant variance
            if group_rewards.std() > 0.01:
                group_advantages = group_advantages / (group_rewards.std() + 1e-8)
            
            advantages[i:i+group_size] = group_advantages
        
        return advantages
    
    def _log_reward_details(self, prompt: str, completion: str, details: RewardDetails):
        """Log detailed reward breakdown"""
        
        logger.info("="*60)
        logger.info(f"Task Type: {details.task_type}")
        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info(f"Completion: {completion[:100]}...")
        
        logger.info(f"Format: {'VALID' if details.format_valid else 'INVALID'} (score: {details.format_score:.3f})")
        
        if details.field_scores:
            logger.info("Field Scores:")
            for field, score in details.field_scores.items():
                weight = P_WEIGHTS.get(field, A_WEIGHTS.get(field, 1.0))
                weighted = details.weighted_scores.get(field, 0.0)
                logger.info(f"  {field:15s}: {score:.3f} (weight: {weight:.1f}, weighted: {weighted:.3f})")
        
        logger.info(f"Total Raw Reward: {details.total_raw_reward:.3f}")
        logger.info(f"Shaped Reward: {details.shaped_reward:.3f}")
        logger.info("="*60)


def compute_grpo_rewards(
    prompts: List[str],
    completions: List[str],
    stockfish_path: Optional[str] = None,
    group_size: int = 8,
    reward_shaping: str = "graduated",
    verbose: bool = False
) -> Tuple[np.ndarray, List[RewardDetails]]:
    """
    Convenience function to compute GRPO rewards with advantages.
    
    Args:
        prompts: List of prompts
        completions: List of generated completions
        stockfish_path: Path to Stockfish for validation
        group_size: Group size for advantage calculation
        reward_shaping: Type of reward shaping
        verbose: Whether to log details for each sample
        
    Returns:
        (advantages, details_list) for use in GRPO training
    """
    scorer = RewardScorer(
        stockfish_path=stockfish_path,
        reward_shaping=reward_shaping
    )
    
    if verbose:
        # Score with detailed logging
        rewards = []
        details_list = []
        for prompt, completion in zip(prompts, completions):
            reward, details = scorer.score_single(prompt, completion, log_details=True)
            rewards.append(reward)
            details_list.append(details)
        
        rewards = np.array(rewards)
        
        # Compute advantages
        advantages = scorer._compute_group_advantages(rewards, group_size)
        
        return advantages, details_list
    else:
        # Batch scoring
        return scorer.score_batch(
            prompts,
            completions,
            compute_advantages=True,
            group_size=group_size
        )


if __name__ == "__main__":
    # Test the reward scorer
    
    print("\n=== Testing P: Task Reward Scoring ===")
    
    p_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Test different quality completions
    completions = [
        "M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4",  # Perfect
        "M: e2e4 d2d4  E: 0.3 0.4  B: e2e4",  # Good format, partial content
        "M: e2e4 d2d4",  # Missing sections
        "random garbage",  # Invalid
    ]
    
    scorer = RewardScorer()
    
    for i, completion in enumerate(completions):
        print(f"\nTest {i+1}:")
        reward, details = scorer.score_single(p_prompt, completion, log_details=True)
        print(f"Final shaped reward: {reward:.3f}")
    
    print("\n=== Testing A: Task Reward Scoring ===")
    
    a_prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+,+"
    
    completions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",  # Perfect
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+false+false",  # Close FEN
        "some_fen+0.5+true+false",  # Wrong format
        "invalid",  # Garbage
    ]
    
    for i, completion in enumerate(completions):
        print(f"\nTest {i+1}:")
        reward, details = scorer.score_single(a_prompt, completion, log_details=True)
        print(f"Final shaped reward: {reward:.3f}")
    
    print("\n=== Testing Batch Scoring with Advantages ===")
    
    prompts = [p_prompt] * 4 + [a_prompt] * 4
    completions = [
        "M: e2e4 d2d4 g1f3  E: 0.3 0.35 0.28  B: e2e4",  # Good P:
        "M: e2e4",  # Bad P:
        "M: a2a3 b2b3  E: 0.1 0.1  B: a2a3",  # OK P:
        "garbage",  # Bad P:
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",  # Good A:
        "bad_fen+0+true+true",  # Bad A:
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1+0.001+false+false",  # OK A:
        "invalid",  # Bad A:
    ]
    
    advantages, details_list = compute_grpo_rewards(
        prompts,
        completions,
        group_size=4,
        verbose=False
    )
    
    print("\nBatch results:")
    for i, (adv, detail) in enumerate(zip(advantages, details_list)):
        print(f"Sample {i+1} ({detail.task_type}): "
              f"raw={detail.total_raw_reward:.3f}, "
              f"shaped={detail.shaped_reward:.3f}, "
              f"advantage={adv:.3f}")