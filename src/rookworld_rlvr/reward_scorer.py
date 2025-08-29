"""
Reward/Advantage Scorer for GRPO Training

Processes prompt+completion tuples and returns shaped rewards with detailed logging.
Implements advantage calculation for group-relative baselines.
"""

import logging
from typing import Tuple, Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import math
import re

from .dataset import parse_p_task, parse_a_task
from .validation import (
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
        format_bonus: float = 0.1,
        continuous_components: Optional[Dict[str, str]] = None
    ):
        """
        Initialize reward scorer.
        
        Args:
            stockfish_path: Path to Stockfish for P: task validation
            reward_shaping: Type of shaping ("graduated", "linear", "binary", "continuous")
            min_reward: Minimum reward (for invalid completions)
            max_reward: Maximum reward (for perfect completions)
            format_bonus: Bonus for correct format even with wrong content
            continuous_components: Dict of component names to scaling functions
                                 e.g. {"fen_similarity": "exponential", "evaluations": "linear"}
        """
        self.stockfish_path = stockfish_path
        self.reward_shaping = reward_shaping
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.format_bonus = format_bonus
        
        # Default continuous components with scaling functions
        self.continuous_components = continuous_components or {
            "fen_similarity": "exponential",  # More reward for near-perfect matches
            "evaluations": "linear",  # Direct proportional to accuracy
        }
        
        logger.info(f"RewardScorer initialized - shaping: {reward_shaping}, "
                   f"range: [{min_reward}, {max_reward}], "
                   f"continuous: {list(self.continuous_components.keys())}")
    
    def score_single(
        self,
        prompt: str,
        completion: str,
        ground_truth: Optional[str] = None,
        log_details: bool = True
    ) -> Tuple[float, RewardDetails]:
        """
        Score a single prompt+completion pair.
        
        Args:
            prompt: The input prompt (P: or A: task)
            completion: The generated completion
            ground_truth: Optional ground truth target for comparison
            log_details: Whether to log detailed validation info
            
        Returns:
            (shaped_reward, RewardDetails) tuple
        """
        # Determine task type from prompt
        task_type = self._identify_task_type(prompt)
        
        if task_type == "P":
            reward, details = self._score_p_task(prompt, completion, ground_truth)
        elif task_type == "A":
            reward, details = self._score_a_task(prompt, completion, ground_truth)
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
    
    def _score_p_task(self, prompt: str, completion: str, ground_truth: Optional[str] = None) -> Tuple[float, RewardDetails]:
        """Score a P: (Policy) task with optional ground truth comparison"""
        
        # Parse prompt to get FEN
        try:
            _, _, prompt_data = parse_p_task(prompt)
            fen = prompt_data.get('fen', '')
        except:
            fen = prompt[3:].strip() if prompt.startswith("P: ") else ""
        
        # Parse ground truth if available
        target_moves = None
        target_evals = None
        target_best = None
        if ground_truth:
            try:
                # Parse target completion
                moves_match = re.search(r'M:\s*([a-h][1-8][a-h][1-8]\w*(?:\s+[a-h][1-8][a-h][1-8]\w*)*)', ground_truth)
                if moves_match:
                    target_moves = moves_match.group(1).split()
                
                evals_match = re.search(r'E:\s*([-\d\.]+(?:\s+[-\d\.]+)*)', ground_truth)
                if evals_match:
                    try:
                        target_evals = [float(x) for x in evals_match.group(1).split()]
                    except:
                        pass
                
                best_match = re.search(r'B:\s*([a-h][1-8][a-h][1-8]\w*)', ground_truth)
                if best_match:
                    target_best = best_match.group(1)
            except:
                pass
        
        # Validate format
        format_score, format_details = validate_p_format(completion)
        format_valid = format_score > 0.5
        
        field_scores = {}
        weighted_scores = {}
        
        if format_valid:
            # Content validation - use ground truth if available
            
            # Best move (highest priority) - compare to ground truth
            if 'best_move' in format_details and target_best:
                # Direct comparison with ground truth
                best_score = 1.0 if format_details['best_move'] == target_best else 0.0
                field_scores['best_move'] = best_score
                weighted_scores['best_move'] = best_score * P_WEIGHTS['best_move']
            elif 'best_move' in format_details and not target_best:
                # Fall back to Stockfish if no ground truth
                best_score = validate_p_best_move(fen, format_details['best_move'], self.stockfish_path)
                field_scores['best_move'] = best_score
                weighted_scores['best_move'] = best_score * P_WEIGHTS['best_move']
            
            # Candidate moves - compare to ground truth
            if 'moves' in format_details and target_moves:
                # Compare moves list with ground truth
                matches = sum(1 for m in format_details['moves'] if m in target_moves)
                candidates_score = matches / len(format_details['moves']) if format_details['moves'] else 0.0
                field_scores['candidates'] = candidates_score
                weighted_scores['candidates'] = candidates_score * P_WEIGHTS['candidates']
            elif 'moves' in format_details:
                # Fall back to Stockfish
                candidates_score = validate_p_candidates(fen, format_details['moves'], self.stockfish_path)
                field_scores['candidates'] = candidates_score
                weighted_scores['candidates'] = candidates_score * P_WEIGHTS['candidates']
            
            # Evaluations - uses continuous scoring with ground truth
            if 'evals' in format_details and target_evals:
                # Compare evaluations with ground truth using MSE
                gen_evals = format_details['evals']
                if len(gen_evals) == len(target_evals):
                    errors = [(g - t)**2 for g, t in zip(gen_evals, target_evals)]
                    mse = sum(errors) / len(errors) if errors else 0.0
                    # Convert MSE to score (lower is better, max MSE ~100 for very bad evals)
                    evals_score = max(0.0, 1.0 - mse / 100.0)
                else:
                    # Length mismatch penalty
                    evals_score = 0.0
                field_scores['evaluations'] = evals_score
                
                # Apply continuous scaling for evaluation accuracy
                if "evaluations" in self.continuous_components:
                    scaling = self.continuous_components["evaluations"]
                    scaled_score = self._apply_scaling(evals_score, scaling)
                    weighted_scores['evaluations'] = scaled_score * P_WEIGHTS['evaluations']
                else:
                    weighted_scores['evaluations'] = evals_score * P_WEIGHTS['evaluations']
            elif 'evals' in format_details:
                # Fall back to Stockfish
                evals_score = validate_p_evaluations(fen, format_details['evals'], self.stockfish_path)
                field_scores['evaluations'] = evals_score
                
                if "evaluations" in self.continuous_components:
                    scaling = self.continuous_components["evaluations"]
                    scaled_score = self._apply_scaling(evals_score, scaling)
                    weighted_scores['evaluations'] = scaled_score * P_WEIGHTS['evaluations']
                else:
                    weighted_scores['evaluations'] = evals_score * P_WEIGHTS['evaluations']
        
        # Add format score
        field_scores['format'] = format_score
        weighted_scores['format'] = format_score * P_WEIGHTS['format']
        
        # Calculate total weighted reward
        total_weight = sum(P_WEIGHTS.values())
        total_weighted = sum(weighted_scores.values()) / total_weight
        
        # Apply reward shaping - use continuous if evaluation accuracy is a major component
        if "evaluations" in self.continuous_components and 'evaluations' in weighted_scores:
            # For P: tasks with continuous evaluation scoring, use weighted score directly
            shaped_reward = self._shape_reward(total_weighted, format_valid, component_name="mixed_continuous")
        else:
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
    
    def _score_a_task(self, prompt: str, completion: str, ground_truth: Optional[str] = None) -> Tuple[float, RewardDetails]:
        """Score an A: (Environment) task with optional ground truth comparison"""
        
        # Parse prompt to get FEN, move, history
        try:
            _, _, prompt_data = parse_a_task(prompt)
            fen = prompt_data.get('fen', '')
            move = prompt_data.get('move', '')
            history = prompt_data.get('history', '')
        except:
            # Fallback parsing
            if "+" in prompt:
                parts = prompt[3:].split("+") if prompt.startswith("A: ") else prompt.split("+")
                fen = parts[0].strip() if len(parts) > 0 else ""
                move = parts[1].strip() if len(parts) > 1 else ""
                history = parts[2].strip() if len(parts) > 2 else ""
            else:
                fen, move, history = "", "", ""
        
        # Parse ground truth if available
        target_fen = None
        target_reward = None
        target_terminated = None
        target_truncated = None
        if ground_truth:
            try:
                # Ground truth format: [new_FEN]+[reward]+[terminated]+[truncated]
                parts = ground_truth.split("+")
                if len(parts) >= 4:
                    target_fen = parts[0].strip()
                    target_reward = float(parts[1].strip())
                    target_terminated = parts[2].strip().lower() in ['true', '1']
                    target_truncated = parts[3].strip().lower() in ['true', '1']
            except:
                pass
        
        # Validate format
        format_score, format_details = validate_a_format(completion)
        format_valid = format_score > 0.5
        
        field_scores = {}
        weighted_scores = {}
        
        if format_valid:
            # Content validation
            
            # FEN match - use ground truth if available
            if 'new_fen' in format_details and target_fen:
                # Direct comparison with ground truth using edit distance
                fen_score = validate_a_fen(target_fen, format_details['new_fen'])
                field_scores['fen_match'] = fen_score
                
                # Apply continuous scaling for FEN similarity
                if "fen_similarity" in self.continuous_components:
                    scaling = self.continuous_components["fen_similarity"]
                    scaled_score = self._apply_scaling(fen_score, scaling)
                    weighted_scores['fen_match'] = scaled_score * A_WEIGHTS['fen_match']
                else:
                    weighted_scores['fen_match'] = fen_score * A_WEIGHTS['fen_match']
            elif 'new_fen' in format_details:
                # Fall back to calculating expected FEN
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
                
                if "fen_similarity" in self.continuous_components:
                    scaling = self.continuous_components["fen_similarity"]
                    scaled_score = self._apply_scaling(fen_score, scaling)
                    weighted_scores['fen_match'] = scaled_score * A_WEIGHTS['fen_match']
                else:
                    weighted_scores['fen_match'] = fen_score * A_WEIGHTS['fen_match']
            
            # Game state flags - use ground truth if available
            if 'terminated' in format_details and 'truncated' in format_details:
                if target_terminated is not None and target_truncated is not None:
                    # Direct comparison with ground truth
                    gen_terminated = format_details['terminated']
                    gen_truncated = format_details['truncated']
                    flags_score = 1.0 if (gen_terminated == target_terminated and gen_truncated == target_truncated) else 0.0
                else:
                    # Fall back to validation
                    terminated = 'true' if format_details['terminated'] else 'false'
                    truncated = 'true' if format_details['truncated'] else 'false'
                    flags_score = validate_a_flags(fen, move, terminated, truncated)
                field_scores['game_state'] = flags_score
                weighted_scores['game_state'] = flags_score * A_WEIGHTS['game_state']
            
            # Reward value - use ground truth if available
            if 'reward' in format_details:
                if target_reward is not None:
                    # Direct comparison with ground truth
                    gen_reward = format_details['reward']
                    # Allow small tolerance for floating point comparison
                    reward_score = 1.0 if abs(gen_reward - target_reward) < 0.01 else 0.0
                else:
                    # Fall back to validation
                    reward_score = validate_a_reward(fen, move, format_details['reward'])
                field_scores['reward_value'] = reward_score
                weighted_scores['reward_value'] = reward_score * A_WEIGHTS['reward_value']
        
        # Add format score
        field_scores['format'] = format_score
        weighted_scores['format'] = format_score * A_WEIGHTS['format']
        
        # Calculate total weighted reward
        total_weight = sum(A_WEIGHTS.values())
        total_weighted = sum(weighted_scores.values()) / total_weight
        
        # Apply reward shaping - use continuous if FEN similarity is a major component
        if "fen_similarity" in self.continuous_components and 'fen_match' in weighted_scores:
            # For A: tasks with continuous FEN scoring, use weighted score directly
            shaped_reward = self._shape_reward(total_weighted, format_valid, component_name="mixed_continuous")
        else:
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
    
    def _apply_scaling(self, value: float, scaling: str) -> float:
        """
        Apply a scaling function to a continuous value.
        
        Args:
            value: Input value (0 to 1)
            scaling: Type of scaling ("linear", "exponential", "sigmoid")
            
        Returns:
            Scaled value (0 to 1)
        """
        if scaling == "exponential":
            # Exponential scaling: rewards near-perfect matches more
            # f(x) = (e^(k*x) - 1) / (e^k - 1) where k=3
            k = 3.0
            return (math.exp(k * value) - 1) / (math.exp(k) - 1)
            
        elif scaling == "sigmoid":
            # Sigmoid scaling: S-curve with steeper middle
            # f(x) = 1 / (1 + e^(-k*(x-0.5))) where k=10
            k = 10.0
            return 1.0 / (1.0 + math.exp(-k * (value - 0.5)))
            
        elif scaling == "quadratic":
            # Quadratic scaling: x^2, rewards high values more
            return value ** 2
            
        else:  # "linear" or default
            return value
    
    def _shape_reward(self, raw_reward: float, format_valid: bool, component_name: Optional[str] = None) -> float:
        """
        Apply reward shaping to raw scores.
        
        Args:
            raw_reward: Raw weighted reward (0 to 1)
            format_valid: Whether format was valid
            component_name: Optional component name for continuous scaling
            
        Returns:
            Shaped reward value
        """
        # Check if this component should use continuous rewards
        if component_name == "mixed_continuous":
            # For mixed continuous, use linear scaling directly
            shaped = self.min_reward + (self.max_reward - self.min_reward) * raw_reward
        elif component_name and component_name in self.continuous_components:
            scaling = self.continuous_components[component_name]
            scaled = self._apply_scaling(raw_reward, scaling)
            # Map to reward range
            shaped = self.min_reward + (self.max_reward - self.min_reward) * scaled
            
        elif self.reward_shaping == "continuous":
            # Global continuous mode - linear by default
            shaped = self.min_reward + (self.max_reward - self.min_reward) * raw_reward
            
        elif self.reward_shaping == "graduated":
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
    continuous_components: Optional[Dict[str, str]] = None,
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
        continuous_components: Components to use continuous rewards for
        verbose: Whether to log details for each sample
        
    Returns:
        (advantages, details_list) for use in GRPO training
    """
    scorer = RewardScorer(
        stockfish_path=stockfish_path,
        reward_shaping=reward_shaping,
        continuous_components=continuous_components
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