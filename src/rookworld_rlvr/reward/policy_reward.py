"""
Policy Task Reward Computation for RookWorld GRPO Training

This module computes structured rewards for the Policy (P:) task, which expects
the model to generate Stockfish-style analysis in the format:
P: <FEN>    M: <move1> <move2> <move3> <move4> <move5>    E: <eval1> <eval2> <eval3> <eval4> <eval5>    B: <best_move>

Two-tier reward system:
1. Structure Verification: Can the output be parsed correctly?
2. Content Verification: Is the parsed content accurate vs Stockfish?
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import chess
import chess.engine
import re


@dataclass 
class PolicyRewardConfig:
    """Configuration for policy task rewards"""
    # Structure rewards
    r_policy_structure: float = 0.2      # Correct format (P:, M:, E:, B: sections)
    r_policy_parse: float = 0.1          # Parseable moves and evaluations
    r_policy_malformed: float = -1.0     # Malformed/unparseable output penalty
    
    # Content rewards
    r_policy_move_match: float = 0.5     # Multi-label: moves match Stockfish top-5
    r_policy_eval_accuracy: float = 0.2  # Regression: evaluation accuracy
    r_policy_best_move: float = 1.0      # Classification: best move matches Stockfish #1
    
    # Content validation thresholds
    eval_tolerance: float = 100.0        # Centipawn tolerance for eval accuracy
    require_exact_count: bool = True     # Require exactly 5 moves and 5 evals


@dataclass
class ParsedPolicyOutput:
    """Parsed policy task output"""
    moves: List[str]           # List of UCI moves
    evaluations: List[float]   # List of evaluation scores
    best_move: str             # Single best move
    is_valid_format: bool      # Whether parsing succeeded
    parsing_errors: List[str]  # Any parsing errors encountered


class PolicyRewardComputer:
    """Computes structured rewards for Policy (P:) task outputs"""
    
    def __init__(self, config: PolicyRewardConfig = None):
        """
        Initialize policy reward computer
        
        Args:
            config: Reward configuration (uses defaults if None)
        """
        self.config = config or PolicyRewardConfig()
    
    def compute_reward(
        self,
        generated_text: str,
        board: chess.Board,
        stockfish_analysis: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute total reward for policy task output
        
        Args:
            generated_text: Model's generated output (should be structured analysis)
            board: Chess board position
            stockfish_analysis: Ground truth from Stockfish
                Expected keys: 'top5_moves', 'top5_evals', 'best_move'
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Parse the generated output
        parsed = self.parse_policy_output(generated_text)
        
        reward_breakdown = {
            "structure_reward": 0.0,
            "parse_reward": 0.0,
            "move_match_reward": 0.0,
            "eval_accuracy_reward": 0.0,
            "best_move_reward": 0.0,
            "malformed_penalty": 0.0,
            "total_reward": 0.0
        }
        
        # Structure verification
        if parsed.is_valid_format:
            reward_breakdown["structure_reward"] = self.config.r_policy_structure
            
            # Parse verification (correct counts and types)
            if self._validate_parsed_content(parsed):
                reward_breakdown["parse_reward"] = self.config.r_policy_parse
                
                # Content verification
                content_rewards = self._compute_content_rewards(parsed, stockfish_analysis)
                reward_breakdown.update(content_rewards)
            else:
                # Parsing succeeded but content validation failed
                reward_breakdown["malformed_penalty"] = self.config.r_policy_malformed * 0.5
        else:
            # Complete parsing failure
            reward_breakdown["malformed_penalty"] = self.config.r_policy_malformed
        
        # Calculate total reward
        total_reward = sum(reward_breakdown.values())
        reward_breakdown["total_reward"] = total_reward
        
        return total_reward, reward_breakdown
    
    def parse_policy_output(self, generated_text: str) -> ParsedPolicyOutput:
        """
        Parse policy task output into structured components
        
        Expected format: " <move1> <move2> <move3> <move4> <move5>    E: <eval1> <eval2> <eval3> <eval4> <eval5>    B: <best_move>"
        
        Args:
            generated_text: Raw model output (typically starts with space)
            
        Returns:
            ParsedPolicyOutput with parsing results
        """
        parsing_errors = []
        
        try:
            # Clean the text
            text = generated_text.strip()
            
            # Look for the pattern: moves followed by E: evaluations followed by B: best_move
            # Pattern: <moves>    E: <evals>    B: <best_move>
            
            # First, try to find the E: and B: markers
            e_match = re.search(r'\s+E:\s+', text)
            b_match = re.search(r'\s+B:\s+', text)
            
            if not e_match or not b_match:
                parsing_errors.append("Missing E: or B: markers")
                return ParsedPolicyOutput([], [], "", False, parsing_errors)
            
            e_start = e_match.start()
            e_end = e_match.end()
            b_start = b_match.start()
            b_end = b_match.end()
            
            if b_start <= e_end:
                parsing_errors.append("B: marker comes before E: section ends")
                return ParsedPolicyOutput([], [], "", False, parsing_errors)
            
            # Extract sections
            moves_section = text[:e_start].strip()
            evals_section = text[e_end:b_start].strip()
            best_move_section = text[b_end:].strip()
            
            # Parse moves
            moves = []
            if moves_section:
                move_tokens = moves_section.split()
                for token in move_tokens:
                    # Basic UCI format validation (allow promotions)
                    if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', token):
                        moves.append(token)
                    else:
                        parsing_errors.append(f"Invalid move format: {token}")
            
            # Parse evaluations
            evaluations = []
            if evals_section:
                eval_tokens = evals_section.split()
                for token in eval_tokens:
                    try:
                        eval_val = float(token)
                        evaluations.append(eval_val)
                    except ValueError:
                        parsing_errors.append(f"Invalid evaluation: {token}")
            
            # Parse best move
            best_move = ""
            if best_move_section:
                best_move_tokens = best_move_section.split()
                if best_move_tokens:
                    candidate = best_move_tokens[0]
                    if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', candidate):
                        best_move = candidate
                    else:
                        parsing_errors.append(f"Invalid best move format: {candidate}")
            
            is_valid = len(parsing_errors) == 0
            
            return ParsedPolicyOutput(
                moves=moves,
                evaluations=evaluations, 
                best_move=best_move,
                is_valid_format=is_valid,
                parsing_errors=parsing_errors
            )
            
        except Exception as e:
            parsing_errors.append(f"Parsing exception: {e}")
            return ParsedPolicyOutput([], [], "", False, parsing_errors)
    
    def _validate_parsed_content(self, parsed: ParsedPolicyOutput) -> bool:
        """
        Validate that parsed content meets requirements
        
        Args:
            parsed: Parsed policy output
            
        Returns:
            True if content is valid for reward computation
        """
        if self.config.require_exact_count:
            # Require exactly 5 moves and 5 evaluations
            if len(parsed.moves) != 5:
                return False
            if len(parsed.evaluations) != 5:
                return False
        else:
            # At least some moves and evaluations
            if len(parsed.moves) == 0:
                return False
            if len(parsed.evaluations) == 0:
                return False
        
        # Must have a best move
        if not parsed.best_move:
            return False
        
        return True
    
    def _compute_content_rewards(
        self,
        parsed: ParsedPolicyOutput,
        stockfish_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute content-based rewards by comparing with Stockfish analysis
        
        Args:
            parsed: Parsed model output
            stockfish_analysis: Ground truth from Stockfish
            
        Returns:
            Dictionary with content reward components
        """
        rewards = {
            "move_match_reward": 0.0,
            "eval_accuracy_reward": 0.0, 
            "best_move_reward": 0.0
        }
        
        stockfish_moves = stockfish_analysis.get('top5_moves', [])
        stockfish_evals = stockfish_analysis.get('top5_evals', [])
        stockfish_best = stockfish_analysis.get('best_move', '')
        
        # Move matching reward (multi-label classification)
        if stockfish_moves:
            move_matches = 0
            for move in parsed.moves:
                if move in stockfish_moves:
                    move_matches += 1
            
            match_ratio = move_matches / max(len(parsed.moves), 1)
            rewards["move_match_reward"] = match_ratio * self.config.r_policy_move_match
        
        # Evaluation accuracy reward (regression)
        if stockfish_evals and parsed.evaluations:
            # Compare up to the minimum length
            min_len = min(len(parsed.evaluations), len(stockfish_evals))
            
            eval_accuracy = 0.0
            for i in range(min_len):
                pred_eval = parsed.evaluations[i]
                true_eval = stockfish_evals[i]
                
                # Compute accuracy based on centipawn error
                error = abs(pred_eval - true_eval)
                # Normalize: perfect score (0 error) = 1.0, max tolerance = 0.0
                accuracy = max(0.0, 1.0 - (error / self.config.eval_tolerance))
                eval_accuracy += accuracy
            
            eval_accuracy /= min_len  # Average accuracy
            rewards["eval_accuracy_reward"] = eval_accuracy * self.config.r_policy_eval_accuracy
        
        # Best move reward (classification)
        if stockfish_best and parsed.best_move:
            if parsed.best_move == stockfish_best:
                rewards["best_move_reward"] = self.config.r_policy_best_move
        
        return rewards


# Stockfish integration stub - will be implemented when Stockfish engine is added
def get_stockfish_analysis_stub(board: chess.Board) -> Dict[str, Any]:
    """
    Stub for Stockfish analysis - returns mock data for testing
    
    This will be replaced with actual Stockfish integration in a future phase.
    """
    # Generate some legal moves for testing
    legal_moves = list(board.legal_moves)
    
    if len(legal_moves) == 0:
        return {
            'top5_moves': [],
            'top5_evals': [],
            'best_move': ''
        }
    
    # Take up to 5 legal moves
    top5_moves = [move.uci() for move in legal_moves[:5]]
    
    # Generate mock evaluations (slightly positive for white, negative for black)
    base_eval = 0.1 if board.turn == chess.WHITE else -0.1
    top5_evals = [base_eval + i * 0.05 for i in range(len(top5_moves))]
    
    best_move = top5_moves[0] if top5_moves else ''
    
    return {
        'top5_moves': top5_moves,
        'top5_evals': top5_evals,
        'best_move': best_move
    }