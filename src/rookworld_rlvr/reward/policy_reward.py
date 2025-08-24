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
import re

from ..engine.stockfish import StockfishEngine, StockfishAnalysis


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
        stockfish_analysis: StockfishAnalysis
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute total reward for policy task output
        
        Args:
            generated_text: Model's generated output (should be structured analysis)
            board: Chess board position
            stockfish_analysis: Ground truth from StockfishAnalysis
        
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
        
        # Graduated reward system with partial credit
        reward_breakdown.update(self._compute_graduated_rewards(parsed, stockfish_analysis))
        
        # Calculate total reward
        total_reward = sum(reward_breakdown.values())
        reward_breakdown["total_reward"] = total_reward
        
        return total_reward, reward_breakdown
    
    def parse_policy_output(self, generated_text: str) -> ParsedPolicyOutput:
        """
        Parse policy task output into structured components with flexible parsing
        
        Expected format: " <move1> <move2> <move3> <move4> <move5>    E: <eval1> <eval2> <eval3> <eval4> <eval5>    B: <best_move>"
        Now supports flexible spacing and partial outputs for curriculum learning.
        
        Args:
            generated_text: Raw model output (typically starts with space)
            
        Returns:
            ParsedPolicyOutput with parsing results
        """
        parsing_errors = []
        
        try:
            # Clean the text
            text = generated_text.strip()
            
            # Try flexible parsing approach - look for E: and B: markers with any amount of whitespace
            # More forgiving regex patterns
            e_match = re.search(r'\s*E:\s*', text, re.IGNORECASE)
            b_match = re.search(r'\s*B:\s*', text, re.IGNORECASE)
            
            # If both markers found, use structured parsing
            if e_match and b_match:
                return self._parse_structured_format(text, e_match, b_match, parsing_errors)
            
            # Fallback: Try to extract moves even without perfect structure
            return self._parse_flexible_format(text, parsing_errors)
            
        except Exception as e:
            parsing_errors.append(f"Parsing exception: {e}")
            return ParsedPolicyOutput([], [], "", False, parsing_errors)
    
    def _parse_structured_format(self, text: str, e_match, b_match, parsing_errors: List[str]) -> ParsedPolicyOutput:
        """Parse text with both E: and B: markers found"""
        try:
            e_start = e_match.start()
            e_end = e_match.end()
            b_start = b_match.start()
            b_end = b_match.end()
            
            # Extract sections
            moves_section = text[:e_start].strip()
            evals_section = text[e_end:b_start].strip()
            best_move_section = text[b_end:].strip()
            
            # Parse moves
            moves = self._extract_moves(moves_section, parsing_errors)
            
            # Parse evaluations  
            evaluations = self._extract_evaluations(evals_section, parsing_errors)
            
            # Parse best move
            best_move = self._extract_best_move(best_move_section, parsing_errors)
            
            # Consider valid if we got some moves, even with minor errors
            is_valid = len(moves) > 0
            
            return ParsedPolicyOutput(
                moves=moves,
                evaluations=evaluations,
                best_move=best_move,
                is_valid_format=is_valid,
                parsing_errors=parsing_errors
            )
        except Exception as e:
            parsing_errors.append(f"Structured parsing error: {e}")
            return ParsedPolicyOutput([], [], "", False, parsing_errors)
    
    def _parse_flexible_format(self, text: str, parsing_errors: List[str]) -> ParsedPolicyOutput:
        """Flexible parsing for partial outputs - extract whatever moves we can find"""
        try:
            # Look for any UCI-format moves in the text
            moves = []
            evaluations = []
            best_move = ""
            
            # Find all potential UCI moves
            uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
            move_matches = re.findall(uci_pattern, text)
            
            # Deduplicate while preserving order
            seen_moves = set()
            for move in move_matches:
                if move not in seen_moves:
                    moves.append(move)
                    seen_moves.add(move)
            
            # Look for evaluation numbers (floats between reasonable bounds)
            eval_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
            eval_matches = re.findall(eval_pattern, text)
            
            for eval_str in eval_matches:
                try:
                    eval_val = float(eval_str)
                    # Accept reasonable evaluation range (-50 to +50)
                    if -50.0 <= eval_val <= 50.0:
                        evaluations.append(eval_val)
                except ValueError:
                    continue
            
            # Use first move as best move if no clear indication
            if moves:
                best_move = moves[0]
            
            # Consider partially valid if we found at least one move
            is_valid = len(moves) > 0
            
            if len(moves) == 0:
                parsing_errors.append("No valid UCI moves found")
            
            return ParsedPolicyOutput(
                moves=moves,
                evaluations=evaluations,
                best_move=best_move,
                is_valid_format=is_valid,
                parsing_errors=parsing_errors
            )
            
        except Exception as e:
            parsing_errors.append(f"Flexible parsing error: {e}")
            return ParsedPolicyOutput([], [], "", False, parsing_errors)
    
    def _extract_moves(self, moves_section: str, parsing_errors: List[str]) -> List[str]:
        """Extract moves from moves section"""
        moves = []
        if moves_section:
            move_tokens = moves_section.split()
            for token in move_tokens:
                # Basic UCI format validation (allow promotions)
                if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', token):
                    moves.append(token)
                else:
                    # Don't treat invalid format as hard error for flexibility
                    parsing_errors.append(f"Invalid move format: {token}")
        return moves
    
    def _extract_evaluations(self, evals_section: str, parsing_errors: List[str]) -> List[float]:
        """Extract evaluations from evaluations section"""
        evaluations = []
        if evals_section:
            eval_tokens = evals_section.split()
            for token in eval_tokens:
                try:
                    eval_val = float(token)
                    # Accept reasonable evaluation range
                    if -50.0 <= eval_val <= 50.0:
                        evaluations.append(eval_val)
                    else:
                        parsing_errors.append(f"Evaluation out of range: {token}")
                except ValueError:
                    parsing_errors.append(f"Invalid evaluation: {token}")
        return evaluations
    
    def _extract_best_move(self, best_move_section: str, parsing_errors: List[str]) -> str:
        """Extract best move from best move section"""
        best_move = ""
        if best_move_section:
            best_move_tokens = best_move_section.split()
            if best_move_tokens:
                candidate = best_move_tokens[0]
                if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', candidate):
                    best_move = candidate
                else:
                    parsing_errors.append(f"Invalid best move format: {candidate}")
        return best_move
    
    def _compute_graduated_rewards(self, parsed: ParsedPolicyOutput, stockfish_analysis: StockfishAnalysis) -> Dict[str, float]:
        """
        Compute graduated rewards with partial credit for curriculum learning
        
        Reward levels:
        - 0.2: Found some valid moves (structure attempt)
        - 0.4: Found moves with some evaluations (partial parse)
        - 0.6: Majority of moves/evals valid (good parse)  
        - 0.8: Most content matches Stockfish (good content)
        - 1.0: Perfect match (full reward)
        
        Args:
            parsed: Parsed policy output
            stockfish_analysis: Ground truth analysis
            
        Returns:
            Dictionary with reward breakdown
        """
        rewards = {
            "structure_reward": 0.0,
            "parse_reward": 0.0, 
            "move_match_reward": 0.0,
            "eval_accuracy_reward": 0.0,
            "best_move_reward": 0.0,
            "malformed_penalty": 0.0
        }
        
        # Level 1: Structure attempt (0.2) - Found at least one valid move
        if len(parsed.moves) > 0:
            rewards["structure_reward"] = 0.2
            
            # Level 2: Partial parse (0.4) - Has both moves and evaluations  
            if len(parsed.moves) > 0 and len(parsed.evaluations) > 0:
                rewards["parse_reward"] = 0.2  # Additional 0.2 (total 0.4)
                
                # Level 3: Good parse (0.6) - Has reasonable number of elements
                move_count_good = 2 <= len(parsed.moves) <= 6
                eval_count_good = 2 <= len(parsed.evaluations) <= 6
                
                if move_count_good and eval_count_good:
                    rewards["move_match_reward"] = 0.2  # Additional 0.2 (total 0.6)
                    
                    # Level 4: Content matching (0.8) - Some moves match Stockfish
                    content_score = self._compute_content_match_score(parsed, stockfish_analysis)
                    if content_score > 0.3:  # At least 30% content match
                        rewards["eval_accuracy_reward"] = 0.2  # Additional 0.2 (total 0.8)
                        
                        # Level 5: Excellent match (1.0) - High content accuracy
                        if content_score > 0.7:  # At least 70% content match
                            rewards["best_move_reward"] = 0.2  # Additional 0.2 (total 1.0)
        
        # Small penalty for completely empty output (but not -1.0)
        if len(parsed.moves) == 0 and len(parsed.evaluations) == 0:
            rewards["malformed_penalty"] = -0.1  # Much smaller penalty
        
        return rewards
    
    def _compute_content_match_score(self, parsed: ParsedPolicyOutput, stockfish_analysis: StockfishAnalysis) -> float:
        """Compute how well parsed content matches Stockfish analysis"""
        if not stockfish_analysis or not parsed.moves:
            return 0.0
        
        total_score = 0.0
        components = 0
        
        # Move matching score
        if stockfish_analysis.top5_moves:
            move_matches = 0
            for move in parsed.moves[:5]:  # Check up to 5 moves
                if move in stockfish_analysis.top5_moves:
                    move_matches += 1
            move_score = move_matches / min(5, len(parsed.moves))
            total_score += move_score
            components += 1
        
        # Best move matching
        if stockfish_analysis.best_move and parsed.best_move:
            if parsed.best_move == stockfish_analysis.best_move:
                total_score += 1.0
            components += 1
        
        # Evaluation accuracy (if both have evaluations)
        if stockfish_analysis.top5_evals and parsed.evaluations:
            eval_score = self._compute_eval_accuracy(parsed.evaluations, stockfish_analysis.top5_evals)
            total_score += eval_score
            components += 1
        
        return total_score / max(1, components)
    
    def _compute_eval_accuracy(self, parsed_evals: List[float], stockfish_evals: List[float]) -> float:
        """Compute evaluation accuracy score"""
        if not parsed_evals or not stockfish_evals:
            return 0.0
        
        # Compare up to min(len) evaluations
        comparisons = min(len(parsed_evals), len(stockfish_evals))
        if comparisons == 0:
            return 0.0
        
        accurate_count = 0
        for i in range(comparisons):
            # Consider accurate if within tolerance
            diff = abs(parsed_evals[i] - stockfish_evals[i])
            if diff <= self.config.eval_tolerance / 100.0:  # Convert centipawns to eval scale
                accurate_count += 1
        
        return accurate_count / comparisons
    
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
        stockfish_analysis: StockfishAnalysis
    ) -> Dict[str, float]:
        """
        Compute content-based rewards by comparing with Stockfish analysis
        
        Args:
            parsed: Parsed model output
            stockfish_analysis: Ground truth from StockfishAnalysis
            
        Returns:
            Dictionary with content reward components
        """
        rewards = {
            "move_match_reward": 0.0,
            "eval_accuracy_reward": 0.0, 
            "best_move_reward": 0.0
        }
        
        stockfish_moves = stockfish_analysis.top5_moves
        stockfish_evals = stockfish_analysis.top5_evals
        stockfish_best = stockfish_analysis.best_move
        
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


# Convenience functions for integration
def compute_policy_reward(
    generated_text: str,
    board: chess.Board,
    stockfish_engine: StockfishEngine,
    config: Optional[PolicyRewardConfig] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function to compute policy reward with automatic Stockfish analysis
    
    Args:
        generated_text: Model's generated output
        board: Chess board position
        stockfish_engine: Stockfish engine instance
        config: Reward configuration (uses defaults if None)
        
    Returns:
        Tuple of (total_reward, reward_breakdown)
    """
    # Get Stockfish analysis
    stockfish_analysis = stockfish_engine.analyze(board)
    
    # Compute reward
    reward_computer = PolicyRewardComputer(config)
    return reward_computer.compute_reward(generated_text, board, stockfish_analysis)


def create_policy_reward_computer(config: Optional[PolicyRewardConfig] = None) -> PolicyRewardComputer:
    """
    Factory function to create policy reward computer
    
    Args:
        config: Reward configuration (uses defaults if None)
        
    Returns:
        PolicyRewardComputer instance
    """
    return PolicyRewardComputer(config)