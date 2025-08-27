"""
Lean Validation System for RookWorld GRPO Training

Simple validation using Stockfish for P: tasks and python-chess for A: tasks.
"""

import logging
import chess
import chess.engine
import re
from typing import List, Dict, Tuple, Optional
import subprocess
import os

logger = logging.getLogger(__name__)


class LeanValidator:
    """Minimal validation system for P: and A: tasks"""
    
    def __init__(self, stockfish_path: Optional[str] = None):
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.engine = None
        
        logger.info(f"Validator initialized with Stockfish: {self.stockfish_path}")
    
    def _find_stockfish(self) -> str:
        """Find Stockfish executable"""
        candidates = [
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish", 
            "stockfish"
        ]
        
        for path in candidates:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, 
                                      timeout=5)
                if result.returncode == 0:
                    logger.info(f"Found Stockfish at: {path}")
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        logger.warning("Stockfish not found, P: task validation will be limited")
        return "stockfish"  # fallback
    
    def start_engine(self):
        """Start Stockfish engine"""
        if self.engine is None:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                logger.info("Stockfish engine started")
            except Exception as e:
                logger.error(f"Failed to start Stockfish: {e}")
                self.engine = None
    
    def stop_engine(self):
        """Stop Stockfish engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None
            logger.info("Stockfish engine stopped")
    
    def validate_policy_completion(
        self, 
        fen: str, 
        completion: str
    ) -> Dict[str, float]:
        """
        Validate a P: task completion using Stockfish
        
        NEW SPECIFICATION:
        Expected format: M: [top-5-moves in UCI] E: [centipawn eval after top-5-moves] B: [best-move in UCI]
        
        Reward priority (decreasing importance):
        1. Best move accuracy (#1 key metric)
        2. Correct format (ignore padding)  
        3. Correctness of generated top move candidates (how many are in actual top 5)
        4. Accuracy of centipawn evaluations (regression scoring)
        """
        
        logger.debug(f"Validating P: completion for FEN: {fen}")
        logger.debug(f"Completion: {completion[:100]}...")
        
        rewards = {
            "best_move": 0.0,       # #1 priority - best move accuracy
            "format": 0.0,          # #2 priority - correct format  
            "move_candidates": 0.0, # #3 priority - top move accuracy
            "evaluations": 0.0      # #4 priority - eval accuracy
        }
        
        try:
            # Parse the completion format
            parsed = self._parse_policy_format(completion)
            if parsed is None:
                logger.debug("Failed to parse policy format")
                return rewards
            
            moves, evaluations, best_move = parsed
            rewards["format"] = 1.0  # Correct format parsed
            
            logger.debug(f"Parsed - moves: {moves}, evals: {evaluations}, best: {best_move}")
            
            # Validate with chess board
            board = chess.Board(fen)
            
            # Get Stockfish analysis for ground truth
            stockfish_moves = []
            stockfish_evals = []
            if self.engine:
                try:
                    info = self.engine.analyse(board, chess.engine.Limit(depth=15), multipv=5)
                    stockfish_moves = [pv.move.uci() for pv in info[:5]]
                    stockfish_evals = [pv.score.relative.score(mate_score=10000) for pv in info[:5]]
                    
                    logger.debug(f"Stockfish top 5: {stockfish_moves}")
                    logger.debug(f"Stockfish evals: {stockfish_evals}")
                    
                except Exception as e:
                    logger.debug(f"Stockfish analysis failed: {e}")
            
            # #1 PRIORITY: Best move accuracy (key metric)
            if best_move and stockfish_moves:
                try:
                    best_move_obj = chess.Move.from_uci(best_move)
                    if best_move_obj in board.legal_moves:
                        if best_move == stockfish_moves[0]:
                            rewards["best_move"] = 1.0  # Perfect match with Stockfish #1
                        elif best_move in stockfish_moves[:3]:
                            rewards["best_move"] = 0.7  # In top 3
                        elif best_move in stockfish_moves:
                            rewards["best_move"] = 0.5  # In top 5
                        else:
                            rewards["best_move"] = 0.1  # Legal but not optimal
                except:
                    rewards["best_move"] = 0.0  # Illegal move
            
            # #3 PRIORITY: Move candidates accuracy (how many generated moves are in actual top 5)
            if moves and stockfish_moves:
                legal_generated_moves = []
                for move_uci in moves:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            legal_generated_moves.append(move_uci)
                    except:
                        pass
                
                # Count matches with Stockfish top 5
                matching_count = sum(1 for move in legal_generated_moves if move in stockfish_moves)
                if legal_generated_moves:
                    rewards["move_candidates"] = matching_count / len(legal_generated_moves)
                    logger.debug(f"Move candidates: {matching_count}/{len(legal_generated_moves)} match top 5")
            
            # #4 PRIORITY: Evaluation accuracy (regression scoring against Stockfish evals)
            if evaluations and stockfish_evals and len(evaluations) == len(stockfish_evals):
                eval_errors = []
                for gen_eval, sf_eval in zip(evaluations, stockfish_evals):
                    if sf_eval is not None:
                        # Calculate relative error (normalize by magnitude + 100 to avoid division by zero)
                        error = abs(gen_eval - sf_eval) / (abs(sf_eval) + 100)
                        eval_errors.append(error)
                
                if eval_errors:
                    # Convert to reward: lower error = higher reward
                    avg_error = sum(eval_errors) / len(eval_errors)
                    rewards["evaluations"] = max(0.0, 1.0 - avg_error)  # Cap at 1.0
                    logger.debug(f"Evaluation error: {avg_error:.3f}, reward: {rewards['evaluations']:.3f}")
            
            logger.debug(f"P: validation rewards: {rewards}")
            
        except Exception as e:
            logger.error(f"Policy validation error: {e}")
        
        return rewards
    
    def validate_environment_completion(
        self, 
        fen: str, 
        move_uci: str,
        history: str,
        completion: str
    ) -> Dict[str, float]:
        """
        Validate an A: task completion using python-chess
        
        NEW SPECIFICATION:
        Expected format: [new FEN]+[reward]+[terminated]+[truncated]
        
        Reward priority (decreasing importance):
        1. Correct format (number of sections, + delimited)
        2. Binary new FEN exact match (or Levenshtein distance for smoothness)  
        3. Binary terminated (game ended) & truncated (illegal move)
        4. Reward value (1.0 for checkmate, 0.5 for draw/stalemate, 0.001 for legal continuation)
        """
        
        logger.debug(f"Validating A: completion for FEN: {fen}, move: {move_uci}")
        logger.debug(f"History: {history}, Completion: {completion[:100]}...")
        
        rewards = {
            "format": 0.0,      # #1 priority - correct format
            "fen_match": 0.0,   # #2 priority - FEN accuracy
            "game_state": 0.0,  # #3 priority - terminated/truncated flags  
            "reward_value": 0.0 # #4 priority - reward accuracy
        }
        
        try:
            board = chess.Board(fen)
            
            # Apply the move
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                logger.debug(f"Invalid move: {move_uci}")
                # For truncated cases, expect truncated=True
                if "+" in completion:
                    parts = completion.split("+")
                    if len(parts) >= 4:
                        rewards["format"] = 1.0
                        if parts[3].strip().lower() in ["true", "1"]:  # truncated flag
                            rewards["game_state"] = 1.0
                return rewards
            
            # Valid move - apply it
            board.push(move)
            resulting_fen = board.fen()
            
            # Parse completion format: [new FEN]+[reward]+[terminated]+[truncated]
            if "+" in completion:
                parts = completion.split("+")
                if len(parts) >= 4:
                    rewards["format"] = 1.0  # Correct number of sections
                    
                    completion_fen = parts[0].strip()
                    completion_reward = parts[1].strip()
                    completion_terminated = parts[2].strip()
                    completion_truncated = parts[3].strip()
                    
                    logger.debug(f"Parsed completion - FEN: {completion_fen[:30]}..., "
                               f"reward: {completion_reward}, term: {completion_terminated}, trunc: {completion_truncated}")
                    
                    # #2 PRIORITY: FEN match (binary or Levenshtein for smoothness)
                    if completion_fen == resulting_fen:
                        rewards["fen_match"] = 1.0  # Perfect match
                    else:
                        # Use Levenshtein distance for partial credit
                        fen_similarity = 1.0 - (self._levenshtein_distance(completion_fen, resulting_fen) / 
                                               max(len(completion_fen), len(resulting_fen)))
                        rewards["fen_match"] = max(0.0, fen_similarity)
                    
                    # #3 PRIORITY: Game state flags (terminated/truncated)
                    expected_terminated = board.is_game_over()
                    expected_truncated = False  # Move was legal
                    
                    terminated_correct = (completion_terminated.lower() in ["true", "1"]) == expected_terminated
                    truncated_correct = (completion_truncated.lower() in ["true", "1"]) == expected_truncated
                    
                    if terminated_correct and truncated_correct:
                        rewards["game_state"] = 1.0
                    elif terminated_correct or truncated_correct:
                        rewards["game_state"] = 0.5
                    
                    # #4 PRIORITY: Reward value accuracy
                    try:
                        reward_val = float(completion_reward)
                        expected_reward = self._calculate_expected_reward(board)
                        
                        # Check if reward matches expected value
                        if abs(reward_val - expected_reward) < 0.1:
                            rewards["reward_value"] = 1.0
                        elif abs(reward_val - expected_reward) < 0.3:
                            rewards["reward_value"] = 0.5
                        else:
                            rewards["reward_value"] = 0.1
                            
                    except ValueError:
                        rewards["reward_value"] = 0.0
                    
                    logger.debug(f"Expected - terminated: {expected_terminated}, reward: {self._calculate_expected_reward(board)}")
                    
            logger.debug(f"A: validation rewards: {rewards}")
                
        except Exception as e:
            logger.error(f"Environment validation error: {e}")
        
        return rewards
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_expected_reward(self, board: chess.Board) -> float:
        """Calculate expected reward based on game state"""
        if board.is_checkmate():
            return 1.0  # Checkmate by active player
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition():
            return 0.5  # Draw or stalemate
        else:
            return 0.001  # Legal move, play continues
    
    def _parse_policy_format(self, completion: str) -> Optional[Tuple[List[str], List[float], str]]:
        """Parse P: task completion format"""
        
        try:
            # Look for M: section
            moves_match = re.search(r'M:\s*([a-h][1-8][a-h][1-8]\w*(?:\s+[a-h][1-8][a-h][1-8]\w*)*)', completion)
            if not moves_match:
                return None
            
            moves = moves_match.group(1).split()
            
            # Look for E: section
            evals_match = re.search(r'E:\s*([-\d\.]+(?:\s+[-\d\.]+)*)', completion)
            evaluations = []
            if evals_match:
                try:
                    evaluations = [float(x) for x in evals_match.group(1).split()]
                except:
                    evaluations = []
            
            # Look for B: section
            best_match = re.search(r'B:\s*([a-h][1-8][a-h][1-8]\w*)', completion)
            best_move = best_match.group(1) if best_match else ""
            
            return moves, evaluations, best_move
            
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None