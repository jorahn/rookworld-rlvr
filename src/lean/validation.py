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
        
        Expected format: M: move1 move2 move3 move4 move5  E: eval1 eval2 eval3 eval4 eval5  B: best_move
        """
        
        logger.debug(f"Validating P: completion for FEN: {fen}")
        logger.debug(f"Completion: {completion[:100]}...")
        
        rewards = {
            "structure": 0.0,  # Can we parse the format?
            "moves": 0.0,      # Are moves valid?
            "evaluations": 0.0, # Are evaluations reasonable?
            "best_move": 0.0   # Is best move good?
        }
        
        try:
            # Parse the completion format
            parsed = self._parse_policy_format(completion)
            if parsed is None:
                logger.debug("Failed to parse policy format")
                return rewards
            
            moves, evaluations, best_move = parsed
            rewards["structure"] = 0.2  # Basic structure reward
            
            logger.debug(f"Parsed - moves: {moves}, evals: {evaluations}, best: {best_move}")
            
            # Validate with chess board
            board = chess.Board(fen)
            
            # Check if moves are legal
            valid_moves = 0
            for move_uci in moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        valid_moves += 1
                except:
                    pass
            
            rewards["moves"] = (valid_moves / len(moves)) * 0.3 if moves else 0.0
            
            # Check best move
            if best_move:
                try:
                    best_move_obj = chess.Move.from_uci(best_move)
                    if best_move_obj in board.legal_moves:
                        rewards["best_move"] = 0.3
                except:
                    pass
            
            # Use Stockfish if available
            if self.engine:
                try:
                    # Get Stockfish top moves
                    info = self.engine.analyse(board, chess.engine.Limit(depth=15), multipv=5)
                    stockfish_moves = [pv.move.uci() for pv in info[:5]]
                    
                    # Compare with generated moves
                    matching_moves = sum(1 for move in moves if move in stockfish_moves)
                    rewards["moves"] += (matching_moves / 5) * 0.2  # Bonus for Stockfish agreement
                    
                    # Check if best move matches Stockfish #1
                    if best_move == stockfish_moves[0]:
                        rewards["best_move"] += 0.2  # Bonus for perfect best move
                    
                    logger.debug(f"Stockfish comparison - matching moves: {matching_moves}/5")
                    
                except Exception as e:
                    logger.debug(f"Stockfish analysis failed: {e}")
            
            logger.debug(f"P: validation rewards: {rewards}")
            
        except Exception as e:
            logger.error(f"Policy validation error: {e}")
        
        return rewards
    
    def validate_environment_completion(
        self, 
        fen: str, 
        move_uci: str,
        completion: str
    ) -> Dict[str, float]:
        """
        Validate an A: task completion using python-chess
        
        Expected: resulting position and game state after the move
        """
        
        logger.debug(f"Validating A: completion for FEN: {fen}, move: {move_uci}")
        logger.debug(f"Completion: {completion[:100]}...")
        
        rewards = {
            "structure": 0.0,  # Basic format
            "position": 0.0,   # Correct resulting position  
            "game_state": 0.0  # Correct game state (check, mate, etc.)
        }
        
        try:
            board = chess.Board(fen)
            
            # Apply the move
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                
                rewards["structure"] = 0.2  # Valid move applied
                
                # Check if completion contains the resulting FEN
                resulting_fen = board.fen()
                if resulting_fen.split()[0] in completion:  # Check position part of FEN
                    rewards["position"] = 0.5
                    logger.debug("Correct resulting position found")
                
                # Check game state indicators
                if board.is_checkmate() and ("checkmate" in completion.lower() or "mate" in completion.lower()):
                    rewards["game_state"] = 0.3
                elif board.is_check() and ("check" in completion.lower()):
                    rewards["game_state"] = 0.2
                elif not board.is_checkmate() and not board.is_check():
                    rewards["game_state"] = 0.1  # Normal continuation
                
                logger.debug(f"A: validation rewards: {rewards}")
                
            else:
                logger.debug(f"Invalid move: {move_uci}")
                
        except Exception as e:
            logger.error(f"Environment validation error: {e}")
        
        return rewards
    
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