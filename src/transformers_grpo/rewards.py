"""
Verifiable reward system for chess GRPO training.
Combines format validation and content scoring.
"""

import re
import chess
import chess.engine
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np


class ChessRewardScorer:
    """Lean reward scorer for chess analysis tasks."""
    
    def __init__(self, stockfish_path: Optional[str] = None):
        self.stockfish_path = stockfish_path
        self.engine = None
        if stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
    
    def __del__(self):
        if self.engine:
            self.engine.quit()
    
    def score_responses(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Score a batch of responses."""
        scores = []
        for prompt, response in zip(prompts, responses):
            score = self._score_single_response(prompt, response)
            scores.append(score)
        return scores
    
    def _score_single_response(self, prompt: str, response: str) -> float:
        """Score a single response with format + content validation."""
        # Extract FEN from prompt (assumes format like "P:<FEN> M:")
        fen_match = re.search(r'P:([^\s]+)', prompt)
        if not fen_match:
            return 0.0
        
        fen = fen_match.group(1)
        
        # Parse board
        try:
            board = chess.Board(fen)
        except ValueError:
            return 0.0
        
        # Format scoring (40% weight)
        format_score = self._score_format(response)
        
        # Content scoring (60% weight) - only if format is reasonable
        content_score = 0.0
        if format_score > 0.3:
            content_score = self._score_content(board, response)
        
        return 0.4 * format_score + 0.6 * content_score
    
    def _score_format(self, response: str) -> float:
        """Score format adherence (0.0 to 1.0)."""
        score = 0.0
        
        # Check for move notation
        if re.search(r'\b[a-h][1-8][a-h][1-8]\b|[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8]', response):
            score += 0.3
        
        # Check for evaluation
        if re.search(r'[+-]?\d+\.?\d*', response):
            score += 0.3
        
        # Check for reasonable structure
        if len(response.strip()) > 10 and len(response.strip()) < 200:
            score += 0.2
        
        # Bonus for chess keywords
        chess_keywords = ['move', 'best', 'analysis', 'position', 'evaluation']
        if any(keyword in response.lower() for keyword in chess_keywords):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_content(self, board: chess.Board, response: str) -> float:
        """Score content accuracy using basic chess validation."""
        if not board.is_valid():
            return 0.0
        
        score = 0.0
        
        # Extract moves from response
        moves = self._extract_moves(response)
        legal_moves = list(board.legal_moves)
        
        if not moves:
            return 0.2  # Minimal score for no moves
        
        # Check if suggested moves are legal
        valid_moves = 0
        for move_str in moves[:3]:  # Check first 3 moves
            try:
                move = board.parse_san(move_str) if len(move_str) < 6 else chess.Move.from_uci(move_str)
                if move in legal_moves:
                    valid_moves += 1
            except (ValueError, chess.InvalidMoveError):
                continue
        
        if moves:
            score += 0.5 * (valid_moves / min(len(moves), 3))
        
        # Stockfish comparison (if available)
        if self.engine and valid_moves > 0:
            try:
                info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
                best_move = info['pv'][0] if 'pv' in info else None
                
                if best_move:
                    best_move_str = board.san(best_move)
                    if best_move_str in response or str(best_move) in response:
                        score += 0.5
            except Exception:
                pass
        else:
            # Basic positional scoring without engine
            score += 0.3
        
        return min(score, 1.0)
    
    def _extract_moves(self, text: str) -> List[str]:
        """Extract chess moves from text."""
        # UCI format
        uci_moves = re.findall(r'\b[a-h][1-8][a-h][1-8][qrbnQRBN]?\b', text)
        
        # SAN format  
        san_moves = re.findall(r'\b(?:[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)\b', text)
        
        return uci_moves + san_moves


def create_reward_function(stockfish_path: Optional[str] = None):
    """Create reward function for TRL training."""
    scorer = ChessRewardScorer(stockfish_path)
    
    def reward_function(samples: List[Dict[str, Any]]) -> List[float]:
        prompts = [sample["prompt"] for sample in samples]
        responses = [sample["response"] for sample in samples]
        return scorer.score_responses(prompts, responses)
    
    return reward_function