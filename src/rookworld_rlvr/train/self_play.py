"""
Self-Play Manager for RookWorld GRPO Training

This module manages parallel self-play games to generate diverse chess positions
for training. It maintains multiple concurrent games and advances them using the
current policy, providing a continuous stream of varied positions.
"""

import chess
import torch
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging

from .config import GRPOConfig
from ..train.policy import CausalLMPolicy


@dataclass
class GameState:
    """State of a single self-play game."""
    
    board: chess.Board              # Current board position
    move_count: int                 # Number of moves played
    game_id: str                    # Unique identifier for this game
    is_finished: bool               # Whether game is over
    termination_reason: str         # How the game ended
    position_history: List[str]     # FEN strings of all positions


class PositionBuffer:
    """Buffer for maintaining diverse training positions.
    
    This buffer stores positions from self-play games and opening variations,
    providing a diverse set of training positions with configurable sampling.
    """
    
    def __init__(self, capacity: int = 1000, opening_weight: float = 0.3):
        """
        Initialize position buffer
        
        Args:
            capacity: Maximum number of positions to store
            opening_weight: Probability of sampling from opening positions vs game positions
        """
        self.capacity = capacity
        self.opening_weight = opening_weight
        
        # Position storage with LRU eviction
        self.positions = deque(maxlen=capacity)
        self.position_metadata = deque(maxlen=capacity)  # source, game_id, move_number
        
        # Standard opening positions for diversity
        self.opening_positions = self._initialize_openings()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized position buffer with capacity {capacity}")
    
    def _initialize_openings(self) -> List[str]:
        """Initialize common opening positions for training diversity."""
        return [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            
            # King's pawn openings
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # 1.e4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # 1.e4 e5
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # 1.e4 e5 2.Nf3
            
            # Queen's pawn openings  
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # 1.d4
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",  # 1.d4 d5
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",  # 1.d4 d5 2.c4
            
            # Sicilian Defense
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # 1.e4 c5
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # 1.e4 c5 2.Nf3
            
            # Indian defenses
            "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",  # 1.d4 g6
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",  # 1.d4 Nf6
            
            # English Opening
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",  # 1.c4
            
            # French Defense
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  # 1.e4 d6
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # 1.e4 e6
            
            # Middlegame positions
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian
            "rnbqkb1r/pp1ppppp/5n2/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 2 3",  # English
            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4",  # Queen's Gambit
            
            # Endgame positions for tactical training
            "8/8/3k4/8/8/3K4/8/8 w - - 0 1",  # King vs King
            "8/8/3k4/8/8/3KP3/8/8 w - - 0 1",  # King and Pawn vs King
            "8/8/3k4/8/3R4/3K4/8/8 w - - 0 1",  # Rook endgame
        ]
    
    def add_position(self, fen: str, metadata: Dict[str, Any] = None):
        """
        Add position to buffer
        
        Args:
            fen: FEN string of the position
            metadata: Additional metadata (source, game_id, etc.)
        """
        self.positions.append(fen)
        self.position_metadata.append(metadata or {})
    
    def add_game_positions(self, game_state: GameState):
        """
        Add all positions from a completed game
        
        Args:
            game_state: Completed game state
        """
        for move_num, fen in enumerate(game_state.position_history):
            metadata = {
                'source': 'self_play',
                'game_id': game_state.game_id,
                'move_number': move_num,
                'total_moves': len(game_state.position_history),
                'termination': game_state.termination_reason
            }
            self.add_position(fen, metadata)
    
    def sample_positions(self, n: int = 1) -> List[str]:
        """
        Sample positions for training
        
        Args:
            n: Number of positions to sample
            
        Returns:
            List of FEN strings
        """
        sampled = []
        for _ in range(n):
            if random.random() < self.opening_weight or len(self.positions) < 10:
                # Sample from openings
                fen = random.choice(self.opening_positions)
            else:
                # Sample from game positions
                fen = random.choice(self.positions)
            sampled.append(fen)
        
        return sampled
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring."""
        return {
            'total_positions': len(self.positions),
            'opening_positions': len(self.opening_positions),
            'capacity': self.capacity,
            'opening_weight': self.opening_weight,
            'utilization': len(self.positions) / self.capacity if self.capacity > 0 else 0.0
        }


class SelfPlayManager:
    """Manages multiple parallel self-play games for position generation.
    
    This manager runs several concurrent games using the current policy,
    providing a continuous stream of diverse positions for GRPO training.
    """
    
    def __init__(self, 
                 config: GRPOConfig,
                 policy: CausalLMPolicy):
        """
        Initialize self-play manager
        
        Args:
            config: GRPO training configuration
            policy: Policy for making moves in self-play
        """
        self.config = config
        self.policy = policy
        
        # Initialize games
        self.games: List[GameState] = []
        for i in range(config.n_parallel_games):
            self.games.append(self._create_new_game(f"game_{i}"))
        
        # Position buffer
        self.position_buffer = PositionBuffer(
            capacity=config.position_buffer_size,
            opening_weight=config.sample_opening_frac
        )
        
        # Statistics
        self.stats = {
            'games_completed': 0,
            'total_moves_played': 0,
            'positions_generated': 0,
            'average_game_length': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized self-play manager with {config.n_parallel_games} games")
    
    def _create_new_game(self, game_id: str) -> GameState:
        """Create a new game state."""
        return GameState(
            board=chess.Board(),
            move_count=0,
            game_id=game_id,
            is_finished=False,
            termination_reason="",
            position_history=[]
        )
    
    def _select_move(self, game_state: GameState) -> Optional[chess.Move]:
        """
        Select move using the current policy
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move or None if no legal moves
        """
        board = game_state.board
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        try:
            # Use policy to score legal moves
            fen = board.fen()
            move_ucis = [move.uci() for move in legal_moves]
            
            # Score moves using the policy
            move_scores = self.policy.score_legal_moves(fen, move_ucis)
            
            # Convert to probabilities with temperature
            probs = torch.softmax(move_scores / self.config.temperature, dim=0)
            
            # Sample move
            move_idx = torch.multinomial(probs, num_samples=1).item()
            return legal_moves[move_idx]
            
        except Exception as e:
            self.logger.warning(f"Policy move selection failed: {e}, using random")
            return random.choice(legal_moves)
    
    def _finalize_game(self, game_state: GameState):
        """
        Finalize completed game and extract positions
        
        Args:
            game_state: Completed game state
        """
        # Determine termination reason
        board = game_state.board
        if board.is_checkmate():
            winner = "white" if board.turn == chess.BLACK else "black"
            game_state.termination_reason = f"checkmate_{winner}_wins"
        elif board.is_stalemate():
            game_state.termination_reason = "stalemate"
        elif board.is_insufficient_material():
            game_state.termination_reason = "insufficient_material"
        elif board.is_seventyfive_moves():
            game_state.termination_reason = "75_move_rule"
        elif board.is_fivefold_repetition():
            game_state.termination_reason = "fivefold_repetition"
        elif game_state.move_count >= self.config.max_game_len:
            game_state.termination_reason = "max_length_reached"
        else:
            game_state.termination_reason = "unknown"
        
        # Add positions to buffer
        self.position_buffer.add_game_positions(game_state)
        
        # Update statistics
        self.stats['games_completed'] += 1
        self.stats['total_moves_played'] += game_state.move_count
        self.stats['positions_generated'] += len(game_state.position_history)
        
        if self.stats['games_completed'] > 0:
            self.stats['average_game_length'] = (
                self.stats['total_moves_played'] / self.stats['games_completed']
            )
        
        self.logger.debug(f"Game {game_state.game_id} finished: {game_state.termination_reason} "
                         f"after {game_state.move_count} moves")
    
    def advance_games(self, n_moves: int = 1):
        """
        Advance all active games by specified number of moves
        
        Args:
            n_moves: Number of moves to advance each game
        """
        for game_state in self.games:
            if game_state.is_finished:
                continue
            
            for _ in range(n_moves):
                if game_state.is_finished:
                    break
                
                # Record current position
                current_fen = game_state.board.fen()
                game_state.position_history.append(current_fen)
                
                # Check termination conditions
                if (game_state.board.is_game_over() or 
                    game_state.move_count >= self.config.max_game_len):
                    game_state.is_finished = True
                    self._finalize_game(game_state)
                    
                    # Start new game
                    new_game_id = f"{game_state.game_id}_restart_{self.stats['games_completed']}"
                    new_game = self._create_new_game(new_game_id)
                    
                    # Replace finished game
                    game_idx = self.games.index(game_state)
                    self.games[game_idx] = new_game
                    break
                
                # Select and make move
                move = self._select_move(game_state)
                if move is None:
                    game_state.is_finished = True
                    self._finalize_game(game_state)
                    break
                
                game_state.board.push(move)
                game_state.move_count += 1
    
    def get_current_positions(self) -> List[str]:
        """Get current positions from all active games."""
        return [game.board.fen() for game in self.games if not game.is_finished]
    
    def sample_training_positions(self, n: int) -> List[str]:
        """
        Sample positions for training
        
        Args:
            n: Number of positions to sample
            
        Returns:
            List of FEN strings for training
        """
        return self.position_buffer.sample_positions(n)
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for monitoring."""
        active_games = sum(1 for game in self.games if not game.is_finished)
        
        game_lengths = [game.move_count for game in self.games]
        
        stats = {
            **self.stats,
            'active_games': active_games,
            'current_game_lengths': {
                'min': min(game_lengths) if game_lengths else 0,
                'max': max(game_lengths) if game_lengths else 0,
                'mean': np.mean(game_lengths) if game_lengths else 0,
                'std': np.std(game_lengths) if game_lengths else 0
            },
            'buffer_stats': self.position_buffer.get_statistics()
        }
        
        return stats
    
    def reset_games(self):
        """Reset all games to starting positions."""
        for i, game_state in enumerate(self.games):
            if not game_state.is_finished:
                self._finalize_game(game_state)
            
            self.games[i] = self._create_new_game(f"reset_game_{i}")
        
        self.logger.info("All self-play games reset")
    
    def save_position_buffer(self, filepath: str):
        """Save position buffer to file for analysis."""
        positions_data = {
            'positions': list(self.position_buffer.positions),
            'metadata': list(self.position_buffer.position_metadata),
            'openings': self.position_buffer.opening_positions,
            'stats': self.get_game_statistics()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(positions_data, f, indent=2)
        
        self.logger.info(f"Position buffer saved to {filepath}")
    
    def load_position_buffer(self, filepath: str):
        """Load position buffer from file."""
        import json
        with open(filepath, 'r') as f:
            positions_data = json.load(f)
        
        # Restore positions
        self.position_buffer.positions.clear()
        self.position_buffer.position_metadata.clear()
        
        for pos, meta in zip(positions_data['positions'], positions_data['metadata']):
            self.position_buffer.add_position(pos, meta)
        
        self.logger.info(f"Position buffer loaded from {filepath}")


def create_self_play_manager(config: GRPOConfig, policy: CausalLMPolicy) -> SelfPlayManager:
    """
    Factory function to create self-play manager
    
    Args:
        config: GRPO training configuration
        policy: Policy for self-play moves
        
    Returns:
        Configured SelfPlayManager instance
    """
    return SelfPlayManager(config, policy)