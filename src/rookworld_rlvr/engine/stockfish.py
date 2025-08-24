"""
Stockfish Engine Integration for RookWorld GRPO Training

This module provides a robust interface to Stockfish chess engine for generating
ground truth analysis data for the Policy (P:) task reward computation.
"""

import chess
import chess.engine
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StockfishAnalysis:
    """Structured Stockfish analysis result for P: task ground truth."""
    
    top5_moves: List[str]
    """Top 5 moves in UCI format (e.g., ['e2e4', 'g1f3', 'd2d4', ...])."""
    
    top5_evals: List[float] 
    """Evaluations in centipawns for top 5 moves (e.g., [15, -5, 10, ...])."""
    
    best_move: str
    """Best move in UCI format. For white: first move. For black: last move (matches RookWorld training)."""
    
    depth: int
    """Search depth achieved by Stockfish."""
    
    analysis_time: float
    """Actual time spent on analysis in seconds."""


class StockfishEngine:
    """Production-ready Stockfish engine wrapper for GRPO training.
    
    This class handles:
    - Engine lifecycle management with proper cleanup
    - Robust error handling and fallback behavior  
    - Analysis caching for efficiency during training
    - RookWorld-compatible output formatting
    """
    
    def __init__(self, 
                 stockfish_path: Optional[str] = None,
                 time_limit: float = 0.1,
                 multipv: int = 5,
                 cache_size: int = 1000):
        """Initialize Stockfish engine.
        
        Args:
            stockfish_path: Path to Stockfish binary. If None, searches PATH.
            time_limit: Analysis time limit per position in seconds.
            multipv: Number of principal variations to analyze (top-N moves).
            cache_size: Maximum number of positions to cache analysis for.
        """
        self.stockfish_path = stockfish_path
        self.time_limit = time_limit
        self.multipv = multipv
        self.cache_size = cache_size
        
        # Analysis cache for efficiency
        self._cache: Dict[str, StockfishAnalysis] = {}
        self._cache_access_order: List[str] = []
        
        # Engine instance (lazy initialization)
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._engine_info: Optional[Dict[str, Any]] = None
        
        # Statistics
        self.stats = {
            'analyses_requested': 0,
            'cache_hits': 0,
            'engine_analyses': 0,
            'fallback_used': 0,
            'total_engine_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_engine(self) -> bool:
        """Initialize Stockfish engine with error handling.
        
        Returns:
            True if engine initialized successfully, False otherwise.
        """
        if self._engine is not None:
            return True
        
        try:
            if self.stockfish_path:
                # Use specified path
                engine_path = Path(self.stockfish_path)
                if not engine_path.exists():
                    self.logger.error(f"Stockfish not found at {engine_path}")
                    return False
            else:
                # Search in PATH
                try:
                    engine_path = "stockfish"
                    # Test if stockfish is available
                    import shutil
                    if shutil.which("stockfish") is None:
                        self.logger.warning("Stockfish not found in PATH")
                        return False
                except ImportError:
                    self.logger.warning("Could not check for Stockfish in PATH")
                    return False
            
            # Initialize engine
            self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            
            # Configure engine options
            self._engine.configure({"MultiPV": self.multipv})
            
            # Get engine info
            self._engine_info = {
                'name': self._engine.id.get('name', 'Unknown'),
                'author': self._engine.id.get('author', 'Unknown'),
                'version': str(self._engine.id)
            }
            
            self.logger.info(f"Initialized Stockfish: {self._engine_info['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Stockfish: {e}")
            self._engine = None
            return False
    
    def _cache_key(self, board: chess.Board) -> str:
        """Generate cache key for a board position."""
        return board.fen()
    
    def _update_cache(self, key: str, analysis: StockfishAnalysis):
        """Update cache with LRU eviction."""
        # Remove if already exists
        if key in self._cache:
            self._cache_access_order.remove(key)
        
        # Add to cache
        self._cache[key] = analysis
        self._cache_access_order.append(key)
        
        # LRU eviction
        while len(self._cache) > self.cache_size:
            oldest_key = self._cache_access_order.pop(0)
            del self._cache[oldest_key]
    
    def _get_from_cache(self, key: str) -> Optional[StockfishAnalysis]:
        """Retrieve from cache and update access order."""
        if key in self._cache:
            # Update access order
            self._cache_access_order.remove(key)
            self._cache_access_order.append(key)
            return self._cache[key]
        return None
    
    def _create_fallback_analysis(self, board: chess.Board) -> StockfishAnalysis:
        """Create fallback analysis when Stockfish is unavailable.
        
        This provides a basic analysis using random legal moves, allowing
        training to proceed even without Stockfish (though with degraded quality).
        """
        import random
        
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            # No legal moves (game over)
            return StockfishAnalysis(
                top5_moves=[],
                top5_evals=[],
                best_move="",
                depth=0,
                analysis_time=0.0
            )
        
        # Randomly sample up to 5 moves
        n_moves = min(5, len(legal_moves))
        sampled_moves = random.sample(legal_moves, n_moves)
        
        # Convert to UCI format
        move_ucis = [move.uci() for move in sampled_moves]
        
        # Pad to 5 moves if needed
        while len(move_ucis) < 5:
            move_ucis.append(move_ucis[0])  # Repeat first move
        
        # Generate random evaluations (centered around 0)
        evals = [random.uniform(-50, 50) for _ in range(5)]
        
        # Best move selection matches RookWorld training pattern
        best_move = move_ucis[-1] if board.turn == chess.BLACK else move_ucis[0]
        
        return StockfishAnalysis(
            top5_moves=move_ucis,
            top5_evals=evals,
            best_move=best_move,
            depth=0,
            analysis_time=0.0
        )
    
    def analyze(self, board: chess.Board) -> StockfishAnalysis:
        """Analyze position and return structured result.
        
        Args:
            board: Chess position to analyze.
            
        Returns:
            StockfishAnalysis with top moves, evaluations, and best move.
        """
        self.stats['analyses_requested'] += 1
        
        # Check cache first
        cache_key = self._cache_key(board)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            return cached_result
        
        # Try to initialize engine if needed
        if not self._initialize_engine():
            self.logger.warning("Using fallback analysis (no Stockfish)")
            self.stats['fallback_used'] += 1
            analysis = self._create_fallback_analysis(board)
            self._update_cache(cache_key, analysis)
            return analysis
        
        # Perform Stockfish analysis
        start_time = time.time()
        
        try:
            # Run multi-PV analysis
            info = self._engine.analyse(
                board, 
                chess.engine.Limit(time=self.time_limit), 
                multipv=self.multipv
            )
            
            analysis_time = time.time() - start_time
            self.stats['total_engine_time'] += analysis_time
            self.stats['engine_analyses'] += 1
            
            # Extract moves and evaluations
            moves = []
            evals = []
            depth = 0
            
            for entry in info:
                if 'pv' in entry and len(entry['pv']) > 0:
                    moves.append(entry['pv'][0].uci())
                    
                    # Convert score to centipawns
                    if 'score' in entry:
                        score = entry['score'].relative
                        if score.is_mate():
                            # Convert mate scores to large numbers
                            mate_score = score.mate()
                            cp_score = 10000 - abs(mate_score) * 100
                            if mate_score < 0:
                                cp_score = -cp_score
                        else:
                            cp_score = score.score() or 0
                        evals.append(cp_score / 100.0)  # Convert to centipawn scale
                    else:
                        evals.append(0.0)
                    
                    # Track depth from first entry
                    if len(moves) == 1 and 'depth' in entry:
                        depth = entry['depth']
            
            # Pad to 5 moves if we got fewer
            while len(moves) < 5:
                if moves:
                    moves.append(moves[0])  # Repeat best move
                    evals.append(evals[0])
                else:
                    # No moves found - likely game over
                    break
            
            # Truncate to 5 moves
            moves = moves[:5]
            evals = evals[:5]
            
            # Select best move using RookWorld convention
            if moves:
                best_move = moves[-1] if board.turn == chess.BLACK else moves[0]
            else:
                best_move = ""
            
            analysis = StockfishAnalysis(
                top5_moves=moves,
                top5_evals=evals,
                best_move=best_move,
                depth=depth,
                analysis_time=analysis_time
            )
            
            # Cache the result
            self._update_cache(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Stockfish analysis failed: {e}, using fallback")
            self.stats['fallback_used'] += 1
            analysis = self._create_fallback_analysis(board)
            self._update_cache(cache_key, analysis)
            return analysis
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache and performance statistics."""
        hit_rate = self.stats['cache_hits'] / max(1, self.stats['analyses_requested'])
        avg_engine_time = self.stats['total_engine_time'] / max(1, self.stats['engine_analyses'])
        
        return {
            **self.stats,
            'cache_hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'avg_engine_time': avg_engine_time,
            'engine_info': self._engine_info
        }
    
    def clear_cache(self):
        """Clear analysis cache."""
        self._cache.clear()
        self._cache_access_order.clear()
        self.logger.info("Stockfish analysis cache cleared")
    
    def close(self):
        """Properly close the engine and clean up resources."""
        if self._engine is not None:
            try:
                self._engine.quit()
                self.logger.info("Stockfish engine closed")
            except Exception as e:
                self.logger.warning(f"Error closing Stockfish: {e}")
            finally:
                self._engine = None
                self._engine_info = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor with cleanup."""
        self.close()


# Convenience functions for easy integration
def create_stockfish_engine(stockfish_path: Optional[str] = None,
                          time_limit: float = 0.1,
                          multipv: int = 5,
                          cache_size: int = 1000) -> StockfishEngine:
    """Create a Stockfish engine instance with standard configuration.
    
    Args:
        stockfish_path: Path to Stockfish binary. If None, searches PATH.
        time_limit: Analysis time limit per position in seconds.
        multipv: Number of principal variations to analyze.
        cache_size: Maximum number of cached analyses.
        
    Returns:
        Configured StockfishEngine instance.
    """
    return StockfishEngine(
        stockfish_path=stockfish_path,
        time_limit=time_limit,
        multipv=multipv,
        cache_size=cache_size
    )


def analyze_position(board: chess.Board,
                    engine: Optional[StockfishEngine] = None,
                    **kwargs) -> StockfishAnalysis:
    """Analyze a single position with automatic engine management.
    
    Args:
        board: Chess position to analyze.
        engine: Existing engine instance. If None, creates temporary engine.
        **kwargs: Arguments for engine creation if engine is None.
        
    Returns:
        StockfishAnalysis result.
    """
    if engine is not None:
        return engine.analyze(board)
    else:
        with create_stockfish_engine(**kwargs) as temp_engine:
            return temp_engine.analyze(board)


def batch_analyze_positions(boards: List[chess.Board],
                          engine: Optional[StockfishEngine] = None,
                          **kwargs) -> List[StockfishAnalysis]:
    """Analyze multiple positions efficiently with caching.
    
    Args:
        boards: List of chess positions to analyze.
        engine: Existing engine instance. If None, creates temporary engine.
        **kwargs: Arguments for engine creation if engine is None.
        
    Returns:
        List of StockfishAnalysis results in same order as input.
    """
    if engine is not None:
        return [engine.analyze(board) for board in boards]
    else:
        with create_stockfish_engine(**kwargs) as temp_engine:
            return [temp_engine.analyze(board) for board in boards]