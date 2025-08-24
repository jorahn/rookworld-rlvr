"""
Evaluation Module for RookWorld GRPO Training

This module provides comprehensive evaluation metrics for chess-specific tasks:
- Legal move rate for Policy (P:) tasks
- Environment accuracy for A: tasks  
- Structured output quality assessment
- Chess tactical benchmarks
"""

import chess
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import json

from .config import GRPOConfig
from ..train.policy import CausalLMPolicy
from ..reward.policy_reward import PolicyRewardComputer, compute_policy_reward
from ..reward.env_reward import EnvRewardComputer, compute_env_reward
from ..environment.chess_env import ChessEnvironment
from ..engine.stockfish import StockfishEngine


@dataclass
class EvaluationMetrics:
    """Container for evaluation results."""
    
    # Policy task metrics
    legal_move_rate: float = 0.0            # Fraction of legal moves generated
    policy_structure_rate: float = 0.0      # Fraction with correct P: format
    policy_parse_rate: float = 0.0          # Fraction that parse correctly  
    move_match_score: float = 0.0           # Average move matching vs Stockfish
    eval_accuracy_score: float = 0.0        # Average evaluation accuracy
    best_move_accuracy: float = 0.0         # Fraction with correct best move
    avg_policy_reward: float = 0.0          # Average total policy reward
    
    # Environment task metrics
    env_structure_rate: float = 0.0         # Fraction with correct A: format
    env_fen_exact_rate: float = 0.0         # Exact FEN match rate
    env_fen_similarity: float = 0.0         # Average FEN similarity
    env_flags_accuracy: float = 0.0         # Game state flags accuracy
    avg_env_reward: float = 0.0             # Average total environment reward
    
    # Generation quality metrics
    avg_generation_time: float = 0.0        # Average generation time per sample
    unique_moves_rate: float = 0.0          # Diversity of generated moves
    repetition_rate: float = 0.0            # Fraction of repeated generations
    
    # Overall metrics
    total_samples: int = 0                  # Number of samples evaluated
    evaluation_time: float = 0.0            # Total evaluation time
    timestamp: float = 0.0                  # When evaluation was run


class ChessEvaluator:
    """Comprehensive evaluator for chess-specific GRPO training metrics.
    
    This evaluator provides:
    - Legal move generation assessment
    - Structured output quality measurement
    - Content accuracy via Stockfish comparison
    - Chess tactical problem solving
    - Training progress monitoring
    """
    
    def __init__(self, 
                 config: GRPOConfig,
                 stockfish_engine: StockfishEngine):
        """
        Initialize chess evaluator
        
        Args:
            config: GRPO training configuration
            stockfish_engine: Stockfish engine for ground truth analysis
        """
        self.config = config
        self.stockfish = stockfish_engine
        
        # Initialize reward computers
        self.policy_reward_computer = PolicyRewardComputer()
        self.env_reward_computer = EnvRewardComputer()
        self.chess_env = ChessEnvironment()
        
        # Test positions for evaluation
        self.test_positions = self._load_test_positions()
        self.tactical_positions = self._load_tactical_positions()
        
        # Evaluation cache for efficiency
        self._cache: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized chess evaluator with {len(self.test_positions)} test positions")
    
    def _load_test_positions(self) -> List[str]:
        """Load standard test positions from config and additional curated sets."""
        positions = list(self.config.test_positions)  # From config
        
        # Add additional middlegame positions
        additional = [
            # Complex middlegame positions
            "r2qkb1r/pb2nppp/1pn1p3/3pP3/2pP4/2P2N2/PP1NBPPP/R1BQK2R w KQkq - 0 8",
            "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 2 6",
            "r1bqk2r/pp2bppp/2n1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R b KQkq - 0 7",
            
            # Tactical positions (pins, forks, skewers)
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 4",
            "rnbqkb1r/ppp2ppp/4pn2/3P4/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 4",
            "r2qkbnr/ppp2ppp/2n1p3/3pPb2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 2 6",
            
            # Endgame positions
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",  # King and pawn vs king
            "4k3/8/8/8/8/4K3/8/4R3 w - - 0 1",  # Rook vs king
            "8/8/2k5/8/8/2K5/8/1Q6 w - - 0 1",  # Queen vs king
        ]
        
        positions.extend(additional)
        return positions
    
    def _load_tactical_positions(self) -> List[Dict[str, str]]:
        """Load tactical test positions with known solutions."""
        return [
            {
                'fen': 'r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 4',
                'description': 'Italian Game tactical shot',
                'best_move': 'Nxe4',  # Knight fork
                'category': 'fork'
            },
            {
                'fen': '2rr3k/pp3pp1/1nnqbN1p/3ppP2/2pPP3/2P3P1/PPBQ4/R4RK1 w - - 0 1',
                'description': 'Mate in 3',
                'best_move': 'Qd4',
                'category': 'mate'
            },
            {
                'fen': 'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
                'description': 'Complex tactical position',
                'best_move': 'Qd8+',
                'category': 'combination'
            }
        ]
    
    def evaluate_policy_task(self, 
                           policy: CausalLMPolicy, 
                           positions: List[str],
                           n_samples_per_position: int = 3) -> Dict[str, float]:
        """
        Evaluate policy (P:) task performance
        
        Args:
            policy: Policy to evaluate
            positions: Chess positions in FEN format
            n_samples_per_position: Number of generations per position
            
        Returns:
            Dictionary of policy task metrics
        """
        results = {
            'legal_moves': 0,
            'structure_correct': 0, 
            'parse_correct': 0,
            'move_matches': 0,
            'eval_accuracy': 0,
            'best_move_correct': 0,
            'total_rewards': 0,
            'total_samples': 0,
            'generation_times': []
        }
        
        for fen in positions[:self.config.eval_positions]:
            board = chess.Board(fen)
            
            # Skip positions with no legal moves
            if not list(board.legal_moves):
                continue
            
            # Generate multiple samples for this position
            for _ in range(n_samples_per_position):
                start_time = time.time()
                
                try:
                    # Generate policy task output
                    prompt = f"P: {fen}    M:"
                    generation_result = policy.generate([prompt], max_new_tokens=50)
                    generated_text = generation_result['texts'][0] if generation_result['texts'] else ""
                    
                    generation_time = time.time() - start_time
                    results['generation_times'].append(generation_time)
                    
                    # Get Stockfish analysis for ground truth
                    stockfish_analysis = self.stockfish.analyze(board)
                    
                    # Compute detailed reward breakdown
                    total_reward, reward_breakdown = compute_policy_reward(
                        generated_text, board, self.stockfish
                    )
                    
                    # Update metrics
                    results['total_samples'] += 1
                    results['total_rewards'] += total_reward
                    
                    # Structure and parsing metrics
                    if reward_breakdown.get('structure_reward', 0) > 0:
                        results['structure_correct'] += 1
                    
                    if reward_breakdown.get('parse_reward', 0) > 0:
                        results['parse_correct'] += 1
                    
                    # Content accuracy metrics
                    if reward_breakdown.get('move_match_reward', 0) > 0:
                        results['move_matches'] += 1
                    
                    if reward_breakdown.get('eval_accuracy_reward', 0) > 0:
                        results['eval_accuracy'] += reward_breakdown['eval_accuracy_reward']
                    
                    if reward_breakdown.get('best_move_reward', 0) > 0:
                        results['best_move_correct'] += 1
                    
                    # Check if any legal move was generated (basic parsing)
                    try:
                        parsed = self.policy_reward_computer.parse_policy_output(generated_text)
                        for move_str in parsed.moves:
                            try:
                                move = chess.Move.from_uci(move_str)
                                if move in board.legal_moves:
                                    results['legal_moves'] += 1
                                    break
                            except:
                                continue
                    except:
                        pass
                        
                except Exception as e:
                    self.logger.warning(f"Policy evaluation failed for {fen}: {e}")
                    results['total_samples'] += 1
        
        # Convert to rates and averages
        total = max(results['total_samples'], 1)
        return {
            'legal_move_rate': results['legal_moves'] / total,
            'policy_structure_rate': results['structure_correct'] / total,
            'policy_parse_rate': results['parse_correct'] / total,
            'move_match_score': results['move_matches'] / total,
            'eval_accuracy_score': results['eval_accuracy'] / total,
            'best_move_accuracy': results['best_move_correct'] / total,
            'avg_policy_reward': results['total_rewards'] / total,
            'avg_generation_time': np.mean(results['generation_times']) if results['generation_times'] else 0.0
        }
    
    def evaluate_environment_task(self, 
                                policy: CausalLMPolicy,
                                positions: List[str],
                                n_samples_per_position: int = 3) -> Dict[str, float]:
        """
        Evaluate environment (A:) task performance
        
        Args:
            policy: Policy to evaluate
            positions: Chess positions in FEN format
            n_samples_per_position: Number of generations per position
            
        Returns:
            Dictionary of environment task metrics
        """
        results = {
            'structure_correct': 0,
            'fen_exact_matches': 0,
            'fen_similarities': [],
            'flags_correct': 0,
            'total_rewards': 0,
            'total_samples': 0,
            'generation_times': []
        }
        
        for fen in positions[:self.config.eval_positions]:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            if not legal_moves:
                continue
            
            for _ in range(n_samples_per_position):
                # Sample random legal move
                move = np.random.choice(legal_moves)
                uci = move.uci()
                
                start_time = time.time()
                
                try:
                    # Generate environment task output
                    prompt = f"A: {fen}+{uci}+"
                    generation_result = policy.generate([prompt], max_new_tokens=32)
                    generated_text = generation_result['texts'][0] if generation_result['texts'] else ""
                    
                    generation_time = time.time() - start_time
                    results['generation_times'].append(generation_time)
                    
                    # Get ground truth by applying the move
                    expected_response = self.chess_env.apply_move(fen, uci)
                    
                    # Compute detailed reward breakdown
                    total_reward, reward_breakdown = compute_env_reward(
                        generated_text, expected_response
                    )
                    
                    # Update metrics
                    results['total_samples'] += 1
                    results['total_rewards'] += total_reward
                    
                    # Structure metrics
                    if reward_breakdown.get('structure_reward', 0) > 0:
                        results['structure_correct'] += 1
                    
                    # FEN accuracy metrics
                    if reward_breakdown.get('fen_exact_reward', 0) > 0:
                        results['fen_exact_matches'] += 1
                    
                    if 'fen_similarity' in reward_breakdown:
                        results['fen_similarities'].append(reward_breakdown['fen_similarity'])
                    
                    # Flags accuracy
                    if reward_breakdown.get('flags_accuracy_reward', 0) > 0:
                        results['flags_correct'] += 1
                        
                except Exception as e:
                    self.logger.warning(f"Environment evaluation failed for {fen}+{uci}: {e}")
                    results['total_samples'] += 1
        
        # Convert to rates and averages
        total = max(results['total_samples'], 1)
        return {
            'env_structure_rate': results['structure_correct'] / total,
            'env_fen_exact_rate': results['fen_exact_matches'] / total,
            'env_fen_similarity': np.mean(results['fen_similarities']) if results['fen_similarities'] else 0.0,
            'env_flags_accuracy': results['flags_correct'] / total,
            'avg_env_reward': results['total_rewards'] / total,
            'avg_generation_time': np.mean(results['generation_times']) if results['generation_times'] else 0.0
        }
    
    def evaluate_tactical_performance(self, policy: CausalLMPolicy) -> Dict[str, float]:
        """
        Evaluate performance on tactical chess positions
        
        Args:
            policy: Policy to evaluate
            
        Returns:
            Dictionary of tactical performance metrics
        """
        results = {
            'total_positions': len(self.tactical_positions),
            'correct_solutions': 0,
            'category_performance': {}
        }
        
        for pos_data in self.tactical_positions:
            fen = pos_data['fen']
            best_move = pos_data['best_move']
            category = pos_data['category']
            
            try:
                # Generate move for tactical position
                prompt = f"P: {fen}    M:"
                generation_result = policy.generate([prompt], max_new_tokens=50)
                generated_text = generation_result['texts'][0] if generation_result['texts'] else ""
                
                # Parse generated moves
                parsed = self.policy_reward_computer.parse_policy_output(generated_text)
                
                # Check if best move is in generated moves or is the best move
                found_solution = False
                if parsed.best_move == best_move:
                    found_solution = True
                elif best_move in parsed.moves:
                    found_solution = True
                
                if found_solution:
                    results['correct_solutions'] += 1
                
                # Track by category
                if category not in results['category_performance']:
                    results['category_performance'][category] = {'correct': 0, 'total': 0}
                
                results['category_performance'][category]['total'] += 1
                if found_solution:
                    results['category_performance'][category]['correct'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Tactical evaluation failed for {fen}: {e}")
        
        # Compute category success rates
        for category, stats in results['category_performance'].items():
            total = max(stats['total'], 1)
            stats['success_rate'] = stats['correct'] / total
        
        overall_rate = results['correct_solutions'] / max(results['total_positions'], 1)
        results['tactical_success_rate'] = overall_rate
        
        return results
    
    def evaluate(self, 
                 policy: CausalLMPolicy,
                 include_tactical: bool = True) -> EvaluationMetrics:
        """
        Run comprehensive evaluation of the policy
        
        Args:
            policy: Policy to evaluate
            include_tactical: Whether to include tactical position evaluation
            
        Returns:
            Complete evaluation metrics
        """
        eval_start_time = time.time()
        
        # Policy task evaluation
        self.logger.info("Evaluating policy (P:) task performance...")
        policy_metrics = self.evaluate_policy_task(policy, self.test_positions)
        
        # Environment task evaluation  
        self.logger.info("Evaluating environment (A:) task performance...")
        env_metrics = self.evaluate_environment_task(policy, self.test_positions)
        
        # Tactical evaluation (optional)
        tactical_metrics = {}
        if include_tactical:
            self.logger.info("Evaluating tactical performance...")
            tactical_metrics = self.evaluate_tactical_performance(policy)
        
        total_eval_time = time.time() - eval_start_time
        
        # Combine metrics
        metrics = EvaluationMetrics(
            # Policy metrics
            legal_move_rate=policy_metrics['legal_move_rate'],
            policy_structure_rate=policy_metrics['policy_structure_rate'],
            policy_parse_rate=policy_metrics['policy_parse_rate'],
            move_match_score=policy_metrics['move_match_score'],
            eval_accuracy_score=policy_metrics['eval_accuracy_score'],
            best_move_accuracy=policy_metrics['best_move_accuracy'],
            avg_policy_reward=policy_metrics['avg_policy_reward'],
            
            # Environment metrics
            env_structure_rate=env_metrics['env_structure_rate'],
            env_fen_exact_rate=env_metrics['env_fen_exact_rate'],
            env_fen_similarity=env_metrics['env_fen_similarity'],
            env_flags_accuracy=env_metrics['env_flags_accuracy'],
            avg_env_reward=env_metrics['avg_env_reward'],
            
            # Generation metrics
            avg_generation_time=(policy_metrics['avg_generation_time'] + env_metrics['avg_generation_time']) / 2,
            
            # Meta metrics
            total_samples=self.config.eval_positions * 3 * 2,  # 3 samples per position, 2 tasks
            evaluation_time=total_eval_time,
            timestamp=time.time()
        )
        
        self.logger.info(f"Evaluation completed in {total_eval_time:.2f}s")
        return metrics
    
    def metrics_to_dict(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Convert EvaluationMetrics to dictionary for logging."""
        return {
            # Policy metrics
            'policy/legal_move_rate': metrics.legal_move_rate,
            'policy/structure_rate': metrics.policy_structure_rate,
            'policy/parse_rate': metrics.policy_parse_rate,
            'policy/move_match_score': metrics.move_match_score,
            'policy/eval_accuracy': metrics.eval_accuracy_score,
            'policy/best_move_accuracy': metrics.best_move_accuracy,
            'policy/avg_reward': metrics.avg_policy_reward,
            
            # Environment metrics
            'environment/structure_rate': metrics.env_structure_rate,
            'environment/fen_exact_rate': metrics.env_fen_exact_rate,
            'environment/fen_similarity': metrics.env_fen_similarity,
            'environment/flags_accuracy': metrics.env_flags_accuracy,
            'environment/avg_reward': metrics.avg_env_reward,
            
            # Performance metrics
            'performance/avg_generation_time': metrics.avg_generation_time,
            'performance/total_samples': metrics.total_samples,
            'performance/evaluation_time': metrics.evaluation_time
        }
    
    def print_evaluation_report(self, metrics: EvaluationMetrics):
        """Print human-readable evaluation report."""
        print("\n" + "="*60)
        print("ROOKWORLD GRPO EVALUATION REPORT")
        print("="*60)
        
        print(f"\nPolicy (P:) Task Performance:")
        print(f"  Legal Move Rate:      {metrics.legal_move_rate:.2%}")
        print(f"  Structure Correct:    {metrics.policy_structure_rate:.2%}")
        print(f"  Parse Success:        {metrics.policy_parse_rate:.2%}")
        print(f"  Move Match Score:     {metrics.move_match_score:.3f}")
        print(f"  Eval Accuracy:        {metrics.eval_accuracy_score:.3f}")
        print(f"  Best Move Accuracy:   {metrics.best_move_accuracy:.2%}")
        print(f"  Average Reward:       {metrics.avg_policy_reward:.3f}")
        
        print(f"\nEnvironment (A:) Task Performance:")
        print(f"  Structure Correct:    {metrics.env_structure_rate:.2%}")
        print(f"  FEN Exact Match:      {metrics.env_fen_exact_rate:.2%}")
        print(f"  FEN Similarity:       {metrics.env_fen_similarity:.3f}")
        print(f"  Flags Accuracy:       {metrics.env_flags_accuracy:.2%}")
        print(f"  Average Reward:       {metrics.avg_env_reward:.3f}")
        
        print(f"\nGeneration Performance:")
        print(f"  Avg Generation Time:  {metrics.avg_generation_time:.3f}s")
        print(f"  Total Samples:        {metrics.total_samples}")
        print(f"  Evaluation Time:      {metrics.evaluation_time:.2f}s")
        
        print("\n" + "="*60)


def create_evaluator(config: GRPOConfig, stockfish_engine: StockfishEngine) -> ChessEvaluator:
    """
    Factory function to create chess evaluator
    
    Args:
        config: GRPO training configuration  
        stockfish_engine: Stockfish engine for analysis
        
    Returns:
        Configured ChessEvaluator instance
    """
    return ChessEvaluator(config, stockfish_engine)