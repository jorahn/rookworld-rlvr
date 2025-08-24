#!/usr/bin/env python3
"""
Full GRPO Training Script with Automatic Batch Size Optimization

This script demonstrates the complete GRPO training pipeline:
- Loads HuggingFace RookWorld-LM weights
- Generates 256 diverse chess positions
- Uses 80/20 Policy/Environment task distribution
- Creates groups of 4 rollouts per position
- Automatically determines optimal batch size based on GPU memory
- Runs full GRPO training with monitoring

Usage:
    python scripts/train_full_grpo.py
    python scripts/train_full_grpo.py --steps 500 --max-positions 128
"""

import argparse
import logging
import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import chess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.model.loader import load_pretrained_model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch
from rookworld_rlvr.train.policy import CausalLMPolicy
from rookworld_rlvr.data.collector import GRPODataCollector, GRPOCollectionConfig
from rookworld_rlvr.environment.chess_env import ChessEnvironment
from rookworld_rlvr.engine.stockfish import StockfishEngine


@dataclass
class MemoryProfile:
    """Memory usage profile for batch size optimization"""
    peak_gb: float
    total_gb: float
    utilization: float
    batch_positions: int
    group_size: int
    effective_batch_size: int


class FullGRPOTrainer:
    """Complete GRPO training with automatic optimization"""
    
    def __init__(self, config: GRPOConfig):
        """Initialize full GRPO trainer"""
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Set up reproducibility
        self.setup_reproducibility()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.trainer = None
        self.stockfish = None
        self.data_collector = None
        self.chess_positions = []
        
        # Training state
        self.optimal_batch_config = None
        self.memory_profiles = []
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path(self.config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "train_full_grpo.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def setup_reproducibility(self):
        """Setup reproducible training environment"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def initialize_models(self):
        """Initialize and load models"""
        self.logger.info("Initializing RookWorld-LM models...")
        
        # Initialize tokenizer
        self.tokenizer = TokenizerBridge()
        
        # Load pre-trained model from HuggingFace
        self.logger.info(f"Loading weights from {self.config.model_name_or_path}")
        self.model = load_pretrained_model(
            self.config.model_name_or_path,
            device=self.config.device
        )
        self.model.train()
        
        # Create reference model (frozen copy)
        model_config = GPT2Config()
        self.ref_model = GPT2Model(model_config)
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
            
        self.logger.info("Models initialized successfully")
        
        # Apply optimizations
        if self.config.use_torch_compile:
            self.logger.info("Applying torch.compile optimization...")
            try:
                self.model = torch.compile(
                    self.model,
                    mode=self.config.torch_compile_mode,
                    backend="inductor"
                )
                self.logger.info("torch.compile applied successfully")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")
                
    def initialize_components(self):
        """Initialize training components"""
        self.logger.info("Initializing training components...")
        
        # Initialize GRPO trainer
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.config)
        
        # Initialize Stockfish for rewards
        self.stockfish = StockfishEngine(
            stockfish_path=self.config.stockfish_path,
            time_limit=self.config.stockfish_time_limit
        )
        
        # Initialize data collector
        collection_config = GRPOCollectionConfig(
            group_size=self.config.group_size,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            mix_env_ratio=self.config.mix_env_ratio
        )
        
        # Create policy wrapper
        policy = CausalLMPolicy(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.config.device
        )
        
        self.data_collector = GRPODataCollector(policy, collection_config)
        
        self.logger.info("Components initialized successfully")
        
    def generate_diverse_positions(self, n_positions: int = 256) -> List[str]:
        """Generate diverse chess positions for training"""
        self.logger.info(f"Generating {n_positions} diverse chess positions...")
        
        positions = []
        chess_env = ChessEnvironment()
        
        # 1. Opening positions (20%)
        opening_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # 1.e4
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # 1.d4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # 1.e4 e5
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # Queen's Gambit setup
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",  # Italian Game
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",  # Ruy Lopez setup
        ]
        
        positions.extend(opening_positions)
        
        # 2. Mid-game positions (60%) - Generate through random play
        target_midgame = int(n_positions * 0.6)
        while len([p for p in positions if p not in opening_positions]) < target_midgame:
            try:
                board = chess.Board()
                # Play 8-20 moves to reach mid-game
                n_moves = random.randint(8, 20)
                
                for _ in range(n_moves):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves or board.is_game_over():
                        break
                    
                    # Weighted move selection (favor center control)
                    move_weights = []
                    for move in legal_moves:
                        weight = 1.0
                        # Prefer center squares
                        if move.to_square in [27, 28, 35, 36]:  # d4, e4, d5, e5
                            weight *= 2.0
                        # Prefer development
                        if board.turn == chess.WHITE and move.from_square in [1, 6, 57, 62]:  # Knights
                            weight *= 1.5
                        move_weights.append(weight)
                    
                    # Weighted random selection
                    total_weight = sum(move_weights)
                    rand_val = random.random() * total_weight
                    
                    cumulative_weight = 0
                    selected_move = legal_moves[0]  # Fallback
                    for move, weight in zip(legal_moves, move_weights):
                        cumulative_weight += weight
                        if rand_val <= cumulative_weight:
                            selected_move = move
                            break
                    
                    board.push(selected_move)
                
                if not board.is_game_over():
                    fen = board.fen()
                    if fen not in positions:
                        positions.append(fen)
                        
            except Exception as e:
                self.logger.warning(f"Failed to generate position: {e}")
                continue
                
        # 3. Endgame positions (20%) - Positions with fewer pieces
        target_endgame = n_positions - len(positions)
        endgame_attempts = 0
        max_endgame_attempts = target_endgame * 10
        
        while len(positions) < n_positions and endgame_attempts < max_endgame_attempts:
            try:
                board = chess.Board()
                # Play many moves to reach endgame
                n_moves = random.randint(30, 80)
                
                for _ in range(n_moves):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves or board.is_game_over():
                        break
                    move = random.choice(legal_moves)
                    board.push(move)
                
                # Check if it's actually an endgame (< 14 pieces)
                piece_count = len([sq for sq in chess.SQUARES if board.piece_at(sq) is not None])
                
                if piece_count < 14 and not board.is_game_over():
                    fen = board.fen()
                    if fen not in positions:
                        positions.append(fen)
                        
                endgame_attempts += 1
                        
            except Exception as e:
                endgame_attempts += 1
                continue
        
        # Fill remaining with more diverse positions if needed
        while len(positions) < n_positions:
            try:
                additional_positions = chess_env.create_sample_positions(n_positions - len(positions))
                for pos in additional_positions:
                    if pos not in positions:
                        positions.append(pos)
                        if len(positions) >= n_positions:
                            break
            except Exception:
                break
        
        final_positions = positions[:n_positions]
        self.logger.info(f"Generated {len(final_positions)} positions:")
        self.logger.info(f"  - Opening positions: {len([p for p in final_positions if p in opening_positions])}")
        self.logger.info(f"  - Mid-game positions: {len(final_positions) - len([p for p in final_positions if p in opening_positions])}")
        
        return final_positions
        
    def profile_memory_usage(self, batch_positions: int, group_size: int) -> MemoryProfile:
        """Profile memory usage for given batch configuration"""
        self.logger.info(f"Profiling memory: batch_positions={batch_positions}, group_size={group_size}")
        
        if not torch.cuda.is_available():
            return MemoryProfile(0, 0, 0, batch_positions, group_size, batch_positions * group_size)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create test batch
        test_positions = self.chess_positions[:batch_positions]
        
        try:
            # Populate data collector with positions
            self.data_collector.add_positions_to_buffer(test_positions)
            
            # Collect a batch
            batch_groups = self.data_collector.collect_mixed_batch(batch_positions)
            
            if not batch_groups:
                raise RuntimeError("Failed to collect batch data")
            
            # Run 3 training steps to get stable memory measurement
            for step in range(3):
                total_loss = 0
                
                for group in batch_groups:
                    # Convert to GRPOBatch
                    grpo_batch = GRPOBatch(
                        input_ids=group['input_ids'],
                        attention_mask=group['attention_mask'],
                        target_start_indices=group['target_start_indices'],
                        old_logprobs=group['old_logprobs'],
                        rewards=group['rewards'],
                        position_fen=group.get('position_fen', 'unknown'),
                        task_type=group.get('task_type', 'policy')
                    )
                    
                    # Forward pass
                    loss, metrics = self.trainer.compute_grpo_loss(grpo_batch)
                    total_loss += loss
                
                # Backward pass
                self.trainer.optimizer.zero_grad()
                total_loss.backward()
                self.trainer.optimizer.step()
                
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            utilization = peak_memory / total_memory
            
            self.logger.info(f"  Peak memory: {peak_memory:.2f}GB ({utilization:.1%} utilization)")
            
            return MemoryProfile(
                peak_gb=peak_memory,
                total_gb=total_memory,
                utilization=utilization,
                batch_positions=batch_positions,
                group_size=group_size,
                effective_batch_size=batch_positions * group_size
            )
            
        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
            return MemoryProfile(999, 1000, 1.0, batch_positions, group_size, batch_positions * group_size)
        finally:
            torch.cuda.empty_cache()
    
    def find_optimal_batch_size(self, target_utilization: float = 0.85) -> MemoryProfile:
        """Find optimal batch size for target GPU utilization"""
        self.logger.info(f"Finding optimal batch size (target: {target_utilization:.1%} GPU utilization)")
        
        # Test configurations starting from conservative values
        test_configs = []
        
        # Start with small batch and scale up
        for batch_pos in [2, 4, 8, 16, 32, 48, 64, 96, 128]:
            test_configs.append((batch_pos, self.config.group_size))
            
        best_config = None
        
        for batch_positions, group_size in test_configs:
            if batch_positions > len(self.chess_positions):
                break
                
            profile = self.profile_memory_usage(batch_positions, group_size)
            self.memory_profiles.append(profile)
            
            # If we exceed target utilization, previous config was optimal
            if profile.utilization > target_utilization:
                break
            
            best_config = profile
            
            # If we're getting close to memory limit, stop
            if profile.utilization > 0.95:
                self.logger.warning("Approaching memory limit, stopping batch size search")
                break
        
        if best_config is None:
            # Fallback to very conservative settings
            best_config = MemoryProfile(
                peak_gb=2.0,
                total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 8.0,
                utilization=0.25,
                batch_positions=2,
                group_size=self.config.group_size,
                effective_batch_size=2 * self.config.group_size
            )
        
        self.optimal_batch_config = best_config
        
        self.logger.info("Optimal batch configuration found:")
        self.logger.info(f"  Batch positions: {best_config.batch_positions}")
        self.logger.info(f"  Group size: {best_config.group_size}")
        self.logger.info(f"  Effective batch size: {best_config.effective_batch_size}")
        self.logger.info(f"  Memory usage: {best_config.peak_gb:.2f}GB ({best_config.utilization:.1%})")
        
        return best_config
    
    def run_training(self):
        """Run complete GRPO training"""
        self.logger.info("Starting GRPO training...")
        
        # Update config with optimal batch size
        if self.optimal_batch_config:
            self.config.batch_positions = self.optimal_batch_config.batch_positions
            
        # Populate data collector with all positions
        self.data_collector.add_positions_to_buffer(self.chess_positions)
        
        # Training loop
        step = 0
        best_reward = -float('inf')
        
        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Total steps: {self.config.steps}")
        self.logger.info(f"  Batch positions: {self.config.batch_positions}")
        self.logger.info(f"  Group size: {self.config.group_size}")
        self.logger.info(f"  Effective batch size: {self.config.batch_positions * self.config.group_size}")
        self.logger.info(f"  Policy/Environment ratio: {(1-self.config.mix_env_ratio)*100:.0f}/{self.config.mix_env_ratio*100:.0f}")
        
        start_time = time.time()
        
        while step < self.config.steps:
            try:
                step_start = time.time()
                
                # Collect batch
                batch_groups = self.data_collector.collect_mixed_batch(self.config.batch_positions)
                
                if not batch_groups:
                    self.logger.warning(f"No batch data collected at step {step}")
                    continue
                
                # Train on batch
                total_loss = 0
                total_rewards = []
                policy_count = 0
                env_count = 0
                
                for group in batch_groups:
                    # Convert to GRPOBatch
                    grpo_batch = GRPOBatch(
                        input_ids=group['input_ids'],
                        attention_mask=group['attention_mask'],
                        target_start_indices=group['target_start_indices'],
                        old_logprobs=group['old_logprobs'],
                        rewards=group['rewards'],
                        position_fen=group.get('position_fen', 'unknown'),
                        task_type=group.get('task_type', 'policy')
                    )
                    
                    # Forward pass
                    loss, metrics = self.trainer.compute_grpo_loss(grpo_batch)
                    total_loss += loss
                    
                    # Collect metrics
                    rewards = group['rewards'].cpu().numpy()
                    total_rewards.extend(rewards)
                    
                    if group.get('task_type') == 'policy':
                        policy_count += 1
                    else:
                        env_count += 1
                
                # Backward pass
                self.trainer.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip_norm
                )
                
                # Optimizer step
                self.trainer.optimizer.step()
                self.trainer.scheduler.step()
                
                # Update trainer state
                self.trainer.step_count += 1
                
                step_time = time.time() - step_start
                
                # Logging
                if step % 10 == 0:
                    mean_reward = np.mean(total_rewards)
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                    
                    self.logger.info(
                        f"Step {step:4d} | "
                        f"Loss: {total_loss.item():.4f} | "
                        f"Reward: {mean_reward:.3f} (best: {best_reward:.3f}) | "
                        f"P/E: {policy_count}/{env_count} | "
                        f"Time: {step_time:.2f}s"
                    )
                
                # Save checkpoint
                if step % 100 == 0 and step > 0:
                    self.save_checkpoint(step)
                
                step += 1
                
            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Training error at step {step}: {e}")
                self.logger.error(f"Continuing training...")
                continue
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s ({total_time/3600:.2f}h)")
        self.logger.info(f"Final best reward: {best_reward:.3f}")
        
        # Save final checkpoint
        self.save_checkpoint(step, is_final=True)
        
    def save_checkpoint(self, step: int, is_final: bool = False):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        if is_final:
            checkpoint_dir = Path(self.config.output_dir) / "final_checkpoint"
            
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': step,
            'config': self.config.__dict__
        }, model_path)
        
        # Save training state
        if hasattr(self.trainer, 'state_dict'):
            trainer_path = checkpoint_dir / "trainer.pt"
            torch.save(self.trainer.state_dict(), trainer_path)
        
        # Save memory profiles
        if self.memory_profiles:
            profiles_path = checkpoint_dir / "memory_profiles.json"
            profiles_data = [
                {
                    'batch_positions': p.batch_positions,
                    'group_size': p.group_size,
                    'effective_batch_size': p.effective_batch_size,
                    'peak_gb': p.peak_gb,
                    'total_gb': p.total_gb,
                    'utilization': p.utilization
                }
                for p in self.memory_profiles
            ]
            with open(profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        self.logger.info("=== FULL GRPO TRAINING PIPELINE ===")
        
        try:
            # 1. Initialize models and components
            self.initialize_models()
            self.initialize_components()
            
            # 2. Generate diverse chess positions
            self.chess_positions = self.generate_diverse_positions(256)
            
            # 3. Find optimal batch size
            self.find_optimal_batch_size()
            
            # 4. Run training
            self.run_training()
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Full GRPO Training with Automatic Batch Size Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--steps", type=int, default=1000,
                       help="Number of training steps")
    parser.add_argument("--max-positions", type=int, default=256,
                       help="Number of diverse chess positions to generate")
    parser.add_argument("--output-dir", type=str, default="outputs/full_grpo",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--stockfish-path", type=str, default=None,
                       help="Path to Stockfish binary")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu, auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = GRPOConfig(
        # Model
        model_name_or_path="jrahn/RookWorld-LM-124M",
        
        # Training
        steps=args.steps,
        batch_positions=4,  # Will be auto-optimized
        group_size=4,       # 4 rollouts per position as requested
        
        # Hyperparameters (validated for stability)
        lr=1e-5,
        kl_coef=0.01,
        clip_range=0.2,
        temperature=0.7,
        
        # Task distribution (80/20)
        mix_env_ratio=0.2,
        
        # Performance optimizations
        use_mixed_precision=torch.cuda.is_available(),  # BF16 for RTX 4090
        use_torch_compile=True,    # 3-5x speedup
        torch_compile_mode="reduce-overhead",
        
        # System
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
        
        # Stockfish
        stockfish_path=args.stockfish_path,
        stockfish_time_limit=0.1,
        
        # Self-play
        n_parallel_games=8,
        position_buffer_size=512,
    )
    
    # Initialize trainer
    trainer = FullGRPOTrainer(config)
    
    # Run complete pipeline
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()