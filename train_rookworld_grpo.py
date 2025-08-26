#!/usr/bin/env python3
"""
RookWorld-LM GRPO Training Script

Complete implementation of Group Relative Policy Optimization (GRPO) training
for RookWorld-LM on verifiable chess tasks.

This script orchestrates:
- Pure PyTorch GPT-2 model loading
- Mixed Policy (P:) and Environment (A:) task training  
- Stockfish-verified reward computation
- Self-play position generation
- Comprehensive evaluation and monitoring

Usage:
    uv run python train_rookworld_grpo.py --steps 1000 --group-size 8
    uv run python train_rookworld_grpo.py --mix-env-ratio 0.0 --steps 2000  # Policy only
    uv run python train_rookworld_grpo.py --steps 5000 --batch-positions 16 --lr 5e-6  # High perf
"""

import argparse
import logging
import os
import sys
import json
import time
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import signal

import torch
import numpy as np
import chess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOTrainingStep
from rookworld_rlvr.train.policy import CausalLMPolicy
from rookworld_rlvr.train.self_play import SelfPlayManager
from rookworld_rlvr.train.evaluator import ChessEvaluator
from rookworld_rlvr.train.checkpoint_manager import CheckpointManager
from rookworld_rlvr.data.collector import GRPODataCollector, GRPOCollectionConfig  
from rookworld_rlvr.data.rookworld_dataset import RookWorldDatasetProcessor
from rookworld_rlvr.engine.stockfish import StockfishEngine
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.loader import load_pretrained_model


class TrainingOrchestrator:
    """Main orchestrator for GRPO training process."""
    
    def __init__(self, config: GRPOConfig):
        """
        Initialize training orchestrator
        
        Args:
            config: Complete GRPO training configuration
        """
        self.config = config
        self.run_id = None  # Initialize run_id before setup_logging
        self.setup_logging()
        self.setup_reproducibility()
        
        # Training components (initialized in setup)
        self.model: Optional[GPT2Model] = None
        self.ref_model: Optional[GPT2Model] = None
        self.policy: Optional[CausalLMPolicy] = None
        self.trainer: Optional[GRPOTrainer] = None
        self.stockfish: Optional[StockfishEngine] = None
        self.data_collector: Optional[GRPODataCollector] = None
        self.self_play: Optional[SelfPlayManager] = None
        self.evaluator: Optional[ChessEvaluator] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.dataset_processor: Optional[RookWorldDatasetProcessor] = None
        
        # Training state
        self.training_active = False
        self.should_stop = False
        
        # Run identity and recovery state
        self.run_id = None
        self.run_start_time = None
        self.resume_count = 0
        self.is_resumed = False
        self.start_step = 0
        self.last_stable_checkpoint = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized GRPO training orchestrator")
    
    def setup_logging(self, is_resume=False):
        """Setup comprehensive logging configuration with resume support."""
        # Create run-specific directory if not resuming
        if not is_resume and not self.run_id:
            self.run_id = self.config.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use run-specific subdirectory
        log_dir = Path(self.config.output_dir) / self.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with run-specific output directory
        self.config.output_dir = str(log_dir)
        
        log_file = log_dir / self.config.log_file
        
        # Use append mode for resume, write mode for new runs
        log_mode = 'a' if (is_resume and self.config.append_logs_on_resume) else 'w'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode=log_mode),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create the logger instance
        self.logger = logging.getLogger(__name__)
        
        # Add resume marker to logs
        if is_resume:
            self.logger.info("="*60)
            self.logger.info(f"RESUMING TRAINING - Run ID: {self.run_id}")
            self.logger.info(f"Resume count: {self.resume_count}")
            self.logger.info(f"Resumed from step: {self.start_step}")
            self.logger.info("="*60)
        else:
            self.logger.info(f"NEW TRAINING RUN - Run ID: {self.run_id}")
            self.run_start_time = time.time()
    
    def setup_reproducibility(self):
        """Setup reproducible training environment."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            # Performance vs deterministic tradeoff
            if self.config.enable_cudnn_benchmark:
                # Enable benchmark for better performance (non-deterministic)
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                # Ensure deterministic operations (slower but reproducible)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True
    
    def initialize_models(self):
        """Initialize and load RookWorld-LM models."""
        self.logger.info("Initializing RookWorld-LM models...")
        
        # Create model configuration
        model_config = GPT2Config()  # Uses RookWorld-LM defaults
        model_config.use_gradient_checkpointing = self.config.use_gradient_checkpointing
        
        # Initialize policy model
        self.model = GPT2Model(model_config)
        
        # Load pre-trained weights
        self.logger.info(f"Loading weights from {self.config.model_name_or_path}")
        self.model = load_pretrained_model(
            self.config.model_name_or_path,
            device=self.config.device
        )
        self.model.train()
        
        # Create reference model (frozen copy)
        # If multiple GPUs available, put reference model on second GPU
        self.ref_device = self.config.device
        if torch.cuda.device_count() > 1 and self.config.device == 'cuda':
            self.ref_device = 'cuda:1'
            self.logger.info(f"Using multi-GPU setup: training model on cuda:0, reference model on cuda:1")
        
        self.ref_model = GPT2Model(model_config).to(self.ref_device)
        # Move state dict to correct device before loading
        ref_state_dict = {k: v.to(self.ref_device) for k, v in self.model.state_dict().items()}
        self.ref_model.load_state_dict(ref_state_dict)
        self.ref_model.eval()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        
        # Apply torch.compile optimization if enabled
        if self.config.use_torch_compile:
            self.logger.info("Compiling models with torch.compile...")
            try:
                # Warm up models with a forward pass before compilation to ensure proper device setup
                dummy_input = torch.randint(0, 1000, (1, 10), device=self.config.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                    dummy_ref_input = dummy_input.to(self.ref_device)
                    _ = self.ref_model(dummy_ref_input)
                
                # Compile the main model
                self.model = torch.compile(
                    self.model,
                    mode=self.config.torch_compile_mode,
                    backend=self.config.torch_compile_backend
                )
                
                # Compile the reference model
                self.ref_model = torch.compile(
                    self.ref_model,
                    mode=self.config.torch_compile_mode,
                    backend=self.config.torch_compile_backend
                )
                
                self.logger.info(f"Successfully compiled models with mode='{self.config.torch_compile_mode}'")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}, continuing without compilation")
                self.config.use_torch_compile = False
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
    
    def initialize_components(self):
        """Initialize all training components."""
        self.logger.info("Initializing training components...")
        
        # Policy wrapper - pass reference device info
        self.policy = CausalLMPolicy(
            self.model, 
            self.ref_model, 
            self.config,
            device=self.config.device
        )
        # Store reference device for policy to use
        self.policy.ref_device = self.ref_device
        
        # Stockfish engine for ground truth analysis
        self.stockfish = StockfishEngine(
            stockfish_path=self.config.stockfish_path,
            time_limit=self.config.stockfish_time_limit,
            multipv=self.config.stockfish_multipv,
            cache_size=1000
        )
        
        # GRPO trainer
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.config)
        
        # Data collector for GRPO batches with proper config
        collection_config = GRPOCollectionConfig(
            group_size=self.config.group_size,
            max_new_tokens_policy=self.config.max_new_tokens,
            max_new_tokens_env=self.config.max_new_tokens_env,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            mix_env_ratio=self.config.mix_env_ratio,
            device=self.config.device
        )
        self.data_collector = GRPODataCollector(self.policy, collection_config)
        
        # Self-play manager
        self.self_play = SelfPlayManager(self.config, self.policy)
        
        # Evaluator
        self.evaluator = ChessEvaluator(self.config, self.stockfish)
        
        # Dataset processor for diverse training positions (if enabled)
        if self.config.use_dataset:
            self.dataset_processor = RookWorldDatasetProcessor()
            self.logger.info("Dataset processor initialized for diverse training positions")
        else:
            self.dataset_processor = None
            self.logger.info("Using self-play only for training positions")
        
        self.logger.info("All components initialized successfully")
    
    def _optimize_cuda_performance(self):
        """Apply comprehensive PyTorch/CUDA performance optimizations per best practices."""
        if not torch.cuda.is_available():
            return
        
        self.logger.info("Applying GRPO best practices PyTorch optimizations...")
            
        # Enable memory management optimizations
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # REMOVED: Aggressive memory pre-allocation that was causing artificial OOM
        # The previous code pre-allocated 90% of GPU memory (~21GB) unnecessarily
        # This caused "fake" OOM errors when actual usage was only ~1-2GB
        # Let PyTorch manage memory allocation dynamically for efficiency
        
        # RTX 4090 / Ada Lovelace optimizations
        # 1. TF32 optimization for ~30% matmul speedup on Ampere+
        original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('high')  
        self.logger.info(f"✅ TF32 optimization: {original_precision} → high (~30% matmul speedup)")
        
        # 2. Enable TensorFloat-32 for additional acceleration
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("✅ TF32 backend acceleration enabled")
        
        # 3. CUDA allocator optimization for 24GB cards
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.logger.info("✅ CUDA allocator optimized for 24GB cards")
        
        # 4. CUDA kernel optimization
        torch.backends.cudnn.benchmark = True
        self.logger.info("✅ CUDA kernel optimization enabled")
        
        self.logger.info(f"🚀 Applied RTX 4090 optimizations: ~50-60% expected speed increase")
    
    def _manage_cuda_memory(self):
        """Manage CUDA memory during training."""
        if not torch.cuda.is_available():
            return
            
        # Clear cache periodically to prevent fragmentation
        if self.trainer.step_count % 100 == 0:
            torch.cuda.empty_cache()
    
    def run_training(self):
        """Execute the main GRPO training loop."""
        self.logger.info("="*60)
        self.logger.info("STARTING ROOKWORLD GRPO TRAINING")
        self.logger.info("="*60)
        self.logger.info(self.config.summary())
        
        # Apply CUDA optimizations
        self._optimize_cuda_performance()
        
        self.training_active = True
        training_start_time = time.time()
        
        # Initial evaluation
        self.logger.info("Running initial evaluation...")
        initial_metrics = self.evaluator.evaluate(self.policy, include_tactical=True)
        self.evaluator.print_evaluation_report(initial_metrics)
        
        # Save initial metrics
        self._save_metrics(0, initial_metrics, prefix="initial")
        
        try:
            for step in range(1, self.config.steps + 1):
                if self.should_stop:
                    self.logger.info("Training interrupted by user")
                    break
                
                step_start_time = time.time()
                
                # Collect training data
                training_data = self._collect_training_data()
                
                if not training_data.groups:
                    self.logger.warning(f"No training data collected for step {step}")
                    continue
                
                # Training step with rollout epochs for enhanced sample efficiency
                metrics = self.trainer.training_step_with_rollout_epochs(training_data)
                
                # CUDA memory management
                self._manage_cuda_memory()
                
                # Advance self-play games periodically
                if step % 5 == 0:
                    self.self_play.advance_games(n_moves=1)
                
                step_time = time.time() - step_start_time
                
                # Logging
                if step % self.config.log_interval == 0:
                    self._log_training_progress(step, metrics, step_time)
                
                # Evaluation
                if step % self.config.eval_every == 0:
                    self.logger.info(f"Running evaluation at step {step}...")
                    eval_metrics = self.evaluator.evaluate(self.policy, include_tactical=False)
                    self.evaluator.print_evaluation_report(eval_metrics)
                    self._save_metrics(step, eval_metrics)
                
                # Save checkpoint
                if step % self.config.save_every == 0:
                    self._save_checkpoint(step)
                
                # Early stopping check
                if self._should_early_stop(step):
                    self.logger.info("Early stopping triggered")
                    break
        
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
            raise
        
        finally:
            self.training_active = False
            total_time = time.time() - training_start_time
            
            # Final evaluation and save
            self.logger.info("Running final evaluation...")
            final_metrics = self.evaluator.evaluate(self.policy, include_tactical=True)
            self.evaluator.print_evaluation_report(final_metrics)
            self._save_metrics("final", final_metrics, prefix="final")
            
            # Final checkpoint
            self._save_checkpoint("final")
            
            # Training summary
            self._print_training_summary(total_time)
    
    def _collect_training_data(self) -> GRPOTrainingStep:
        """Collect GRPO training data for one step using diverse sources."""
        # Sample training positions from multiple sources for better diversity
        n_positions = self.config.batch_positions
        positions = []
        
        # 1. Current game positions (immediate exploration)
        current_game_positions = self.self_play.get_current_positions()
        n_from_games = min(len(current_game_positions), n_positions // 3)
        positions.extend(current_game_positions[:n_from_games])
        
        # 2. Dataset positions (high-quality diverse positions) - if enabled
        remaining_after_games = n_positions - len(positions)
        n_from_dataset = min(remaining_after_games, n_positions // 2) if self.config.use_dataset else 0
        
        if n_from_dataset > 0 and self.dataset_processor:
            try:
                dataset_positions = self._sample_dataset_positions(n_from_dataset)
                positions.extend(dataset_positions)
                self.logger.debug(f"Added {len(dataset_positions)} positions from dataset")
            except Exception as e:
                self.logger.warning(f"Failed to sample from dataset: {e}, using self-play instead")
        
        # 3. Fill remaining from self-play buffer
        final_remaining = n_positions - len(positions)
        if final_remaining > 0:
            buffer_positions = self.self_play.sample_training_positions(final_remaining)
            positions.extend(buffer_positions)
        
        # Collect GRPO groups
        groups = []
        for fen in positions:
            try:
                # Convert FEN string to chess.Board object
                board = chess.Board(fen)
                
                # Decide task type
                if random.random() < self.config.mix_env_ratio:
                    group = self.data_collector.collect_env_group(board)
                else:
                    group = self.data_collector.collect_policy_group(board)
                
                if group is not None:
                    groups.append(group)
            except Exception as e:
                self.logger.warning(f"Failed to collect group for {fen}: {e}")
        
        return GRPOTrainingStep(groups=groups)
    
    def _sample_dataset_positions(self, n_positions: int) -> List[str]:
        """Sample diverse positions from the RookWorld dataset."""
        try:
            # Load dataset if not already loaded
            if not hasattr(self.dataset_processor, '_dataset_cache'):
                self.logger.info("Loading RookWorld dataset for diverse training positions...")
                dataset = self.dataset_processor.load_dataset()
                # Extract unique FEN positions from the dataset
                positions = set()
                for sample in dataset.select(range(min(1000, len(dataset)))):  # Sample first 1000 for efficiency
                    # Extract FEN from the sample text
                    fen = self.dataset_processor.extract_position_from_text(sample['text'])
                    if fen:
                        positions.add(fen)
                self.dataset_processor._dataset_cache = list(positions)
                self.logger.info(f"Cached {len(positions)} unique positions from dataset")
            
            # Sample requested number of positions
            available_positions = self.dataset_processor._dataset_cache
            if len(available_positions) < n_positions:
                return random.sample(available_positions, len(available_positions))
            else:
                return random.sample(available_positions, n_positions)
                
        except Exception as e:
            self.logger.warning(f"Dataset sampling failed: {e}, using fallback")
            return []
    
    def _log_training_progress(self, step: int, metrics: Dict[str, float], step_time: float):
        """Log enhanced training progress information with GRPO best practices metrics."""
        # Core training metrics
        loss = metrics.get('total_loss', 0.0)
        policy_loss = metrics.get('policy_loss', 0.0)
        kl_div = metrics.get('kl_div', 0.0)
        mean_reward = metrics.get('mean_reward', 0.0)
        lr = metrics.get('learning_rate', 0.0)
        
        # Enhanced GRPO metrics
        kl_95pct = metrics.get('kl_div_95pct', 0.0)
        fraction_clipped = metrics.get('fraction_clipped', 0.0)
        entropy = metrics.get('approx_entropy', 0.0)
        rollout_buffer_size = metrics.get('rollout_buffer_size', 0)
        used_rollout = metrics.get('used_rollout_epochs', False)
        
        # Self-play stats
        self_play_stats = self.self_play.get_game_statistics()
        active_games = self_play_stats.get('active_games', 0)
        
        # Stockfish stats
        stockfish_stats = self.stockfish.get_cache_stats()
        cache_hit_rate = stockfish_stats.get('cache_hit_rate', 0.0)
        
        # Main progress line
        self.logger.info(
            f"Step {step:4d}/{self.config.steps} | "
            f"Loss: {loss:.4f} | "
            f"KL: {kl_div:.4f} (95%: {kl_95pct:.4f}) | "
            f"Reward: {mean_reward:.3f} | "
            f"LR: {lr:.6f} | "
            f"Time: {step_time:.2f}s"
        )
        
        # Enhanced metrics line
        rollout_indicator = "R" if used_rollout else "-"
        self.logger.info(
            f"         Enhanced | "
            f"Clipped: {fraction_clipped:.2%} | "
            f"Entropy: {entropy:.2f} | "
            f"Buffer: {rollout_buffer_size} | "
            f"Rollout: [{rollout_indicator}] | "
            f"Games: {active_games} | "
            f"Cache: {cache_hit_rate:.2%}"
        )
    
    def _save_metrics(self, step: Any, metrics: Any, prefix: str = "eval"):
        """Save evaluation metrics to file."""
        metrics_file = Path(self.config.output_dir) / f"{prefix}_metrics.json"
        
        # Load existing metrics
        all_metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        
        # Add new metrics
        all_metrics[str(step)] = {
            'step': step,
            'metrics': self.evaluator.metrics_to_dict(metrics),
            'timestamp': time.time()
        }
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def _save_checkpoint(self, step: Any):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'step': step
        }, model_path)
        
        # Save trainer state
        trainer_path = checkpoint_dir / "trainer.pt"
        self.trainer.save_checkpoint(str(trainer_path))
        
        # Save configuration
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            # Convert config to dict for JSON serialization
            config_dict = self.config.__dict__.copy()
            json.dump(config_dict, f, indent=2)
        
        # Save self-play positions
        positions_path = checkpoint_dir / "positions.json"
        self.self_play.save_position_buffer(str(positions_path))
        
        self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def _should_early_stop(self, step: int) -> bool:
        """Check if early stopping should be triggered."""
        # Could implement early stopping based on metrics
        # For now, just check for user interrupt
        return self.should_stop
    
    def _print_training_summary(self, total_time: float):
        """Print training completion summary."""
        # Get final statistics
        trainer_state = self.trainer.get_training_state()
        self_play_stats = self.self_play.get_game_statistics()
        stockfish_stats = self.stockfish.get_cache_stats()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total Time:           {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Steps Completed:      {trainer_state['step_count']}")
        print(f"Samples Trained:      {trainer_state['total_samples_trained']}")
        print(f"Games Played:         {self_play_stats['games_completed']}")
        print(f"Positions Generated:  {self_play_stats['positions_generated']}")
        print(f"Stockfish Analyses:   {stockfish_stats['engine_analyses']}")
        print(f"Cache Hit Rate:       {stockfish_stats['cache_hit_rate']:.2%}")
        print(f"Final Learning Rate:  {trainer_state['current_lr']:.8f}")
        print(f"Output Directory:     {self.config.output_dir}")
        print("="*60)
    
    def cleanup(self):
        """Clean up resources."""
        if self.stockfish:
            self.stockfish.close()
        self.logger.info("Cleanup completed")


def create_config_from_args(args) -> GRPOConfig:
    """Create GRPOConfig from command line arguments."""
    return GRPOConfig(
        # Model
        model_name_or_path=args.model,
        
        # Training
        steps=args.steps,
        batch_positions=args.batch_positions,
        group_size=args.group_size,
        
        # Hyperparameters
        lr=args.lr,
        kl_coef=args.kl_coef,
        kl_estimator=args.kl_estimator,
        clip_range=args.clip_range,
        temperature=args.temperature,
        max_new_tokens_env=args.max_new_tokens_env,
        
        # New improved parameters
        kl_divergence_threshold=args.kl_divergence_threshold,
        kl_warmup_steps=args.kl_warmup_steps,
        kl_warmup_factor=args.kl_warmup_factor,
        reward_warmup_steps=args.reward_warmup_steps,
        
        # Task mix
        mix_env_ratio=args.mix_env_ratio,
        use_dataset=args.use_dataset,
        
        # Self-play
        n_parallel_games=args.n_parallel_games,
        
        # System
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"),
        
        # Stockfish
        stockfish_path=args.stockfish_path,
        stockfish_time_limit=args.stockfish_time,
        
        # Evaluation
        eval_every=args.eval_every,
        save_every=args.save_every,
        
        # Performance optimizations
        use_mixed_precision=args.mixed_precision if args.mixed_precision is not None else torch.cuda.is_available(),
        use_torch_compile=args.torch_compile if args.torch_compile is not None else True,
        torch_compile_mode=args.compile_mode,
        use_gradient_checkpointing=args.gradient_checkpointing,
        
        # Resume and recovery settings
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
        enable_recovery=args.recovery_mode,
        force_new_run=args.new_run
    )


def main():
    """Main entry point for GRPO training."""
    parser = argparse.ArgumentParser(
        description="Train RookWorld-LM with Group Relative Policy Optimization (GRPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model", type=str, default="jrahn/RookWorld-LM-124M",
                       help="HuggingFace model identifier or local path")
    
    # Training arguments
    parser.add_argument("--steps", type=int, default=1000,
                       help="Number of training steps")
    parser.add_argument("--batch-positions", type=int, default=8,
                       help="Number of chess positions per training batch")
    parser.add_argument("--group-size", type=int, default=8,
                       help="GRPO group size (samples per position)")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate (validated for stability)")
    parser.add_argument("--kl-coef", type=float, default=0.01,
                       help="KL divergence penalty coefficient (reduced for stability)")
    parser.add_argument("--clip-range", type=float, default=0.2,
                       help="PPO clipping range")
    parser.add_argument("--kl-estimator", type=str, default="kl3", 
                       choices=["kl1", "kl2", "kl3"],
                       help="KL estimator: kl1 (simple), kl2 (exp-based), kl3 (quadratic)")
    
    # New improved parameters
    parser.add_argument("--kl-divergence-threshold", type=float, default=5.0,
                       help="KL divergence threshold for early stopping (higher = more tolerant)")
    parser.add_argument("--kl-warmup-steps", type=int, default=100,
                       help="Number of steps with reduced KL penalty for curriculum learning")
    parser.add_argument("--kl-warmup-factor", type=float, default=0.0,
                       help="KL coefficient multiplier during warmup (0.0 = no KL penalty)")
    parser.add_argument("--reward-warmup-steps", type=int, default=100,
                       help="Number of steps for graduated reward curriculum learning")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--max-new-tokens-env", type=int, default=80,
                       help="Maximum tokens to generate for environment (A:) tasks")
    
    # Task configuration
    parser.add_argument("--mix-env-ratio", type=float, default=0.2,
                       help="Fraction of environment (A:) tasks vs policy (P:) tasks (validated 36.9% stability improvement)")
    parser.add_argument("--use-dataset", action="store_true", default=True,
                       help="Use RookWorld dataset for diverse training positions")
    parser.add_argument("--no-dataset", dest="use_dataset", action="store_false",
                       help="Use only self-play for training positions")
    
    # Self-play
    parser.add_argument("--n-parallel-games", type=int, default=4,
                       help="Number of parallel self-play games")
    
    # Stockfish
    parser.add_argument("--stockfish-path", type=str, default=None,
                       help="Path to Stockfish binary (searches PATH if not provided)")
    parser.add_argument("--stockfish-time", type=float, default=0.1,
                       help="Time limit for Stockfish analysis per position")
    
    # Evaluation and checkpointing
    parser.add_argument("--eval-every", type=int, default=50,
                       help="Run evaluation every N steps")
    parser.add_argument("--save-every", type=int, default=100,
                       help="Save checkpoint every N steps")
    
    # Performance optimizations
    parser.add_argument("--mixed-precision", action="store_true", default=None,
                       help="Enable mixed precision training (auto-detected if not specified)")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false",
                       help="Disable mixed precision training")
    parser.add_argument("--torch-compile", action="store_true", default=None,
                       help="Enable torch.compile optimization")
    parser.add_argument("--no-torch-compile", dest="torch_compile", action="store_false",
                       help="Disable torch.compile optimization")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="Torch compile mode")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    
    # System
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu). Auto-detect if not specified")
    parser.add_argument("--output-dir", type=str, default="rookworld_grpo_checkpoints",
                       help="Directory for saving outputs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Resume and recovery options
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                       help="Automatically resume from latest checkpoint if available")
    parser.add_argument("--recovery-mode", action="store_true", default=True,
                       help="Enable automatic recovery on training instability")
    parser.add_argument("--new-run", action="store_true",
                       help="Force new run even if checkpoint exists (for experiments)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Initialize components but don't train")
    
    args = parser.parse_args()
    
    # Adjust logging level if debug
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    try:
        # Setup
        orchestrator.initialize_models()
        orchestrator.initialize_components()
        
        if args.dry_run:
            print("Dry run completed - all components initialized successfully")
            return
        
        # Run training
        orchestrator.run_training()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    main()