#!/usr/bin/env python3
"""
Lean GRPO Training Script for RookWorld-LM

Minimal implementation without dead code or memory leaks.
Key features:
- Training model on cuda:0, frozen reference model on cuda:1
- Extensive logging of all data flow
- Simple reward shaping with Stockfish validation
- No mixed precision, minimal config
- Only uses prepared RookWorld dataset (no self-play)
"""

import argparse
import logging
import torch
from transformers import GPT2Tokenizer
import time
from typing import Dict

# Import lean components
from model import LeanRookWorldModel
from dataset import LeanRookWorldDataset
from validation import LeanValidator
from grpo import LeanGRPOTrainer


def setup_logging(log_level: str = "INFO"):
    """Setup comprehensive logging"""
    
    # Create formatter with timestamp and detailed info
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('lean_grpo_training.log')
    file_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Also log to specific module loggers
    for module in ['lean.model', 'lean.dataset', 'lean.validation', 'lean.grpo']:
        logger = logging.getLogger(module)
        logger.setLevel(getattr(logging, log_level.upper()))


def setup_models_and_tokenizer():
    """Setup models on different GPUs and tokenizer"""
    
    logger = logging.getLogger(__name__)
    logger.info("=== MODEL SETUP ===")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - this implementation requires 2 GPUs")
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        logger.warning("Only 1 GPU available, using cuda:0 for both models")
        train_device = "cuda:0"
        ref_device = "cuda:0"
    else:
        train_device = "cuda:0"  # Training model
        ref_device = "cuda:1"    # Frozen reference model
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"
    
    logger.info(f"Tokenizer loaded - vocab size: {len(tokenizer)}")
    
    # Load training model
    logger.info(f"Loading training model on {train_device}...")
    train_model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
    train_model.to_device(train_device)
    train_model.train()  # Training mode
    
    # Load reference model (frozen)
    logger.info(f"Loading reference model on {ref_device}...")
    ref_model = LeanRookWorldModel("jrahn/RookWorld-LM-124M")
    ref_model.to_device(ref_device)
    ref_model.eval()  # Frozen mode
    
    # Freeze reference model parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    logger.info("Models loaded and placed on GPUs")
    logger.info(f"Training model device: {next(train_model.parameters()).device}")
    logger.info(f"Reference model device: {next(ref_model.parameters()).device}")
    
    return train_model, ref_model, tokenizer


def log_memory_usage():
    """Log GPU memory usage"""
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i} memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")


def main():
    parser = argparse.ArgumentParser(description="Lean GRPO Training for RookWorld-LM")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size") 
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--stockfish-path", default=None, help="Path to Stockfish executable")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=== LEAN GRPO TRAINING START ===")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Setup models and tokenizer
        train_model, ref_model, tokenizer = setup_models_and_tokenizer()
        
        # Setup dataset
        logger.info("=== DATASET SETUP ===")
        dataset = LeanRookWorldDataset()
        dataset.load()
        
        # Setup validator
        logger.info("=== VALIDATOR SETUP ===")
        validator = LeanValidator(stockfish_path=args.stockfish_path)
        validator.start_engine()
        
        # Setup GRPO trainer
        logger.info("=== TRAINER SETUP ===")
        trainer = LeanGRPOTrainer(
            model=train_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            group_size=args.group_size,
            clip_range=args.clip_range,
            kl_coef=args.kl_coef,
            learning_rate=args.learning_rate
        )
        
        # Training loop
        logger.info("=== TRAINING LOOP START ===")
        
        for step in range(args.steps):
            step_start_time = time.time()
            
            logger.info(f"\\n{'='*50}")
            logger.info(f"TRAINING STEP {step + 1}/{args.steps}")
            logger.info(f"{'='*50}")
            
            # Log initial memory
            log_memory_usage()
            
            # Get training batch from dataset
            logger.info("Getting training batch...")
            batch_data = dataset.get_training_batch(args.batch_size)
            
            prompts = [prompt for _, prompt, _ in batch_data]
            task_types = [task_type for task_type, _, _ in batch_data]
            
            logger.info(f"Batch loaded - prompts: {len(prompts)}")
            logger.info(f"Task distribution: P={sum(1 for t in task_types if t == 'P')}, "
                       f"A={sum(1 for t in task_types if t == 'A')}")
            
            # Log sample prompts
            for i, (task_type, prompt, _) in enumerate(batch_data[:2]):
                logger.info(f"Sample {i+1} ({task_type}): {prompt[:100]}...")
            
            # Collect rollouts
            logger.info("Collecting rollouts...")
            batch = trainer.collect_rollouts(prompts, task_types, validator)
            
            # Log rollout results
            logger.info(f"Rollouts collected - completions: {len(batch.completions)}")
            for i, completion in enumerate(batch.completions[:2]):
                logger.info(f"Completion {i+1}: {completion[:100]}...")
            
            # Training step
            logger.info("Performing training step...")
            metrics = trainer.train_step(batch)
            
            # Log metrics
            step_time = time.time() - step_start_time
            logger.info(f"\\nSTEP {step + 1} METRICS:")
            logger.info(f"  Total Loss:     {metrics['total_loss']:.4f}")
            logger.info(f"  Policy Loss:    {metrics['policy_loss']:.4f}")
            logger.info(f"  KL Penalty:     {metrics['kl_penalty']:.4f}")
            logger.info(f"  KL Divergence:  {metrics['kl_divergence']:.4f}")
            logger.info(f"  Mean Reward:    {metrics['reward_mean']:.4f}")
            logger.info(f"  Ratio Mean:     {metrics['ratio_mean']:.4f}")
            logger.info(f"  Grad Norm:      {metrics['grad_norm']:.4f}")
            logger.info(f"  Step Time:      {step_time:.2f}s")
            
            # Log memory after step
            log_memory_usage()
            
            # Log tensor shapes and memory info
            logger.debug(f"Batch tensor info:")
            logger.debug(f"  Rewards shape: {batch.rewards.shape}, device: {batch.rewards.device}")
            logger.debug(f"  Logprobs shape: {batch.logprobs.shape}, device: {batch.logprobs.device}")
            logger.debug(f"  Ref logprobs shape: {batch.ref_logprobs.shape}, device: {batch.ref_logprobs.device}")
            
            # Cleanup
            del batch
            torch.cuda.empty_cache()
            
            logger.info(f"Step {step + 1} completed in {step_time:.2f}s")
        
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if 'validator' in locals():
            validator.stop_engine()
        logger.info("=== CLEANUP COMPLETED ===")


if __name__ == "__main__":
    main()