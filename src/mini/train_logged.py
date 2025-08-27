"""
GRPO training script with detailed logging for debugging

Run 100 steps with batch size 32 and group size 8.
"""

import argparse
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime

import torch
import torch.optim as optim
import tiktoken
import numpy as np

# Mini modules
from config import GRPOConfig
from grpo import (
    compute_log_probs,
    compute_advantages,
    grpo_loss,
    create_prompt_mask,
    ReferenceModel,
    AdaptiveKLController,
    ValueFunction
)
from loader import load_rookworld_model
from dataset import load_and_prepare_samples
from reward_scorer import RewardScorer


def setup_logging(log_dir: str = "logs"):
    """Setup detailed logging to file and console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/grpo_training_{timestamp}.log"
    
    # Setup file handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Setup console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Setup logger
    logger = logging.getLogger('grpo_training')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def log_rollout_details(logger, rollout_data, step):
    """Log detailed rollout information."""
    logger.debug(f"\n=== ROLLOUT DETAILS (Step {step}) ===")
    logger.debug(f"Number of samples: {len(rollout_data['rewards'])}")
    logger.debug(f"Rewards: min={min(rollout_data['rewards']):.3f}, "
                f"max={max(rollout_data['rewards']):.3f}, "
                f"mean={np.mean(rollout_data['rewards']):.3f}, "
                f"std={np.std(rollout_data['rewards']):.3f}")
    
    # Log reward distribution
    reward_counts = {}
    for r in rollout_data['rewards']:
        reward_counts[f"{r:.1f}"] = reward_counts.get(f"{r:.1f}", 0) + 1
    logger.debug(f"Reward distribution: {reward_counts}")
    
    # Log sample completions
    if 'completions' in rollout_data:
        logger.debug("\nSample completions:")
        for i, (completion, reward) in enumerate(list(zip(rollout_data['completions'], rollout_data['rewards']))[:3]):
            logger.debug(f"\n[Sample {i+1}] Reward: {reward:.3f}")
            logger.debug(f"Completion: {completion[:200]}...")  # First 200 chars


def log_training_metrics(logger, metrics, step, elapsed_time):
    """Log detailed training metrics."""
    logger.info(f"\nStep {step} | Time: {elapsed_time:.2f}s")
    logger.info(f"  Loss: {metrics.get('total_loss', 0):.4f}")
    logger.info(f"  PG Loss: {metrics.get('pg_loss', 0):.4f}")
    logger.info(f"  KL Divergence: {metrics.get('kl_div', 0):.4f}")
    
    if 'kl_forward' in metrics:
        logger.debug(f"  KL Forward: {metrics['kl_forward']:.4f}")
    if 'kl_reverse' in metrics:
        logger.debug(f"  KL Reverse: {metrics['kl_reverse']:.4f}")
    if 'kl_symmetric' in metrics:
        logger.debug(f"  KL Symmetric: {metrics['kl_symmetric']:.4f}")
    
    if 'value_loss' in metrics and metrics['value_loss'] > 0:
        logger.info(f"  Value Loss: {metrics['value_loss']:.4f}")
    if 'entropy' in metrics:
        logger.debug(f"  Entropy: {metrics['entropy']:.4f}")
    if 'ratio_outliers' in metrics:
        logger.debug(f"  Ratio Outliers: {metrics['ratio_outliers']:.3%}")
    
    # Log gradient norms
    if 'grad_norm' in metrics:
        logger.debug(f"  Gradient Norm: {metrics['grad_norm']:.4f}")


def collect_rollouts(
    model,
    samples: List[Tuple],
    tokenizer,
    config: GRPOConfig,
    logger,
    step: int,
    baseline_tracker: Optional[Dict] = None
) -> Dict:
    """
    Generate K completions per prompt and compute rewards.
    """
    model.eval()
    
    all_sequences = []
    all_attention_masks = []
    all_rewards = []
    all_prompt_lengths = []
    all_completions = []  # Store for logging
    
    pad_id = 50256  # GPT-2 EOS token ID
    
    # Create reward scorer
    scorer = RewardScorer(reward_shaping=config.reward_shaping)
    
    for sample_idx, (task_type, prompt, ground_truth, data) in enumerate(samples):
        logger.debug(f"Processing sample {sample_idx+1}/{len(samples)}: {task_type} task")
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        prompt_length = len(prompt_ids)
        
        # Generate K completions
        sample_rewards = []
        sample_sequences = []
        sample_masks = []
        sample_completions = []
        
        for k in range(config.k_samples):
            # Generate completion
            prompt_tensor = torch.tensor(prompt_ids, device=config.device).unsqueeze(0)
            
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_tensor,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    pad_token_id=pad_id
                )
            
            # Decode completion
            completion_ids = output_ids[0, len(prompt_ids):].tolist()
            completion = tokenizer.decode(completion_ids)
            
            # Clean completion
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            # Score completion - prompt already has correct format
            reward, _ = scorer.score_single(prompt, completion, log_details=False)
            
            sample_rewards.append(reward)
            # Detach and move to CPU to avoid memory accumulation
            sample_sequences.append(output_ids[0].detach().cpu())
            sample_masks.append(torch.ones_like(output_ids[0]).detach().cpu())
            sample_completions.append(completion)
            
            if k == 0:  # Log first completion per sample
                logger.debug(f"  K={k+1}, Reward={reward:.3f}, Completion: {completion[:100]}...")
        
        # Store group data
        all_rewards.extend(sample_rewards)
        all_sequences.extend(sample_sequences)
        all_attention_masks.extend(sample_masks)
        all_prompt_lengths.extend([prompt_length] * config.k_samples)
        all_completions.extend(sample_completions)
    
    # Pad sequences for batch processing
    max_len = max(seq.shape[0] for seq in all_sequences)
    padded_sequences = []
    padded_masks = []
    
    for seq, mask in zip(all_sequences, all_attention_masks):
        seq_len = seq.shape[0]
        if seq_len < max_len:
            # Create padding on CPU first
            padding = torch.full((max_len - seq_len,), pad_id)
            seq = torch.cat([padding, seq])
            mask = torch.cat([torch.zeros(max_len - seq_len), mask])
        padded_sequences.append(seq)
        padded_masks.append(mask)
    
    # Stack on CPU first, then move to GPU
    sequences = torch.stack(padded_sequences).to(config.device)
    attention_masks = torch.stack(padded_masks).to(config.device)
    rewards = torch.tensor(all_rewards, device=config.device)
    
    # Compute advantages with enhanced baseline
    advantages = compute_advantages(
        rewards,
        group_size=config.k_samples,
        baseline_type=config.baseline_type,
        baseline_tracker=baseline_tracker
    )
    
    logger.debug(f"\nAdvantages: mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}")
    
    return {
        "sequences": sequences,
        "attention_masks": attention_masks,
        "rewards": all_rewards,
        "advantages": advantages,
        "prompt_lengths": all_prompt_lengths,
        "completions": all_completions
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k_samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()
    
    # Setup logging
    logger, log_file = setup_logging(args.log_dir)
    logger.info(f"Logging to: {log_file}")
    
    # Create config
    config = GRPOConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        k_samples=args.k_samples,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        n_train_samples=200,  # Ensure enough samples
        n_eval_samples=50,
        log_freq=1,  # Log every step
        eval_freq=20
    )
    
    # Log configuration
    logger.info("\n=== CONFIGURATION ===")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    
    # Load model
    logger.info("\n=== LOADING MODEL ===")
    model = load_rookworld_model(config.model_path, device=config.device)
    logger.info(f"Loaded model from {config.model_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Create reference model
    ref_model = ReferenceModel(model)
    logger.info("Created reference model for KL regularization")
    
    # Setup enhanced features
    kl_controller = None
    if config.adaptive_kl:
        kl_controller = AdaptiveKLController(
            init_kl_coef=config.kl_coef,
            target_kl=config.kl_target,
            horizon=config.kl_horizon
        )
        logger.info(f"Initialized adaptive KL controller (target={config.kl_target})")
    
    value_function = None
    if config.value_loss_coef > 0:
        value_function = ValueFunction(model.config.n_embd).to(config.device)
        logger.info("Initialized value function")
    
    baseline_tracker = {"ema": 0.0, "count": 0} if config.baseline_type == "ema" else None
    
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load data
    logger.info(f"\n=== LOADING DATA ===")
    logger.info(f"Loading {config.n_train_samples} training samples...")
    train_samples = load_and_prepare_samples(
        n_samples=config.n_train_samples,
        seed=config.data_seed
    )
    logger.info(f"Loaded {len(train_samples)} training samples")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95)
    )
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    training_history = []
    
    for step in range(1, config.max_steps + 1):
        start_time = time.time()
        
        # Sample batch
        batch_indices = np.random.choice(
            len(train_samples),
            size=min(config.batch_size, len(train_samples)),
            replace=False
        )
        batch_samples = [train_samples[i] for i in batch_indices]
        
        logger.debug(f"\nStep {step}: Sampled {len(batch_samples)} prompts")
        
        # Collect rollouts
        rollout_data = collect_rollouts(
            model, batch_samples, tokenizer, config, logger, step, baseline_tracker
        )
        
        # Log rollout details
        log_rollout_details(logger, rollout_data, step)
        
        # Training step
        model.train()
        
        # Compute log probs
        policy_log_probs = compute_log_probs(
            model,
            rollout_data["sequences"],
            rollout_data["attention_masks"]
        )
        
        ref_log_probs = ref_model.compute_log_probs(
            rollout_data["sequences"],
            rollout_data["attention_masks"]
        )
        
        # Create prompt mask
        prompt_mask = create_prompt_mask(
            rollout_data["sequences"],
            rollout_data["prompt_lengths"]
        )
        
        # Compute value estimates if using value function
        values = None
        value_targets = None
        if value_function is not None:
            with torch.no_grad():
                # Simplified: just use rewards tensor as both values and targets
                values = torch.tensor(rollout_data["rewards"], device=config.device)
                value_targets = torch.tensor(rollout_data["rewards"], device=config.device)
        
        # Update KL coefficient if using adaptive control
        current_kl_coef = config.kl_coef
        if kl_controller is not None:
            with torch.no_grad():
                completion_mask = (1 - prompt_mask)
                temp_kl = ((policy_log_probs - ref_log_probs) * completion_mask).sum(dim=1).mean().item()
            current_kl_coef = kl_controller.update(temp_kl)
            logger.debug(f"Adaptive KL coefficient: {current_kl_coef:.6f}")
        
        # Compute enhanced loss
        loss, metrics = grpo_loss(
            policy_log_probs,
            ref_log_probs,
            rollout_data["advantages"],
            prompt_mask,
            kl_coef=current_kl_coef,
            clip_range=config.clip_range,
            kl_type=config.kl_type,
            values=values,
            value_targets=value_targets,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        metrics['grad_norm'] = grad_norm.item()
        
        optimizer.step()
        
        elapsed_time = time.time() - start_time
        metrics['total_loss'] = loss.item()
        
        # Log training metrics
        log_training_metrics(logger, metrics, step, elapsed_time)
        
        # Store history before deleting variables
        training_history.append({
            'step': step,
            'loss': metrics['total_loss'],
            'mean_reward': np.mean(rollout_data['rewards']),
            'kl_div': metrics.get('kl_div', 0),
            'elapsed_time': elapsed_time
        })
        
        # Clear reference model cache periodically to prevent memory buildup
        if step % 5 == 0:
            ref_model.clear_cache()
        
        # Explicitly free GPU memory after logging
        del loss, policy_log_probs, ref_log_probs, rollout_data
        torch.cuda.empty_cache()
        
        # Periodic evaluation
        if step % config.eval_freq == 0:
            logger.info(f"\n=== EVALUATION (Step {step}) ===")
            # Simple eval: report training metrics
            logger.info(f"Mean reward (last batch): {np.mean(rollout_data['rewards']):.3f}")
            logger.info(f"Reward distribution: {[f'{r:.1f}' for r in rollout_data['rewards'][:10]]}")
        
        # Save checkpoint periodically
        if step % 50 == 0:
            checkpoint_path = f"{config.checkpoint_dir}/checkpoint_step{step}.pt"
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'history': training_history
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final results
    results_file = log_file.replace('.log', '_results.json')
    with open(results_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"\n=== TRAINING COMPLETE ===")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Log file: {log_file}")
    
    # Final statistics
    final_rewards = [h['mean_reward'] for h in training_history[-10:]]
    logger.info(f"\nFinal 10-step statistics:")
    logger.info(f"  Mean reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    logger.info(f"  Final loss: {training_history[-1]['loss']:.4f}")


if __name__ == "__main__":
    main()