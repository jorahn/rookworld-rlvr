"""
GRPO training script for mini implementation

Minimal training loop for fine-tuning RookWorld-LM.
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
from reward_scorer import compute_grpo_rewards


def collect_rollouts(
    model,
    samples: List[Tuple],
    tokenizer,
    config: GRPOConfig,
    baseline_tracker: Optional[Dict] = None
) -> Dict:
    """
    Generate K completions per prompt and compute rewards.
    
    Args:
        model: Current policy model
        samples: List of (task_type, prompt, ground_truth, data) tuples
        tokenizer: Tokenizer for encoding/decoding
        config: Training configuration
        
    Returns:
        Dictionary with sequences, rewards, advantages, etc.
    """
    all_prompts = []
    all_completions = []
    all_sequences = []
    all_prompt_lengths = []
    all_attention_masks = []
    
    model.eval()
    
    print(f"Generating {config.k_samples} completions per prompt...")
    
    for task_type, prompt, _, _ in samples:
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)
        
        # Generate K completions
        for _ in range(config.k_samples):
            input_tensor = torch.tensor([prompt_ids], device=config.device)
            
            # Generate completion
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    pad_token_id=50256  # GPT-2 EOS token ID
                )
            
            # Decode completion
            generated_ids = generated[0].cpu().tolist()
            completion_ids = generated_ids[prompt_len:]
            completion = tokenizer.decode(completion_ids)
            
            # Clean up
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            # Store
            all_prompts.append(prompt)
            all_completions.append(completion)
            all_sequences.append(generated_ids)
            all_prompt_lengths.append(prompt_len)
    
    # Compute rewards
    print("Computing rewards...")
    advantages, reward_details = compute_grpo_rewards(
        all_prompts,
        all_completions,
        group_size=config.k_samples,
        reward_shaping=config.reward_shaping,
        continuous_components=config.continuous_components,
        verbose=False
    )
    
    rewards = torch.tensor(
        [d.shaped_reward for d in reward_details],
        device=config.device,
        dtype=torch.float32
    )
    
    # Recompute advantages with enhanced method
    advantages = compute_advantages(
        rewards,
        group_size=config.k_samples,
        baseline_type=config.baseline_type,
        baseline_tracker=baseline_tracker,
        use_gae=config.use_gae,
        gae_lambda=config.gae_lambda
    ).cpu().numpy()  # Convert back to numpy for compatibility
    
    # Prepare tensors for training
    max_len = max(len(seq) for seq in all_sequences)
    pad_id = 50256  # GPT-2 EOS token ID
    
    # Pad sequences and create attention masks
    padded_sequences = []
    attention_masks = []
    
    for seq in all_sequences:
        # Right padding for training
        padding = max_len - len(seq)
        padded_seq = seq + [pad_id] * padding
        mask = [1.0] * len(seq) + [0.0] * padding
        
        padded_sequences.append(padded_seq)
        attention_masks.append(mask)
    
    sequences = torch.tensor(padded_sequences, device=config.device)
    attention_masks = torch.tensor(attention_masks, device=config.device)
    prompt_lengths = torch.tensor(all_prompt_lengths, device=config.device)
    advantages_tensor = torch.tensor(advantages, device=config.device, dtype=torch.float32)
    
    return {
        "sequences": sequences,
        "attention_masks": attention_masks,
        "prompt_lengths": prompt_lengths,
        "rewards": rewards,
        "advantages": advantages_tensor,
        "prompts": all_prompts,
        "completions": all_completions
    }


def evaluate(
    model,
    eval_samples: List[Tuple],
    tokenizer,
    config: GRPOConfig
) -> Dict:
    """
    Evaluate model on validation samples.
    
    Args:
        model: Current policy model
        eval_samples: Evaluation samples
        tokenizer: Tokenizer
        config: Configuration
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    
    prompts = []
    completions = []
    
    with torch.no_grad():
        for task_type, prompt, _, _ in eval_samples:
            # Generate single completion for eval
            prompt_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([prompt_ids], device=config.device)
            
            generated = model.generate(
                input_tensor,
                max_new_tokens=config.max_new_tokens,
                temperature=0.7,  # Lower temperature for eval
                top_k=config.top_k,
                top_p=config.top_p,
                pad_token_id=50256  # GPT-2 EOS token ID
            )
            
            # Decode
            generated_ids = generated[0].cpu().tolist()
            completion = tokenizer.decode(generated_ids[len(prompt_ids):])
            
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            prompts.append(prompt)
            completions.append(completion)
    
    # Score completions
    _, reward_details = compute_grpo_rewards(
        prompts,
        completions,
        group_size=1,
        reward_shaping=config.reward_shaping,
        verbose=False
    )
    
    # Compute metrics
    rewards = [d.shaped_reward for d in reward_details]
    format_valid = [d.format_valid for d in reward_details]
    
    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "format_valid_rate": np.mean(format_valid),
        "min_reward": min(rewards),
        "max_reward": max(rewards)
    }
    
    return metrics


def train(config: GRPOConfig):
    """
    Main training loop.
    
    Args:
        config: Training configuration
    """
    print("=" * 60)
    print("GRPO TRAINING - MINI IMPLEMENTATION")
    print("=" * 60)
    
    # Setup
    device = torch.device(config.device)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load model
    print(f"\nLoading model: {config.model_path}")
    model = load_rookworld_model(config.model_path, device=config.device)
    model.train()
    
    # Create reference model
    print("Creating reference model...")
    ref_model = ReferenceModel(model)
    
    # Initialize adaptive KL controller if enabled
    kl_controller = None
    if config.adaptive_kl:
        print("Initializing adaptive KL controller...")
        kl_controller = AdaptiveKLController(
            init_kl_coef=config.kl_coef,
            target_kl=config.kl_target,
            horizon=config.kl_horizon
        )
    
    # Initialize value function if using GAE
    value_function = None
    value_optimizer = None
    if config.use_gae:
        print("Initializing value function...")
        value_function = ValueFunction(hidden_size=768).to(config.device)
        value_optimizer = optim.AdamW(
            value_function.parameters(),
            lr=config.learning_rate * 0.1  # Smaller LR for value function
        )
    
    # Initialize baseline tracker for advanced methods
    baseline_tracker = {} if config.baseline_type in ["ema", "adaptive"] else None
    
    # Load data
    print(f"\nLoading {config.n_train_samples} training samples...")
    train_samples = load_and_prepare_samples(
        n_samples=config.n_train_samples,
        seed=config.data_seed
    )
    
    print(f"Loading {config.n_eval_samples} evaluation samples...")
    eval_samples = load_and_prepare_samples(
        n_samples=config.n_eval_samples,
        seed=config.data_seed + 1000  # Different seed for eval
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95)
    )
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initial evaluation
    print("\n" + "=" * 60)
    print("INITIAL EVALUATION")
    print("=" * 60)
    eval_metrics = evaluate(model, eval_samples, tokenizer, config)
    print(f"Mean reward: {eval_metrics['mean_reward']:.3f}")
    print(f"Format valid: {eval_metrics['format_valid_rate']*100:.1f}%")
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    for step in range(1, config.max_steps + 1):
        start_time = time.time()
        
        # Sample batch
        batch_indices = np.random.choice(
            len(train_samples),
            size=min(config.batch_size, len(train_samples)),
            replace=False
        )
        batch_samples = [train_samples[i] for i in batch_indices]
        
        # Collect rollouts
        rollout_data = collect_rollouts(
            model, batch_samples, tokenizer, config, baseline_tracker
        )
        
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
            # Get hidden states from model (need to modify this based on actual model structure)
            with torch.no_grad():
                model_outputs = model(rollout_data["sequences"], rollout_data["attention_masks"])
                hidden_states = model_outputs.get("hidden_states", model_outputs["logits"])  # Fallback
                if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                    # Average over completion tokens for value estimation
                    completion_mask_expanded = (1 - prompt_mask).unsqueeze(-1)
                    values = value_function(hidden_states)
                    values = (values * (1 - prompt_mask)).sum(dim=1) / ((1 - prompt_mask).sum(dim=1) + 1e-8)
                    value_targets = rollout_data["rewards"]  # Use actual rewards as targets
        
        # Update KL coefficient if using adaptive control
        current_kl_coef = config.kl_coef
        if kl_controller is not None:
            # First compute current KL to update controller
            with torch.no_grad():
                completion_mask = (1 - prompt_mask)
                temp_kl = ((policy_log_probs - ref_log_probs) * completion_mask).sum(dim=1).mean().item()
            current_kl_coef = kl_controller.update(temp_kl)
        
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
        if value_optimizer is not None:
            value_optimizer.zero_grad()
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if value_function is not None:
            torch.nn.utils.clip_grad_norm_(value_function.parameters(), config.grad_clip)
        
        # Update
        optimizer.step()
        if value_optimizer is not None:
            value_optimizer.step()
        
        # Logging
        if step % config.log_freq == 0:
            elapsed = time.time() - start_time
            print(f"\n[Step {step}/{config.max_steps}] Time: {elapsed:.2f}s")
            print(f"  Total Loss: {metrics['total_loss']:.4f}")
            print(f"  PG Loss: {metrics['pg_loss']:.4f}")
            print(f"  KL Div ({config.kl_type}): {metrics['kl_div']:.4f}")
            print(f"  KL Coef: {metrics['kl_coef']:.4f}")
            if 'value_loss' in metrics and metrics['value_loss'] > 0:
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
            if 'entropy_loss' in metrics and metrics['entropy_loss'] != 0:
                print(f"  Entropy Loss: {metrics['entropy_loss']:.4f}")
            print(f"  Mean Reward: {rollout_data['rewards'].mean().item():.3f}")
            print(f"  Advantage: {metrics['advantage_mean']:.3f} ± {metrics['advantage_std']:.3f}")
            print(f"  Ratio: {metrics['ratio_mean']:.3f} ± {metrics['ratio_std']:.3f}")
            print(f"  Clipped: {metrics['clipped_frac']*100:.1f}% | Outliers: {metrics['ratio_outliers']*100:.1f}%")
            
            # Show all KL types for monitoring
            if config.kl_type != "forward":
                print(f"  KL Forward: {metrics['kl_forward']:.4f}")
            if config.kl_type != "reverse":
                print(f"  KL Reverse: {metrics['kl_reverse']:.4f}")
            if config.kl_type != "symmetric":
                print(f"  KL Symmetric: {metrics['kl_symmetric']:.4f}")
        
        # Evaluation
        if step % config.eval_freq == 0:
            print(f"\n[Step {step}] Evaluating...")
            eval_metrics = evaluate(model, eval_samples, tokenizer, config)
            print(f"  Mean reward: {eval_metrics['mean_reward']:.3f}")
            print(f"  Format valid: {eval_metrics['format_valid_rate']*100:.1f}%")
        
        # Save checkpoint
        if step % config.save_freq == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    eval_metrics = evaluate(model, eval_samples, tokenizer, config)
    print(f"Mean reward: {eval_metrics['mean_reward']:.3f}")
    print(f"Format valid: {eval_metrics['format_valid_rate']*100:.1f}%")
    print(f"Reward range: [{eval_metrics['min_reward']:.3f}, {eval_metrics['max_reward']:.3f}]")
    
    # Save final model
    final_path = Path(config.checkpoint_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model: {final_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="GRPO training for RookWorld-LM")
    
    # Key hyperparameters
    parser.add_argument("--steps", type=int, help="Number of training steps")
    parser.add_argument("--k_samples", type=int, help="Completions per prompt")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--kl_coef", type=float, help="KL coefficient")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create config
    config = GRPOConfig()
    
    # Override with command line arguments
    if args.steps:
        config.max_steps = args.steps
    if args.k_samples:
        config.k_samples = args.k_samples
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.kl_coef:
        config.kl_coef = args.kl_coef
    
    # Print configuration
    print("Configuration:")
    print(f"  Steps: {config.max_steps}")
    print(f"  K samples: {config.k_samples}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  KL coefficient: {config.kl_coef}")
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()