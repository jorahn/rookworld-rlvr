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
import subprocess

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


class TrainingHistoryWriter:
    """Writes training history to disk to avoid memory accumulation."""
    
    def __init__(self, log_dir: str, max_memory_entries: int = 10):
        """
        Initialize history writer.
        
        Args:
            log_dir: Directory for logs
            max_memory_entries: Maximum entries to keep in memory
        """
        self.log_dir = log_dir
        self.history_file = f"{log_dir}/training_history.jsonl"
        self.max_memory_entries = max_memory_entries
        self.memory_buffer = []
    
    def add_entry(self, entry: dict):
        """Add entry to disk immediately, keep minimal buffer in memory."""
        # Write to disk (append mode)
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Keep only last N entries in memory for quick access
        self.memory_buffer.append(entry)
        if len(self.memory_buffer) > self.max_memory_entries:
            self.memory_buffer.pop(0)
    
    def get_recent(self, n: int = 10) -> list:
        """Get last n entries from memory buffer."""
        return self.memory_buffer[-n:]
    
    def get_all(self) -> list:
        """Read all history from disk if needed."""
        entries = []
        if Path(self.history_file).exists():
            with open(self.history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        return entries


def get_nvidia_smi_memory():
    """
    Get actual GPU memory usage using nvidia-smi for current process.
    Returns memory in GB.
    """
    try:
        pid = os.getpid()
        # Query memory for specific process
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output to find our process
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    proc_pid, mem_mb = parts
                    if int(proc_pid.strip()) == pid:
                        return float(mem_mb.strip()) / 1024  # Convert to GB
        
        # If process not found, try global GPU memory
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        memory_mb = float(result.stdout.strip())
        return memory_mb / 1024  # Convert to GB
    except:
        # Fallback to torch if nvidia-smi fails
        return torch.cuda.memory_allocated() / 1024**3

def log_gpu_memory(logger, step: int, force: bool = False, interval: int = 10) -> bool:
    """
    Log GPU memory usage and detect potential leaks.
    
    Args:
        logger: Logger instance
        step: Current training step
        force: Force logging regardless of interval
        interval: Log every N steps
        
    Returns:
        True if emergency cleanup needed
    """
    if step % interval == 0 or force:
        # Get PyTorch tracked memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        # Get actual memory from nvidia-smi
        actual = get_nvidia_smi_memory()
        
        # Calculate untracked memory
        untracked = actual - allocated
        
        logger.info(f"[Memory] Step {step}: Actual={actual:.2f}GB (PyTorch={allocated:.2f}GB, "
                   f"Untracked={untracked:.2f}GB)")
        
        # Check for memory leak - if untracked memory is growing
        if untracked > 5.0:  # More than 5GB untracked is suspicious
            logger.warning(f"⚠️ MEMORY LEAK: {untracked:.2f}GB untracked memory!")
        
        # Warning thresholds (using actual memory now)
        if actual > 18.0:  # 18GB warning (have 24GB)
            logger.warning(f"⚠️ High memory usage: {actual:.2f}GB actual")
            return True  # Signal for emergency cleanup
        
        # Check for memory leak (growing usage)
        if step > 100 and allocated > 10.0:
            growth_rate = (allocated - 5.0) / (step / 100)  # GB per 100 steps above baseline
            if growth_rate > 1.0:
                logger.warning(f"⚠️ Possible memory leak: {growth_rate:.2f}GB/100steps growth")
                return True
    
    return False


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
    
    # Create reward scorer with continuous components
    scorer = RewardScorer(
        reward_shaping=config.reward_shaping,
        continuous_components=config.continuous_components
    )
    
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
            
            # Score completion with ground truth comparison
            reward, _ = scorer.score_single(prompt, completion, ground_truth=ground_truth, log_details=False)
            
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
    
    # Stack on CPU first, then move to GPU only when needed
    # Keep sequences on CPU initially to save memory
    sequences = torch.stack(padded_sequences)
    attention_masks = torch.stack(padded_masks)
    
    # Move to GPU just before use
    sequences = sequences.to(config.device)
    attention_masks = attention_masks.to(config.device)
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
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--n_train_samples", type=int, default=1000, help="Number of training samples to load")
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
        n_train_samples=args.n_train_samples,
        n_eval_samples=50,
        log_freq=1,  # Log every step for detailed monitoring
        eval_freq=args.eval_freq,
        save_freq=args.save_freq
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
    
    # Create reference model with caching disabled
    ref_model = ReferenceModel(model, cache_size=0)  # No caching to prevent memory leak
    logger.info("Created reference model for KL regularization (caching disabled)")
    
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
    
    # Initialize history writer
    history_writer = TrainingHistoryWriter(args.log_dir, max_memory_entries=10)
    
    # Log initial memory
    initial_pytorch_memory = torch.cuda.memory_allocated() / 1024**3
    initial_actual_memory = get_nvidia_smi_memory()
    logger.info(f"Initial GPU memory: PyTorch={initial_pytorch_memory:.2f}GB, Actual={initial_actual_memory:.2f}GB")
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    for step in range(1, config.max_steps + 1):
        start_time = time.time()
        
        # Monitor memory at start of step
        needs_cleanup = log_gpu_memory(logger, step, interval=10)
        
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
        
        # Extract tensors and ensure proper cleanup
        sequences = rollout_data["sequences"]
        attention_masks = rollout_data["attention_masks"]
        advantages = rollout_data["advantages"]
        prompt_lengths = rollout_data["prompt_lengths"]
        
        # Compute log probs with chunked processing
        policy_log_probs = compute_log_probs(
            model,
            sequences,
            attention_masks,
            chunk_size=config.log_prob_chunk_size  # Process in chunks to save memory
        )
        
        # Get ref_log_probs with chunked processing
        with torch.no_grad():
            ref_log_probs_cpu = ref_model.compute_log_probs(
                sequences,
                attention_masks,
                return_on_cpu=True  # Returns CPU tensor
            )
            ref_log_probs = ref_log_probs_cpu.to(config.device)  # Move to GPU for loss computation
            
        # Force cleanup of intermediate tensors
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Create prompt mask
        prompt_mask = create_prompt_mask(
            sequences,
            prompt_lengths
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
                del completion_mask  # Free immediately
            current_kl_coef = kl_controller.update(temp_kl)
            logger.debug(f"Adaptive KL coefficient: {current_kl_coef:.6f}")
        
        # Compute enhanced loss
        loss, metrics = grpo_loss(
            policy_log_probs,
            ref_log_probs,
            advantages,
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
        optimizer.zero_grad(set_to_none=True)  # More aggressive gradient clearing
        loss.backward()
        
        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        metrics['grad_norm'] = grad_norm.item()
        
        optimizer.step()
        
        # Detach loss to prevent gradient accumulation
        loss = loss.detach()
        
        # Clear any remaining gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
        
        elapsed_time = time.time() - start_time
        metrics['total_loss'] = loss.item()
        
        # Log training metrics
        log_training_metrics(logger, metrics, step, elapsed_time)
        
        # Store history and eval data before deleting variables
        mean_reward = np.mean(rollout_data['rewards'])
        reward_sample = [f'{r:.1f}' for r in rollout_data['rewards'][:10]]
        
        # Write history to disk instead of accumulating in RAM
        history_entry = {
            'step': step,
            'loss': metrics['total_loss'],
            'mean_reward': mean_reward,
            'kl_div': metrics.get('kl_div', 0),
            'memory_gb': torch.cuda.memory_allocated() / 1024**3,
            'actual_memory_gb': get_nvidia_smi_memory(),  # Track actual memory too
            'elapsed_time': elapsed_time
        }
        history_writer.add_entry(history_entry)
        
        # Periodic evaluation (before cleanup)
        if step % config.eval_freq == 0:
            logger.info(f"\n=== EVALUATION (Step {step}) ===")
            # Simple eval: report training metrics
            logger.info(f"Mean reward (last batch): {mean_reward:.3f}")
            logger.info(f"Reward distribution: {reward_sample}")
        
        # Aggressive cleanup - clear cache every step
        ref_model.clear_cache()
        
        # Explicitly free GPU memory after logging
        # Delete all large tensors explicitly
        del loss, policy_log_probs, ref_log_probs, ref_log_probs_cpu
        del sequences, attention_masks, advantages, prompt_mask
        if values is not None:
            del values, value_targets
        
        # Clear the entire rollout_data dictionary
        rollout_data.clear()
        del rollout_data
        
        # Force synchronization to ensure operations complete
        torch.cuda.synchronize()
        
        # Aggressive garbage collection every step
        import gc
        gc.collect()
        
        # Clear GPU cache every step to prevent accumulation  
        torch.cuda.empty_cache()
        
        # Clear Python's cyclic garbage collector
        gc.collect()
        
        # Emergency cleanup if high memory
        if needs_cleanup:
            # Extra aggressive cleanup
            for _ in range(3):  # Multiple passes to ensure cleanup
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logger.info("Performed emergency memory cleanup")
        
        # Save checkpoint periodically
        if step % config.save_freq == 0:
            checkpoint_path = f"{config.checkpoint_dir}/checkpoint_step{step}.pt"
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'recent_history': history_writer.get_recent(10)  # Only save recent history
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final results from disk
    results_file = log_file.replace('.log', '_results.json')
    all_history = history_writer.get_all()  # Read from disk
    with open(results_file, 'w') as f:
        json.dump(all_history, f, indent=2)
    
    logger.info(f"\n=== TRAINING COMPLETE ===")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"History file: {history_writer.history_file}")
    
    # Final statistics from recent buffer
    final_history = history_writer.get_recent(10)
    if final_history:
        final_rewards = [h['mean_reward'] for h in final_history]
        logger.info(f"\nFinal 10-step statistics:")
        logger.info(f"  Mean reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
        logger.info(f"  Final loss: {final_history[-1]['loss']:.4f}")
        logger.info(f"  Final memory: {final_history[-1].get('memory_gb', 0):.2f}GB")


if __name__ == "__main__":
    main()