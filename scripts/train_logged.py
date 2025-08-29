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

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import from rookworld_rlvr package
from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.grpo import (
    compute_log_probs,
    compute_advantages,
    grpo_loss,
    create_prompt_mask,
    ReferenceModel,
    AdaptiveKLController,
    ValueFunction
)
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.reward_scorer import RewardScorer

# Import batch generation modules
from rookworld_rlvr.batch_generation import (
    collect_rollouts_batched,
    collect_rollouts_task_specific_batched
)

# Import learning rate scheduler
from rookworld_rlvr.scheduler import create_lr_scheduler, visualize_lr_schedule


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
    rewards = rollout_data['rewards']
    if torch.is_tensor(rewards):
        logger.debug(f"Rewards: min={rewards.min().item():.3f}, "
                    f"max={rewards.max().item():.3f}, "
                    f"mean={rewards.mean().item():.3f}, "
                    f"std={rewards.std().item():.3f}")
    else:
        logger.debug(f"Rewards: min={min(rewards):.3f}, "
                    f"max={max(rewards):.3f}, "
                    f"mean={np.mean(rewards):.3f}, "
                    f"std={np.std(rewards):.3f}")
    
    # Log reward distribution  
    reward_counts = {}
    rewards_list = rewards.cpu().tolist() if torch.is_tensor(rewards) else rewards
    for r in rewards_list:
        reward_counts[f"{r:.1f}"] = reward_counts.get(f"{r:.1f}", 0) + 1
    logger.debug(f"Reward distribution: {reward_counts}")
    
    # Log sample completions
    if 'completions' in rollout_data:
        logger.debug("\nSample completions:")
        for i, (completion, reward) in enumerate(list(zip(rollout_data['completions'], rollout_data['rewards']))[:3]):
            logger.debug(f"\n[Sample {i+1}] Reward: {reward:.3f}")
            logger.debug(f"Completion: {completion[:200]}...")  # First 200 chars


def debug_advantage_computation(rollout_data, config, logger, step):
    """
    Debug advantage computation to identify why std=0.000 consistently.
    """
    rewards = rollout_data['rewards']
    advantages = rollout_data.get('advantages')
    
    if advantages is None:
        logger.warning(f"⚠️ No advantages found in rollout_data")
        return
        
    logger.info(f"=== ADVANTAGE COMPUTATION DEBUG (Step {step}) ===")
    
    # Analyze reward grouping
    batch_size = len(rewards) // config.k_samples
    logger.info(f"  Total rewards: {len(rewards)}, Expected groups: {batch_size}, K_samples: {config.k_samples}")
    
    # Check group structure
    for group_idx in range(min(3, batch_size)):  # Check first 3 groups
        start_idx = group_idx * config.k_samples
        end_idx = start_idx + config.k_samples
        group_rewards = rewards[start_idx:end_idx]
        group_advantages = advantages[start_idx:end_idx]
        
        if torch.is_tensor(group_rewards):
            group_reward_mean = group_rewards.mean().item()
            group_reward_std = group_rewards.std().item()
            group_adv_mean = group_advantages.mean().item() if torch.is_tensor(group_advantages) else np.mean(group_advantages)
            group_adv_std = group_advantages.std().item() if torch.is_tensor(group_advantages) else np.std(group_advantages)
        else:
            group_reward_mean = np.mean(group_rewards)
            group_reward_std = np.std(group_rewards)
            group_adv_mean = np.mean(group_advantages)
            group_adv_std = np.std(group_advantages)
            
        logger.info(f"  Group {group_idx}: rewards μ={group_reward_mean:.3f}±{group_reward_std:.3f}, advantages μ={group_adv_mean:.3f}±{group_adv_std:.3f}")
        
        # Check for identical rewards within group (indicates issue)
        if group_reward_std == 0.0:
            logger.warning(f"    ⚠️ Group {group_idx} has identical rewards: {group_rewards}")
    
    # Overall advantage stats
    if torch.is_tensor(advantages):
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        adv_min = advantages.min().item()
        adv_max = advantages.max().item()
    else:
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        adv_min = np.min(advantages)
        adv_max = np.max(advantages)
        
    logger.info(f"  Overall Advantages: μ={adv_mean:.3f}, σ={adv_std:.3f}, min={adv_min:.3f}, max={adv_max:.3f}")
    
    if adv_std == 0.0:
        logger.error(f"🚨 ZERO ADVANTAGE VARIANCE: All advantages identical! This breaks GRPO learning.")


def analyze_batch_generation_quality(rollout_data, logger, step):
    """
    Analyze batch generation quality and log detailed metrics.
    """
    rewards = rollout_data['rewards']
    sequences = rollout_data['sequences']
    completions = rollout_data.get('completions', [])
    
    # Reward analysis
    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    reward_min = rewards.min().item()
    reward_max = rewards.max().item()
    
    # Sequence length analysis
    seq_lengths = [seq.shape[0] for seq in sequences] if isinstance(sequences[0], torch.Tensor) else [len(seq) for seq in sequences]
    length_mean = np.mean(seq_lengths)
    length_std = np.std(seq_lengths)
    
    # Completion quality analysis
    completion_lengths = [len(comp) for comp in completions]
    comp_length_mean = np.mean(completion_lengths) if completion_lengths else 0
    
    # Zero reward detection (indicates issues)
    zero_rewards = (rewards == 0).sum().item()
    negative_rewards = (rewards < 0).sum().item()
    
    # Reward quantization analysis
    reward_values = rewards.cpu().tolist() if torch.is_tensor(rewards) else rewards
    unique_rewards = len(set([round(r, 3) for r in reward_values]))
    
    # Log detailed analysis
    logger.info(f"=== BATCH GENERATION QUALITY ANALYSIS (Step {step}) ===")
    logger.info(f"  Reward Stats: μ={reward_mean:.3f}, σ={reward_std:.3f}, min={reward_min:.3f}, max={reward_max:.3f}")
    logger.info(f"  Reward Granularity: {unique_rewards} unique values from {len(reward_values)} samples")
    logger.info(f"  Sequence Lengths: μ={length_mean:.1f}, σ={length_std:.1f}")
    logger.info(f"  Completion Lengths: μ={comp_length_mean:.1f}")
    logger.info(f"  Quality Issues: {zero_rewards}/{len(rewards)} zero rewards, {negative_rewards}/{len(rewards)} negative rewards")
    
    # Quality warnings
    if unique_rewards < 10 and len(reward_values) > 20:
        logger.warning(f"⚠️ HIGHLY QUANTIZED REWARDS: Only {unique_rewards} unique values - reward system may be collapsing")
    
    if zero_rewards > len(rewards) * 0.2:  # More than 20% zero rewards
        logger.warning(f"⚠️ HIGH ZERO REWARD RATE: {zero_rewards/len(rewards)*100:.1f}% - possible generation issues")
    
    if reward_std == 0.0:
        logger.warning(f"⚠️ ZERO REWARD VARIANCE: All rewards identical - possible advantage computation issue")
    
    if negative_rewards > len(rewards) * 0.5:  # More than 50% negative rewards
        logger.warning(f"⚠️ HIGH NEGATIVE REWARD RATE: {negative_rewards/len(rewards)*100:.1f}% - model may be degrading")


def run_ab_comparison(model, samples, tokenizer, config, logger, step):
    """
    Run A/B test comparing individual vs batch generation on same samples.
    """
    logger.info(f"=== A/B COMPARISON TEST (Step {step}) ===")
    
    # Test with smaller sample to avoid timeout
    test_samples = samples[:2]  # Just 2 prompts for quick comparison
    
    # A: Individual generation
    logger.info("Running individual generation...")
    config_individual = config
    config_individual.use_batch_generation = False
    
    individual_start = time.time()
    individual_data = collect_rollouts(
        model, test_samples, tokenizer, config_individual, logger, step
    )
    individual_time = time.time() - individual_start
    
    # B: Batch generation
    logger.info("Running batch generation...")
    config_batch = config
    config_batch.use_batch_generation = True
    
    batch_start = time.time()
    if config.batch_generation_mode == "task_specific":
        batch_data = collect_rollouts_task_specific_batched(
            model, test_samples, tokenizer, config_batch, batch_size=config.batch_generation_size
        )
    else:
        batch_data = collect_rollouts_batched(
            model, test_samples, tokenizer, config_batch, batch_size=config.batch_generation_size
        )
    batch_time = time.time() - batch_start
    
    # Compare results
    ind_rewards = individual_data['rewards']
    batch_rewards = batch_data['rewards']
    
    ind_mean = ind_rewards.mean().item() if torch.is_tensor(ind_rewards) else np.mean(ind_rewards)
    batch_mean = batch_rewards.mean().item() if torch.is_tensor(batch_rewards) else np.mean(batch_rewards)
    
    speedup = individual_time / batch_time if batch_time > 0 else 0
    quality_delta = abs(batch_mean - ind_mean) / abs(ind_mean) if ind_mean != 0 else 0
    
    logger.info(f"  Individual: {individual_time:.2f}s, reward={ind_mean:.3f}")
    logger.info(f"  Batch: {batch_time:.2f}s, reward={batch_mean:.3f}")
    logger.info(f"  Speedup: {speedup:.2f}x, Quality Δ: {quality_delta:.1%}")
    
    if quality_delta > 0.3:  # >30% quality difference
        logger.warning(f"⚠️ SIGNIFICANT QUALITY DIFFERENCE: Batch generation may have issues")
    
    return individual_data, batch_data


class EarlyStopping:
    """
    Early stopping monitor for training health and recovery detection.
    """
    
    def __init__(self, window_size: int = 5, patience: int = 10):
        """
        Initialize early stopping monitor.
        
        Args:
            window_size: Number of recent steps to monitor for recovery
            patience: Number of steps to wait before stopping
        """
        self.window_size = window_size
        self.patience = patience
        self.unhealthy_steps = 0
        self.best_metric = float('-inf')
        self.step_history = []
        
    def check_training_health(self, metrics: dict, step: int) -> tuple:
        """
        Check training health and return (should_stop, reason).
        
        Args:
            metrics: Training metrics dict
            step: Current training step
            
        Returns:
            (should_stop: bool, reason: str)
        """
        # Extract key health indicators
        grad_norm = metrics.get('grad_norm', 0.0)
        total_loss = metrics.get('total_loss', 0.0)
        ratio_mean = metrics.get('ratio_mean', 1.0)
        clipped_frac = metrics.get('clipped_frac', 0.0)
        
        # Health criteria
        healthy = True
        issues = []
        
        # Gradient explosion
        if grad_norm > 10.0:
            healthy = False
            issues.append(f"gradient explosion ({grad_norm:.1f})")
        
        # Loss explosion
        if abs(total_loss) > 1000:
            healthy = False
            issues.append(f"loss explosion ({total_loss:.0f})")
            
        # Policy ratio explosion
        if ratio_mean > 1000:
            healthy = False
            issues.append(f"policy explosion ({ratio_mean:.0f})")
            
        # Excessive clipping
        if clipped_frac > 0.95:
            healthy = False
            issues.append(f"excessive clipping ({clipped_frac*100:.0f}%)")
        
        # Update health tracking
        if healthy:
            self.unhealthy_steps = 0
        else:
            self.unhealthy_steps += 1
            
        # Track progress metric (use negative loss for maximization)
        current_metric = -abs(total_loss) if abs(total_loss) < 1000 else -1000
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            
        # Decision logic
        should_stop = False
        reason = ""
        
        if self.unhealthy_steps >= self.patience:
            should_stop = True
            reason = f"Training unhealthy for {self.unhealthy_steps} steps: {', '.join(issues)}"
        
        return should_stop, reason
    
    def get_status(self) -> dict:
        """Get current early stopping status."""
        return {
            'unhealthy_steps': self.unhealthy_steps,
            'patience': self.patience,
            'best_metric': self.best_metric,
            'steps_until_stop': max(0, self.patience - self.unhealthy_steps)
        }


def detect_training_instability(metrics, step, logger):
    """
    Detect training instability patterns and log warnings.
    """
    # Policy ratio explosion detection
    ratio_mean = metrics.get('ratio_mean', 1.0)
    ratio_std = metrics.get('ratio_std', 0.0)
    
    if ratio_mean > 1000 or ratio_std > 1000:
        logger.error(f"🚨 POLICY RATIO EXPLOSION: mean={ratio_mean:.0f}, std={ratio_std:.0f}")
        logger.error("   This indicates severe training instability!")
        
    # Loss explosion detection
    total_loss = metrics.get('total_loss', 0.0)
    pg_loss = metrics.get('pg_loss', 0.0)
    
    if total_loss > 1000000:  # Loss > 1M indicates explosion
        logger.error(f"🚨 LOSS EXPLOSION: total={total_loss:.0f}, pg={pg_loss:.0f}")
    
    # KL divergence explosion detection
    kl_div = metrics.get('kl_div', 0.0)
    kl_forward = metrics.get('kl_forward', 0.0)
    
    if kl_div > 5.0 or kl_forward > 10.0:
        logger.error(f"🚨 KL DIVERGENCE EXPLOSION: kl_div={kl_div:.3f}, kl_forward={kl_forward:.3f}")
        logger.error("   Model policy is diverging from reference model!")
    
    # Clipping rate detection
    clipped_frac = metrics.get('clipped_frac', 0.0)
    
    if clipped_frac > 0.9:
        logger.warning(f"⚠️ HIGH CLIPPING RATE: {clipped_frac*100:.1f}% - training may be unstable")


def manage_checkpoints(checkpoint_dir: str, max_checkpoints: int, logger):
    """
    Manage rolling checkpoint retention (keep only last N checkpoints).
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        logger: Logger instance
    """
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return
            
        # Get all checkpoint files
        checkpoint_files = list(checkpoint_path.glob("checkpoint_step*.pt"))
        
        if len(checkpoint_files) <= max_checkpoints:
            return  # No cleanup needed
            
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        to_remove = len(checkpoint_files) - max_checkpoints
        removed_count = 0
        
        for checkpoint_file in checkpoint_files[:to_remove]:
            try:
                checkpoint_file.unlink()
                removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old checkpoints (keeping last {max_checkpoints})")
            
    except Exception as e:
        logger.warning(f"Checkpoint cleanup failed: {e}")


def log_training_metrics(logger, metrics, step, elapsed_time, timing_breakdown=None):
    """Log detailed training metrics with timing breakdown."""
    logger.info(f"\nStep {step} | Time: {elapsed_time:.2f}s")
    
    # Log timing breakdown if available
    if timing_breakdown:
        logger.info(f"=== STEP TIMING BREAKDOWN ===")
        total_time = elapsed_time
        logger.info(f"  Rollout: {timing_breakdown.get('rollout', 0):.2f}s ({timing_breakdown.get('rollout', 0)/total_time*100:.1f}%)")
        logger.info(f"    Generation: {timing_breakdown.get('generation', 0):.2f}s ({timing_breakdown.get('generation', 0)/total_time*100:.1f}%)")
        logger.info(f"    Reward scoring: {timing_breakdown.get('reward_scoring', 0):.2f}s ({timing_breakdown.get('reward_scoring', 0)/total_time*100:.1f}%)")
        logger.info(f"  Training computation: {timing_breakdown.get('training', 0):.2f}s ({timing_breakdown.get('training', 0)/total_time*100:.1f}%)")
        logger.info(f"    Log probs: {timing_breakdown.get('logprobs', 0):.2f}s")
        logger.info(f"    Loss computation: {timing_breakdown.get('loss_computation', 0):.2f}s")
        logger.info(f"    Backward pass: {timing_breakdown.get('backward', 0):.2f}s")
        logger.info(f"  Memory cleanup: {timing_breakdown.get('cleanup', 0):.2f}s")
    
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
    
    # Log gradient norms and learning rate
    if 'grad_norm' in metrics:
        logger.debug(f"  Gradient Norm: {metrics['grad_norm']:.4f}")
    
    if 'learning_rate' in metrics:
        logger.debug(f"  Learning Rate: {metrics['learning_rate']:.2e}")
    
    # Log training health indicators
    if 'ratio_mean' in metrics and metrics['ratio_mean'] > 100:
        logger.warning(f"  ⚠️ High Policy Ratio: {metrics['ratio_mean']:.1f}")
    if 'clipped_frac' in metrics and metrics['clipped_frac'] > 0.8:
        logger.warning(f"  ⚠️ High Clipping: {metrics['clipped_frac']*100:.0f}%")


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
    Generate K completions per prompt and compute rewards with timing profiling.
    Supports both individual and batch generation based on config.
    """
    model.eval()
    
    # Initialize timing profiler
    timing_profile = {
        'data_prep': 0.0,
        'generation': 0.0,
        'reward_scoring': 0.0,
        'batch_preparation': 0.0,
        'total_rollout': 0.0,
        'batch_generation_mode': 'individual'  # Track which mode was used
    }
    rollout_start_time = time.time()
    
    # Batch generation path - delegate to optimized functions
    if config.use_batch_generation:
        logger.info(f"Using BATCH GENERATION mode: {config.batch_generation_mode}")
        timing_profile['batch_generation_mode'] = config.batch_generation_mode
        
        generation_start = time.time()
        
        if config.batch_generation_mode == "task_specific":
            rollout_data = collect_rollouts_task_specific_batched(
                model=model,
                samples=samples,
                tokenizer=tokenizer,
                config=config,
                baseline_tracker=baseline_tracker,
                batch_size=config.batch_generation_size
            )
        else:  # "mixed" mode
            rollout_data = collect_rollouts_batched(
                model=model,
                samples=samples,
                tokenizer=tokenizer,
                config=config,
                baseline_tracker=baseline_tracker,
                batch_size=config.batch_generation_size
            )
        
        generation_time = time.time() - generation_start
        timing_profile['generation'] = generation_time
        timing_profile['total_rollout'] = generation_time
        
        # Add completions for logging
        rollout_data['completions'] = rollout_data.get('completions', [''] * len(rollout_data['rewards']))
        rollout_data['timing_profile'] = timing_profile
        
        # Log batch generation stats
        logger.info(f"Batch generation completed in {generation_time:.2f}s")
        logger.info(f"Generated {len(rollout_data['rewards'])} completions")
        logger.info(f"Mean reward: {rollout_data['rewards'].mean().item():.3f}")
        
        return rollout_data
    
    # Individual generation path (original code)
    logger.debug(f"Using INDIVIDUAL generation mode")
    timing_profile['batch_generation_mode'] = 'individual'
    
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
        
        # Data preparation timing
        data_prep_start = time.time()
        prompt_ids = tokenizer.encode(prompt, disallowed_special=())
        prompt_length = len(prompt_ids)
        timing_profile['data_prep'] += time.time() - data_prep_start
        
        # Generate K completions
        sample_rewards = []
        sample_sequences = []
        sample_masks = []
        sample_completions = []
        
        for k in range(config.k_samples):
            # Generation timing
            generation_start = time.time()
            
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
            
            timing_profile['generation'] += time.time() - generation_start
            
            # Store completion (scoring will be done in batch)
            sample_sequences.append(output_ids[0].detach().cpu())
            sample_masks.append(torch.ones_like(output_ids[0]).detach().cpu())
            sample_completions.append(completion)
        
        # Vectorized reward scoring (batch score all k_samples for this prompt)
        reward_start = time.time()
        prompt_list = [prompt] * config.k_samples
        batch_rewards, _ = scorer.score_batch(prompt_list, sample_completions, compute_advantages=False)
        timing_profile['reward_scoring'] += time.time() - reward_start
        
        # Log first completion with its reward
        if len(sample_completions) > 0:
            logger.debug(f"  K=1, Reward={batch_rewards[0]:.3f}, Completion: {sample_completions[0][:100]}...")
        
        # Store results
        sample_rewards = batch_rewards.tolist()
        
        # Store group data
        all_rewards.extend(sample_rewards)
        all_sequences.extend(sample_sequences)
        all_attention_masks.extend(sample_masks)
        all_prompt_lengths.extend([prompt_length] * config.k_samples)
        all_completions.extend(sample_completions)
    
    # Batch preparation timing
    batch_prep_start = time.time()
    
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
    
    timing_profile['batch_preparation'] += time.time() - batch_prep_start
    
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
    
    # Finalize timing profile
    timing_profile['total_rollout'] = time.time() - rollout_start_time
    
    logger.debug(f"\nAdvantages: mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}")
    
    # Log detailed timing breakdown
    total_time = timing_profile['total_rollout']
    logger.info(f"\n=== ROLLOUT TIMING BREAKDOWN (Step {step}) ===")
    logger.info(f"Total rollout time: {total_time:.3f}s")
    logger.info(f"  Data preparation: {timing_profile['data_prep']:.3f}s ({timing_profile['data_prep']/total_time*100:.1f}%)")
    logger.info(f"  Generation: {timing_profile['generation']:.3f}s ({timing_profile['generation']/total_time*100:.1f}%)")
    logger.info(f"  Reward scoring: {timing_profile['reward_scoring']:.3f}s ({timing_profile['reward_scoring']/total_time*100:.1f}%)")
    logger.info(f"  Batch preparation: {timing_profile['batch_preparation']:.3f}s ({timing_profile['batch_preparation']/total_time*100:.1f}%)")
    
    return {
        "sequences": sequences,
        "attention_masks": attention_masks,
        "rewards": all_rewards,
        "advantages": advantages,
        "prompt_lengths": all_prompt_lengths,
        "completions": all_completions,
        "timing_profile": timing_profile
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
    parser.add_argument("--use_bf16", action="store_true", help="Enable BF16 mixed precision training")
    parser.add_argument("--use_torch_compile", action="store_true", help="Enable torch.compile() optimization")
    
    # Batch generation optimization arguments
    parser.add_argument("--use_batch_generation", action="store_true", help="Enable batch generation for 3x speedup")
    parser.add_argument("--batch_generation_mode", type=str, default="mixed", choices=["mixed", "task_specific"], 
                       help="Batch generation strategy: mixed or task_specific")
    parser.add_argument("--batch_generation_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--ab_test_mode", action="store_true", help="Run A/B test comparing individual vs batch generation")
    parser.add_argument("--reward_shaping", type=str, default="graduated", choices=["graduated", "linear", "binary"], 
                       help="Reward shaping strategy: graduated, linear, or binary")
    
    # Learning rate schedule arguments
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "linear", "constant"],
                       help="Learning rate schedule: cosine, linear, or constant")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR as fraction of initial LR")
    
    # Early stopping and checkpoint management
    parser.add_argument("--early_stop_window", type=int, default=5, help="Early stop if no recovery within N steps")
    parser.add_argument("--max_checkpoints", type=int, default=5, help="Keep only last N checkpoints")
    
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
        save_freq=args.save_freq,
        use_bf16=args.use_bf16,  # Enable BF16 mixed precision
        use_torch_compile=args.use_torch_compile,  # Enable torch.compile()
        
        # Batch generation optimizations
        use_batch_generation=args.use_batch_generation,
        batch_generation_mode=args.batch_generation_mode,
        batch_generation_size=args.batch_generation_size,
        
        # Reward shaping
        reward_shaping=args.reward_shaping,
        
        # Learning rate schedule
        lr_schedule_type=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    
    # Log configuration with optimization highlights
    logger.info("\n=== CONFIGURATION ===")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    
    # Highlight performance optimizations
    logger.info("\n🚀 PERFORMANCE OPTIMIZATIONS:")
    if config.use_batch_generation:
        logger.info(f"  ⚡ Batch Generation: ENABLED ({config.batch_generation_mode} mode, batch_size={config.batch_generation_size})")
        logger.info("     Expected speedup: ~3x generation time")
    else:
        logger.info("  ⚡ Batch Generation: DISABLED")
    
    logger.info(f"  🔥 BF16 Mixed Precision: {'ENABLED' if config.use_bf16 else 'DISABLED'}")
    logger.info(f"  🏎️ Torch Compile: {'ENABLED' if config.use_torch_compile else 'DISABLED'}")
    logger.info(f"  🎯 TF32 Acceleration: {'ENABLED' if config.enable_tf32 else 'DISABLED'}")
    logger.info(f"  🧠 GAE: {'DISABLED (batch compatibility)' if config.use_batch_generation else ('ENABLED' if config.use_gae else 'DISABLED')}")
    
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
    
    # Initialize value function if using GAE (but disable if using batch generation)
    value_function = None
    effective_use_value_function = config.value_loss_coef > 0 and not config.use_batch_generation
    if effective_use_value_function:
        value_function = ValueFunction(model.config.n_embd).to(config.device)
        logger.info("Initialized value function")
    elif config.value_loss_coef > 0 and config.use_batch_generation:
        logger.info("Value function disabled due to batch generation compatibility")
    
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
    
    # Setup learning rate scheduler
    lr_scheduler = create_lr_scheduler(optimizer, config)
    if lr_scheduler is not None:
        logger.info(f"Initialized {config.lr_schedule_type} LR scheduler:")
        logger.info(f"  Warmup steps: {config.warmup_steps}")
        logger.info(f"  Initial LR: {config.learning_rate}")
        logger.info(f"  Min LR: {config.learning_rate * config.min_lr_ratio}")
        
        # Preview schedule
        schedule_preview = visualize_lr_schedule(config, steps_to_show=min(20, config.max_steps))
        logger.info(f"  LR Schedule Preview (first 20 steps):")
        for step, lr in schedule_preview[:10]:
            logger.info(f"    Step {step:2d}: {lr:.2e}")
        if len(schedule_preview) > 10:
            logger.info(f"    ... (showing first 10/{len(schedule_preview)} steps)")
    else:
        logger.info("Using constant learning rate (no scheduler)")
    
    # Initialize early stopping monitor
    early_stopping = EarlyStopping(
        window_size=args.early_stop_window,
        patience=args.early_stop_window
    )
    logger.info(f"Initialized early stopping (recovery window: {args.early_stop_window} steps)")
    
    # Setup performance optimizations
    if config.enable_tf32:
        # Enable TF32 for faster training on Ampere+ GPUs (RTX 30/40 series, A100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 acceleration for Ampere+ GPUs")
    
    if config.tensor_core_precision:
        # Optimize for Tensor Core utilization
        torch.set_float32_matmul_precision(config.tensor_core_precision)
        logger.info(f"Set Tensor Core precision to '{config.tensor_core_precision}' for maximum utilization")
    
    # Setup gradient scaler for mixed precision training
    scaler = None
    if config.use_bf16:
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Enabled BF16 mixed precision training with gradient scaling")
    
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
        
        # Enhanced monitoring for batch generation
        if config.use_batch_generation:
            analyze_batch_generation_quality(rollout_data, logger, step)
            debug_advantage_computation(rollout_data, config, logger, step)
        
        # Log rollout details
        log_rollout_details(logger, rollout_data, step)
        
        # Training step timing
        training_start_time = time.time()
        model.train()
        
        # Extract rollout timing profile
        rollout_timing = rollout_data.get("timing_profile", {})
        
        # Extract tensors and ensure proper cleanup
        sequences = rollout_data["sequences"]
        attention_masks = rollout_data["attention_masks"]
        advantages = rollout_data["advantages"]
        prompt_lengths = rollout_data["prompt_lengths"]
        
        # Log probability computation timing
        logprobs_start = time.time()
        
        # Compute log probs with chunked processing and optional BF16
        policy_log_probs = compute_log_probs(
            model,
            sequences,
            attention_masks,
            chunk_size=config.log_prob_chunk_size,  # Process in chunks to save memory
            use_bf16=config.use_bf16  # Enable BF16 mixed precision
        )
        
        # Get ref_log_probs with chunked processing (keep in FP32 for stability)
        with torch.no_grad():
            ref_log_probs_cpu = ref_model.compute_log_probs(
                sequences,
                attention_masks,
                return_on_cpu=True,  # Returns CPU tensor
                use_bf16=False  # Keep reference model in FP32 for numerical stability
            )
            ref_log_probs = ref_log_probs_cpu.to(config.device)  # Move to GPU for loss computation
        
        logprobs_time = time.time() - logprobs_start
            
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
        
        # Loss computation timing
        loss_start = time.time()
        
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
        
        loss_computation_time = time.time() - loss_start
        
        # Backward pass timing
        backward_start = time.time()
        
        # Backward pass with optional mixed precision
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            # BF16 mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        
        metrics['grad_norm'] = grad_norm.item()
        
        # Update learning rate schedule
        if lr_scheduler is not None:
            lr_scheduler.step(step)
            current_lr = lr_scheduler.get_current_lr()
            metrics['learning_rate'] = current_lr
        
        backward_time = time.time() - backward_start
        
        # Detach loss to prevent gradient accumulation
        loss = loss.detach()
        
        # Clear any remaining gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
        
        # Calculate total training time
        training_computation_time = time.time() - training_start_time
        elapsed_time = time.time() - start_time
        metrics['total_loss'] = loss.item()
        
        # Compile comprehensive timing breakdown
        step_timing_breakdown = {
            'rollout': rollout_timing.get('total_rollout', 0),
            'generation': rollout_timing.get('generation', 0),
            'reward_scoring': rollout_timing.get('reward_scoring', 0),
            'training': training_computation_time,
            'logprobs': logprobs_time,
            'loss_computation': loss_computation_time,
            'backward': backward_time,
            'cleanup': elapsed_time - rollout_timing.get('total_rollout', 0) - training_computation_time
        }
        
        # Enhanced training stability detection
        detect_training_instability(metrics, step, logger)
        
        # Check early stopping
        should_stop, stop_reason = early_stopping.check_training_health(metrics, step)
        if should_stop:
            logger.error(f"🛑 EARLY STOPPING TRIGGERED: {stop_reason}")
            logger.info(f"Training terminated at step {step}/{config.max_steps}")
            break
            
        # Log early stopping status if unhealthy
        if early_stopping.unhealthy_steps > 0:
            status = early_stopping.get_status()
            logger.warning(f"⚠️ Training unhealthy for {status['unhealthy_steps']}/{status['patience']} steps")
        
        # Log training metrics with timing breakdown
        log_training_metrics(logger, metrics, step, elapsed_time, step_timing_breakdown)
        
        # Store history and eval data before deleting variables
        rewards = rollout_data['rewards']
        mean_reward = rewards.mean().item() if torch.is_tensor(rewards) else np.mean(rewards)
        rewards_list = rewards.cpu().tolist() if torch.is_tensor(rewards) else rewards
        reward_sample = [f'{r:.1f}' for r in rewards_list[:10]]
        
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
        
        # Save checkpoint periodically with rolling management
        if step % config.save_freq == 0:
            checkpoint_path = f"{config.checkpoint_dir}/checkpoint_step{step}.pt"
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            checkpoint_data = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'recent_history': history_writer.get_recent(10),
                'early_stopping_state': early_stopping.get_status()
            }
            
            if lr_scheduler is not None:
                checkpoint_data['lr_scheduler_state'] = lr_scheduler.get_schedule_info()
            
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Manage checkpoint retention
            manage_checkpoints(config.checkpoint_dir, args.max_checkpoints, logger)
    
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