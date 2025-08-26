"""
GRPO (Group Relative Policy Optimization) Trainer for RookWorld-LM

This module implements the core GRPO algorithm with PPO-style clipped policy gradients,
KL regularization, and group-relative baselines for stable policy learning on chess tasks.

GRPO algorithm overview:
1. Sample G responses per position (group_size = G)
2. Compute group-relative baseline (mean reward within group)
3. Use PPO-style clipped objective with advantages = rewards - baseline
4. Add KL penalty term to prevent policy from drifting too far from reference
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import time
import os

from .config import GRPOConfig


class RolloutBuffer:
    """Buffer for storing and reusing rollouts across multiple training epochs"""
    
    def __init__(self, capacity: int = 100):
        """
        Initialize rollout buffer
        
        Args:
            capacity: Maximum number of samples to store (reduced default for memory efficiency)
        """
        self.capacity = capacity
        self.data = []
        self.epoch_count = 0
        self.max_epochs = 2  # Default max epochs per rollout batch
        self.memory_cleanup_interval = 20  # Clear buffer every N steps
        
    def store_rollout(self, batch: 'GRPOBatch', ref_logprobs: torch.Tensor):
        """Store a rollout with cached reference logprobs"""
        # Ensure all tensors are properly detached and moved to CPU if needed
        detached_batch = self._detach_batch_tensors(batch)
        detached_ref_logprobs = ref_logprobs.detach().cpu()
        
        self.data.append({
            'batch': detached_batch,
            'ref_logprobs': detached_ref_logprobs,
            'epoch_count': 0
        })
        
        # Remove oldest if over capacity
        if len(self.data) > self.capacity:
            # Explicitly delete old data to free memory
            old_item = self.data.pop(0)
            self._cleanup_rollout_item(old_item)
    
    def get_epoch_iterator(self, batch_size: int, n_epochs: int = 2):
        """Get iterator for training epochs over stored rollouts"""
        if not self.data:
            return []
        
        # Filter rollouts that haven't exceeded max epochs
        available_data = [item for item in self.data if item['epoch_count'] < self.max_epochs]
        
        if not available_data:
            # All rollouts exhausted, clear buffer
            self.data.clear()
            return []
        
        # Create epoch batches
        epoch_batches = []
        for epoch in range(n_epochs):
            # Shuffle available data
            indices = torch.randperm(len(available_data))
            
            for i in range(0, len(available_data), batch_size):
                batch_indices = indices[i:i+batch_size]
                epoch_batch = [available_data[idx.item()] for idx in batch_indices]
                epoch_batches.append(epoch_batch)
        
        # Increment epoch counts
        for item in available_data:
            item['epoch_count'] += n_epochs
        
        return epoch_batches
    
    def should_collect_new_rollouts(self) -> bool:
        """Check if we need to collect new rollouts"""
        # Need new rollouts if buffer is empty or all rollouts are exhausted
        fresh_rollouts = [item for item in self.data if item['epoch_count'] < self.max_epochs]
        return len(fresh_rollouts) < self.capacity // 4  # Collect when < 25% fresh
    
    def _detach_batch_tensors(self, batch: 'GRPOBatch') -> 'GRPOBatch':
        """Detach all tensors in batch to prevent memory leaks"""
        import copy
        
        # Create a new batch with detached tensors moved to CPU to prevent GPU memory accumulation
        detached_batch = GRPOBatch(
            input_ids=batch.input_ids.detach().cpu() if batch.input_ids is not None else None,
            attention_mask=batch.attention_mask.detach().cpu() if batch.attention_mask is not None else None,
            target_start_indices=batch.target_start_indices.detach().cpu() if batch.target_start_indices is not None else None,
            old_logprobs=batch.old_logprobs.detach().cpu() if batch.old_logprobs is not None else None,
            rewards=batch.rewards.detach().cpu() if batch.rewards is not None else None,
            position_fen=batch.position_fen,
            task_type=batch.task_type
        )
                
        return detached_batch
    
    def _cleanup_rollout_item(self, item):
        """Cleanup memory from a rollout item"""
        try:
            # Delete tensors explicitly
            if 'batch' in item:
                del item['batch']
            if 'ref_logprobs' in item:
                del item['ref_logprobs']
        except Exception:
            pass  # Ignore cleanup errors
    
    def clear_buffer(self):
        """Clear entire buffer and free memory"""
        for item in self.data:
            self._cleanup_rollout_item(item)
        self.data.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def clear(self):
        """Clear all stored rollouts (legacy method)"""
        self.clear_buffer()
    
    def __len__(self) -> int:
        """Number of stored rollouts"""
        return len(self.data)


@dataclass
class GRPOBatch:
    """Batch of GRPO training data for a single group."""
    
    input_ids: torch.Tensor          # [group_size, seq_len]
    attention_mask: torch.Tensor     # [group_size, seq_len]
    target_start_indices: torch.Tensor  # [group_size] - where target generation starts
    old_logprobs: torch.Tensor       # [group_size] - logprobs from sampling policy
    rewards: torch.Tensor            # [group_size] - task-specific rewards
    
    # Metadata for debugging and logging
    position_fen: str                # Chess position this group is for
    task_type: str                   # 'policy' or 'environment'


@dataclass
class GRPOTrainingStep:
    """Complete training step with multiple groups."""
    
    groups: List[GRPOBatch]          # List of GRPO groups
    
    def __len__(self) -> int:
        """Number of groups in this training step."""
        return len(self.groups)
    
    def get_total_samples(self) -> int:
        """Total number of samples across all groups."""
        return sum(len(group.rewards) for group in self.groups)


class AdaptiveKLController:
    """Adaptive KL coefficient controller for stable policy updates.
    
    Dynamically adjusts KL penalty based on observed KL divergence to maintain
    target level of policy drift from reference model.
    """
    
    def __init__(self, init_kl_coef: float, target_kl: Optional[float] = None):
        """
        Initialize adaptive KL controller
        
        Args:
            init_kl_coef: Initial KL coefficient value
            target_kl: Target KL divergence. If None, uses fixed coefficient.
        """
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.adaptation_rate = 0.1  # How aggressively to adapt
    
    def update(self, observed_kl: float):
        """Update KL coefficient based on observed KL divergence.
        
        Args:
            observed_kl: Average KL divergence from latest training step
        """
        if self.target_kl is None:
            return  # Fixed KL coefficient
        
        # Proportional control: increase coef if KL too high, decrease if too low
        error_ratio = observed_kl / self.target_kl
        
        if error_ratio > 1.5:
            # KL too high - increase penalty
            self.kl_coef *= (1.0 + self.adaptation_rate)
        elif error_ratio < 0.5:
            # KL too low - decrease penalty to allow more exploration
            self.kl_coef *= (1.0 - self.adaptation_rate)
        
        # Clamp to reasonable range
        self.kl_coef = max(0.001, min(1.0, self.kl_coef))
    
    def get_coefficient(self) -> float:
        """Get current KL coefficient value."""
        return self.kl_coef


class GRPOTrainer:
    """GRPO trainer implementing group-relative policy optimization.
    
    This trainer implements the core GRPO algorithm with:
    - Group-relative baselines for variance reduction
    - PPO-style clipped policy gradients for stable updates
    - KL regularization to prevent policy drift
    - Support for mixed policy/environment tasks
    - RTX 4090 / Ada Lovelace optimizations
    """
    
    def _setup_pytorch_optimizations(self):
        """Setup PyTorch optimizations for RTX 4090 / Ada Lovelace"""
        # Enable TF32 for faster matmuls on Ampere/Ada GPUs
        torch.set_float32_matmul_precision("high")
        
        # CUDA memory allocator optimization for 24GB cards
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Enable optimized CUDA kernels
        torch.backends.cudnn.benchmark = True
    
    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing with best practices"""
        try:
            # Use the recommended non-reentrant version for DDP compatibility
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                # Fallback for models without the method
                self.model.gradient_checkpointing = True
        except TypeError:
            # Fallback if use_reentrant parameter not supported
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing = True
            logging.getLogger(__name__).warning(
                "Gradient checkpointing enabled without use_reentrant=False. "
                "Consider upgrading PyTorch for better DDP compatibility."
            )
    
    def __init__(self, 
                 model: nn.Module,
                 ref_model: nn.Module, 
                 config: GRPOConfig):
        """
        Initialize GRPO trainer with RTX 4090 optimizations
        
        Args:
            model: Policy model to train (RookWorld-LM)
            ref_model: Frozen reference model for KL penalty
            config: GRPO training configuration
        """
        # RTX 4090 / Ada Lovelace optimizations
        self._setup_pytorch_optimizations()
        
        self.model = model
        self.ref_model = ref_model
        self.config = config
        
        # Setup gradient checkpointing if enabled
        if config.use_gradient_checkpointing:
            self._setup_gradient_checkpointing()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        self.ref_model.eval()
        
        # Optimized AdamW with modern LLM training settings
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.95),  # Modern LLM beta2 setting
            eps=1e-8,
            weight_decay=config.weight_decay,
            foreach=True  # Faster grouped operations
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.steps,
            eta_min=config.lr * 0.1
        )
        
        # Adaptive KL controller
        self.kl_controller = AdaptiveKLController(
            init_kl_coef=config.kl_coef,
            target_kl=config.kl_target
        )
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if config.use_mixed_precision else None
        
        # Training state
        self.step_count = 0
        self.total_samples_trained = 0
        self.nan_skip_count = 0
        self.consecutive_nan_count = 0
        
        # Recovery state
        self.last_stable_checkpoint = None
        self.recovery_attempt_count = 0
        
        # Metrics tracking
        self.metrics_history: List[Dict[str, float]] = []
        
        # Reward normalization tracking
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_momentum = 0.99  # Exponential moving average momentum
        
        # Rollout buffer for sample efficiency (reduced capacity for memory efficiency)
        self.rollout_buffer = RolloutBuffer(capacity=100)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized GRPO trainer with config: {config}")
        self.logger.info("Rollout buffer initialized for multi-epoch training")
        
        # Log warmup configuration
        if config.kl_warmup_steps > 0:
            self.logger.info(f"KL warmup enabled: {config.kl_warmup_steps} steps with factor {config.kl_warmup_factor}")
        if config.reward_warmup_steps > 0:
            self.logger.info(f"Reward warmup enabled: {config.reward_warmup_steps} steps for curriculum learning")
    
    def get_current_kl_coefficient(self) -> float:
        """Get current KL coefficient with warmup applied."""
        base_coef = self.kl_controller.get_coefficient()
        
        # Apply KL warmup
        if self.step_count < self.config.kl_warmup_steps:
            warmup_factor = self.config.kl_warmup_factor
            return base_coef * warmup_factor
        
        return base_coef
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards using exponential moving average of mean/std.
        
        Args:
            rewards: Raw reward tensor [batch_size]
            
        Returns:
            Normalized rewards [batch_size]
        """
        # Skip normalization during warmup to let statistics stabilize
        if self.step_count < max(self.config.kl_warmup_steps, self.config.reward_warmup_steps):
            return rewards
        
        # Update running statistics with exponential moving average
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item() + 1e-8  # Add small epsilon to prevent division by zero
        
        self.reward_mean = self.reward_momentum * self.reward_mean + (1 - self.reward_momentum) * batch_mean
        self.reward_std = self.reward_momentum * self.reward_std + (1 - self.reward_momentum) * batch_std
        
        # Normalize rewards
        normalized_rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        
        return normalized_rewards
    
    def compute_logprobs(self, 
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        target_start_indices: torch.Tensor,
                        use_ref_model: bool = False) -> torch.Tensor:
        """
        Compute token-mean log probabilities for target sequences
        
        Args:
            input_ids: Input token sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] 
            target_start_indices: Where target generation starts [batch_size]
            use_ref_model: Whether to use reference model instead of policy model
            
        Returns:
            Token-mean log probabilities [batch_size]
        """
        model = self.ref_model if use_ref_model else self.model
        
        # FIXED: Ensure consistent model state for reproducible logprob computation
        original_training_mode = model.training
        model.eval()  # Force eval mode for consistency
        
        # FIXED: Always disable gradients for logprob computation for consistency
        # This ensures identical behavior between test components and production code
        with torch.set_grad_enabled(False):
            # Move tensors to appropriate device for multi-GPU setup
            if use_ref_model:
                # Move to reference model device (might be different GPU) - non-blocking for performance
                ref_device = next(model.parameters()).device
                input_ids = input_ids.to(ref_device, non_blocking=True)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(ref_device, non_blocking=True)
            
            # Forward pass with optional mixed precision (BF16 for RTX 4090)
            if self.config.use_mixed_precision:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]
            
            # Shift for autoregressive loss (predict next token)
            shift_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            shift_labels = input_ids[:, 1:]   # [batch_size, seq_len-1]
            
            # Handle attention mask (create all-ones if None)
            if attention_mask is None:
                batch_size, seq_len = input_ids.shape
                shift_attention = torch.ones(batch_size, seq_len - 1, dtype=torch.long, device=input_ids.device)
            else:
                shift_attention = attention_mask[:, 1:]  # [batch_size, seq_len-1]
            
            # Convert to log probabilities
            log_probs = torch.log_softmax(shift_logits, dim=-1)  # [batch_size, seq_len-1, vocab_size]
            
            # Gather log probs for actual tokens
            token_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size, seq_len-1]
            
            # Create target mask (only count tokens after target_start_indices)
            batch_size, seq_len = token_log_probs.shape
            target_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
            
            for i in range(batch_size):
                start_idx = max(0, target_start_indices[i] - 1)  # -1 for shift
                target_mask[i, start_idx:] = shift_attention[i, start_idx:].bool()
            
            # Apply mask and compute mean
            masked_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)
            token_counts = target_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
            
            mean_log_probs = masked_log_probs.sum(dim=1) / token_counts
        
        # Restore original training mode
        model.train(original_training_mode)
        
        # Move result back to training device if we used reference model
        if use_ref_model:
            mean_log_probs = mean_log_probs.to(self.config.device)
            # Cleanup intermediate tensors to prevent GPU memory accumulation
            if input_ids.device != self.config.device:
                del input_ids
                if attention_mask is not None:
                    del attention_mask
                # Force cleanup of GPU cache after multi-GPU operations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return mean_log_probs
    
    def _compute_logprobs_with_gradients(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_start_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logprobs with gradients enabled for training
        
        This is used during loss computation when we need gradients for backprop.
        For consistent evaluation, use compute_logprobs() instead.
        """
        model = self.model  # Always use policy model for gradient computation
        
        # Forward pass with gradients enabled
        if self.config.use_mixed_precision:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]
        
        # Shift for autoregressive loss (predict next token)
        shift_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        shift_labels = input_ids[:, 1:]   # [batch_size, seq_len-1]
        
        # Handle attention mask (create all-ones if None)
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            shift_attention = torch.ones(batch_size, seq_len - 1, dtype=torch.long, device=input_ids.device)
        else:
            shift_attention = attention_mask[:, 1:]  # [batch_size, seq_len-1]
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # [batch_size, seq_len-1, vocab_size]
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Create target mask (only count tokens after target_start_indices)
        batch_size, seq_len = token_log_probs.shape
        target_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        
        for i in range(batch_size):
            start_idx = max(0, target_start_indices[i] - 1)  # -1 for shift
            target_mask[i, start_idx:] = shift_attention[i, start_idx:].bool()
        
        # Apply mask and compute mean
        masked_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)
        token_counts = target_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        
        mean_log_probs = masked_log_probs.sum(dim=1) / token_counts
        
        return mean_log_probs
    
    def compute_grpo_loss(self, batch: GRPOBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss for a single group
        
        Args:
            batch: GRPO batch data for one group
            
        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        # Current policy log probabilities - ENABLE GRADIENTS FOR TRAINING
        # Temporarily enable gradients for backward pass computation
        self.model.train()  # Ensure training mode for gradient computation
        current_logprobs = self._compute_logprobs_with_gradients(
            batch.input_ids,
            batch.attention_mask, 
            batch.target_start_indices
        )
        
        # Reference policy log probabilities (for KL penalty)
        with torch.no_grad():
            ref_logprobs = self.compute_logprobs(
                batch.input_ids,
                batch.attention_mask,
                batch.target_start_indices, 
                use_ref_model=True
            )
        
        # Apply reward normalization for stability
        normalized_rewards = self.normalize_rewards(batch.rewards)
        
        # Group-relative baseline (key innovation of GRPO)
        baseline = normalized_rewards.mean()
        advantages = normalized_rewards - baseline
        
        # PPO-style clipped objective
        logprob_ratio = torch.exp(current_logprobs - batch.old_logprobs)
        
        # Clipped surrogate objective
        unclipped_objective = logprob_ratio * advantages
        clipped_ratio = torch.clamp(
            logprob_ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range
        )
        clipped_objective = clipped_ratio * advantages
        
        # Take minimum (conservative estimate)
        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
        
        # Token-level KL divergence penalty (per GRPO best practices)
        kl_div, kl_per_token = self._compute_token_level_kl(
            current_logprobs, ref_logprobs, batch
        )
        kl_loss = self.get_current_kl_coefficient() * kl_div
        
        # Total loss
        total_loss = policy_loss + kl_loss
        
        # Enhanced metrics for comprehensive logging per best practices
        metrics = self._compute_enhanced_metrics(
            policy_loss, kl_loss, total_loss, kl_div,
            batch, advantages, logprob_ratio, current_logprobs, ref_logprobs,
            kl_per_token, normalized_rewards
        )
        
        return total_loss, metrics
    
    def _compute_token_level_kl(self, current_logprobs, ref_logprobs, batch):
        """Compute token-level KL divergence with proper masking and estimator options"""
        
        # For now, we'll use the sequence-level logprobs and extend to token-level
        # This is a simplified implementation - full token-level requires modifying logprob computation
        
        # Compute basic KL divergence per sample
        kl_per_sample = current_logprobs - ref_logprobs
        
        # Apply KL estimator (kl1, kl2, kl3 variants from best practices)
        if self.config.kl_estimator == "kl1":
            # Simple difference (what we had before)
            kl_values = kl_per_sample
        elif self.config.kl_estimator == "kl2":
            # Exponential-based estimator: exp(kl) - 1 - kl
            kl_values = torch.exp(kl_per_sample) - 1 - kl_per_sample
        else:  # kl3 (default)
            # Quadratic approximation: 0.5 * kl^2
            kl_values = 0.5 * kl_per_sample ** 2
        
        # Average across samples in the batch
        mean_kl = kl_values.mean()
        
        return mean_kl, kl_values
    
    def _compute_enhanced_metrics(self, policy_loss, kl_loss, total_loss, kl_div,
                                 batch, advantages, logprob_ratio, current_logprobs, ref_logprobs,
                                 kl_per_token=None, normalized_rewards=None):
        """Compute enhanced metrics per GRPO best practices"""
        
        # Basic metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(), 
            'kl_div': kl_div.item(),
            'kl_coef': self.get_current_kl_coefficient(),
            'total_loss': total_loss.item(),
            'baseline': batch.rewards.mean().item(),
            'mean_reward': batch.rewards.mean().item(),
            'std_reward': batch.rewards.std().item(),
            'mean_advantage': advantages.mean().item(),
            'mean_logprob_ratio': logprob_ratio.mean().item(),
        }
        
        # Enhanced KL monitoring using token-level information
        if kl_per_token is not None and len(kl_per_token) > 1:
            # Convert to float for quantile calculation
            kl_float = kl_per_token.float()
            metrics['kl_div_95pct'] = torch.quantile(kl_float, 0.95).item()
            metrics['kl_div_5pct'] = torch.quantile(kl_float, 0.05).item()
            metrics['kl_estimator'] = self.config.kl_estimator
        else:
            # Fallback to simple difference-based percentiles
            kl_diffs = current_logprobs - ref_logprobs
            if len(kl_diffs) > 1:
                metrics['kl_div_95pct'] = torch.quantile(kl_diffs, 0.95).item()
                metrics['kl_div_5pct'] = torch.quantile(kl_diffs, 0.05).item()
            else:
                metrics['kl_div_95pct'] = metrics['kl_div']
                metrics['kl_div_5pct'] = metrics['kl_div']
        
        # Clipping analysis
        clip_mask = torch.abs(logprob_ratio - 1.0) > self.config.clip_range
        metrics['fraction_clipped'] = clip_mask.float().mean().item()
        
        # Reward distribution analysis (raw rewards)
        rewards_cpu = batch.rewards.cpu().numpy()
        metrics['reward_min'] = rewards_cpu.min()
        metrics['reward_max'] = rewards_cpu.max()
        metrics['reward_25pct'] = float(np.percentile(rewards_cpu, 25))
        metrics['reward_75pct'] = float(np.percentile(rewards_cpu, 75))
        
        # Normalized reward statistics (if available)
        if normalized_rewards is not None:
            norm_rewards_cpu = normalized_rewards.cpu().numpy()
            metrics['norm_reward_mean'] = norm_rewards_cpu.mean()
            metrics['norm_reward_std'] = norm_rewards_cpu.std()
            metrics['reward_mean_ema'] = self.reward_mean
            metrics['reward_std_ema'] = self.reward_std
        
        # Task-specific metrics if we can identify task type
        if hasattr(batch, 'task_type'):
            metrics['task_type'] = batch.task_type
        
        # Advantage distribution
        advantages_cpu = advantages.cpu().numpy()
        metrics['advantage_min'] = advantages_cpu.min()
        metrics['advantage_max'] = advantages_cpu.max()
        metrics['fraction_positive_advantages'] = (advantages > 0).float().mean().item()
        
        # Policy ratio analysis for stability monitoring
        metrics['logprob_ratio_min'] = logprob_ratio.min().item()
        metrics['logprob_ratio_max'] = logprob_ratio.max().item()
        
        # Entropy estimation (approximate from logprobs)
        # Note: This is a rough approximation - true entropy would need full distribution
        with torch.no_grad():
            approx_entropy = -current_logprobs.mean().item()
            metrics['approx_entropy'] = approx_entropy
        
        return metrics
    
    def _log_training_health(self, aggregated_metrics, mean_kl_div, kl_95pct):
        """Log comprehensive training health metrics per GRPO best practices"""
        
        # Calculate aggregate statistics across all groups
        clip_fraction = np.mean(aggregated_metrics.get('fraction_clipped', [0.0]))
        mean_reward = np.mean(aggregated_metrics.get('mean_reward', [0.0]))
        entropy = np.mean(aggregated_metrics.get('approx_entropy', [0.0]))
        
        # Clipping analysis
        if clip_fraction > 0.5:
            self.logger.warning(f"High clipping rate: {clip_fraction:.2%} of samples clipped")
        elif clip_fraction > 0.3:
            self.logger.info(f"Moderate clipping: {clip_fraction:.2%} of samples clipped")
        
        # Entropy monitoring for exploration
        if entropy < 0.1:
            self.logger.warning(f"Low entropy detected: {entropy:.4f} - policy may be collapsing")
        
        # Task-specific reward analysis (for our chess domain)
        task_rewards = {}
        for key, values in aggregated_metrics.items():
            if 'task_type' in key:
                continue
            if 'reward' in key:
                task_rewards[key] = np.mean(values)
        
        # Log task distribution if available
        if 'task_type' in aggregated_metrics:
            task_types = aggregated_metrics['task_type']
            if isinstance(task_types, list):
                task_counts = {}
                for task in task_types:
                    task_counts[task] = task_counts.get(task, 0) + 1
                self.logger.debug(f"Task distribution: {task_counts}")
        
        # Periodic detailed health report (every 10 steps)
        if self.step_count % 10 == 0:
            reward_std = np.mean(aggregated_metrics.get('std_reward', [0.0]))
            logprob_ratio_max = np.mean(aggregated_metrics.get('logprob_ratio_max', [1.0]))
            
            self.logger.info(
                f"Health Report Step {self.step_count}: "
                f"KL={mean_kl_div:.4f} (95p={kl_95pct:.4f}), "
                f"Reward={mean_reward:.3f}±{reward_std:.3f}, "
                f"Clip={clip_fraction:.2%}, "
                f"Entropy={entropy:.4f}, "
                f"MaxRatio={logprob_ratio_max:.3f}"
            )
    
    def training_step(self, step_data: GRPOTrainingStep) -> Dict[str, float]:
        """
        Perform one GRPO training step on multiple groups
        
        Args:
            step_data: Training step with multiple GRPO groups
            
        Returns:
            Aggregated metrics for logging
        """
        self.model.train()
        
        total_loss = 0.0
        aggregated_metrics = {}
        
        # Process each group
        for group_idx, batch in enumerate(step_data.groups):
            # Move to device
            batch.input_ids = batch.input_ids.to(self.config.device)
            batch.attention_mask = batch.attention_mask.to(self.config.device)
            batch.target_start_indices = batch.target_start_indices.to(self.config.device)
            batch.old_logprobs = batch.old_logprobs.to(self.config.device)
            batch.rewards = batch.rewards.to(self.config.device)
            
            # Compute loss for this group
            loss, metrics = self.compute_grpo_loss(batch)
            total_loss += loss
            
            # Aggregate metrics
            for key, value in metrics.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value)
        
        # Average loss across groups
        if len(step_data.groups) > 0:
            total_loss = total_loss / len(step_data.groups)
        
        # Backward pass with gradient accumulation and mixed precision
        if self.config.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.config.gradient_accumulation_steps
        
        # NaN/Inf guard: Check loss before backward pass
        if not torch.isfinite(total_loss):
            self.logger.warning(f"NaN/Inf detected in loss at step {self.step_count}, skipping update")
            self.optimizer.zero_grad()
            self.nan_skip_count += 1
            self.consecutive_nan_count += 1
            
            # Initialize final_metrics for NaN case
            final_metrics = {
                'loss': float('nan'),
                'nan_skip': 1,
                'nan_skip_count': self.nan_skip_count,
                'consecutive_nan_count': self.consecutive_nan_count
            }
            
            # Attempt recovery if enabled and we have a stable checkpoint
            if self.consecutive_nan_count >= 10:
                if (self.config.enable_recovery and 
                    self.last_stable_checkpoint and 
                    self.recovery_attempt_count < 3):  # Max 3 recovery attempts
                    
                    self.logger.warning(f"Training instability detected (step {self.step_count}), attempting recovery...")
                    
                    # Save debug checkpoint before recovery
                    debug_checkpoint_path = f"debug_nan_step{self.step_count}_attempt{self.recovery_attempt_count + 1}"
                    try:
                        self._save_debug_checkpoint(debug_checkpoint_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to save debug checkpoint: {e}")
                    
                    # Attempt recovery
                    recovery_success = self._attempt_recovery()
                    
                    if recovery_success:
                        self.logger.info(f"Recovery successful from step {self.step_count}, continuing training")
                        return final_metrics  # Continue training from recovered state
                    else:
                        self.recovery_attempt_count += 1
                        if self.recovery_attempt_count >= 3:
                            self.logger.error("Maximum recovery attempts reached, stopping training")
                        else:
                            self.logger.warning(f"Recovery attempt {self.recovery_attempt_count} failed, will retry")
                            return final_metrics  # Try again next step
                
                # If recovery disabled or failed, stop training
                self.logger.error(f"Too many consecutive NaN losses ({self.consecutive_nan_count}), stopping training")
                raise RuntimeError("Training diverged: too many consecutive NaN losses")
            
            return final_metrics
        
        # Reset consecutive NaN count on successful loss
        self.consecutive_nan_count = 0
        self.recovery_attempt_count = 0  # Reset recovery attempts on successful step
        
        # Backward pass with gradient scaling if using mixed precision
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Explicit gradient cleanup between accumulation steps to prevent memory buildup
        if (self.step_count + 1) % self.config.gradient_accumulation_steps != 0:
            # Clear intermediate gradients that won't be used until accumulation is complete
            for param in self.model.parameters():
                if param.grad is not None:
                    # Keep gradients but cleanup any unnecessary references
                    param.grad.detach_()
        
        # Additional cleanup for multi-GPU setups - clear any residual tensors
        if torch.cuda.device_count() > 1 and self.step_count % 5 == 0:
            torch.cuda.empty_cache()
        
        # Optimizer step (every gradient_accumulation_steps)
        if (self.step_count + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping on unscaled gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
                
                # Step with gradient scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Update KL controller
        mean_kl_div = np.mean(aggregated_metrics.get('kl_div', [0.0]))
        self.kl_controller.update(mean_kl_div)
        
        # Enhanced KL monitoring per best practices (track tail as well as mean)
        kl_95pct = np.mean(aggregated_metrics.get('kl_div_95pct', [mean_kl_div]))
        kl_tail_heavy = kl_95pct > 2 * abs(mean_kl_div) if mean_kl_div != 0 else False
        
        # Log enhanced KL statistics
        if kl_tail_heavy:
            self.logger.warning(f"KL tail heavy: mean={mean_kl_div:.4f}, 95pct={kl_95pct:.4f}")
        
        # KL divergence monitoring and early stopping with configurable threshold
        if abs(mean_kl_div) > self.config.kl_divergence_threshold or kl_95pct > self.config.kl_divergence_threshold * 2:
            self.logger.error(f"Extreme KL divergence detected: mean={mean_kl_div:.3f}, 95pct={kl_95pct:.3f}")
            if self.config.enable_recovery and self.last_stable_checkpoint:
                self.logger.warning("Attempting recovery from extreme KL divergence...")
                recovery_success = self._attempt_recovery()
                if not recovery_success:
                    raise RuntimeError(f"Training diverged: extreme KL divergence mean={mean_kl_div:.3f}")
            else:
                raise RuntimeError(f"Training diverged: extreme KL divergence mean={mean_kl_div:.3f}")
        elif abs(mean_kl_div) > self.config.kl_divergence_threshold / 2.5 or kl_95pct > self.config.kl_divergence_threshold / 1.25:
            self.logger.warning(f"High KL divergence detected: mean={mean_kl_div:.3f}, 95pct={kl_95pct:.3f}")
        
        # Enhanced logging for training health
        self._log_training_health(aggregated_metrics, mean_kl_div, kl_95pct)
        
        # Prepare final metrics
        final_metrics = {}
        for key, values in aggregated_metrics.items():
            # Check if values are numeric before averaging
            if values and all(isinstance(v, (int, float)) for v in values):
                final_metrics[key] = float(np.mean(values))
            elif len(values) == 1:
                # Single value, use as-is
                final_metrics[key] = values[0]
            else:
                # Multiple non-numeric values, use the most common one
                final_metrics[key] = max(set(values), key=values.count)
        
        # Add training state info
        final_metrics.update({
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step': self.step_count,
            'total_samples': self.total_samples_trained,
            'num_groups': len(step_data.groups),
            'nan_skip': 0,  # Default to 0 if no NaN detected
            'nan_skip_count': self.nan_skip_count,
            'consecutive_nan_count': self.consecutive_nan_count
        })
        
        # Update training state
        self.step_count += 1
        self.total_samples_trained += step_data.get_total_samples()
        self.metrics_history.append(final_metrics)
        
        return final_metrics
    
    def store_rollout_for_epochs(self, batch: GRPOBatch, ref_logprobs: torch.Tensor):
        """Store a rollout in buffer for multi-epoch training"""
        self.rollout_buffer.store_rollout(batch, ref_logprobs.detach().clone())
        
        # More frequent memory cleanup to prevent accumulation
        if self.step_count % 5 == 0:  # Every 5 steps instead of 20
            self._cleanup_memory()
        
    def training_step_with_rollout_epochs(self, step_data: GRPOTrainingStep) -> Dict[str, float]:
        """
        Enhanced training step that can use rollout epochs for efficiency
        
        This method implements the GRPO best practice of:
        1. Collect rollouts with cached ref logprobs
        2. Train for 1-2 epochs over the rollouts
        3. Refresh rollouts when exhausted
        """
        # Store new rollouts in buffer
        for batch in step_data.groups:
            # Compute and cache reference logprobs
            with torch.no_grad():
                ref_logprobs = self.compute_logprobs(
                    batch.input_ids.to(self.config.device),
                    batch.attention_mask.to(self.config.device),
                    batch.target_start_indices.to(self.config.device),
                    use_ref_model=True
                )
                self.store_rollout_for_epochs(batch, ref_logprobs)
        
        # If we have enough stored rollouts, train on them for multiple epochs
        if not self.rollout_buffer.should_collect_new_rollouts():
            return self._train_on_buffered_rollouts()
        else:
            # Otherwise, use standard single-step training
            return self.training_step(step_data)
    
    def _train_on_buffered_rollouts(self) -> Dict[str, float]:
        """Train on buffered rollouts for multiple epochs"""
        if len(self.rollout_buffer) == 0:
            return {}
            
        # Get epoch iterator (2 epochs by default)
        epoch_batches = self.rollout_buffer.get_epoch_iterator(
            batch_size=self.config.batch_positions, 
            n_epochs=2
        )
        
        if not epoch_batches:
            return {}
        
        self.logger.debug(f"Training on {len(epoch_batches)} epoch batches from rollout buffer")
        
        aggregated_metrics = {}
        total_loss = 0.0
        
        # Train on each epoch batch
        for epoch_batch in epoch_batches:
            # Convert stored rollouts back to training format
            batch_groups = []
            
            for stored_item in epoch_batch:
                batch = stored_item['batch']
                # Move batch to device
                batch.input_ids = batch.input_ids.to(self.config.device)
                batch.attention_mask = batch.attention_mask.to(self.config.device)
                batch.target_start_indices = batch.target_start_indices.to(self.config.device)
                batch.old_logprobs = batch.old_logprobs.to(self.config.device)
                batch.rewards = batch.rewards.to(self.config.device)
                
                # Compute current policy logprobs (with gradients)
                self.model.train()
                current_logprobs = self._compute_logprobs_with_gradients(
                    batch.input_ids,
                    batch.attention_mask, 
                    batch.target_start_indices
                )
                
                # Use cached reference logprobs
                ref_logprobs = stored_item['ref_logprobs'].to(self.config.device)
                
                # Compute loss for this group
                loss, metrics = self._compute_loss_with_cached_ref(
                    batch, current_logprobs, ref_logprobs
                )
                total_loss += loss
                
                # Aggregate metrics
                for key, value in metrics.items():
                    if key not in aggregated_metrics:
                        aggregated_metrics[key] = []
                    aggregated_metrics[key].append(value)
        
        # Standard training step completion
        if len(epoch_batches) > 0:
            total_loss = total_loss / len(epoch_batches)
            
            # Backward pass and optimizer step
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        # Prepare final metrics
        final_metrics = {}
        for key, values in aggregated_metrics.items():
            # Check if values are numeric before averaging
            if values and all(isinstance(v, (int, float)) for v in values):
                final_metrics[key] = float(np.mean(values))
            elif len(values) == 1:
                # Single value, use as-is
                final_metrics[key] = values[0]
            else:
                # Multiple non-numeric values, use the most common one
                final_metrics[key] = max(set(values), key=values.count)
        
        final_metrics.update({
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step': self.step_count,
            'rollout_buffer_size': len(self.rollout_buffer),
            'used_rollout_epochs': True
        })
        
        self.step_count += 1
        self.metrics_history.append(final_metrics)
        
        return final_metrics
    
    def _compute_loss_with_cached_ref(self, batch, current_logprobs, ref_logprobs):
        """Compute GRPO loss using cached reference logprobs"""
        
        # Apply reward normalization for stability
        normalized_rewards = self.normalize_rewards(batch.rewards)
        
        # Group-relative baseline
        baseline = normalized_rewards.mean()
        advantages = normalized_rewards - baseline
        
        # PPO-style clipped objective
        logprob_ratio = torch.exp(current_logprobs - batch.old_logprobs)
        
        unclipped_objective = logprob_ratio * advantages
        clipped_ratio = torch.clamp(
            logprob_ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range
        )
        clipped_objective = clipped_ratio * advantages
        
        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
        
        # Token-level KL divergence penalty
        kl_div, kl_per_token = self._compute_token_level_kl(
            current_logprobs, ref_logprobs, batch
        )
        kl_loss = self.get_current_kl_coefficient() * kl_div
        
        total_loss = policy_loss + kl_loss
        
        # Basic metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item(),
            'mean_reward': batch.rewards.mean().item(),
            'baseline': baseline.item()
        }
        
        return total_loss, metrics
    
    def get_metrics_summary(self, last_n_steps: int = 10) -> Dict[str, float]:
        """
        Get summary statistics for recent training steps
        
        Args:
            last_n_steps: Number of recent steps to summarize
            
        Returns:
            Summary metrics
        """
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-last_n_steps:]
        
        summary = {}
        for key in recent_metrics[0].keys():
            values = [m[key] for m in recent_metrics if key in m]
            if values:
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
        
        return summary
    
    def save_checkpoint(self, path: str, include_optimizer: bool = True):
        """
        Save training checkpoint
        
        Args:
            path: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'step_count': self.step_count,
            'total_samples_trained': self.total_samples_trained,
            'kl_coef': self.get_current_kl_coefficient(),
            'metrics_history': self.metrics_history,
            'config': self.config,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'nan_skip_count': self.nan_skip_count,
            'consecutive_nan_count': self.consecutive_nan_count
        }
        
        if include_optimizer:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            })
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        Load training checkpoint
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step_count = checkpoint['step_count']
        self.total_samples_trained = checkpoint['total_samples_trained']
        self.kl_controller.kl_coef = checkpoint['kl_coef']
        self.metrics_history = checkpoint.get('metrics_history', [])
        self.nan_skip_count = checkpoint.get('nan_skip_count', 0)
        self.reward_mean = checkpoint.get('reward_mean', 0.0)
        self.reward_std = checkpoint.get('reward_std', 1.0)
        self.consecutive_nan_count = checkpoint.get('consecutive_nan_count', 0)
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def set_last_stable_checkpoint(self, checkpoint_path: str):
        """Set the path to the last stable checkpoint for recovery."""
        self.last_stable_checkpoint = checkpoint_path
        self.logger.debug(f"Updated last stable checkpoint: {checkpoint_path}")
    
    def _save_debug_checkpoint(self, debug_name: str):
        """Save debug checkpoint for troubleshooting."""
        debug_path = f"/tmp/grpo_debug_{debug_name}.pt"
        
        debug_data = {
            'step_count': self.step_count,
            'consecutive_nan_count': self.consecutive_nan_count,
            'nan_skip_count': self.nan_skip_count,
            'current_lr': self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.lr,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'debug_timestamp': time.time()
        }
        
        torch.save(debug_data, debug_path)
        self.logger.debug(f"Debug checkpoint saved: {debug_path}")
    
    def _attempt_recovery(self) -> bool:
        """Attempt to recover from training instability by loading last stable checkpoint."""
        if not self.last_stable_checkpoint:
            self.logger.warning("No stable checkpoint available for recovery")
            return False
        
        try:
            # Load stable checkpoint
            self.logger.info(f"Loading stable checkpoint: {self.last_stable_checkpoint}")
            self.load_checkpoint(self.last_stable_checkpoint, load_optimizer=True)
            
            # Reduce learning rate as recovery strategy
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.lr
            new_lr = current_lr * self.config.recovery_lr_factor
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # Reset NaN counters
            self.consecutive_nan_count = 0
            self.nan_skip_count = 0  # Reset for this recovery attempt
            
            self.logger.info(f"Recovery completed: reduced LR from {current_lr:.2e} to {new_lr:.2e}")
            return True
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False
    
    def _cleanup_memory(self):
        """Cleanup memory and perform garbage collection"""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                self.logger.info(f"Memory cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for monitoring."""
        state = {
            'step_count': self.step_count,
            'total_samples_trained': self.total_samples_trained,
            'current_lr': self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.lr,
            'kl_coefficient': self.get_current_kl_coefficient(),
            'model_device': next(self.model.parameters()).device,
            'training_mode': self.model.training,
            'nan_skip_count': self.nan_skip_count,
            'consecutive_nan_count': self.consecutive_nan_count
        }
        
        # Add memory information if CUDA available
        if torch.cuda.is_available():
            state.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
            })
            
        return state