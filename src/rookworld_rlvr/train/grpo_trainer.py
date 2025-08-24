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

from .config import GRPOConfig


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
    """
    
    def __init__(self, 
                 model: nn.Module,
                 ref_model: nn.Module, 
                 config: GRPOConfig):
        """
        Initialize GRPO trainer
        
        Args:
            model: Policy model to train (RookWorld-LM)
            ref_model: Frozen reference model for KL penalty
            config: GRPO training configuration
        """
        self.model = model
        self.ref_model = ref_model
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        self.ref_model.eval()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            eps=1e-8
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
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
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
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized GRPO trainer with config: {config}")
    
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
            # Forward pass with optional mixed precision (BF16 for RTX 4090)
            if self.config.use_mixed_precision and not use_ref_model:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
        
        # Group-relative baseline (key innovation of GRPO)
        baseline = batch.rewards.mean()
        advantages = batch.rewards - baseline
        
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
        
        # KL divergence penalty
        kl_div = (current_logprobs - ref_logprobs).mean()
        kl_loss = self.kl_controller.get_coefficient() * kl_div
        
        # Total loss
        total_loss = policy_loss + kl_loss
        
        # Metrics for logging
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(), 
            'kl_div': kl_div.item(),
            'kl_coef': self.kl_controller.get_coefficient(),
            'total_loss': total_loss.item(),
            'baseline': baseline.item(),
            'mean_reward': batch.rewards.mean().item(),
            'std_reward': batch.rewards.std().item(),
            'mean_advantage': advantages.mean().item(),
            'mean_logprob_ratio': logprob_ratio.mean().item(),
            'fraction_clipped': (torch.abs(logprob_ratio - 1.0) > self.config.clip_range).float().mean().item()
        }
        
        return total_loss, metrics
    
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
            
            # Add NaN skip flag to metrics
            final_metrics['nan_skip'] = 1
            final_metrics['nan_skip_count'] = self.nan_skip_count
            final_metrics['consecutive_nan_count'] = self.consecutive_nan_count
            
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
        
        # KL divergence monitoring and early stopping
        if abs(mean_kl_div) > 5.0:
            self.logger.error(f"Extreme KL divergence detected: {mean_kl_div:.3f}, indicating training instability")
            if self.config.enable_recovery and self.last_stable_checkpoint:
                self.logger.warning("Attempting recovery from extreme KL divergence...")
                recovery_success = self._attempt_recovery()
                if not recovery_success:
                    raise RuntimeError(f"Training diverged: extreme KL divergence {mean_kl_div:.3f}")
            else:
                raise RuntimeError(f"Training diverged: extreme KL divergence {mean_kl_div:.3f}")
        elif abs(mean_kl_div) > 2.0:
            self.logger.warning(f"High KL divergence detected: {mean_kl_div:.3f}, monitoring for instability")
        
        # Prepare final metrics
        final_metrics = {}
        for key, values in aggregated_metrics.items():
            final_metrics[key] = float(np.mean(values))
        
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
            'kl_coef': self.kl_controller.get_coefficient(),
            'metrics_history': self.metrics_history,
            'config': self.config,
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
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for monitoring."""
        return {
            'step_count': self.step_count,
            'total_samples_trained': self.total_samples_trained,
            'current_lr': self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.lr,
            'kl_coefficient': self.kl_controller.get_coefficient(),
            'model_device': next(self.model.parameters()).device,
            'training_mode': self.model.training,
            'nan_skip_count': self.nan_skip_count,
            'consecutive_nan_count': self.consecutive_nan_count
        }