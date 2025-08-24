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
        
        with torch.set_grad_enabled(not use_ref_model):
            # Forward pass with optional mixed precision
            if self.config.use_mixed_precision and not use_ref_model:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Shift for autoregressive loss (predict next token)
            shift_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            shift_labels = input_ids[:, 1:]   # [batch_size, seq_len-1]
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
        # Current policy log probabilities
        current_logprobs = self.compute_logprobs(
            batch.input_ids,
            batch.attention_mask, 
            batch.target_start_indices,
            use_ref_model=False
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
        
        # Prepare final metrics
        final_metrics = {}
        for key, values in aggregated_metrics.items():
            final_metrics[key] = float(np.mean(values))
        
        # Add training state info
        final_metrics.update({
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step': self.step_count,
            'total_samples': self.total_samples_trained,
            'num_groups': len(step_data.groups)
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
            'config': self.config
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
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for monitoring."""
        return {
            'step_count': self.step_count,
            'total_samples_trained': self.total_samples_trained,
            'current_lr': self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.lr,
            'kl_coefficient': self.kl_controller.get_coefficient(),
            'model_device': next(self.model.parameters()).device,
            'training_mode': self.model.training
        }