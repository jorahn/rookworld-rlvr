"""
Learning rate schedulers for GRPO training

Implements warmup + cosine annealing + linear decay schedule
to prevent policy collapse and improve training stability.
"""

import math
from typing import Optional


class LearningRateScheduler:
    """
    Advanced learning rate scheduler for GRPO training.
    
    Implements:
    1. Warmup: Linear increase from 0 to peak LR
    2. Cosine annealing: Smooth decay from peak to min LR  
    3. Linear decay: Final linear annealing to near-zero
    """
    
    def __init__(
        self,
        optimizer,
        max_steps: int,
        warmup_steps: int = 50,
        min_lr_ratio: float = 0.1,
        schedule_type: str = "cosine",
        cosine_restart_steps: Optional[int] = None
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            max_steps: Total training steps
            warmup_steps: Number of warmup steps
            min_lr_ratio: Minimum LR as fraction of initial LR
            schedule_type: "cosine", "linear", or "constant"
            cosine_restart_steps: Optional cosine restart period
        """
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.schedule_type = schedule_type
        self.cosine_restart_steps = cosine_restart_steps
        
        # Store initial learning rates for each parameter group
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.min_lrs = [lr * min_lr_ratio for lr in self.initial_lrs]
        
        self.current_step = 0
        
    def get_lr(self, step: int) -> float:
        """
        Calculate learning rate for given step.
        
        Args:
            step: Current training step (0-indexed)
            
        Returns:
            Learning rate value
        """
        if step < self.warmup_steps:
            # Warmup: Linear increase from 0 to initial_lr
            warmup_factor = step / self.warmup_steps
            return self.initial_lrs[0] * warmup_factor
        
        if self.schedule_type == "constant":
            return self.initial_lrs[0]
        
        elif self.schedule_type == "linear":
            # Linear decay from initial_lr to min_lr
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            
            return self.initial_lrs[0] * (1.0 - progress) + self.min_lrs[0] * progress
        
        elif self.schedule_type == "cosine":
            # Cosine annealing from initial_lr to min_lr
            effective_step = step - self.warmup_steps
            effective_max_steps = self.max_steps - self.warmup_steps
            
            # Handle cosine restarts
            if self.cosine_restart_steps is not None:
                effective_step = effective_step % self.cosine_restart_steps
                effective_max_steps = self.cosine_restart_steps
            
            # Cosine annealing formula
            progress = effective_step / effective_max_steps
            progress = min(1.0, progress)
            
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            return self.min_lrs[0] + (self.initial_lrs[0] - self.min_lrs[0]) * cosine_factor
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def step(self, step: Optional[int] = None):
        """
        Update learning rate for current step.
        
        Args:
            step: Training step (if None, uses internal counter)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        current_lr = self.get_lr(self.current_step)
        
        # Update all parameter groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i < len(self.initial_lrs):
                # Scale by the ratio if multiple parameter groups have different initial LRs
                scale_factor = self.initial_lrs[i] / self.initial_lrs[0] if self.initial_lrs[0] != 0 else 1.0
                param_group['lr'] = current_lr * scale_factor
            else:
                param_group['lr'] = current_lr
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer.param_groups:
            return self.optimizer.param_groups[0]['lr']
        return 0.0
    
    def get_schedule_info(self) -> dict:
        """Get information about the learning rate schedule."""
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'warmup_steps': self.warmup_steps,
            'schedule_type': self.schedule_type,
            'current_lr': self.get_current_lr(),
            'initial_lr': self.initial_lrs[0] if self.initial_lrs else 0.0,
            'min_lr': self.min_lrs[0] if self.min_lrs else 0.0,
            'warmup_complete': self.current_step >= self.warmup_steps,
            'progress': self.current_step / self.max_steps if self.max_steps > 0 else 0.0
        }


def create_lr_scheduler(optimizer, config) -> Optional[LearningRateScheduler]:
    """
    Create learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: GRPOConfig with scheduler settings
        
    Returns:
        LearningRateScheduler instance or None if constant LR
    """
    if config.lr_schedule_type == "constant":
        return None
        
    return LearningRateScheduler(
        optimizer=optimizer,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
        schedule_type=config.lr_schedule_type,
        cosine_restart_steps=config.cosine_restart_steps
    )


def visualize_lr_schedule(config, steps_to_show: int = 100):
    """
    Generate learning rate schedule preview for validation.
    
    Args:
        config: GRPOConfig with scheduler settings
        steps_to_show: Number of steps to preview
        
    Returns:
        List of (step, lr) tuples
    """
    import torch
    import torch.optim as optim
    
    # Create dummy optimizer and scheduler
    dummy_model = torch.nn.Linear(1, 1)
    dummy_optimizer = optim.AdamW(dummy_model.parameters(), lr=config.learning_rate)
    
    scheduler = create_lr_scheduler(dummy_optimizer, config)
    if scheduler is None:
        return [(i, config.learning_rate) for i in range(steps_to_show)]
    
    schedule_preview = []
    for step in range(min(steps_to_show, config.max_steps)):
        lr = scheduler.get_lr(step)
        schedule_preview.append((step, lr))
    
    return schedule_preview