"""
Core GRPO algorithm for mini implementation

Minimal implementation of Group Relative Policy Optimization.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import copy


def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute log probabilities for a sequence.
    
    Args:
        model: GPT2Model instance
        input_ids: Token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Log probabilities [batch_size, seq_len-1]
    """
    # Get model outputs
    outputs = model(input_ids, attention_mask)
    logits = outputs["logits"]
    
    # Shift for autoregressive: predict next token
    logits = logits[:, :-1, :]  # [batch, seq-1, vocab]
    labels = input_ids[:, 1:]  # [batch, seq-1]
    
    # Compute log probs
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask padding if attention mask provided
    if attention_mask is not None:
        mask = attention_mask[:, 1:]  # Shift mask too
        token_log_probs = token_log_probs * mask
    
    return token_log_probs


def compute_advantages(
    rewards: torch.Tensor,
    group_size: int
) -> torch.Tensor:
    """
    Compute group-relative advantages.
    
    For each group of K samples (from same prompt),
    subtract the group mean as baseline.
    
    Args:
        rewards: Reward values [batch_size]
        group_size: Number of samples per prompt (K)
        
    Returns:
        Advantages [batch_size]
    """
    batch_size = rewards.shape[0]
    n_groups = batch_size // group_size
    
    advantages = torch.zeros_like(rewards)
    
    for i in range(n_groups):
        start = i * group_size
        end = (i + 1) * group_size
        group_rewards = rewards[start:end]
        
        # Group mean as baseline
        baseline = group_rewards.mean()
        advantages[start:end] = group_rewards - baseline
    
    # Normalize advantages
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def grpo_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    prompt_mask: torch.Tensor,
    kl_coef: float = 0.02,
    clip_range: float = 0.2
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss with KL regularization.
    
    Args:
        policy_log_probs: Current policy log probs [batch_size, seq_len]
        ref_log_probs: Reference model log probs [batch_size, seq_len]
        advantages: Advantage values [batch_size]
        prompt_mask: Mask for prompt tokens (1) vs completion (0) [batch_size, seq_len]
        kl_coef: KL penalty coefficient
        clip_range: PPO-style clipping range
        
    Returns:
        loss: Scalar loss value
        metrics: Dictionary of metrics for logging
    """
    # Only compute loss on completion tokens
    completion_mask = (1 - prompt_mask)
    
    # Sum log probs over completion tokens
    policy_completion_log_probs = (policy_log_probs * completion_mask).sum(dim=1)
    ref_completion_log_probs = (ref_log_probs * completion_mask).sum(dim=1)
    
    # Compute probability ratio
    log_ratio = policy_completion_log_probs - ref_completion_log_probs
    ratio = torch.exp(log_ratio)
    
    # PPO-style clipping
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    # Policy gradient loss (maximize advantages)
    pg_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()
    
    # KL divergence (on completion tokens only)
    kl_div = (policy_log_probs - ref_log_probs) * completion_mask
    kl_div = kl_div.sum(dim=1).mean()
    
    # Total loss
    loss = pg_loss + kl_coef * kl_div
    
    # Metrics
    metrics = {
        "pg_loss": pg_loss.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "clipped_frac": (ratio != clipped_ratio).float().mean().item()
    }
    
    return loss, metrics


class ReferenceModel:
    """
    Frozen reference model for KL regularization.
    
    Maintains a frozen copy of the initial policy.
    """
    
    def __init__(self, model):
        """
        Create frozen copy of model.
        
        Args:
            model: GPT2Model to copy
        """
        # Deep copy and freeze
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities with frozen model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities [batch_size, seq_len-1]
        """
        with torch.no_grad():
            return compute_log_probs(self.model, input_ids, attention_mask)


def create_prompt_mask(
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor
) -> torch.Tensor:
    """
    Create mask distinguishing prompt from completion tokens.
    
    Args:
        input_ids: Token IDs [batch_size, seq_len]
        prompt_lengths: Length of prompt for each sample [batch_size]
        
    Returns:
        Mask where 1=prompt, 0=completion [batch_size, seq_len-1]
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros(batch_size, seq_len - 1, device=input_ids.device)
    
    for i, length in enumerate(prompt_lengths):
        # -1 because we shift for autoregressive
        mask[i, :length-1] = 1.0
    
    return mask