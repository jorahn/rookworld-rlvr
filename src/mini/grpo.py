"""
Core GRPO algorithm for mini implementation

Minimal implementation of Group Relative Policy Optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import copy
import numpy as np


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


def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    kl_type: str = "forward"
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference model.
    
    Args:
        policy_log_probs: Current policy log probs [batch_size, seq_len]
        ref_log_probs: Reference model log probs [batch_size, seq_len]
        completion_mask: Mask for completion tokens [batch_size, seq_len]
        kl_type: Type of KL divergence ("forward", "reverse", "symmetric")
        
    Returns:
        KL divergence [batch_size] 
    """
    if kl_type == "forward":  # KL1: KL(policy || ref)
        # D_KL(P||Q) = sum(P * log(P/Q))
        kl_div = torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
        kl_div = (kl_div * completion_mask).sum(dim=1)
        
    elif kl_type == "reverse":  # KL2: KL(ref || policy)
        # D_KL(Q||P) = sum(Q * log(Q/P))
        kl_div = torch.exp(ref_log_probs) * (ref_log_probs - policy_log_probs)
        kl_div = (kl_div * completion_mask).sum(dim=1)
        
    elif kl_type == "symmetric":  # KL3: Symmetric KL = 0.5 * (KL1 + KL2)
        # Symmetric KL divergence
        kl_forward = torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs)
        kl_reverse = torch.exp(ref_log_probs) * (ref_log_probs - policy_log_probs)
        
        kl_forward = (kl_forward * completion_mask).sum(dim=1)
        kl_reverse = (kl_reverse * completion_mask).sum(dim=1)
        
        kl_div = 0.5 * (kl_forward + kl_reverse)
        
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")
    
    return kl_div


class AdaptiveKLController:
    """
    Adaptive KL coefficient controller.
    
    Adjusts KL coefficient based on observed KL divergence to maintain target.
    """
    
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon
        self.kl_history = []
        
    def update(self, kl_div: float) -> float:
        """
        Update KL coefficient based on observed KL divergence.
        
        Args:
            kl_div: Current KL divergence
            
        Returns:
            Updated KL coefficient
        """
        self.kl_history.append(kl_div)
        
        # Keep only recent history
        if len(self.kl_history) > self.horizon:
            self.kl_history = self.kl_history[-self.horizon:]
        
        # Compute recent average KL
        recent_kl = np.mean(self.kl_history[-100:])  # Last 100 steps
        
        # Adaptive adjustment
        if recent_kl > 2 * self.target_kl:
            # KL too high - increase penalty
            self.kl_coef *= 1.5
        elif recent_kl < 0.5 * self.target_kl:
            # KL too low - decrease penalty
            self.kl_coef *= 0.8
        
        # Clamp to reasonable range
        self.kl_coef = max(0.001, min(1.0, self.kl_coef))
        
        return self.kl_coef


class ValueFunction(nn.Module):
    """
    Simple value function for advantage estimation.
    
    Takes token embeddings and predicts scalar value.
    """
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimates.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Values [batch_size, seq_len]
        """
        return self.value_head(hidden_states).squeeze(-1)


def compute_advantages(
    rewards: torch.Tensor,
    group_size: int,
    baseline_type: str = "group_mean",
    baseline_tracker: Optional[Dict] = None,
    values: Optional[torch.Tensor] = None,
    use_gae: bool = False,
    gae_lambda: float = 0.95
) -> torch.Tensor:
    """
    Compute group-relative advantages with advanced baseline methods.
    
    Args:
        rewards: Reward values [batch_size]
        group_size: Number of samples per prompt (K)
        baseline_type: Type of baseline ("group_mean", "ema", "learned", "adaptive")
        baseline_tracker: Tracker for EMA/adaptive baselines
        values: Value function estimates for GAE [batch_size] 
        use_gae: Whether to use Generalized Advantage Estimation
        gae_lambda: GAE lambda parameter
        
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
        
        # Compute baseline using specified method
        if baseline_type == "group_mean":
            baseline = group_rewards.mean()
        elif baseline_type == "ema":
            if baseline_tracker is None:
                baseline_tracker = {"ema_baseline": 0.0, "count": 0}
            
            current_mean = group_rewards.mean().item()
            if baseline_tracker["count"] == 0:
                baseline_tracker["ema_baseline"] = current_mean
            else:
                alpha = 0.1  # EMA smoothing factor
                baseline_tracker["ema_baseline"] = (
                    alpha * current_mean + (1 - alpha) * baseline_tracker["ema_baseline"]
                )
            baseline_tracker["count"] += 1
            baseline = torch.tensor(baseline_tracker["ema_baseline"], device=rewards.device)
        elif baseline_type == "learned":
            # Use value function if provided, otherwise fall back to group mean
            if values is not None:
                baseline = values[start:end].mean()
            else:
                baseline = group_rewards.mean()
        elif baseline_type == "adaptive":
            # Adaptive baseline that adjusts based on reward variance
            group_std = group_rewards.std()
            if group_std > 0.1:  # High variance - use more conservative baseline
                baseline = group_rewards.median()  
            else:  # Low variance - use mean
                baseline = group_rewards.mean()
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")
        
        # Compute advantages
        group_advantages = group_rewards - baseline
        
        # Apply GAE if requested and values available
        if use_gae and values is not None:
            group_values = values[start:end]
            # Simplified GAE for group setting
            deltas = group_advantages
            gae_advantages = torch.zeros_like(deltas)
            gae_adv = 0
            
            # Backward pass for GAE
            for t in reversed(range(len(deltas))):
                gae_adv = deltas[t] + gae_lambda * gae_adv
                gae_advantages[t] = gae_adv
                
            advantages[start:end] = gae_advantages
        else:
            advantages[start:end] = group_advantages
    
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
    clip_range: float = 0.2,
    kl_type: str = "forward",
    values: Optional[torch.Tensor] = None,
    value_targets: Optional[torch.Tensor] = None,
    value_loss_coef: float = 0.1,
    entropy_coef: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    Enhanced GRPO loss with multiple KL types and value function.
    
    Args:
        policy_log_probs: Current policy log probs [batch_size, seq_len]
        ref_log_probs: Reference model log probs [batch_size, seq_len]
        advantages: Advantage values [batch_size]
        prompt_mask: Mask for prompt tokens (1) vs completion (0) [batch_size, seq_len]
        kl_coef: KL penalty coefficient
        clip_range: PPO-style clipping range
        kl_type: Type of KL divergence ("forward", "reverse", "symmetric")
        values: Value function estimates [batch_size]
        value_targets: Target values for value function [batch_size]
        value_loss_coef: Value function loss coefficient
        entropy_coef: Entropy regularization coefficient
        
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
    ratio = torch.exp(torch.clamp(log_ratio, -20, 20))  # Numerical stability
    
    # PPO-style clipping
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    # Policy gradient loss (maximize advantages)
    pg_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()
    
    # Enhanced KL divergence with multiple types
    kl_div = compute_kl_divergence(
        policy_log_probs, ref_log_probs, completion_mask, kl_type
    ).mean()
    
    # Start with policy gradient and KL terms
    total_loss = pg_loss + kl_coef * kl_div
    
    # Value function loss (if value function is used)
    value_loss = torch.tensor(0.0, device=policy_log_probs.device)
    if values is not None and value_targets is not None:
        value_loss = F.mse_loss(values, value_targets)
        total_loss += value_loss_coef * value_loss
    
    # Entropy regularization (encourage exploration)
    entropy_loss = torch.tensor(0.0, device=policy_log_probs.device)
    if entropy_coef > 0:
        # Approximate entropy from log probabilities
        policy_probs = torch.exp(policy_log_probs)
        entropy = -(policy_probs * policy_log_probs * completion_mask).sum(dim=1).mean()
        entropy_loss = -entropy_coef * entropy  # Negative because we want to maximize entropy
        total_loss += entropy_loss
    
    # Advanced metrics
    with torch.no_grad():
        # Compute all KL types for monitoring
        kl_forward = compute_kl_divergence(
            policy_log_probs, ref_log_probs, completion_mask, "forward"
        ).mean().item()
        kl_reverse = compute_kl_divergence(
            policy_log_probs, ref_log_probs, completion_mask, "reverse"
        ).mean().item()
        kl_symmetric = compute_kl_divergence(
            policy_log_probs, ref_log_probs, completion_mask, "symmetric"
        ).mean().item()
        
        # Ratio statistics
        ratio_mean = ratio.mean().item()
        ratio_std = ratio.std().item()
        clipped_frac = (ratio != clipped_ratio).float().mean().item()
        
        # Advantage statistics
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        
        # Check for ratio outliers (indicator of training instability)
        ratio_outliers = ((ratio < 0.5) | (ratio > 2.0)).float().mean().item()
    
    # Comprehensive metrics
    metrics = {
        "pg_loss": pg_loss.item(),
        "kl_div": kl_div.item(),
        "kl_forward": kl_forward,
        "kl_reverse": kl_reverse, 
        "kl_symmetric": kl_symmetric,
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "total_loss": total_loss.item(),
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "ratio_outliers": ratio_outliers,
        "advantage_mean": adv_mean,
        "advantage_std": adv_std,
        "clipped_frac": clipped_frac,
        "kl_coef": kl_coef
    }
    
    return total_loss, metrics


class ReferenceModel:
    """
    Enhanced frozen reference model for KL regularization.
    
    Maintains a frozen copy of the initial policy with caching for efficiency.
    """
    
    def __init__(self, model, cache_size: int = 1000):
        """
        Create frozen copy of model with optional caching.
        
        Args:
            model: GPT2Model to copy
            cache_size: Maximum number of cached computations
        """
        # Deep copy and freeze
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Cache for efficiency (optional)
        self.cache_size = cache_size
        self.cache = {}
        
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        Compute log probabilities with frozen model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            use_cache: Whether to use caching (for repeated computations)
            
        Returns:
            Log probabilities [batch_size, seq_len-1]
        """
        if use_cache:
            # Create cache key (simplified - in practice you'd use a proper hash)
            cache_key = tuple(input_ids.view(-1).tolist())
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Compute and cache
            with torch.no_grad():
                result = compute_log_probs(self.model, input_ids, attention_mask)
            
            # Manage cache size
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            return result
        else:
            with torch.no_grad():
                return compute_log_probs(self.model, input_ids, attention_mask)
    
    def clear_cache(self):
        """Clear the computation cache."""
        self.cache.clear()


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