"""
Mini implementation - Clean dataset processing and validation

This package implements proper handling of the RookWorld dataset with:
- Correct preprocessing (A: prefix for non-P: tasks)
- Separate parsing for P: and A: tasks
- Prioritized validation with weighted scoring
- Reward scoring and advantage computation for GRPO
"""

from .dataset import (
    preprocess_sample,
    parse_p_task,
    parse_a_task,
    load_and_prepare_samples,
    get_batch_by_type
)

from .validation import (
    validate_p_format,
    validate_a_format,
    validate_p_task,
    validate_a_task,
    P_WEIGHTS,
    A_WEIGHTS
)

from .reward_scorer import (
    RewardScorer,
    RewardDetails,
    compute_grpo_rewards
)

__all__ = [
    # Dataset functions
    'preprocess_sample',
    'parse_p_task',
    'parse_a_task',
    'load_and_prepare_samples',
    'get_batch_by_type',
    
    # Validation functions
    'validate_p_format',
    'validate_a_format',
    'validate_p_task',
    'validate_a_task',
    'P_WEIGHTS',
    'A_WEIGHTS',
    
    # Reward scoring
    'RewardScorer',
    'RewardDetails',
    'compute_grpo_rewards',
]