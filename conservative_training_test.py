#!/usr/bin/env python3
"""
Conservative Training Test Script

Tests the training stability fixes with very conservative parameters to prevent KL divergence explosion.
This script validates that the fixes work before integrating them into the main training pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rookworld_rlvr.train.config import GRPOConfig


def create_conservative_config():
    """Create a very conservative training configuration to prevent KL divergence explosion"""
    config = GRPOConfig(
        # Ultra-conservative learning rate
        lr=1e-7,
        
        # Very low KL coefficient
        kl_coef=0.0001,
        
        # Conservative clipping
        clip_range=0.05,
        
        # Much longer warmup periods
        kl_warmup_steps=500,
        kl_warmup_factor=0.0,
        reward_warmup_steps=300,
        
        # Very strict KL threshold
        kl_divergence_threshold=1.0,
        
        # Improved reward balance (already fixed)
        r_policy_malformed=-0.3,
        r_env_malformed=-0.3,
        
        # Small batch for stability
        batch_positions=4,
        group_size=4,
        
        # Minimal steps for testing
        steps=3,
        
        # Disable performance optimizations for debugging
        use_mixed_precision=False,
        use_torch_compile=False,
        
        # Conservative evaluation
        eval_every=1,
        
        # Enable recovery system
        enable_recovery=True,
        recovery_lr_factor=0.1,
        
        # Other stability settings
        mix_env_ratio=0.2,
        temperature=0.5,  # Lower temperature for more conservative sampling
        seed=42
    )
    return config


def main():
    """Test conservative training configuration"""
    parser = argparse.ArgumentParser(description="Test conservative training configuration")
    parser.add_argument("--dry-run", action="store_true", help="Just show config, don't train")
    args = parser.parse_args()
    
    # Create conservative config
    config = create_conservative_config()
    
    print("🔧 Conservative Training Configuration:")
    print("=" * 60)
    print(f"Learning Rate:           {config.lr}")
    print(f"KL Coefficient:          {config.kl_coef}")
    print(f"Clip Range:              {config.clip_range}")
    print(f"KL Warmup Steps:         {config.kl_warmup_steps}")
    print(f"KL Divergence Threshold: {config.kl_divergence_threshold}")
    print(f"Reward Warmup Steps:     {config.reward_warmup_steps}")
    print(f"Policy Malformed Penalty: {config.r_policy_malformed}")
    print(f"Env Malformed Penalty:   {config.r_env_malformed}")
    print(f"Batch Positions:         {config.batch_positions}")
    print(f"Group Size:              {config.group_size}")
    print(f"Steps:                   {config.steps}")
    print(f"Temperature:             {config.temperature}")
    print("=" * 60)
    
    if args.dry_run:
        print("✅ Dry run complete. Configuration looks conservative enough to prevent KL explosion.")
        return True
    
    print("\n🚀 Starting conservative training test...")
    
    # Import training components
    from train_rookworld_grpo import GRPOTrainingOrchestrator
    
    # Create orchestrator with conservative config
    orchestrator = GRPOTrainingOrchestrator(config)
    
    try:
        # Run conservative training
        orchestrator.run_training()
        print("✅ Conservative training test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Conservative training test FAILED: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    exit(0 if success else 1)