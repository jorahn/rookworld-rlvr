#!/usr/bin/env python3
"""
Direct runner for single batch training test.

This script can be executed directly to run the single batch test
without pytest, making it easier to integrate with the existing train.sh workflow.

Usage:
    python run_single_batch_test.py
    uv run python run_single_batch_test.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the test class
from tests.test_single_batch_training import TestSingleBatchTraining
from src.rookworld_rlvr.train.config import GRPOConfig


def main():
    """Run single batch training test with comprehensive logging."""
    print("🚀 Running Single Batch Training Test")
    print("=" * 50)
    
    # Configure logging for console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create test instance
    test = TestSingleBatchTraining()
    test.setup_method()
    
    # Configure minimal test settings
    config = GRPOConfig(
        # Core training settings
        steps=1,                      # Single step
        batch_positions=2,            # 2 samples
        group_size=2,                 # Group size 2
        
        # Task configuration  
        mix_env_ratio=0.5,            # 1 policy + 1 environment
        
        # Model settings (use base GPT-2 for testing)
        model_name="gpt2",
        max_new_tokens_policy=50,
        max_new_tokens_env=80,
        temperature=0.3,
        
        # Training hyperparameters
        lr=1e-6,
        kl_coef=0.001,
        clip_range=0.05,
        
        # Performance settings (disabled for testing)
        mixed_precision=False,
        torch_compile=False,
        
        # Logging
        log_level="INFO",
        verbose_logging=True
    )
    
    # Test positions 
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"  # After 1.e4
    ]
    
    print(f"Configuration:")
    print(f"  Steps: {config.steps}")
    print(f"  Batch positions: {config.batch_positions}")
    print(f"  Group size: {config.group_size}")
    print(f"  Mix env ratio: {config.mix_env_ratio}")
    print(f"  Temperature: {config.temperature}")
    print("=" * 50)
    
    # Run the test
    try:
        results = test.test_single_batch_training(config, positions)
        
        print("\n🎉 SINGLE BATCH TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"✅ Processed {len(results['samples'])} samples")
        print(f"✅ Policy samples: {len([s for s in results['samples'] if s['task_type'] == 'policy'])}")
        print(f"✅ Environment samples: {len([s for s in results['samples'] if s['task_type'] == 'environment'])}")
        
        if results['samples']:
            avg_reward = sum(s['total_reward'] for s in results['samples']) / len(results['samples'])
            print(f"✅ Average reward: {avg_reward:.4f}")
        
        if 'step_data' in results and 'policy_loss' in results['step_data']:
            print(f"✅ Final loss: {results['step_data']['policy_loss']:.6f}")
        
        print("\n📋 DETAILED SAMPLE LOG:")
        print("=" * 50)
        for i, sample in enumerate(results['samples']):
            print(f"\n--- SAMPLE {i+1} ({sample['task_type'].upper()}) ---")
            print(f"Prompt: {sample['prompt'][:100]}..." if len(sample['prompt']) > 100 else f"Prompt: {sample['prompt']}")
            print(f"Completion: {sample['completion'][:100]}..." if len(sample['completion']) > 100 else f"Completion: {sample['completion']}")
            print(f"Format Valid: {sample['format_validation_passed']}")
            print(f"Total Reward: {sample['total_reward']:.4f}")
            if sample['reward_components']:
                print("Reward Components:")
                for component, value in sample['reward_components'].items():
                    print(f"  {component}: {value:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("This likely means some dependencies are missing.")
        print("Try running: uv sync")
        return False
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        print("Check the logs above for detailed error information.")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)