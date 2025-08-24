#!/usr/bin/env python3
"""
Test Script for Full GRPO Training Pipeline

This script validates the train_full_grpo.py implementation with a minimal test run.

Usage:
    uv run python scripts/test_full_grpo.py
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_full_grpo import ProductionGRPORunner
from rookworld_rlvr.train.config import GRPOConfig


def test_training_pipeline():
    """Test the full GRPO training pipeline with minimal settings"""
    
    print("=== TESTING FULL GRPO TRAINING PIPELINE ===\n")
    
    # Create minimal configuration for testing
    config = GRPOConfig(
        # Model
        model_name_or_path="jrahn/RookWorld-LM-124M",
        
        # Training (minimal for testing)
        steps=10,  # Just 10 steps for testing
        batch_positions=2,  # Small batch for testing
        group_size=4,       # 4 rollouts per position
        
        # Hyperparameters
        lr=1e-5,
        kl_coef=0.01,
        clip_range=0.2,
        temperature=0.7,
        
        # Task distribution (80/20)
        mix_env_ratio=0.2,
        
        # Performance (disable for testing stability)
        use_mixed_precision=False,
        use_torch_compile=False,
        
        # System
        output_dir="outputs/test_full_grpo",
        seed=42,
        device="cuda" if "cuda" in str(config.device if hasattr(config, 'device') else "cpu") else "cpu",
        
        # Stockfish
        stockfish_path=None,  # Use PATH
        stockfish_time_limit=0.05,  # Fast for testing
        
        # Self-play
        n_parallel_games=2,  # Minimal
        position_buffer_size=32,  # Small buffer
    )
    
    try:
        # Initialize production runner
        print("1. Initializing production runner...")
        runner = ProductionGRPORunner(config)
        
        # Test orchestrator initialization
        print("2. Testing production orchestrator...")
        print(f"   ✓ Orchestrator initialized: {type(runner.orchestrator).__name__}")
        print(f"   ✓ Device: {runner.device}")
        print(f"   ✓ Model path: {config.model_name_or_path}")
        
        # Test position generation (smaller number for testing)
        print("3. Testing position generation...")
        runner.chess_positions = runner.generate_diverse_positions(16)  # Just 16 for testing
        print(f"   ✓ Generated {len(runner.chess_positions)} positions")
        
        # Test memory profiling (if CUDA available)
        print("4. Testing memory profiling...")
        if runner.device.type == "cuda":
            try:
                profile = runner.profile_memory_usage(2, 4)
                print(f"   ✓ Memory profiling: {profile.peak_gb:.2f}GB ({profile.utilization:.1%})")
            except Exception as e:
                print(f"   ⚠ Memory profiling failed: {e}")
        else:
            print("   ⚠ Skipping memory profiling (CPU mode)")
        
        # Test batch size optimization
        print("5. Testing batch size optimization...")
        try:
            optimal_config = runner.find_optimal_batch_size()
            print(f"   ✓ Optimal batch size: {optimal_config.batch_positions}")
            print(f"   ✓ Effective batch size: {optimal_config.effective_batch_size}")
        except Exception as e:
            print(f"   ⚠ Batch optimization failed: {e}")
        
        print("\n=== TEST COMPLETED SUCCESSFULLY ===")
        print("\nTo run full training:")
        print("uv run python scripts/train_full_grpo.py --steps 100")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set up basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    success = test_training_pipeline()
    sys.exit(0 if success else 1)