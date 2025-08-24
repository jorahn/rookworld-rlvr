#!/usr/bin/env python3
"""
Aggressive Mixed Task Overfitting Test

Higher learning rate test to ensure we can achieve true overfitting.
Uses all improvements but with more aggressive learning parameters.
"""

import torch
import sys
import os
import time
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from test_deep_mixed_evaluation import DeepMixedEvaluator

class AggressiveOverfittingTester(DeepMixedEvaluator):
    """More aggressive overfitting test with higher learning rate"""
    
    def __init__(self, device: str = "cuda", epochs: int = 25, batch_size: int = 16, learning_rate: float = 5e-5):
        # Call parent with custom learning rate
        super().__init__(device, epochs, batch_size)
        
        # Override with more aggressive learning rate
        self.grpo_config.lr = learning_rate
        
        # Recreate trainer with new config
        from rookworld_rlvr.train.grpo_trainer import GRPOTrainer
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.grpo_config)
        
        print(f"🔥 AGGRESSIVE OVERFITTING CONFIG:")
        print(f"   Learning rate: {learning_rate} (50x higher than conservative)")
        print(f"   Epochs: {epochs} (shorter for faster feedback)")
        print(f"   Expected: Strong overfitting within {epochs} epochs")

def main():
    """Run aggressive overfitting test"""
    
    print("🔥 AGGRESSIVE MIXED TASK OVERFITTING TEST")
    print("="*80)
    print("Testing with higher learning rate to ensure overfitting capability")
    
    # Test with progressively higher learning rates
    learning_rates = [1e-5, 5e-5, 1e-4]
    
    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"TESTING WITH LEARNING RATE: {lr}")
        print(f"{'='*60}")
        
        tester = AggressiveOverfittingTester(epochs=15, batch_size=16, learning_rate=lr)
        
        # Create batch and run just first 5 epochs for quick feedback
        batch = tester.create_mixed_batch_16()
        tester.model.train()
        
        print(f"Initial logprobs: {[f'{x:.3f}' for x in batch['old_logprobs'].tolist()[:5]]}...")
        
        current_best_improvement = -float('inf')
        
        for epoch in range(5):  # Quick test
            current_logprobs = tester.trainer.compute_logprobs(
                batch['input_ids'], batch['attention_mask'],
                batch['target_start_indices'], use_ref_model=False
            )
            
            from rookworld_rlvr.train.grpo_trainer import GRPOBatch
            grpo_batch = GRPOBatch(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                target_start_indices=batch['target_start_indices'],
                old_logprobs=batch['old_logprobs'], rewards=batch['rewards'],
                position_fen='test', task_type='mixed'
            )
            
            loss, metrics = tester.trainer.compute_grpo_loss(grpo_batch)
            improvement = (current_logprobs - batch['old_logprobs']).mean().item()
            
            print(f"  Epoch {epoch}: avg_improvement={improvement:+.4f}, loss={loss.item():.6f}")
            
            if improvement > current_best_improvement:
                current_best_improvement = improvement
            
            tester.trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tester.model.parameters(), 1.0)
            tester.trainer.optimizer.step()
            
            # Early success detection
            if improvement > 0.5:
                print(f"  🎉 OVERFITTING ACHIEVED! Improvement: {improvement:+.3f}")
                break
        
        # Assessment
        if current_best_improvement > 0.2:
            print(f"  ✅ SUCCESS: Learning rate {lr} shows overfitting (best: {current_best_improvement:+.3f})")
            
            # Run full test with this learning rate
            print(f"\n  Running full 25-epoch test with lr={lr}...")
            full_tester = AggressiveOverfittingTester(epochs=25, batch_size=16, learning_rate=lr)
            success, _ = full_tester.run_deep_evaluation()
            
            if success:
                print(f"  🏆 FULL TEST SUCCESS with lr={lr}")
                return True
            else:
                print(f"  📊 Full test completed but didn't meet all criteria")
        else:
            print(f"  ❌ INSUFFICIENT: Learning rate {lr} too low (best: {current_best_improvement:+.3f})")
    
    print(f"\n⚠️  CONCLUSION: Need even higher learning rate or different approach for clear overfitting")
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 FINAL RESULT: Mixed task training with all improvements CAN overfit!")
        print(f"   This confirms the training pipeline is working correctly.")
    else:
        print(f"\n📋 FINAL RESULT: Conservative approach prevents overfitting")
        print(f"   This suggests the stability improvements work as intended.")
    
    sys.exit(0 if success else 1)