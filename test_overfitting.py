#!/usr/bin/env python3
"""
Overfitting Test for GRPO Implementation

Tests that the model can overfit to a single batch over multiple iterations.
This validates that the training loop and gradients are working correctly.
"""

import torch
import time
import sys
import os
import logging
import json
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rookworld_rlvr.model.config import ROOKWORLD_CONFIG, GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

class OverfittingTester:
    def __init__(self, device: str = "cuda", iterations: int = 50):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.iterations = iterations
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = TokenizerBridge()
        self.config = ROOKWORLD_CONFIG
        
        # Create models
        self.model = GPT2Model(self.config).to(self.device)
        self.ref_model = GPT2Model(self.config).to(self.device)
        
        # Copy weights to reference model
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        
        # GRPO config optimized for overfitting test
        self.grpo_config = GRPOConfig(
            lr=1e-4,  # Higher learning rate for faster overfitting
            group_size=2,  # Minimum allowed for GRPO
            steps=iterations,
            batch_positions=1,
            use_mixed_precision=False,  # Keep disabled for now
            use_torch_compile=False,    # Keep disabled for now
            clip_range=0.2,
            kl_coef=0.01,  # Lower KL penalty to allow more aggressive updates
            device=device
        )
        
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.grpo_config)
        
    def create_single_batch(self) -> Dict[str, torch.Tensor]:
        """Create a small training batch to overfit on (duplicated for group_size=2)"""
        
        # Single chess position with two different moves (same position, different rewards)
        texts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",  # Good move
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: a2a3"   # Suboptimal move
        ]
        
        # Encode both texts
        all_tokens = []
        target_start_indices = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.append(tokens)
            
            # Find target start (after "M:")
            target_start_idx = 0
            for i, token in enumerate(tokens):
                decoded = self.tokenizer.decode([token])
                if decoded.strip() == 'M:':
                    target_start_idx = i + 1
                    break
            target_start_indices.append(target_start_idx)
        
        # Pad to same length
        max_len = max(len(tokens) for tokens in all_tokens)
        input_ids = []
        attention_mask = []
        
        for tokens in all_tokens:
            # Pad with pad token
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            input_ids.append(padded)
            
            # Attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            attention_mask.append(mask)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        target_start_indices = torch.tensor(target_start_indices, device=self.device)
        
        # Get initial logprobs from reference model
        with torch.no_grad():
            old_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        # Different rewards: e2e4 is better than a2a3
        rewards = torch.tensor([1.0, 0.3], dtype=torch.float32, device=self.device)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_start_indices': target_start_indices,
            'old_logprobs': old_logprobs,
            'rewards': rewards,
            'texts': texts
        }
    
    def run_overfitting_test(self):
        """Run overfitting test over multiple iterations"""
        
        print("="*80)
        print("GRPO OVERFITTING TEST")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Iterations: {self.iterations}")
        print(f"Learning rate: {self.grpo_config.lr}")
        print(f"Expected: Loss should decrease, logprobs should increase")
        print("")
        
        # Create single batch
        batch = self.create_single_batch()
        print(f"Training texts:")
        for i, text in enumerate(batch['texts']):
            print(f"  {i+1}. {text} (reward: {batch['rewards'][i]:.1f})")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Target start indices: {batch['target_start_indices'].tolist()}")
        print(f"Initial old logprobs: [{batch['old_logprobs'][0]:.4f}, {batch['old_logprobs'][1]:.4f}]")
        print("")
        
        # Track metrics
        metrics_history = []
        
        # Training loop
        self.model.train()
        
        for iteration in range(self.iterations):
            start_time = time.time()
            
            # Get current logprobs
            current_logprobs = self.trainer.compute_logprobs(
                batch['input_ids'],
                batch['attention_mask'],
                batch['target_start_indices'],
                use_ref_model=False
            )
            
            # Create GRPO batch object
            grpo_batch = GRPOBatch(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_start_indices=batch['target_start_indices'],
                old_logprobs=batch['old_logprobs'],  # Use initial logprobs as reference
                rewards=batch['rewards'],
                position_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                task_type="policy"
            )
            
            # Compute loss
            loss, loss_metrics = self.trainer.compute_grpo_loss(grpo_batch)
            
            # Backward pass
            self.trainer.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.grpo_config.grad_clip_norm
            )
            
            # Optimizer step
            self.trainer.optimizer.step()
            
            iteration_time = time.time() - start_time
            
            # Track metrics
            metrics = {
                'iteration': iteration,
                'loss': loss.item(),
                'current_logprobs': current_logprobs.tolist(),
                'logprob_improvement': (current_logprobs - batch['old_logprobs']).tolist(),
                'grad_norm': grad_norm.item(),
                'time': iteration_time,
                **loss_metrics
            }
            metrics_history.append(metrics)
            
            # Print progress
            if iteration % 10 == 0 or iteration < 5 or iteration >= self.iterations - 5:
                avg_current = current_logprobs.mean().item()
                avg_improvement = (current_logprobs - batch['old_logprobs']).mean().item()
                print(f"Iter {iteration:3d}: "
                      f"Loss={loss.item():8.6f}, "
                      f"AvgLogProb={avg_current:8.4f} "
                      f"(Δ{avg_improvement:+7.4f}), "
                      f"GradNorm={grad_norm.item():.4f}")
        
        print("")
        
        # Final analysis
        final_metrics = metrics_history[-1]
        initial_logprobs = batch['old_logprobs'].tolist()
        final_logprobs = final_metrics['current_logprobs']
        logprob_improvements = [final - initial for final, initial in zip(final_logprobs, initial_logprobs)]
        avg_improvement = sum(logprob_improvements) / len(logprob_improvements)
        
        print("="*80)
        print("OVERFITTING TEST RESULTS")
        print("="*80)
        print(f"Initial logprobs: [{initial_logprobs[0]:.6f}, {initial_logprobs[1]:.6f}]")
        print(f"Final logprobs:   [{final_logprobs[0]:.6f}, {final_logprobs[1]:.6f}]")
        print(f"Improvements:     [{logprob_improvements[0]:+.6f}, {logprob_improvements[1]:+.6f}]")
        print(f"Avg improvement:  {avg_improvement:+.6f}")
        print(f"Final loss:       {final_metrics['loss']:.6f}")
        print(f"Final grad norm:  {final_metrics['grad_norm']:.6f}")
        
        # Overfitting success criteria
        success_criteria = {
            'logprob_improved': avg_improvement > 0.1,  # At least 0.1 average improvement
            'loss_decreased': final_metrics['loss'] < metrics_history[0]['loss'],
            'gradients_flowing': final_metrics['grad_norm'] > 1e-6
        }
        
        print("")
        print("Success Criteria:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")
        
        overall_success = all(success_criteria.values())
        print("")
        if overall_success:
            print("🎉 OVERFITTING TEST PASSED")
            print("   The model successfully overfits to the single example")
            print("   Training loop and gradients are working correctly")
        else:
            print("❌ OVERFITTING TEST FAILED") 
            print("   The model fails to overfit - indicates training issues")
            
        # Save detailed metrics
        with open('overfitting_metrics.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"\nDetailed metrics saved to: overfitting_metrics.json")
        
        return overall_success, metrics_history

def main():
    """Run the overfitting test"""
    tester = OverfittingTester(iterations=100)  # More iterations for clear overfitting
    success, metrics = tester.run_overfitting_test()
    
    if not success:
        print("\n⚠️  Consider investigating:")
        print("   - Learning rate too low")
        print("   - Gradient clipping too aggressive") 
        print("   - KL penalty too high")
        print("   - Target masking issues")
        print("   - Loss computation bugs")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)