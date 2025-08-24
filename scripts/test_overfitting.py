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
from typing import Dict, Any, List

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
    def __init__(self, device: str = "cuda", iterations: int = 50, use_mixed_tasks: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.iterations = iterations
        self.use_mixed_tasks = use_mixed_tasks
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
        # Adjust group size based on whether we're using mixed tasks
        group_size = 5 if use_mixed_tasks else 2
        
        self.grpo_config = GRPOConfig(
            lr=1e-6,  # Much lower learning rate to prevent instability
            group_size=group_size,  # 5 for mixed (4 policy + 1 env), 2 for policy-only
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
        """Create a small training batch to overfit on"""
        
        if self.use_mixed_tasks:
            return self._create_mixed_batch()
        else:
            return self._create_policy_only_batch()
    
    def _create_policy_only_batch(self) -> Dict[str, torch.Tensor]:
        """Create policy-only batch (original implementation)"""
        
        # Single chess position with two different moves (same position, different rewards)
        texts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",  # Good move
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: a2a3"   # Suboptimal move
        ]
        task_types = ["policy", "policy"]
        
        return self._process_batch(texts, task_types)
    
    def _create_mixed_batch(self) -> Dict[str, torch.Tensor]:
        """Create mixed policy + environment batch with realistic 80%/20% split"""
        
        # 4 policy tasks (80%) + 1 environment task (20%)
        texts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",    # Policy
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: d2d4",    # Policy
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: g1f3",    # Policy
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: b1c3",    # Policy
            "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+"          # Environment (20%)
        ]
        task_types = ["policy", "policy", "policy", "policy", "environment"]
        
        return self._process_batch(texts, task_types)
    
    def _find_target_start(self, tokens: List[int], task_type: str) -> int:
        """Find target start index based on task type"""
        
        if task_type == "policy":
            # Policy task: Find target start after "M:"
            for j in range(len(tokens) - 1):
                current_decoded = self.tokenizer.decode([tokens[j]]).strip()
                next_decoded = self.tokenizer.decode([tokens[j + 1]]).strip()
                if current_decoded == 'M' and next_decoded == ':':
                    return j + 2  # Start after both 'M' and ':'
                elif current_decoded.endswith('M') and next_decoded == ':':
                    return j + 2
                elif current_decoded == 'M:':
                    return j + 1
        
        elif task_type == "environment":
            # Environment task: Find target start after first "+"
            for j in range(len(tokens)):
                current_decoded = self.tokenizer.decode([tokens[j]]).strip()
                if current_decoded == '+':
                    return j + 1  # Start after first '+'
        
        # Default fallback
        return len(tokens) - 1
    
    def _create_rewards(self, task_types: List[str]) -> torch.Tensor:
        """Create rewards based on task types"""
        rewards = []
        for task_type in task_types:
            if task_type == "policy":
                # Policy tasks get varied rewards (simulate move quality differences)
                reward = 1.0  # All positive for overfitting test
            else:  # environment
                # Environment tasks get consistent reward
                reward = 0.8  # Slightly lower but still positive
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def _process_batch(self, texts: List[str], task_types: List[str]) -> Dict[str, torch.Tensor]:
        
        # Encode texts and find target start indices
        all_tokens = []
        target_start_indices = []
        
        for i, text in enumerate(texts):
            tokens = self.tokenizer.encode(text)
            all_tokens.append(tokens)
            
            task_type = task_types[i]
            target_start_idx = self._find_target_start(tokens, task_type)
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
        
        # Get initial logprobs from reference model (this simulates data generation)
        # In real training, old_logprobs would come from the policy that generated the data
        with torch.no_grad():
            old_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        # Create rewards based on task types
        rewards = self._create_rewards(task_types)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_start_indices': target_start_indices,
            'old_logprobs': old_logprobs,
            'rewards': rewards,
            'texts': texts,
            'task_types': task_types
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