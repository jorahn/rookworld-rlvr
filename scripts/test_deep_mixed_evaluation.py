#!/usr/bin/env python3
"""
Deep Mixed Task Evaluation with All Improvements

Comprehensive test with:
- Batch size 16 (13 policy + 3 environment = ~20% env ratio)
- 50 epochs of same samples (overfitting test)
- Mixed task training (P: and A: formats)
- Reduced learning rate (1e-6)
- Fixed target start index detection
- Step-by-step in-depth logging
- All recent improvements incorporated
"""

import torch
import time
import sys
import os
import logging
import json
import numpy as np
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

class DeepMixedEvaluator:
    def __init__(self, device: str = "cuda", epochs: int = 50, batch_size: int = 16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = TokenizerBridge()
        self.config = ROOKWORLD_CONFIG
        
        # Create models
        self.model = GPT2Model(self.config).to(self.device)
        self.ref_model = GPT2Model(self.config).to(self.device)
        
        # Copy weights to reference model and freeze
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # GRPO config with all improvements
        self.grpo_config = GRPOConfig(
            lr=1e-6,  # Reduced learning rate for stability
            group_size=batch_size,  # Match batch size
            steps=epochs,
            batch_positions=1,
            use_mixed_precision=False,  # Disabled for numerical stability
            use_torch_compile=False,    # Disabled for debugging clarity
            clip_range=0.2,            # Standard PPO clipping
            kl_coef=0.01,              # KL penalty coefficient
            grad_clip_norm=1.0,        # Gradient clipping
            device=str(device)
        )
        
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.grpo_config)
        
        print(f"Initialized Deep Mixed Evaluator:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.grpo_config.lr}")
        print(f"  Expected to see clear overfitting over {epochs} epochs")
        
    def create_mixed_batch_16(self) -> Dict[str, torch.Tensor]:
        """Create batch of 16 samples with 80% policy (13) + 20% environment (3)"""
        
        base_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        # 13 policy tasks (80%) with varied moves for diversity
        policy_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "f2f4", "e2e3", "d2d3", "g1h3", "b1a3", "c2c4", "c2c3", "a2a4", "h2h4"]
        policy_texts = [f"P: {base_fen}    M: {move}" for move in policy_moves]
        policy_types = ["policy"] * 13
        
        # 3 environment tasks (20%)
        env_moves = ["e2e4", "d2d4", "g1f3"]
        env_texts = [f"A: {base_fen}+{move}+" for move in env_moves]
        env_types = ["environment"] * 3
        
        # Combine
        all_texts = policy_texts + env_texts
        all_types = policy_types + env_types
        
        assert len(all_texts) == 16, f"Expected 16 samples, got {len(all_texts)}"
        assert len([t for t in all_types if t == "policy"]) == 13, "Expected 13 policy tasks"
        assert len([t for t in all_types if t == "environment"]) == 3, "Expected 3 environment tasks"
        
        return self._process_batch(all_texts, all_types)
    
    def _find_target_start(self, tokens: List[int], task_type: str) -> int:
        """Find target start index based on task type (FIXED VERSION)"""
        
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
        
        # Default fallback (should not happen with correct formats)
        return len(tokens) - 1
    
    def _process_batch(self, texts: List[str], task_types: List[str]) -> Dict[str, torch.Tensor]:
        """Process batch of mixed texts into training tensors"""
        
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
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            input_ids.append(padded)
            attention_mask.append(mask)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        target_start_indices = torch.tensor(target_start_indices, device=self.device)
        
        # Get reference logprobs (from frozen reference model)
        with torch.no_grad():
            old_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        # Create realistic rewards (all positive for overfitting)
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
    
    def _create_rewards(self, task_types: List[str]) -> torch.Tensor:
        """Create rewards based on task types"""
        rewards = []
        for i, task_type in enumerate(task_types):
            if task_type == "policy":
                # Policy tasks: varied rewards but all positive
                base_reward = 0.8
                variation = 0.1 * (i % 5)  # Small variation for diversity
                reward = base_reward + variation
            else:  # environment
                # Environment tasks: consistent positive reward
                reward = 0.9
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def run_deep_evaluation(self):
        """Run comprehensive evaluation across 50 epochs"""
        
        print("\n" + "="*80)
        print("DEEP MIXED TASK EVALUATION - ALL IMPROVEMENTS APPLIED")
        print("="*80)
        print(f"Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size} (13 policy + 3 environment)")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.grpo_config.lr}")
        print(f"  Mixed precision: {self.grpo_config.use_mixed_precision}")
        print(f"  Torch compile: {self.grpo_config.use_torch_compile}")
        print(f"  KL coefficient: {self.grpo_config.kl_coef}")
        print(f"  Gradient clipping: {self.grpo_config.grad_clip_norm}")
        
        # Create batch
        batch = self.create_mixed_batch_16()
        
        print(f"\nBatch Analysis:")
        print(f"  Input shape: {batch['input_ids'].shape}")
        print(f"  Target start indices: {batch['target_start_indices'].tolist()}")
        print(f"  Initial reference logprobs: {[f'{x:.4f}' for x in batch['old_logprobs'].tolist()]}")
        print(f"  Rewards: {[f'{x:.2f}' for x in batch['rewards'].tolist()]}")
        print(f"  Task breakdown: {13} policy, {3} environment")
        
        # Detailed sample inspection
        print(f"\nSample Breakdown:")
        for i in range(min(5, len(batch['texts']))):  # Show first 5 samples
            text = batch['texts'][i]
            task_type = batch['task_types'][i]
            target_idx = batch['target_start_indices'][i]
            reward = batch['rewards'][i]
            
            # Get actual target tokens
            input_ids_sample = batch['input_ids'][i]
            seq_len = (batch['attention_mask'][i] == 1).sum().item()
            target_tokens = input_ids_sample[target_idx:seq_len].tolist()
            target_text = self.tokenizer.decode(target_tokens) if target_tokens else ""
            
            print(f"  {i+1}. {task_type.upper()}: {text[:60]}...")
            print(f"     Target idx: {target_idx}, Target: '{target_text.strip()}', Reward: {reward:.2f}")
        
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Expected: Clear overfitting with logprob improvements > 2.0")
        print("")
        
        # Track metrics
        metrics_history = []
        
        # Training loop
        self.model.train()
        
        # Pre-training baseline
        print("EPOCH    AVG_LOGPROB    IMPROVEMENT    LOSS        KL_DIV      GRAD_NORM   TIME")
        print("-" * 80)
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
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
                old_logprobs=batch['old_logprobs'],
                rewards=batch['rewards'],
                position_fen=f"mixed_batch_epoch_{epoch}",
                task_type="mixed"
            )
            
            # Compute loss with detailed metrics
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
            
            epoch_time = time.time() - epoch_start_time
            
            # Calculate metrics
            avg_logprob = current_logprobs.mean().item()
            improvement = (current_logprobs - batch['old_logprobs']).mean().item()
            kl_div = loss_metrics.get('kl_loss', 0.0) / self.grpo_config.kl_coef if 'kl_loss' in loss_metrics else 0.0
            
            # Track metrics
            metrics = {
                'epoch': epoch,
                'avg_logprob': avg_logprob,
                'improvement': improvement,
                'loss': loss.item(),
                'kl_divergence': kl_div,
                'grad_norm': grad_norm.item(),
                'time': epoch_time,
                'current_logprobs': current_logprobs.tolist(),
                'individual_improvements': (current_logprobs - batch['old_logprobs']).tolist(),
                **loss_metrics
            }
            metrics_history.append(metrics)
            
            # Print progress (every 5 epochs or first/last few)
            if epoch % 5 == 0 or epoch < 5 or epoch >= self.epochs - 5:
                print(f"{epoch:5d}    {avg_logprob:10.4f}    {improvement:10.4f}    "
                      f"{loss.item():10.6f}  {kl_div:10.4f}  {grad_norm.item():10.4f}  "
                      f"{epoch_time:8.3f}s")
            
            # Detailed logging every 10 epochs
            if epoch % 10 == 0:
                print(f"\n  Detailed Analysis at Epoch {epoch}:")
                
                # Per-task-type analysis
                policy_indices = [i for i, t in enumerate(batch['task_types']) if t == 'policy']
                env_indices = [i for i, t in enumerate(batch['task_types']) if t == 'environment']
                
                policy_logprobs = current_logprobs[policy_indices]
                env_logprobs = current_logprobs[env_indices]
                
                policy_improvements = (policy_logprobs - batch['old_logprobs'][policy_indices]).mean().item()
                env_improvements = (env_logprobs - batch['old_logprobs'][env_indices]).mean().item()
                
                print(f"    Policy tasks (13): avg_logprob={policy_logprobs.mean().item():.4f}, "
                      f"avg_improvement={policy_improvements:+.4f}")
                print(f"    Environment tasks (3): avg_logprob={env_logprobs.mean().item():.4f}, "
                      f"avg_improvement={env_improvements:+.4f}")
                
                # Check for concerning patterns
                max_improvement = (current_logprobs - batch['old_logprobs']).max().item()
                min_improvement = (current_logprobs - batch['old_logprobs']).min().item()
                
                if max_improvement - min_improvement > 3.0:
                    print(f"    ⚠️  WARNING: Large improvement spread ({min_improvement:+.3f} to {max_improvement:+.3f})")
                
                if abs(kl_div) > 2.0:
                    print(f"    ⚠️  WARNING: Large KL divergence detected ({kl_div:+.3f})")
                
                if grad_norm.item() > 5.0:
                    print(f"    ⚠️  WARNING: Large gradient norm ({grad_norm.item():.3f})")
                
                print("")
        
        print("\n" + "="*80)
        print("FINAL EVALUATION RESULTS")
        print("="*80)
        
        # Final analysis
        final_metrics = metrics_history[-1]
        initial_logprobs = batch['old_logprobs'].tolist()
        final_logprobs = final_metrics['current_logprobs']
        final_improvements = final_metrics['individual_improvements']
        
        avg_improvement = np.mean(final_improvements)
        max_improvement = np.max(final_improvements)
        min_improvement = np.min(final_improvements)
        std_improvement = np.std(final_improvements)
        
        print(f"Training Summary:")
        print(f"  Initial avg logprob: {np.mean(initial_logprobs):.6f}")
        print(f"  Final avg logprob:   {final_metrics['avg_logprob']:.6f}")
        print(f"  Average improvement: {avg_improvement:+.6f}")
        print(f"  Improvement range:   [{min_improvement:+.6f}, {max_improvement:+.6f}]")
        print(f"  Improvement std:     {std_improvement:.6f}")
        print(f"  Final loss:          {final_metrics['loss']:.6f}")
        print(f"  Final KL divergence: {final_metrics['kl_divergence']:.6f}")
        print(f"  Final grad norm:     {final_metrics['grad_norm']:.6f}")
        
        # Task-specific analysis
        policy_indices = [i for i, t in enumerate(batch['task_types']) if t == 'policy']
        env_indices = [i for i, t in enumerate(batch['task_types']) if t == 'environment']
        
        policy_improvements = [final_improvements[i] for i in policy_indices]
        env_improvements = [final_improvements[i] for i in env_indices]
        
        print(f"\nTask-Specific Results:")
        print(f"  Policy tasks (13):     avg={np.mean(policy_improvements):+.4f}, "
              f"std={np.std(policy_improvements):.4f}")
        print(f"  Environment tasks (3): avg={np.mean(env_improvements):+.4f}, "
              f"std={np.std(env_improvements):.4f}")
        
        # Overfitting success criteria
        success_criteria = {
            'significant_improvement': avg_improvement > 1.0,  # At least 1.0 average improvement
            'all_samples_improved': min_improvement > 0.1,    # All samples should improve
            'no_catastrophic_divergence': std_improvement < 2.0,  # Reasonable spread
            'stable_training': abs(final_metrics['kl_divergence']) < 5.0,  # Controlled KL
            'loss_converged': final_metrics['loss'] < metrics_history[10]['loss']  # Loss improved from early epochs
        }
        
        print(f"\nOverfitting Success Criteria:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")
        
        overall_success = all(success_criteria.values())
        
        print(f"\n{'='*50}")
        if overall_success:
            print("🎉 DEEP EVALUATION: OVERFITTING SUCCESS!")
            print("   Mixed task training with all improvements working correctly")
            print(f"   Average improvement: {avg_improvement:+.3f}")
            print(f"   Training stability: ✅ Controlled (std={std_improvement:.3f})")
            print(f"   Task balance: ✅ Both policy and environment tasks learning")
        else:
            print("❌ DEEP EVALUATION: ISSUES DETECTED")
            failed_criteria = [k for k, v in success_criteria.items() if not v]
            print(f"   Failed criteria: {', '.join(failed_criteria)}")
            print(f"   Further investigation needed")
        
        # Save detailed metrics
        output_file = 'deep_mixed_evaluation_metrics.json'
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.grpo_config.lr,
                    'device': str(self.device)
                },
                'batch_info': {
                    'task_types': batch['task_types'],
                    'initial_logprobs': initial_logprobs,
                    'rewards': batch['rewards'].tolist(),
                    'target_start_indices': batch['target_start_indices'].tolist()
                },
                'training_history': metrics_history,
                'final_analysis': {
                    'avg_improvement': avg_improvement,
                    'improvement_range': [min_improvement, max_improvement],
                    'improvement_std': std_improvement,
                    'success_criteria': {k: bool(v) for k, v in success_criteria.items()},
                    'overall_success': bool(overall_success)
                }
            }, f, indent=2)
        
        print(f"\nDetailed metrics saved to: {output_file}")
        return overall_success, metrics_history

def main():
    """Run the comprehensive deep mixed task evaluation"""
    evaluator = DeepMixedEvaluator(epochs=50, batch_size=16)
    success, metrics = evaluator.run_deep_evaluation()
    
    if not success:
        print("\n⚠️  Some success criteria not met. Consider:")
        print("   - Increasing number of epochs")
        print("   - Adjusting learning rate")
        print("   - Checking for numerical stability issues")
        print("   - Analyzing task balance")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)