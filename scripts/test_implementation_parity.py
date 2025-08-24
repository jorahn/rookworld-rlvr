#!/usr/bin/env python3
"""
Implementation Parity Test

Compares test components with full training implementation to ensure all insights
from testing have been properly ported to the production code. Tests on identical
data and settings to verify numerical parity.
"""

import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


class TestComponentsImplementation:
    """Test components implementation for comparison"""
    
    def __init__(self, model: GPT2Model, ref_model: GPT2Model, device: torch.device):
        self.model = model
        self.ref_model = ref_model
        self.device = device
        self.tokenizer = TokenizerBridge()
    
    def find_target_start_manual(self, tokens: List[int], task_type: str) -> int:
        """Manual target detection implementation from test components"""
        
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
    
    def compute_logprobs_manual(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_start_indices: torch.Tensor,
        use_ref_model: bool = False
    ) -> torch.Tensor:
        """Manual logprob computation from test components"""
        
        model = self.ref_model if use_ref_model else self.model
        
        with torch.set_grad_enabled(not use_ref_model):
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]
            
            # Shift for autoregressive loss
            shift_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            shift_labels = input_ids[:, 1:]   # [batch_size, seq_len-1]
            
            # Convert to log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather logprobs for actual tokens
            token_logprobs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Apply target masking manually
            batch_size, seq_len = token_logprobs.shape
            target_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
            
            for i in range(batch_size):
                start_idx = max(0, target_start_indices[i] - 1)  # -1 for shift
                if attention_mask is not None:
                    shift_attention = attention_mask[i, 1:]
                    target_mask[i, start_idx:] = shift_attention[start_idx:].bool()
                else:
                    target_mask[i, start_idx:] = True
            
            # Compute mean logprobs manually
            masked_logprobs = token_logprobs.masked_fill(~target_mask, 0.0)
            token_counts = target_mask.sum(dim=1).clamp(min=1)
            mean_logprobs = masked_logprobs.sum(dim=1) / token_counts
            
            return mean_logprobs
    
    def compute_grpo_loss_manual(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_start_indices: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        clip_range: float = 0.2,
        kl_coef: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Manual GRPO loss computation from test components"""
        
        # Current policy log probabilities
        current_logprobs = self.compute_logprobs_manual(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        # Reference policy log probabilities
        with torch.no_grad():
            ref_logprobs = self.compute_logprobs_manual(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        # Group-relative baseline
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        # PPO-style clipped objective
        logprob_ratio = torch.exp(current_logprobs - old_logprobs)
        
        # Clipped surrogate objective
        unclipped_objective = logprob_ratio * advantages
        clipped_ratio = torch.clamp(logprob_ratio, 1.0 - clip_range, 1.0 + clip_range)
        clipped_objective = clipped_ratio * advantages
        
        # Take minimum (conservative estimate)
        policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
        
        # KL divergence penalty
        kl_div = (current_logprobs - ref_logprobs).mean()
        kl_loss = kl_coef * kl_div
        
        # Total loss
        total_loss = policy_loss + kl_loss
        
        # Metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item(),
            'baseline': baseline.item(),
            'mean_reward': rewards.mean().item(),
            'mean_logprob_ratio': logprob_ratio.mean().item()
        }
        
        return total_loss, metrics


class ImplementationParityTester:
    """Main class for testing implementation parity"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        print(f"🔍 IMPLEMENTATION PARITY TEST")
        print(f"Device: {self.device}")
        print("="*80)
        
        # Set reproducible seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize components
        self.tokenizer = TokenizerBridge()
        
        # Create models with same initialization
        model_config = GPT2Config()
        self.model = GPT2Model(model_config).to(self.device)
        self.ref_model = GPT2Model(model_config).to(self.device)
        
        # Copy model weights to reference
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        
        # Initialize GRPO config and trainer
        self.config = GRPOConfig(
            lr=1e-5,
            group_size=8,
            use_mixed_precision=False,
            use_torch_compile=False,
            kl_coef=0.01,
            clip_range=0.2,
            device=str(self.device)
        )
        
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.config)
        self.test_impl = TestComponentsImplementation(self.model, self.ref_model, self.device)
        
        print("✅ All components initialized")
    
    def create_test_batch(self) -> Dict[str, Any]:
        """Create identical test batch for both implementations"""
        
        # Test data: 6 policy (75%) + 2 environment (25%) tasks
        test_cases = [
            ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4", "policy"),
            ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: d2d4", "policy"),
            ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: g1f3", "policy"),
            ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: b1c3", "policy"),
            ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: c2c4", "policy"),
            ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: f2f4", "policy"),
            ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+result", "environment"),
            ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+d2d4+result", "environment"),
        ]
        
        texts = [text for text, _ in test_cases]
        task_types = [task_type for _, task_type in test_cases]
        
        # Tokenize and find targets
        all_tokens = []
        target_start_indices = []
        
        for i, (text, task_type) in enumerate(test_cases):
            tokens = self.tokenizer.encode(text)
            all_tokens.append(tokens)
            
            # Get target indices from both implementations
            manual_target = self.test_impl.find_target_start_manual(tokens, task_type)
            tokenizer_target = self.tokenizer.get_target_start_index(text, task_type)
            
            print(f"Text {i+1} ({task_type}): manual_target={manual_target}, tokenizer_target={tokenizer_target}")
            
            target_start_indices.append(tokenizer_target)  # Use production implementation
        
        # Pad to same length
        max_len = max(len(tokens) for tokens in all_tokens)
        input_ids = []
        attention_mask = []
        
        for tokens in all_tokens:
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            input_ids.append(padded)
            
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            attention_mask.append(mask)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        target_start_indices = torch.tensor(target_start_indices, device=self.device)
        
        # Create rewards and old_logprobs
        rewards = torch.tensor([1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 0.8, 0.9], device=self.device)
        
        # Get old_logprobs from reference model (simulates data generation)
        with torch.no_grad():
            old_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_start_indices': target_start_indices,
            'old_logprobs': old_logprobs,
            'rewards': rewards,
            'task_types': task_types,
            'texts': texts
        }
    
    def test_target_detection_parity(self, batch: Dict[str, Any]) -> bool:
        """Test target detection parity between implementations"""
        
        print(f"\n1. TARGET DETECTION PARITY")
        print("-" * 50)
        
        all_match = True
        
        for i, (text, task_type) in enumerate(zip(batch['texts'], batch['task_types'])):
            tokens = self.tokenizer.encode(text)
            
            manual_target = self.test_impl.find_target_start_manual(tokens, task_type)
            tokenizer_target = self.tokenizer.get_target_start_index(text, task_type)
            batch_target = batch['target_start_indices'][i].item()
            
            match = (manual_target == tokenizer_target == batch_target)
            status = "✅" if match else "❌"
            
            print(f"  {status} Text {i+1} ({task_type}): manual={manual_target}, tokenizer={tokenizer_target}, batch={batch_target}")
            
            if not match:
                all_match = False
        
        print(f"\nTarget Detection Parity: {'✅ PASS' if all_match else '❌ FAIL'}")
        return all_match
    
    def test_logprob_computation_parity(self, batch: Dict[str, Any]) -> bool:
        """Test logprob computation parity between implementations"""
        
        print(f"\n2. LOGPROB COMPUTATION PARITY")
        print("-" * 50)
        
        # Test both policy and reference models
        test_cases = [
            ("Policy Model", False),
            ("Reference Model", True)
        ]
        
        all_pass = True
        
        for model_name, use_ref_model in test_cases:
            print(f"\n{model_name}:")
            
            # Manual implementation
            manual_logprobs = self.test_impl.compute_logprobs_manual(
                batch['input_ids'], batch['attention_mask'], 
                batch['target_start_indices'], use_ref_model
            )
            
            # Production implementation
            prod_logprobs = self.trainer.compute_logprobs(
                batch['input_ids'], batch['attention_mask'],
                batch['target_start_indices'], use_ref_model
            )
            
            # Compare
            max_diff = torch.abs(manual_logprobs - prod_logprobs).max().item()
            tolerance = 1e-6
            match = max_diff < tolerance
            
            print(f"  Manual logprobs: {[f'{x:.6f}' for x in manual_logprobs.tolist()[:4]]}...")
            print(f"  Production logprobs: {[f'{x:.6f}' for x in prod_logprobs.tolist()[:4]]}...")
            print(f"  Max difference: {max_diff:.8f} (tolerance: {tolerance})")
            print(f"  Result: {'✅ PASS' if match else '❌ FAIL'}")
            
            if not match:
                all_pass = False
        
        print(f"\nLogprob Computation Parity: {'✅ PASS' if all_pass else '❌ FAIL'}")
        return all_pass
    
    def test_loss_computation_parity(self, batch: Dict[str, Any]) -> bool:
        """Test GRPO loss computation parity between implementations"""
        
        print(f"\n3. GRPO LOSS COMPUTATION PARITY")
        print("-" * 50)
        
        # Manual implementation
        manual_loss, manual_metrics = self.test_impl.compute_grpo_loss_manual(
            batch['input_ids'], batch['attention_mask'], batch['target_start_indices'],
            batch['old_logprobs'], batch['rewards'], 
            clip_range=self.config.clip_range, kl_coef=self.config.kl_coef
        )
        
        # Production implementation
        grpo_batch = GRPOBatch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target_start_indices=batch['target_start_indices'],
            old_logprobs=batch['old_logprobs'],
            rewards=batch['rewards'],
            position_fen="test_position",
            task_type="mixed"
        )
        
        prod_loss, prod_metrics = self.trainer.compute_grpo_loss(grpo_batch)
        
        # Compare key metrics
        comparison_metrics = ['policy_loss', 'kl_loss', 'kl_div', 'total_loss', 'mean_reward']
        tolerance = 1e-6
        all_match = True
        
        print(f"Metric Comparisons:")
        for metric in comparison_metrics:
            manual_val = manual_metrics[metric]
            prod_val = prod_metrics[metric]
            diff = abs(manual_val - prod_val)
            match = diff < tolerance
            
            status = "✅" if match else "❌"
            print(f"  {status} {metric}: manual={manual_val:.6f}, prod={prod_val:.6f}, diff={diff:.8f}")
            
            if not match:
                all_match = False
        
        # Compare total loss tensors
        loss_diff = torch.abs(manual_loss - prod_loss).item()
        loss_match = loss_diff < tolerance
        
        print(f"\nTotal Loss Tensor:")
        print(f"  Manual loss: {manual_loss.item():.6f}")
        print(f"  Production loss: {prod_loss.item():.6f}")
        print(f"  Difference: {loss_diff:.8f}")
        print(f"  Result: {'✅ PASS' if loss_match else '❌ FAIL'}")
        
        final_pass = all_match and loss_match
        print(f"\nGRPO Loss Computation Parity: {'✅ PASS' if final_pass else '❌ FAIL'}")
        return final_pass
    
    def test_gradient_flow_parity(self, batch: Dict[str, Any]) -> bool:
        """Test gradient flow and optimization step parity"""
        
        print(f"\n4. GRADIENT FLOW PARITY")
        print("-" * 50)
        
        # Store initial parameters
        initial_params = {}
        for name, param in self.model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Create GRPO batch
        grpo_batch = GRPOBatch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target_start_indices=batch['target_start_indices'],
            old_logprobs=batch['old_logprobs'],
            rewards=batch['rewards'],
            position_fen="test_position",
            task_type="mixed"
        )
        
        # Compute loss and do backward pass
        loss, metrics = self.trainer.compute_grpo_loss(grpo_batch)
        
        self.trainer.optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        has_grads = True
        nan_grads = False
        grad_norms = []
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                has_grads = False
                print(f"  ❌ No gradient for {name}")
            else:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_grads = True
                    print(f"  ❌ NaN/Inf gradient for {name}")
        
        # Take optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.trainer.optimizer.step()
        
        # Check parameter updates
        params_updated = False
        for name, param in self.model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-10):
                params_updated = True
                break
        
        # Results
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Has gradients: {'✅' if has_grads else '❌'}")
        print(f"  NaN/Inf gradients: {'❌' if nan_grads else '✅'}")
        print(f"  Average grad norm: {avg_grad_norm:.8f}")
        print(f"  Parameters updated: {'✅' if params_updated else '❌'}")
        
        gradient_flow_ok = has_grads and not nan_grads and params_updated and avg_grad_norm > 0
        print(f"\nGradient Flow Parity: {'✅ PASS' if gradient_flow_ok else '❌ FAIL'}")
        
        return gradient_flow_ok
    
    def run_comprehensive_parity_test(self) -> Dict[str, bool]:
        """Run comprehensive parity test between implementations"""
        
        print("Creating test batch...")
        batch = self.create_test_batch()
        print(f"✅ Test batch created: {len(batch['texts'])} samples")
        
        # Run all parity tests
        results = {}
        
        results['target_detection'] = self.test_target_detection_parity(batch)
        results['logprob_computation'] = self.test_logprob_computation_parity(batch)  
        results['loss_computation'] = self.test_loss_computation_parity(batch)
        results['gradient_flow'] = self.test_gradient_flow_parity(batch)
        
        return results
    
    def print_final_results(self, results: Dict[str, bool]) -> bool:
        """Print final parity test results"""
        
        print(f"\n{'='*80}")
        print("IMPLEMENTATION PARITY TEST RESULTS")
        print(f"{'='*80}")
        
        all_passed = True
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
            if not passed:
                all_passed = False
        
        print(f"\n{'='*80}")
        if all_passed:
            print("🎉 ALL PARITY TESTS PASSED!")
            print("All insights from test components have been correctly implemented")
            print("in the production training code.")
        else:
            print("⚠️  PARITY ISSUES DETECTED")
            print("Some differences found between test components and production code.")
            print("Investigation and fixes may be needed.")
        print(f"{'='*80}")
        
        return all_passed


def main():
    """Run implementation parity test"""
    
    tester = ImplementationParityTester()
    
    try:
        results = tester.run_comprehensive_parity_test()
        success = tester.print_final_results(results)
        return success
    except Exception as e:
        print(f"\n❌ PARITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)