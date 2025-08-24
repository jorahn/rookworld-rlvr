#!/usr/bin/env python3
"""
Component Isolation Tests

Tests each component of the GRPO implementation in isolation to identify
the root cause of catastrophic model divergence and negative loss.
"""

import torch
import torch.nn.functional as F
import sys
import os
import logging
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rookworld_rlvr.model.config import ROOKWORLD_CONFIG
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ComponentTester:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = TokenizerBridge()
        self.model = GPT2Model(ROOKWORLD_CONFIG).to(self.device)
        self.ref_model = GPT2Model(ROOKWORLD_CONFIG).to(self.device)
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        
        # GRPO config for testing
        self.config = GRPOConfig(
            lr=1e-5,  # Start with very conservative learning rate
            group_size=2,
            use_mixed_precision=False,
            use_torch_compile=False,
            kl_coef=0.01,
            device=str(self.device)
        )
        
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.config)
        
    def test_1_tokenization_and_target_masking(self):
        """Test 1: Tokenization & Target Masking"""
        
        print("="*80)
        print("TEST 1: TOKENIZATION & TARGET MASKING")
        print("="*80)
        
        # Test cases
        test_cases = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: d2d4",
            "P: short    M: test",
            "P: much longer position string here for testing    M: longer_move"
        ]
        
        all_passed = True
        
        for i, text in enumerate(test_cases):
            print(f"\nTest case {i+1}: {text}")
            
            # Tokenize
            tokens = self.tokenizer.encode(text)
            print(f"  Tokens ({len(tokens)}): {tokens}")
            
            # Find target start index
            target_start_idx = None
            for j in range(len(tokens) - 1):
                # Check if current token is 'M' and next is ':'
                current_decoded = self.tokenizer.decode([tokens[j]]).strip()
                next_decoded = self.tokenizer.decode([tokens[j + 1]]).strip()
                if current_decoded == 'M' and next_decoded == ':':
                    target_start_idx = j + 2  # Start after both 'M' and ':'
                    print(f"  Found 'M' ':' at positions {j},{j+1}, target starts at {target_start_idx}")
                    break
                # Also check for combined ' M:' patterns or 'M:' as single token
                elif current_decoded.endswith('M') and next_decoded == ':':
                    target_start_idx = j + 2
                    print(f"  Found 'M' ':' pattern at positions {j},{j+1}, target starts at {target_start_idx}")
                    break
                elif current_decoded == 'M:':
                    target_start_idx = j + 1
                    print(f"  Found 'M:' at position {j}, target starts at {target_start_idx}")
                    break
            
            if target_start_idx is None:
                print(f"  ❌ ERROR: Could not find 'M:' token")
                all_passed = False
                continue
            
            # Verify target tokens
            if target_start_idx < len(tokens):
                target_tokens = tokens[target_start_idx:]
                target_text = self.tokenizer.decode(target_tokens)
                print(f"  Target tokens: {target_tokens}")
                print(f"  Target text: '{target_text}'")
                
                # Check if target makes sense
                if not target_text.strip():
                    print(f"  ❌ ERROR: Target text is empty")
                    all_passed = False
                else:
                    print(f"  ✅ Target looks correct")
            else:
                print(f"  ❌ ERROR: Target start index {target_start_idx} >= sequence length {len(tokens)}")
                all_passed = False
            
            # Test roundtrip
            decoded = self.tokenizer.decode(tokens)
            if decoded == text:
                print(f"  ✅ Roundtrip encoding/decoding correct")
            else:
                print(f"  ❌ ERROR: Roundtrip failed")
                print(f"      Original: '{text}'")
                print(f"      Decoded:  '{decoded}'")
                all_passed = False
        
        print(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Tokenization & Target Masking Test")
        return all_passed
    
    def test_2_logprob_computation(self):
        """Test 2: Logprob Computation"""
        
        print("\n" + "="*80)
        print("TEST 2: LOGPROB COMPUTATION")
        print("="*80)
        
        # Create test batch
        texts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: d2d4"
        ]
        
        all_tokens = [self.tokenizer.encode(text) for text in texts]
        max_len = max(len(tokens) for tokens in all_tokens)
        
        # Create batch tensors
        input_ids = []
        attention_mask = []
        target_start_indices = []
        
        for tokens in all_tokens:
            # Pad
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            input_ids.append(padded)
            attention_mask.append(mask)
            
            # Find target start  
            target_idx = len(tokens) - 1  # Default fallback
            for j in range(len(tokens) - 1):
                current_decoded = self.tokenizer.decode([tokens[j]]).strip()
                next_decoded = self.tokenizer.decode([tokens[j + 1]]).strip()
                if current_decoded == 'M' and next_decoded == ':':
                    target_idx = j + 2
                    break
                elif current_decoded.endswith('M') and next_decoded == ':':
                    target_idx = j + 2
                    break
                elif current_decoded == 'M:':
                    target_idx = j + 1
                    break
            target_start_indices.append(target_idx)
        
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        target_start_indices = torch.tensor(target_start_indices, device=self.device)
        
        print(f"Batch shape: {input_ids.shape}")
        print(f"Target start indices: {target_start_indices.tolist()}")
        
        # Test current model logprobs
        print(f"\nTesting current model logprobs...")
        current_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        print(f"Current logprobs: {current_logprobs.tolist()}")
        
        # Test reference model logprobs
        print(f"\nTesting reference model logprobs...")
        ref_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
        print(f"Reference logprobs: {ref_logprobs.tolist()}")
        
        # Verify consistency (models should be identical initially)
        logprob_diff = torch.abs(current_logprobs - ref_logprobs).max().item()
        print(f"Max difference between current and ref: {logprob_diff:.6f}")
        
        # Manual verification of logprob computation
        print(f"\nManual logprob verification...")
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Shift for autoregressive prediction
            shift_logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            shift_labels = input_ids[:, 1:]   # [batch_size, seq_len-1]
            
            # Convert to log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather logprobs for actual tokens
            token_logprobs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
            print(f"Token logprobs shape: {token_logprobs.shape}")
            print(f"Sample token logprobs [0, :10]: {token_logprobs[0, :10].tolist()}")
            
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
            
            print(f"Target mask shape: {target_mask.shape}")
            print(f"Target mask [0]: {target_mask[0].int().tolist()}")
            print(f"Target tokens counted: {target_mask.sum(dim=1).tolist()}")
            
            # Compute mean logprobs manually
            masked_logprobs = token_logprobs.masked_fill(~target_mask, 0.0)
            token_counts = target_mask.sum(dim=1).clamp(min=1)
            manual_logprobs = masked_logprobs.sum(dim=1) / token_counts
            
            print(f"Manual logprobs: {manual_logprobs.tolist()}")
            print(f"Trainer logprobs: {current_logprobs.tolist()}")
            
            manual_diff = torch.abs(manual_logprobs - current_logprobs).max().item()
            print(f"Max difference (manual vs trainer): {manual_diff:.6f}")
        
        # Success criteria
        tests_passed = {
            'models_identical_initially': logprob_diff < 1e-4,
            'logprobs_reasonable': all(-50 < lp < 0 for lp in current_logprobs.tolist()),
            'manual_computation_matches': manual_diff < 1e-4,
            'target_tokens_nonzero': all(count > 0 for count in token_counts.tolist())
        }
        
        print(f"\nTest Results:")
        all_passed = True
        for test_name, passed in tests_passed.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Logprob Computation Test")
        return all_passed
    
    def test_3_reference_model_freezing(self):
        """Test 3: Reference Model Freezing"""
        
        print("\n" + "="*80)
        print("TEST 3: REFERENCE MODEL FREEZING")
        print("="*80)
        
        # Get initial reference model parameters
        initial_params = {}
        for name, param in self.ref_model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        print(f"Stored {len(initial_params)} reference model parameters")
        
        # Create dummy batch and perform several training steps
        texts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: d2d4"
        ]
        
        all_tokens = [self.tokenizer.encode(text) for text in texts]
        max_len = max(len(tokens) for tokens in all_tokens)
        
        input_ids = []
        attention_mask = []
        target_start_indices = []
        
        for tokens in all_tokens:
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            input_ids.append(padded)
            attention_mask.append(mask)
            
            # Find target start (FIXED VERSION)
            target_idx = len(tokens) - 1  # Default fallback
            for j, token in enumerate(tokens):
                decoded_token = self.tokenizer.decode([token]).strip()
                if decoded_token == 'M:':
                    target_idx = j + 1
                    break
            target_start_indices.append(target_idx)
        
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        target_start_indices = torch.tensor(target_start_indices, device=self.device)
        rewards = torch.tensor([1.0, 0.5], device=self.device)
        
        # Get initial reference logprobs
        with torch.no_grad():
            initial_ref_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        print(f"Initial reference logprobs: {initial_ref_logprobs.tolist()}")
        
        # Perform several training steps
        grpo_batch = GRPOBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_start_indices=target_start_indices,
            old_logprobs=initial_ref_logprobs,
            rewards=rewards,
            position_fen="test",
            task_type="policy"
        )
        
        print(f"\\nPerforming 5 training steps...")
        for step in range(5):
            # Forward and backward pass
            loss, metrics = self.trainer.compute_grpo_loss(grpo_batch)
            self.trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.trainer.optimizer.step()
            
            print(f"  Step {step}: loss = {loss.item():.6f}")
        
        # Check if reference model parameters changed
        print(f"\\nChecking reference model parameter changes...")
        param_changes = {}
        max_change = 0.0
        
        for name, param in self.ref_model.named_parameters():
            change = torch.abs(param - initial_params[name]).max().item()
            param_changes[name] = change
            max_change = max(max_change, change)
        
        # Show top 5 changes
        sorted_changes = sorted(param_changes.items(), key=lambda x: x[1], reverse=True)
        print(f"Top 5 parameter changes:")
        for name, change in sorted_changes[:5]:
            print(f"  {name}: {change:.2e}")
        
        print(f"Maximum parameter change: {max_change:.2e}")
        
        # Test reference logprobs consistency
        with torch.no_grad():
            final_ref_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        print(f"Final reference logprobs: {final_ref_logprobs.tolist()}")
        
        logprob_change = torch.abs(final_ref_logprobs - initial_ref_logprobs).max().item()
        print(f"Reference logprob change: {logprob_change:.6f}")
        
        # Success criteria
        tests_passed = {
            'parameters_unchanged': max_change < 1e-6,
            'logprobs_unchanged': logprob_change < 1e-4,
            'model_in_eval_mode': not self.ref_model.training
        }
        
        print(f"\\nTest Results:")
        all_passed = True
        for test_name, passed in tests_passed.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\\n{'✅ PASSED' if all_passed else '❌ FAILED'}: Reference Model Freezing Test")
        return all_passed
    
    def test_4_kl_divergence_calculation(self):
        """Test 4: KL Divergence Calculation"""
        
        print("\\n" + "="*80)
        print("TEST 4: KL DIVERGENCE CALCULATION")
        print("="*80)
        
        # Test with known KL divergences
        test_cases = [
            {
                'name': 'Identical distributions',
                'current': torch.tensor([-1.0, -2.0]),
                'reference': torch.tensor([-1.0, -2.0]),
                'expected_kl': 0.0
            },
            {
                'name': 'Small positive divergence',
                'current': torch.tensor([-1.1, -2.1]),
                'reference': torch.tensor([-1.0, -2.0]),
                'expected_kl': -0.1
            },
            {
                'name': 'Large negative divergence (problematic)',
                'current': torch.tensor([-10.0, -20.0]),
                'reference': torch.tensor([-1.0, -2.0]),
                'expected_kl': -10.5
            }
        ]
        
        all_passed = True
        
        for i, case in enumerate(test_cases):
            print(f"\\nTest case {i+1}: {case['name']}")
            
            current_logprobs = case['current'].to(self.device)
            ref_logprobs = case['reference'].to(self.device)
            
            # Manual KL computation
            manual_kl = (current_logprobs - ref_logprobs).mean().item()
            expected_kl = case['expected_kl']
            
            print(f"  Current logprobs: {current_logprobs.tolist()}")
            print(f"  Reference logprobs: {ref_logprobs.tolist()}")
            print(f"  Computed KL: {manual_kl:.6f}")
            print(f"  Expected KL: {expected_kl:.6f}")
            
            # Check if KL is reasonable
            kl_diff = abs(manual_kl - expected_kl)
            kl_reasonable = kl_diff < 0.01
            
            if case['name'] == 'Large negative divergence (problematic)':
                print(f"  ⚠️  WARNING: Large negative KL detected (indicates divergence)")
                if abs(manual_kl) > 5.0:
                    print(f"  ❌ KL magnitude too large: {abs(manual_kl):.1f} > 5.0")
                    all_passed = False
            else:
                status = "✅ PASS" if kl_reasonable else "❌ FAIL"
                print(f"  KL computation: {status}")
                if not kl_reasonable:
                    all_passed = False
        
        # Test KL controller
        print(f"\\nTesting Adaptive KL Controller...")
        controller = self.trainer.kl_controller
        
        initial_coef = controller.get_coefficient()
        print(f"Initial KL coefficient: {initial_coef}")
        
        # Test with reasonable KL
        controller.update(0.1)  # Small positive KL
        reasonable_coef = controller.get_coefficient()
        print(f"After small KL (0.1): {reasonable_coef}")
        
        # Test with large KL
        controller.update(10.0)  # Large KL
        large_kl_coef = controller.get_coefficient()
        print(f"After large KL (10.0): {large_kl_coef}")
        
        # Controller should increase coefficient for large KL
        controller_working = large_kl_coef >= reasonable_coef
        print(f"Controller increases coef for large KL: {'✅ PASS' if controller_working else '❌ FAIL'}")
        
        if not controller_working:
            all_passed = False
        
        print(f"\\n{'✅ PASSED' if all_passed else '❌ FAILED'}: KL Divergence Calculation Test")
        return all_passed

def main():
    """Run all component tests"""
    
    print("GRPO COMPONENT ISOLATION TESTS")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - using CPU")
    
    tester = ComponentTester()
    
    # Run tests
    test_results = {
        'tokenization': tester.test_1_tokenization_and_target_masking(),
        'logprob_computation': tester.test_2_logprob_computation(),
        'reference_freezing': tester.test_3_reference_model_freezing(),
        'kl_divergence': tester.test_4_kl_divergence_calculation()
    }
    
    # Summary
    print("\\n" + "="*80)
    print("COMPONENT TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\\n" + "="*80)
    if all_passed:
        print("🎉 ALL COMPONENT TESTS PASSED")
        print("Components are working correctly - issue may be in integration")
    else:
        print("❌ COMPONENT TESTS FAILED")
        print("Found issues in individual components that need fixing")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)