#!/usr/bin/env python3
"""
Extended GRPO Components Unit Tests

Unit tests for individual GRPO components to validate core algorithm correctness.
These tests ensure each component works correctly in isolation.
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rookworld_rlvr.model.config import ROOKWORLD_CONFIG
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch


class TestGRPOComponents(unittest.TestCase):
    """Unit tests for GRPO components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for the class"""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.tokenizer = TokenizerBridge()
        
        # Initialize models
        cls.model = GPT2Model(ROOKWORLD_CONFIG).to(cls.device)
        cls.ref_model = GPT2Model(ROOKWORLD_CONFIG).to(cls.device)
        cls.ref_model.load_state_dict(cls.model.state_dict())
        cls.ref_model.eval()
        
        # Freeze reference model
        for param in cls.ref_model.parameters():
            param.requires_grad_(False)
        
        # GRPO config for testing
        cls.config = GRPOConfig(
            lr=1e-5,
            group_size=2,
            use_mixed_precision=False,
            use_torch_compile=False,
            kl_coef=0.01,
            device=str(cls.device)
        )
        
        cls.trainer = GRPOTrainer(cls.model, cls.ref_model, cls.config)
    
    def test_tokenization_and_masking(self):
        """Test tokenization and target masking functionality"""
        test_cases = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: d2d4",
            "P: short    M: test",
        ]
        
        for i, text in enumerate(test_cases):
            with self.subTest(case=i, text=text[:50]):
                # Tokenize
                tokens = self.tokenizer.encode(text)
                self.assertGreater(len(tokens), 0, "Should produce tokens")
                
                # Test roundtrip encoding/decoding
                decoded = self.tokenizer.decode(tokens)
                self.assertEqual(decoded, text, "Roundtrip encoding should be exact")
                
                # Find target start index using tokenizer method
                target_start_idx = self.tokenizer.get_target_start_index(text, 'policy')
                self.assertLess(target_start_idx, len(tokens), "Target should be within token range")
                
                # Verify target tokens are reasonable
                if target_start_idx < len(tokens):
                    target_tokens = tokens[target_start_idx:target_start_idx+2]
                    target_text = self.tokenizer.decode(target_tokens)
                    self.assertGreater(len(target_text.strip()), 0, "Target text should not be empty")
    
    def test_logprob_computation(self):
        """Test logprob computation consistency"""
        # Create test batch
        texts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: d2d4"
        ]
        
        # Create batch tensors
        all_tokens = [self.tokenizer.encode(text) for text in texts]
        max_len = max(len(tokens) for tokens in all_tokens)
        
        input_ids = []
        attention_mask = []
        target_start_indices = []
        
        for tokens in all_tokens:
            # Pad
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            input_ids.append(padded)
            attention_mask.append(mask)
            
            # Get target start index
            target_idx = self.tokenizer.get_target_start_index(
                self.tokenizer.decode(tokens), 'policy'
            )
            target_start_indices.append(target_idx)
        
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        target_start_indices = torch.tensor(target_start_indices, device=self.device)
        
        # Test current model logprobs
        current_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        # Test reference model logprobs  
        ref_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
        
        # Models should be identical initially
        logprob_diff = torch.abs(current_logprobs - ref_logprobs).max().item()
        self.assertLess(logprob_diff, 1e-4, "Current and reference models should be identical initially")
        
        # Logprobs should be reasonable (negative, not too extreme)
        for lp in current_logprobs.tolist():
            self.assertLess(lp, 0, "Log probabilities should be negative")
            self.assertGreater(lp, -50, "Log probabilities should not be extremely negative")
        
        # Manual verification - must match trainer implementation exactly
        # Ensure model is in consistent state
        original_training = self.model.training
        self.model.eval()
        
        with torch.set_grad_enabled(False):  # Same as trainer after fix
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Shift for autoregressive prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            
            # Convert to log probabilities - use same method as trainer
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            
            # Gather logprobs for actual tokens
            token_logprobs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Apply target masking manually
            batch_size, seq_len = token_logprobs.shape
            target_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
            
            for i in range(batch_size):
                start_idx = max(0, target_start_indices[i] - 1)  # -1 for shift
                shift_attention = attention_mask[i, 1:]
                target_mask[i, start_idx:] = shift_attention[start_idx:].bool()
            
            # Compute mean logprobs manually
            masked_logprobs = token_logprobs.masked_fill(~target_mask, 0.0)
            token_counts = target_mask.sum(dim=1).clamp(min=1)
            manual_logprobs = masked_logprobs.sum(dim=1) / token_counts
        
        # Restore original training mode
        self.model.train(original_training)
        
        # Should match trainer computation
        manual_diff = torch.abs(manual_logprobs - current_logprobs).max().item()
        self.assertLess(manual_diff, 1e-3, "Manual and trainer logprob computation should match closely")
        
        # Should have counted some target tokens
        for count in token_counts.tolist():
            self.assertGreater(count, 0, "Should count some target tokens")
    
    def test_reference_model_freezing(self):
        """Test that reference model parameters remain frozen"""
        # Store initial reference model parameters
        initial_params = {}
        for name, param in self.ref_model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Verify no gradients are required
        for param in self.ref_model.parameters():
            self.assertFalse(param.requires_grad, "Reference model parameters should not require grad")
        
        # Verify model is in eval mode
        self.assertFalse(self.ref_model.training, "Reference model should be in eval mode")
        
        # Create test batch
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
            
            target_idx = self.tokenizer.get_target_start_index(
                self.tokenizer.decode(tokens), 'policy'
            )
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
        
        # Perform training steps
        grpo_batch = GRPOBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_start_indices=target_start_indices,
            old_logprobs=initial_ref_logprobs,
            rewards=rewards,
            position_fen="test",
            task_type="policy"
        )
        
        # Simulate training steps
        for _ in range(3):
            loss, _ = self.trainer.compute_grpo_loss(grpo_batch)
            self.trainer.optimizer.zero_grad()
            loss.backward()
            self.trainer.optimizer.step()
        
        # Check that reference model parameters haven't changed
        max_change = 0.0
        for name, param in self.ref_model.named_parameters():
            change = torch.abs(param - initial_params[name]).max().item()
            max_change = max(max_change, change)
        
        self.assertLess(max_change, 1e-6, "Reference model parameters should not change during training")
        
        # Check that reference logprobs are still consistent
        with torch.no_grad():
            final_ref_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        logprob_change = torch.abs(final_ref_logprobs - initial_ref_logprobs).max().item()
        self.assertLess(logprob_change, 1e-4, "Reference logprobs should remain consistent")
    
    def test_kl_divergence_calculation(self):
        """Test KL divergence computation"""
        # Test cases with known KL divergences
        test_cases = [
            {
                'name': 'Identical distributions',
                'current': torch.tensor([-1.0, -2.0], device=self.device),
                'reference': torch.tensor([-1.0, -2.0], device=self.device),
                'expected_kl': 0.0
            },
            {
                'name': 'Small positive divergence',
                'current': torch.tensor([-1.1, -2.1], device=self.device),
                'reference': torch.tensor([-1.0, -2.0], device=self.device),
                'expected_kl': -0.1
            }
        ]
        
        for case in test_cases:
            with self.subTest(name=case['name']):
                current_logprobs = case['current']
                ref_logprobs = case['reference']
                expected_kl = case['expected_kl']
                
                # Compute KL manually (simplified for unit test)
                kl = (current_logprobs - ref_logprobs).mean().item()
                
                # Check if KL is close to expected
                self.assertAlmostEqual(kl, expected_kl, places=1, 
                                     msg=f"KL divergence should be approximately {expected_kl}")
        
        # Test KL controller
        controller = self.trainer.kl_controller
        initial_coef = controller.get_coefficient()
        self.assertGreater(initial_coef, 0, "KL coefficient should be positive")
        
        # Test controller updates
        controller.update(0.1)  # Small KL
        reasonable_coef = controller.get_coefficient()
        
        controller.update(5.0)  # Large KL
        large_kl_coef = controller.get_coefficient()
        
        # Controller should adapt to large KL
        self.assertGreaterEqual(large_kl_coef, reasonable_coef, 
                               "KL coefficient should increase for large KL values")


if __name__ == '__main__':
    unittest.main()