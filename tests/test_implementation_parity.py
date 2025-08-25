#!/usr/bin/env python3
"""
Implementation Parity Unit Tests

Unit tests to ensure test and production implementation parity for GRPO training.
These tests catch regressions and ensure numerical consistency.
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch


class TestImplementationParity(unittest.TestCase):
    """Unit tests for implementation parity between test and production code"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for the class"""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.tokenizer = TokenizerBridge()
        
        # Initialize models
        model_config = GPT2Config()
        cls.model = GPT2Model(model_config).to(cls.device)
        cls.ref_model = GPT2Model(model_config).to(cls.device)
        
        # Copy weights and freeze reference model
        cls.ref_model.load_state_dict(cls.model.state_dict())
        cls.ref_model.eval()
        for param in cls.ref_model.parameters():
            param.requires_grad_(False)
        
        # GRPO config
        cls.config = GRPOConfig(
            lr=1e-5,
            group_size=2,
            use_mixed_precision=False,
            use_torch_compile=False,
            kl_coef=0.01,
            device=str(cls.device)
        )
        
        cls.trainer = GRPOTrainer(cls.model, cls.ref_model, cls.config)
    
    def setUp(self):
        """Reset model state before each test"""
        # Store and restore initial model state to prevent test interference
        self.initial_model_state = self.model.state_dict()
        
    def tearDown(self):
        """Restore model state after each test"""
        self.model.load_state_dict(self.initial_model_state)
        
    def compute_logprobs_manual(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_start_indices: torch.Tensor,
        use_ref_model: bool = False
    ) -> torch.Tensor:
        """Manual logprob computation for comparison with production code"""
        
        model = self.ref_model if use_ref_model else self.model
        
        # Ensure consistent model state
        original_training_mode = model.training
        model.eval()  # Force eval mode for consistency
        
        with torch.set_grad_enabled(False):  # Always disable gradients for consistency
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Shift for autoregressive loss
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            
            # Handle attention mask
            if attention_mask is None:
                batch_size, seq_len = input_ids.shape
                shift_attention = torch.ones(batch_size, seq_len - 1, dtype=torch.long, device=input_ids.device)
            else:
                shift_attention = attention_mask[:, 1:]
            
            # Convert to log probabilities
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            
            # Gather logprobs for actual tokens
            token_logprobs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Apply target masking
            batch_size, seq_len = token_logprobs.shape
            target_mask = torch.zeros_like(token_logprobs, dtype=torch.bool)
            
            for i in range(batch_size):
                start_idx = max(0, target_start_indices[i] - 1)  # -1 for shift
                target_mask[i, start_idx:] = shift_attention[i, start_idx:].bool()
            
            # Compute mean logprobs
            masked_logprobs = token_logprobs.masked_fill(~target_mask, 0.0)
            token_counts = target_mask.sum(dim=1).clamp(min=1)
            mean_logprobs = masked_logprobs.sum(dim=1) / token_counts
        
        # Restore original training mode
        model.train(original_training_mode)
        
        return mean_logprobs
    
    def test_logprob_computation_parity(self):
        """Test that manual and production logprob computation match exactly"""
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
        
        # Test policy model parity
        manual_logprobs = self.compute_logprobs_manual(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        production_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        policy_diff = torch.abs(manual_logprobs - production_logprobs).max().item()
        self.assertLess(policy_diff, 1e-4, 
                       f"Policy model logprob difference too large: {policy_diff}")
        
        # Test reference model parity
        manual_ref_logprobs = self.compute_logprobs_manual(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
        
        production_ref_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=True
        )
        
        ref_diff = torch.abs(manual_ref_logprobs - production_ref_logprobs).max().item()
        self.assertLess(ref_diff, 1e-4, 
                       f"Reference model logprob difference too large: {ref_diff}")
    
    def test_target_detection_consistency(self):
        """Verify target detection between implementations"""
        test_cases = [
            ("P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4", "policy"),
            ("A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+result", "environment"),
        ]
        
        for text, task_type in test_cases:
            with self.subTest(task_type=task_type):
                # Test tokenizer method
                tokenizer_target = self.tokenizer.get_target_start_index(text, task_type)
                
                # Manual detection
                tokens = self.tokenizer.encode(text)
                if task_type == "policy":
                    manual_target = self._find_policy_target_manual(tokens)
                else:
                    manual_target = self._find_env_target_manual(tokens)
                
                self.assertEqual(tokenizer_target, manual_target,
                               f"Target detection mismatch for {task_type} task")
    
    def _find_policy_target_manual(self, tokens: List[int]) -> int:
        """Manual policy target detection"""
        for j in range(len(tokens) - 1):
            current_decoded = self.tokenizer.decode([tokens[j]]).strip()
            next_decoded = self.tokenizer.decode([tokens[j + 1]]).strip()
            if current_decoded == 'M' and next_decoded == ':':
                return j + 2
            elif current_decoded == 'M:':
                return j + 1
        return len(tokens) - 1
    
    def _find_env_target_manual(self, tokens: List[int]) -> int:
        """Manual environment target detection"""
        plus_count = 0
        for j in range(len(tokens)):
            current_decoded = self.tokenizer.decode([tokens[j]]).strip()
            if current_decoded == '+':
                plus_count += 1
                if plus_count == 2:
                    return j + 1  # After second +
        return len(tokens) - 1
    
    def test_model_state_consistency(self):
        """Verify model state management doesn't cause discrepancies"""
        # Create test batch
        text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4"
        tokens = self.tokenizer.encode(text)
        
        input_ids = torch.tensor([tokens], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        target_start_indices = torch.tensor([
            self.tokenizer.get_target_start_index(text, 'policy')
        ], device=self.device)
        
        # Test in different model modes
        self.model.train()
        train_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        self.model.eval()  
        eval_logprobs = self.trainer.compute_logprobs(
            input_ids, attention_mask, target_start_indices, use_ref_model=False
        )
        
        # Should be identical due to gradient context management
        mode_diff = torch.abs(train_logprobs - eval_logprobs).max().item()
        self.assertLess(mode_diff, 1e-6,
                       f"Model mode should not affect logprob computation: {mode_diff}")
        
        # Test with gradients enabled/disabled externally
        with torch.set_grad_enabled(True):
            grad_enabled_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=False
            )
        
        with torch.set_grad_enabled(False):
            grad_disabled_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=False
            )
        
        # Should be identical due to internal gradient management
        grad_diff = torch.abs(grad_enabled_logprobs - grad_disabled_logprobs).max().item()
        self.assertLess(grad_diff, 1e-6,
                       f"External gradient context should not affect logprob computation: {grad_diff}")
    
    def test_numerical_precision_stability(self):
        """Test that computations are numerically stable across multiple calls"""
        text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4"
        tokens = self.tokenizer.encode(text)
        
        input_ids = torch.tensor([tokens], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        target_start_indices = torch.tensor([
            self.tokenizer.get_target_start_index(text, 'policy')
        ], device=self.device)
        
        # Multiple calls should yield identical results
        results = []
        for _ in range(10):
            logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=False
            )
            results.append(logprobs)
        
        # All results should be identical
        for i in range(1, len(results)):
            diff = torch.abs(results[i] - results[0]).max().item()
            self.assertEqual(diff, 0.0, 
                           f"Multiple calls should yield identical results, got diff: {diff}")


if __name__ == '__main__':
    unittest.main()