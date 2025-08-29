#!/usr/bin/env python3
"""
Numerical Golden Standard Tests

Creates reference numerical outputs for critical functions to detect
any changes introduced by performance optimizations.
"""

import torch
import time
import json
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.config import GRPOConfig
from rookworld_rlvr.loader import load_rookworld_model
from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.grpo import (
    compute_log_probs, 
    ReferenceModel, 
    grpo_loss, 
    compute_advantages, 
    create_prompt_mask
)
from rookworld_rlvr.reward_scorer import compute_grpo_rewards


class GoldenStandardTester:
    """Creates and validates against golden standard numerical outputs."""
    
    def __init__(self, device: str = "cuda", seed: int = 42):
        self.device = device  
        self.seed = seed
        self.golden_file = Path(__file__).parent / "golden_standard.json"
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
    
    def tensor_to_json_safe(self, tensor):
        """Convert tensor to JSON-safe format with precision."""
        if isinstance(tensor, torch.Tensor):
            return {
                'values': tensor.detach().cpu().numpy().tolist(),
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype)
            }
        return tensor
    
    def create_test_data(self):
        """Create reproducible test data for golden standard."""
        # Fixed test sequences for reproducible results
        batch_size, seq_len = 2, 20
        
        # Use fixed random seed for reproducible "random" data
        torch.manual_seed(self.seed)
        sequences = torch.randint(100, 1000, (batch_size, seq_len), device=self.device)
        attention_masks = torch.ones_like(sequences)
        rewards = torch.tensor([0.8, 0.6], device=self.device)
        prompt_lengths = torch.tensor([8, 10])
        
        return sequences, attention_masks, rewards, prompt_lengths
    
    def test_log_prob_computation(self, model) -> Dict[str, Any]:
        """Test log probability computation with fixed inputs."""
        print("Creating golden standard for log prob computation...")
        
        sequences, attention_masks, _, _ = self.create_test_data()
        
        # Compute log probs
        with torch.no_grad():
            log_probs = compute_log_probs(model, sequences, attention_masks)
        
        return {
            'input_sequences': self.tensor_to_json_safe(sequences),
            'attention_masks': self.tensor_to_json_safe(attention_masks), 
            'output_log_probs': self.tensor_to_json_safe(log_probs),
            'log_prob_mean': float(log_probs.mean().item()),
            'log_prob_std': float(log_probs.std().item())
        }
    
    def test_grpo_loss_computation(self, model) -> Dict[str, Any]:
        """Test GRPO loss computation with fixed inputs."""
        print("Creating golden standard for GRPO loss...")
        
        sequences, attention_masks, rewards, prompt_lengths = self.create_test_data()
        
        # Create reference model
        ref_model = ReferenceModel(model)
        
        with torch.no_grad():
            # Compute log probs
            policy_log_probs = compute_log_probs(model, sequences, attention_masks)
            ref_log_probs = ref_model.compute_log_probs(sequences, attention_masks)
            
            # Ensure same device
            if ref_log_probs.device != self.device:
                ref_log_probs = ref_log_probs.to(self.device)
            
            # Compute advantages and masks
            advantages = compute_advantages(rewards, group_size=1)
            prompt_mask = create_prompt_mask(sequences, prompt_lengths)
            
            # Compute loss
            loss, metrics = grpo_loss(
                policy_log_probs,
                ref_log_probs,
                advantages,
                prompt_mask,
                kl_coef=0.02,
                clip_range=0.2
            )
        
        return {
            'policy_log_probs': self.tensor_to_json_safe(policy_log_probs),
            'ref_log_probs': self.tensor_to_json_safe(ref_log_probs),
            'advantages': self.tensor_to_json_safe(advantages),
            'loss_value': float(loss.item()),
            'pg_loss': float(metrics['pg_loss']),
            'kl_divergence': float(metrics['kl_div'])
        }
    
    def test_reward_scoring_computation(self) -> Dict[str, Any]:
        """Test reward scoring with fixed inputs."""
        print("Creating golden standard for reward scoring...")
        
        # Fixed test prompts and completions
        test_cases = [
            (
                "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "M: e2e4 d2d4 g1f3 c2c4 b1c3  E: 0.3 0.35 0.28 0.32 0.29  B: e2e4"
            ),
            (
                "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
            )
        ]
        
        prompts = [case[0] for case in test_cases]
        completions = [case[1] for case in test_cases]
        
        advantages, details = compute_grpo_rewards(
            prompts,
            completions,
            group_size=1,
            reward_shaping="graduated",
            verbose=False
        )
        
        return {
            'test_prompts': prompts,
            'test_completions': completions,
            'advantages': advantages.tolist() if hasattr(advantages, 'tolist') else advantages,
            'reward_details': [
                {
                    'shaped_reward': d.shaped_reward,
                    'format_valid': d.format_valid,
                    'task_type': d.task_type
                } for d in details
            ]
        }
    
    def test_model_generation(self, model) -> Dict[str, Any]:
        """Test model generation with fixed inputs."""
        print("Creating golden standard for model generation...")
        
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Fixed test prompt
        test_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        prompt_tokens = tokenizer.encode(test_prompt)
        input_tensor = torch.tensor([prompt_tokens], device=self.device)
        
        # Generate with fixed parameters
        with torch.no_grad():
            torch.manual_seed(self.seed)  # Ensure deterministic generation
            generated = model.generate(
                input_tensor,
                max_new_tokens=50,  # Short for reproducibility
                temperature=0.01,   # Low temperature for consistency
                top_k=1            # Deterministic
            )
        
        generated_tokens = generated[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_tokens)
        completion = generated_text[len(test_prompt):].strip()
        
        return {
            'input_prompt': test_prompt,
            'input_tokens': prompt_tokens,
            'generated_tokens': generated_tokens,
            'generated_text': generated_text,
            'completion': completion,
            'generation_params': {
                'max_new_tokens': 50,
                'temperature': 0.01,
                'top_k': 1,
                'seed': self.seed
            }
        }
    
    def create_golden_standard(self) -> Dict[str, Any]:
        """Create complete golden standard test outputs."""
        print("🏆 Creating Golden Standard Test Outputs")
        print("=" * 60)
        
        # Load model
        model = load_rookworld_model(device=self.device)
        
        golden_standard = {
            'metadata': {
                'timestamp': time.time(),
                'torch_version': torch.__version__,
                'device': self.device,
                'seed': self.seed
            }
        }
        
        if self.device == "cuda":
            golden_standard['metadata']['gpu_name'] = torch.cuda.get_device_name()
            golden_standard['metadata']['cuda_version'] = torch.version.cuda
        
        # Create golden standards for each component
        golden_standard['log_prob_computation'] = self.test_log_prob_computation(model)
        golden_standard['grpo_loss_computation'] = self.test_grpo_loss_computation(model)
        golden_standard['reward_scoring'] = self.test_reward_scoring_computation()
        golden_standard['model_generation'] = self.test_model_generation(model)
        
        return golden_standard
    
    def save_golden_standard(self, golden_data: Dict[str, Any]) -> None:
        """Save golden standard to file."""
        with open(self.golden_file, 'w') as f:
            json.dump(golden_data, f, indent=2)
        print(f"✅ Golden standard saved to {self.golden_file}")
    
    def load_golden_standard(self) -> Dict[str, Any]:
        """Load existing golden standard."""
        if self.golden_file.exists():
            with open(self.golden_file, 'r') as f:
                return json.load(f)
        return {}
    
    def validate_against_golden(self, current_data: Dict[str, Any], tolerance: float = 1e-4) -> bool:
        """Validate current outputs against golden standard."""
        golden = self.load_golden_standard()
        if not golden:
            print("⚠️ No golden standard found - create one first")
            return False
        
        print("🔍 Validating Against Golden Standard")
        print("=" * 50)
        
        all_passed = True
        
        # Compare numerical outputs with tolerance
        numerical_tests = [
            ('log_prob_computation.log_prob_mean', 'Log Prob Mean'),
            ('log_prob_computation.log_prob_std', 'Log Prob Std'),
            ('grpo_loss_computation.loss_value', 'GRPO Loss'),
            ('grpo_loss_computation.pg_loss', 'Policy Gradient Loss'),
            ('grpo_loss_computation.kl_divergence', 'KL Divergence'),
        ]
        
        for path, name in numerical_tests:
            try:
                golden_val = golden
                current_val = current_data
                
                for key in path.split('.'):
                    golden_val = golden_val[key]
                    current_val = current_val[key]
                
                diff = abs(golden_val - current_val)
                rel_diff = diff / abs(golden_val) if abs(golden_val) > 0 else diff
                
                if rel_diff <= tolerance:
                    print(f"✅ {name}: {rel_diff:.2e} (within tolerance)")
                else:
                    print(f"❌ {name}: {rel_diff:.2e} (exceeds tolerance {tolerance})")
                    print(f"   Golden: {golden_val:.6f}")
                    print(f"   Current: {current_val:.6f}")
                    all_passed = False
                    
            except KeyError as e:
                print(f"⚠️ {name}: Missing key {e}")
                all_passed = False
        
        # Compare text outputs
        text_tests = [
            ('model_generation.completion', 'Model Generation'),
        ]
        
        for path, name in text_tests:
            try:
                golden_val = golden
                current_val = current_data
                
                for key in path.split('.'):
                    golden_val = golden_val[key]
                    current_val = current_val[key]
                
                if golden_val == current_val:
                    print(f"✅ {name}: Exact match")
                else:
                    print(f"⚠️ {name}: Text differs")
                    print(f"   Golden:  '{golden_val[:50]}...'")
                    print(f"   Current: '{current_val[:50]}...'")
                    # Text differences are warnings, not failures for now
                    
            except KeyError as e:
                print(f"⚠️ {name}: Missing key {e}")
        
        if all_passed:
            print("\n✅ All numerical tests passed within tolerance")
        else:
            print("\n❌ Some numerical tests failed - check for regressions")
        
        return all_passed


def test_create_golden_standard():
    """Main test to create golden standard."""
    tester = GoldenStandardTester()
    
    golden_data = tester.create_golden_standard()
    tester.save_golden_standard(golden_data)
    
    print("\n🏆 Golden standard test outputs created")
    return golden_data


def test_validate_golden_standard():
    """Test validation against existing golden standard."""
    tester = GoldenStandardTester()
    
    # Create current outputs  
    current_data = tester.create_golden_standard()
    
    # Validate against saved golden standard
    passed = tester.validate_against_golden(current_data)
    
    assert passed, "Golden standard validation failed"
    print("✅ Golden standard validation passed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['create', 'validate'], default='create')
    args = parser.parse_args()
    
    if args.mode == 'create':
        test_create_golden_standard()
    elif args.mode == 'validate':
        test_validate_golden_standard()