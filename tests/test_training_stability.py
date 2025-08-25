#!/usr/bin/env python3
"""
Comprehensive Training Stability Tests

Tests to validate fixes for KL divergence explosion and training instability issues.
These tests should pass before deploying training fixes to production.
"""

import pytest
import torch
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch, GRPOTrainingStep
from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model


class TestTrainingStability:
    """Comprehensive tests for training stability fixes"""

    def setup_method(self):
        """Set up test environment"""
        self.logger = logging.getLogger("TrainingStabilityTest")
        logging.basicConfig(level=logging.INFO)
        
        # Test device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def create_test_model(self) -> GPT2Model:
        """Create a small test model for stability testing"""
        config = GPT2Config(
            vocab_size=1000,  # Small vocab for testing
            n_positions=128,   # Short sequences
            n_embd=128,        # Small embedding dim
            n_head=4,          # Few attention heads  
            n_layer=2          # Shallow model
        )
        model = GPT2Model(config).to(self.device)
        return model
    
    def create_test_batch_with_extreme_rewards(self, group_size: int = 4) -> GRPOBatch:
        """Create test batch with extreme rewards to simulate the problematic scenario"""
        seq_len = 32
        
        # Create batch tensors
        input_ids = torch.randint(0, 1000, (group_size, seq_len), device=self.device)
        attention_mask = torch.ones((group_size, seq_len), device=self.device)
        target_start_indices = torch.full((group_size,), seq_len // 2, device=self.device)
        old_logprobs = torch.randn(group_size, device=self.device) * 0.1  # Small initial logprobs
        
        # Create extreme negative rewards (simulating parsing failures)
        extreme_rewards = torch.tensor([-1.0] * group_size, device=self.device)
        
        return GRPOBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_start_indices=target_start_indices,
            old_logprobs=old_logprobs,
            rewards=extreme_rewards,
            position_fen="test_position",
            task_type="policy"
        )
    
    def test_nan_detection_and_recovery(self):
        """Test that NaN losses are detected and handled gracefully"""
        # Create conservative config
        config = GRPOConfig(
            lr=1e-6,
            kl_coef=0.001,
            clip_range=0.1,
            device=self.device,
            enable_recovery=True
        )
        
        model = self.create_test_model()
        ref_model = self.create_test_model()
        ref_model.load_state_dict(model.state_dict())
        
        trainer = GRPOTrainer(model, ref_model, config)
        
        # Create batch with extreme rewards
        batch = self.create_test_batch_with_extreme_rewards()
        step_data = GRPOTrainingStep(groups=[batch])
        
        # Run training step
        metrics = trainer.training_step(step_data)
        
        # Verify NaN handling
        if 'nan_skip' in metrics:
            self.logger.info("✅ NaN detected and handled gracefully")
            assert metrics['nan_skip'] == 1
            assert 'nan_skip_count' in metrics
        else:
            # If no NaN, verify loss is finite
            assert torch.isfinite(torch.tensor(metrics['loss']))
            self.logger.info("✅ Training step completed without NaN")
    
    def test_graduated_reward_impact(self):
        """Test that graduated rewards (vs binary) improve training stability"""
        config = GRPOConfig(
            lr=1e-6,
            kl_coef=0.001,
            clip_range=0.1,
            device=self.device
        )
        
        model1 = self.create_test_model()
        ref_model1 = self.create_test_model() 
        ref_model1.load_state_dict(model1.state_dict())
        trainer1 = GRPOTrainer(model1, ref_model1, config)
        
        model2 = self.create_test_model()
        model2.load_state_dict(model1.state_dict())  # Same initialization
        ref_model2 = self.create_test_model()
        ref_model2.load_state_dict(model2.state_dict())
        trainer2 = GRPOTrainer(model2, ref_model2, config)
        
        # Create batch with binary extreme rewards
        batch1 = self.create_test_batch_with_extreme_rewards()
        
        # Create batch with graduated rewards
        batch2 = self.create_test_batch_with_extreme_rewards()
        batch2.rewards = torch.tensor([-0.3, -0.5, -0.2, -0.4], device=self.device)  # Less extreme
        
        # Test both scenarios
        step_data1 = GRPOTrainingStep(groups=[batch1])
        step_data2 = GRPOTrainingStep(groups=[batch2])
        
        metrics1 = trainer1.training_step(step_data1)
        metrics2 = trainer2.training_step(step_data2)
        
        # Graduated rewards should be more stable
        nan_count1 = metrics1.get('nan_skip', 0)
        nan_count2 = metrics2.get('nan_skip', 0)
        
        self.logger.info(f"Binary rewards NaN count: {nan_count1}")
        self.logger.info(f"Graduated rewards NaN count: {nan_count2}")
        
        # Graduated rewards should be more stable (less/no NaNs)
        assert nan_count2 <= nan_count1
        
    def test_learning_rate_sensitivity(self):
        """Test training stability across different learning rates"""
        learning_rates = [1e-7, 1e-6, 1e-5]
        results = {}
        
        for lr in learning_rates:
            config = GRPOConfig(
                lr=lr,
                kl_coef=0.001,
                clip_range=0.1,
                device=self.device
            )
            
            model = self.create_test_model()
            ref_model = self.create_test_model()
            ref_model.load_state_dict(model.state_dict())
            trainer = GRPOTrainer(model, ref_model, config)
            
            # Test with moderate negative rewards
            batch = self.create_test_batch_with_extreme_rewards()
            batch.rewards = torch.tensor([-0.5, -0.3, -0.7, -0.4], device=self.device)
            
            step_data = GRPOTrainingStep(groups=[batch])
            metrics = trainer.training_step(step_data)
            
            results[lr] = {
                'nan_skip': metrics.get('nan_skip', 0),
                'loss': metrics.get('loss', 0.0),
                'finite': torch.isfinite(torch.tensor(metrics.get('loss', float('inf'))))
            }
            
            self.logger.info(f"LR {lr}: NaN skip: {results[lr]['nan_skip']}, Loss finite: {results[lr]['finite']}")
        
        # Lower learning rates should be more stable
        assert results[1e-7]['nan_skip'] <= results[1e-5]['nan_skip']
        
    def test_kl_warmup_effectiveness(self):
        """Test that KL warmup prevents early training instability"""
        # Test without warmup
        config_no_warmup = GRPOConfig(
            lr=1e-6,
            kl_coef=0.1,  # High KL coefficient
            kl_warmup_steps=0,
            kl_warmup_factor=1.0,
            device=self.device
        )
        
        # Test with warmup  
        config_with_warmup = GRPOConfig(
            lr=1e-6,
            kl_coef=0.1,  # High KL coefficient
            kl_warmup_steps=100,
            kl_warmup_factor=0.0,  # Start with 0 KL
            device=self.device
        )
        
        for config, label in [(config_no_warmup, "No Warmup"), (config_with_warmup, "With Warmup")]:
            model = self.create_test_model()
            ref_model = self.create_test_model()
            ref_model.load_state_dict(model.state_dict())
            trainer = GRPOTrainer(model, ref_model, config)
            
            # Force step count to 0 for warmup test
            trainer.step_count = 0
            
            batch = self.create_test_batch_with_extreme_rewards()
            batch.rewards = torch.tensor([-0.5, -0.3, -0.7, -0.4], device=self.device)
            
            step_data = GRPOTrainingStep(groups=[batch])
            metrics = trainer.training_step(step_data)
            
            nan_skip = metrics.get('nan_skip', 0)
            loss_finite = torch.isfinite(torch.tensor(metrics.get('loss', float('inf'))))
            
            self.logger.info(f"{label}: NaN skip: {nan_skip}, Loss finite: {loss_finite}")
            
            if label == "With Warmup":
                # Warmup should improve stability
                assert nan_skip == 0 or loss_finite, "KL warmup should prevent immediate instability"
    
    def test_batch_reward_distribution_impact(self):
        """Test how different reward distributions affect training stability"""
        config = GRPOConfig(
            lr=1e-6,
            kl_coef=0.001,
            clip_range=0.1,
            device=self.device
        )
        
        reward_distributions = {
            "all_negative": [-1.0, -1.0, -1.0, -1.0],
            "mixed": [-1.0, -0.5, 0.2, 0.8],
            "mostly_positive": [0.1, 0.3, 0.7, -0.2],
            "all_positive": [0.2, 0.4, 0.6, 0.8]
        }
        
        results = {}
        
        for dist_name, rewards in reward_distributions.items():
            model = self.create_test_model()
            ref_model = self.create_test_model()
            ref_model.load_state_dict(model.state_dict())
            trainer = GRPOTrainer(model, ref_model, config)
            
            batch = self.create_test_batch_with_extreme_rewards()
            batch.rewards = torch.tensor(rewards, device=self.device)
            
            step_data = GRPOTrainingStep(groups=[batch])
            metrics = trainer.training_step(step_data)
            
            results[dist_name] = {
                'nan_skip': metrics.get('nan_skip', 0),
                'loss': metrics.get('loss', 0.0),
                'finite': torch.isfinite(torch.tensor(metrics.get('loss', float('inf'))))
            }
            
            self.logger.info(f"{dist_name}: NaN: {results[dist_name]['nan_skip']}, Finite: {results[dist_name]['finite']}")
        
        # All negative rewards should be least stable
        assert results["all_negative"]['nan_skip'] >= results["mixed"]['nan_skip']
        assert results["all_negative"]['nan_skip'] >= results["mostly_positive"]['nan_skip']
    
    def test_final_metrics_initialization_bug_fix(self):
        """Test that the final_metrics UnboundLocalError bug is fixed"""
        config = GRPOConfig(
            lr=1e-3,  # High LR to force NaN
            kl_coef=1.0,  # High KL to force NaN
            device=self.device
        )
        
        model = self.create_test_model()
        ref_model = self.create_test_model()
        ref_model.load_state_dict(model.state_dict())
        trainer = GRPOTrainer(model, ref_model, config)
        
        # Create batch with extreme rewards to force NaN
        batch = self.create_test_batch_with_extreme_rewards()
        step_data = GRPOTrainingStep(groups=[batch])
        
        # This should not crash with UnboundLocalError
        try:
            metrics = trainer.training_step(step_data)
            self.logger.info("✅ No UnboundLocalError crash")
            
            # Should return valid metrics dict even on NaN
            assert isinstance(metrics, dict)
            if metrics.get('nan_skip', 0) > 0:
                assert 'nan_skip_count' in metrics
                assert 'consecutive_nan_count' in metrics
                
        except UnboundLocalError as e:
            if 'final_metrics' in str(e):
                pytest.fail("final_metrics UnboundLocalError bug still exists!")
            else:
                raise


def run_stability_tests():
    """Run all stability tests"""
    test_instance = TestTrainingStability()
    test_instance.setup_method()
    
    tests = [
        test_instance.test_nan_detection_and_recovery,
        test_instance.test_graduated_reward_impact,
        test_instance.test_learning_rate_sensitivity,
        test_instance.test_kl_warmup_effectiveness,
        test_instance.test_batch_reward_distribution_impact,
        test_instance.test_final_metrics_initialization_bug_fix
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n🧪 Running {test.__name__}...")
            test()
            print(f"✅ {test.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_stability_tests()
    if success:
        print("🎉 All training stability tests passed!")
        exit(0)
    else:
        print("💥 Some training stability tests failed!")
        exit(1)