#!/usr/bin/env python3
"""
Performance Baseline Tests

Establishes performance baselines before implementing optimizations.
This test must pass with consistent metrics before any optimization work begins.
"""

import torch
import time
import json
import sys
import os
from pathlib import Path
import gc
import numpy as np
from typing import Dict, Any, List

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


class PerformanceBaseline:
    """Performance baseline measurement and validation."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.baseline_file = Path(__file__).parent / "performance_baseline.json"
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        if self.device == "cuda":
            torch.cuda.synchronize()
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            }
        return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}
    
    def calculate_model_flops(self, model, seq_len: int = 150) -> int:
        """
        Calculate theoretical FLOPs for a forward pass through the model.
        Based on GPT-2 architecture: 6 * n_params * seq_len
        """
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Standard formula for transformer FLOPs per forward pass
        # 6 * n_params * seq_len (accounts for attention, MLP, etc.)
        flops_per_forward = 6 * n_params * seq_len
        return flops_per_forward
    
    def get_gpu_peak_flops(self) -> float:
        """
        Get theoretical peak FLOPS for the GPU.
        RTX 4090: ~165 TFLOPS (BF16), ~83 TFLOPS (FP32)
        """
        if self.device != "cuda":
            return 0
        
        gpu_name = torch.cuda.get_device_name().lower()
        
        # Peak TFLOPS for common GPUs (BF16/Tensor Core performance)
        gpu_peak_flops = {
            'rtx 4090': 165e12,      # 165 TFLOPS BF16
            'rtx 3090': 71e12,       # 71 TFLOPS BF16  
            'a100': 312e12,          # 312 TFLOPS BF16
            'v100': 125e12,          # 125 TFLOPS FP16
            'rtx 4080': 122e12,      # 122 TFLOPS BF16
            'rtx 3080': 58e12,       # 58 TFLOPS BF16
        }
        
        for gpu, flops in gpu_peak_flops.items():
            if gpu in gpu_name:
                return flops
        
        # Default fallback (conservative estimate)
        return 50e12  # 50 TFLOPS
    
    def calculate_mfu(self, model_flops: int, elapsed_time: float, batch_size: int = 1) -> float:
        """
        Calculate Model FLOPs Utilization (MFU).
        
        MFU = (Actual FLOPs / Peak FLOPs) * 100%
        Where Actual FLOPs = model_flops * batch_size / elapsed_time
        """
        peak_flops = self.get_gpu_peak_flops()
        if peak_flops == 0:
            return 0.0
        
        actual_flops_per_second = (model_flops * batch_size) / elapsed_time
        mfu = (actual_flops_per_second / peak_flops) * 100
        return mfu
    
    def benchmark_model_loading(self) -> Dict[str, Any]:
        """Benchmark model loading performance."""
        print("📊 Benchmarking model loading...")
        
        start_time = time.time()
        model = load_rookworld_model(device=self.device)
        load_time = time.time() - start_time
        
        memory_stats = self.get_memory_stats()
        
        # Test model functionality
        test_input = torch.randint(0, 1000, (1, 10), device=self.device)
        
        with torch.no_grad():
            start_inference = time.time()
            output = model(test_input)
            inference_time = time.time() - start_inference
        
        return {
            'load_time_seconds': load_time,
            'inference_time_ms': inference_time * 1000,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'memory_after_load': memory_stats
        }
    
    def benchmark_data_loading(self, n_samples: int = 100) -> Dict[str, Any]:
        """Benchmark dataset loading and preprocessing."""
        print("📊 Benchmarking data loading...")
        
        start_time = time.time()
        samples = load_and_prepare_samples(n_samples=n_samples, seed=42)
        load_time = time.time() - start_time
        
        # Analyze sample distribution
        p_tasks = [s for s in samples if s[0] == 'P']
        a_tasks = [s for s in samples if s[0] == 'A']
        
        return {
            'load_time_seconds': load_time,
            'samples_loaded': len(samples),
            'p_task_count': len(p_tasks),
            'a_task_count': len(a_tasks),
            'p_task_ratio': len(p_tasks) / len(samples),
            'avg_prompt_length': np.mean([len(s[1]) for s in samples]),
            'avg_completion_length': np.mean([len(s[2]) for s in samples])
        }
    
    def benchmark_generation_step(self, model, samples: List, k_samples: int = 8) -> Dict[str, Any]:
        """Benchmark a single generation step."""
        print("📊 Benchmarking generation step...")
        
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Use first few samples for testing
        test_samples = samples[:2]
        prompts = [s[1] for s in test_samples]
        
        # Tokenize prompts
        encoded_prompts = []
        for prompt in prompts:
            encoded = tokenizer.encode(prompt)
            encoded_prompts.append(torch.tensor(encoded, device=self.device).unsqueeze(0))
        
        # Benchmark generation
        start_time = time.time()
        memory_before = self.get_memory_stats()
        
        generations = []
        for prompt_tensor in encoded_prompts:
            with torch.no_grad():
                for k in range(k_samples):
                    generated = model.generate(
                        prompt_tensor,
                        max_new_tokens=100,  # Standard test length
                        temperature=0.8,
                        top_k=50,
                        top_p=0.95
                    )
                    generations.append(generated)
        
        generation_time = time.time() - start_time
        memory_after = self.get_memory_stats()
        
        # Calculate MFU for generation
        avg_seq_len = sum(len(g[0]) for g in generations) / len(generations)
        model_flops = self.calculate_model_flops(model, int(avg_seq_len))
        mfu = self.calculate_mfu(model_flops, generation_time, len(generations))
        peak_flops_tflops = self.get_gpu_peak_flops() / 1e12
        
        return {
            'total_generation_time_seconds': generation_time,
            'generations_per_second': len(generations) / generation_time,
            'memory_increase_gb': memory_after['allocated_gb'] - memory_before['allocated_gb'],
            'samples_tested': len(test_samples),
            'k_samples': k_samples,
            'total_generations': len(generations),
            'average_sequence_length': avg_seq_len,
            'model_flops_per_forward': model_flops,
            'mfu_percent': mfu,
            'gpu_peak_tflops': peak_flops_tflops,
            'actual_tflops': (model_flops * len(generations) / generation_time) / 1e12
        }
    
    def benchmark_grpo_computation(self, model, samples: List, k_samples: int = 8) -> Dict[str, Any]:
        """Benchmark GRPO loss computation."""
        print("📊 Benchmarking GRPO computation...")
        
        # Create test data
        batch_size = 4
        seq_len = 150
        
        # Generate dummy sequences
        sequences = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)
        attention_masks = torch.ones_like(sequences)
        rewards = torch.rand(batch_size, device=self.device)
        prompt_lengths = torch.randint(20, 50, (batch_size,))
        
        # Create reference model
        ref_model = ReferenceModel(model)
        
        start_time = time.time()
        memory_before = self.get_memory_stats()
        
        # Compute advantages
        advantages = compute_advantages(rewards, group_size=k_samples)
        
        # Compute log probabilities  
        policy_log_probs = compute_log_probs(model, sequences, attention_masks)
        ref_log_probs = ref_model.compute_log_probs(sequences, attention_masks)
        
        # Ensure tensors are on the same device
        if isinstance(ref_log_probs, torch.Tensor) and ref_log_probs.device != self.device:
            ref_log_probs = ref_log_probs.to(self.device)
        
        # Create mask and compute loss
        prompt_mask = create_prompt_mask(sequences, prompt_lengths)
        loss, metrics = grpo_loss(
            policy_log_probs,
            ref_log_probs, 
            advantages,
            prompt_mask,
            kl_coef=0.02,
            clip_range=0.2
        )
        
        computation_time = time.time() - start_time
        memory_after = self.get_memory_stats()
        
        # Calculate MFU for GRPO computation (includes forward + backward passes)
        # Forward pass + backward pass ≈ 3x forward pass FLOPs
        model_flops = self.calculate_model_flops(model, seq_len)
        total_flops = model_flops * batch_size * 3  # 3x for forward + backward
        mfu = self.calculate_mfu(total_flops, computation_time, 1)
        
        return {
            'grpo_computation_time_seconds': computation_time,
            'loss_value': loss.item(),
            'memory_increase_gb': memory_after['allocated_gb'] - memory_before['allocated_gb'],
            'kl_divergence': metrics['kl_div'],
            'policy_loss': metrics['pg_loss'],
            'sequences_processed': batch_size,
            'model_flops_per_sequence': model_flops,
            'total_flops': total_flops,
            'mfu_percent': mfu,
            'actual_tflops': total_flops / (computation_time * 1e12)
        }
    
    def benchmark_reward_scoring(self, samples: List, n_samples: int = 20) -> Dict[str, Any]:
        """Benchmark reward scoring performance."""
        print("📊 Benchmarking reward scoring...")
        
        test_samples = samples[:n_samples]
        prompts = [s[1] for s in test_samples]
        completions = [s[2] for s in test_samples]  # Use ground truth
        
        start_time = time.time()
        advantages, details = compute_grpo_rewards(
            prompts,
            completions,
            group_size=1,
            reward_shaping="graduated",
            verbose=False
        )
        scoring_time = time.time() - start_time
        
        # Analyze reward quality
        reward_values = [d.shaped_reward for d in details]
        
        return {
            'scoring_time_seconds': scoring_time,
            'samples_per_second': len(test_samples) / scoring_time,
            'samples_scored': len(test_samples),
            'mean_reward': np.mean(reward_values),
            'reward_std': np.std(reward_values),
            'min_reward': np.min(reward_values),
            'max_reward': np.max(reward_values),
            'format_valid_ratio': sum(d.format_valid for d in details) / len(details)
        }
    
    def run_full_baseline(self) -> Dict[str, Any]:
        """Run complete performance baseline benchmark."""
        print("🚀 Running Performance Baseline Benchmark")
        print("=" * 60)
        
        # Clear any existing GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        baseline_metrics = {
            'timestamp': time.time(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': self.device
        }
        
        if self.device == "cuda":
            baseline_metrics['cuda_version'] = torch.version.cuda
            baseline_metrics['gpu_name'] = torch.cuda.get_device_name()
        
        # Run benchmarks
        try:
            model = load_rookworld_model(device=self.device)
            samples = load_and_prepare_samples(n_samples=50, seed=42)
            
            baseline_metrics['model_loading'] = self.benchmark_model_loading()
            baseline_metrics['data_loading'] = self.benchmark_data_loading(n_samples=100)
            baseline_metrics['generation'] = self.benchmark_generation_step(model, samples, k_samples=4)
            baseline_metrics['grpo_computation'] = self.benchmark_grpo_computation(model, samples, k_samples=4)
            baseline_metrics['reward_scoring'] = self.benchmark_reward_scoring(samples, n_samples=20)
            
        except Exception as e:
            baseline_metrics['error'] = str(e)
            raise
        
        return baseline_metrics
    
    def save_baseline(self, metrics: Dict[str, Any]) -> None:
        """Save baseline metrics to file."""
        with open(self.baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Baseline saved to {self.baseline_file}")
    
    def load_baseline(self) -> Dict[str, Any]:
        """Load existing baseline metrics."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def compare_with_baseline(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics with saved baseline."""
        baseline = self.load_baseline()
        if not baseline:
            print("⚠️ No baseline found - run baseline test first")
            return {}
        
        comparison = {}
        
        # Define metrics to compare (path to metric in nested dict)
        metrics_to_compare = [
            ('generation.generations_per_second', 'Generation Speed'),
            ('generation.mfu_percent', 'Generation MFU'),
            ('generation.actual_tflops', 'Generation TFLOPS'),
            ('grpo_computation.grpo_computation_time_seconds', 'GRPO Computation Time'),
            ('grpo_computation.mfu_percent', 'GRPO MFU'),
            ('grpo_computation.actual_tflops', 'GRPO TFLOPS'),
            ('reward_scoring.samples_per_second', 'Reward Scoring Speed'),
            ('model_loading.load_time_seconds', 'Model Load Time'),
            ('generation.memory_increase_gb', 'Generation Memory Usage'),
            ('grpo_computation.memory_increase_gb', 'GRPO Memory Usage')
        ]
        
        for metric_path, display_name in metrics_to_compare:
            try:
                # Navigate nested dict path
                baseline_val = baseline
                current_val = current_metrics
                
                for key in metric_path.split('.'):
                    baseline_val = baseline_val[key]
                    current_val = current_val[key]
                
                # Calculate percentage change
                if baseline_val != 0:
                    change_pct = ((current_val - baseline_val) / baseline_val) * 100
                else:
                    change_pct = 0
                
                comparison[metric_path] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'change_percent': change_pct,
                    'display_name': display_name
                }
            except KeyError:
                # Metric doesn't exist in one of the versions
                continue
        
        return comparison


def test_establish_baseline():
    """Main test function to establish performance baseline."""
    baseline = PerformanceBaseline()
    
    print("🎯 Establishing Performance Baseline")
    print("This test creates reference metrics for optimization comparison")
    print("=" * 70)
    
    # Run full benchmark
    metrics = baseline.run_full_baseline()
    
    # Save baseline
    baseline.save_baseline(metrics)
    
    # Print summary
    print("\n📋 BASELINE SUMMARY")
    print("=" * 50)
    print(f"Model Load Time: {metrics['model_loading']['load_time_seconds']:.2f}s")
    print(f"Data Load Time (100 samples): {metrics['data_loading']['load_time_seconds']:.2f}s") 
    print(f"Generation Speed: {metrics['generation']['generations_per_second']:.2f} gen/s")
    print(f"Generation MFU: {metrics['generation']['mfu_percent']:.2f}% ({metrics['generation']['actual_tflops']:.2f} TFLOPS)")
    print(f"GRPO Computation Time: {metrics['grpo_computation']['grpo_computation_time_seconds']:.3f}s")
    print(f"GRPO MFU: {metrics['grpo_computation']['mfu_percent']:.2f}% ({metrics['grpo_computation']['actual_tflops']:.2f} TFLOPS)")
    print(f"Reward Scoring Speed: {metrics['reward_scoring']['samples_per_second']:.2f} samples/s")
    print(f"Memory Usage: {metrics['model_loading']['memory_after_load']['allocated_gb']:.2f}GB")
    print(f"GPU Peak Performance: {metrics['generation']['gpu_peak_tflops']:.0f} TFLOPS")
    
    # Validate baseline quality
    assert metrics['generation']['generations_per_second'] > 0.1, "Generation too slow"
    assert metrics['reward_scoring']['format_valid_ratio'] > 0.5, "Poor format validity"
    assert metrics['grpo_computation']['loss_value'] is not None, "GRPO computation failed"
    
    print("\n✅ Performance baseline established successfully")
    return metrics


def test_validate_against_baseline():
    """Test current performance against saved baseline."""
    baseline_checker = PerformanceBaseline()
    
    print("🔍 Validating Current Performance Against Baseline")
    print("=" * 60)
    
    # Load existing baseline
    baseline_metrics = baseline_checker.load_baseline()
    if not baseline_metrics:
        print("⚠️ No baseline found - skipping validation")
        return
    
    # Run current benchmark
    current_metrics = baseline_checker.run_full_baseline()
    
    # Compare
    comparison = baseline_checker.compare_with_baseline(current_metrics)
    
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    for metric_path, data in comparison.items():
        change = data['change_percent']
        symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print(f"{symbol} {data['display_name']}: {change:+.1f}%")
        print(f"   Baseline: {data['baseline']:.3f}")
        print(f"   Current:  {data['current']:.3f}")
        
        # Performance regression detection
        if 'time' in metric_path.lower() or 'memory' in metric_path.lower():
            # For time/memory metrics, increases are bad
            if change > 10:  # More than 10% regression
                print(f"   ⚠️ WARNING: Significant regression detected!")
        else:
            # For speed metrics, decreases are bad  
            if change < -10:  # More than 10% slowdown
                print(f"   ⚠️ WARNING: Significant slowdown detected!")
        
        print()
    
    print("✅ Performance validation complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['baseline', 'validate'], default='baseline')
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()
    
    if args.mode == 'baseline':
        test_establish_baseline()
    elif args.mode == 'validate':
        test_validate_against_baseline()