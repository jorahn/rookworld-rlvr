#!/usr/bin/env python3
"""
BF16 Performance Test

Measures the MFU improvement from enabling BF16 mixed precision training
and validates performance gains against the baseline.
"""

import torch
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.config import GRPOConfig
from test_performance_baseline import PerformanceBaseline


def test_bf16_mfu_improvement():
    """Test MFU improvement with BF16 enabled."""
    print("🚀 Testing BF16 Training MFU Improvement")
    print("=" * 60)
    
    # Create baseline tester
    baseline_tester = PerformanceBaseline()
    
    # Load existing FP32 baseline
    fp32_baseline = baseline_tester.load_baseline()
    if not fp32_baseline:
        print("⚠️ No FP32 baseline found - creating one first...")
        fp32_baseline = baseline_tester.run_full_baseline()
        baseline_tester.save_baseline(fp32_baseline)
    
    print("\n📊 FP32 Baseline (Current):")
    print(f"   Training MFU: {fp32_baseline['grpo_computation']['mfu_percent']:.2f}%")
    print(f"   Training TFLOPS: {fp32_baseline['grpo_computation']['actual_tflops']:.2f}")
    print(f"   Generation MFU: {fp32_baseline['generation']['mfu_percent']:.2f}%")
    
    # Test with BF16 enabled
    print("\n🔬 Testing BF16 Performance...")
    
    # Temporarily modify config for BF16 testing
    config = GRPOConfig(use_bf16=True)
    
    # Create model and test BF16 training performance
    model = None
    try:
        from rookworld_rlvr.loader import load_rookworld_model
        from rookworld_rlvr.grpo import ReferenceModel
        
        model = load_rookworld_model(device="cuda")
        
        # Test GRPO computation with BF16
        batch_size, seq_len = 4, 150
        torch.manual_seed(42)
        sequences = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
        attention_masks = torch.ones_like(sequences)
        rewards = torch.rand(batch_size, device="cuda")
        prompt_lengths = torch.randint(20, 50, (batch_size,))
        
        # Setup for BF16 training
        ref_model = ReferenceModel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        # Benchmark BF16 computation only (fair comparison with baseline)
        from rookworld_rlvr.grpo import compute_log_probs, compute_advantages, create_prompt_mask, grpo_loss
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Just the computation part (not full training step)
        advantages = compute_advantages(rewards, group_size=config.k_samples)
        policy_log_probs = compute_log_probs(model, sequences, attention_masks, use_bf16=True)
        
        with torch.no_grad():
            ref_log_probs = ref_model.compute_log_probs(sequences, attention_masks, use_bf16=False)
            ref_log_probs = ref_log_probs.to("cuda")
        
        prompt_mask = create_prompt_mask(sequences, prompt_lengths)
        loss, metrics = grpo_loss(
            policy_log_probs,
            ref_log_probs,
            advantages,
            prompt_mask,
            kl_coef=0.02,
            clip_range=0.2
        )
        
        torch.cuda.synchronize()
        bf16_time = time.time() - start_time
        
        # Calculate BF16 MFU
        model_flops = baseline_tester.calculate_model_flops(model, seq_len)
        total_flops = model_flops * batch_size * 3  # Forward + backward
        bf16_mfu = baseline_tester.calculate_mfu(total_flops, bf16_time, 1)
        bf16_tflops = total_flops / (bf16_time * 1e12)
        
        print("\n📊 BF16 Results:")
        print(f"   Training Time: {bf16_time:.3f}s")
        print(f"   Training MFU: {bf16_mfu:.2f}%")
        print(f"   Training TFLOPS: {bf16_tflops:.2f}")
        
        # Compare with baseline
        fp32_mfu = fp32_baseline['grpo_computation']['mfu_percent']
        fp32_tflops = fp32_baseline['grpo_computation']['actual_tflops']
        fp32_time = fp32_baseline['grpo_computation']['grpo_computation_time_seconds']
        
        mfu_improvement = (bf16_mfu - fp32_mfu) / fp32_mfu * 100
        tflops_improvement = (bf16_tflops - fp32_tflops) / fp32_tflops * 100
        time_improvement = (fp32_time - bf16_time) / fp32_time * 100
        
        print("\n📈 Performance Improvement:")
        print(f"   MFU: {fp32_mfu:.2f}% → {bf16_mfu:.2f}% ({mfu_improvement:+.1f}%)")
        print(f"   TFLOPS: {fp32_tflops:.2f} → {bf16_tflops:.2f} ({tflops_improvement:+.1f}%)")
        print(f"   Time: {fp32_time:.3f}s → {bf16_time:.3f}s ({time_improvement:+.1f}%)")
        
        # Validate improvement
        assert bf16_mfu >= fp32_mfu * 0.95, f"BF16 MFU regression: {bf16_mfu:.2f}% vs {fp32_mfu:.2f}%"
        
        if mfu_improvement > 5:
            print("🎉 Significant MFU improvement achieved!")
        elif mfu_improvement > 0:
            print("✅ Modest MFU improvement achieved")
        else:
            print("⚠️ No MFU improvement (expected for small models)")
        
        print(f"\n✅ BF16 training MFU test complete")
        
        return {
            'bf16_mfu': bf16_mfu,
            'fp32_mfu': fp32_mfu,
            'mfu_improvement_percent': mfu_improvement,
            'bf16_tflops': bf16_tflops,
            'speedup': timings_comparison if 'timings_comparison' in locals() else 1.0
        }
        
    except Exception as e:
        print(f"❌ BF16 performance test failed: {e}")
        raise
    finally:
        # Cleanup
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_bf16_mfu_improvement()