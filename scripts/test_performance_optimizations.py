#!/usr/bin/env python3
"""
Performance Optimization Test

Tests BF16 mixed precision, torch.compile, and larger batch sizes to achieve >15% MFU.
This addresses the critical performance issues identified in the GitHub issue.
"""

import torch
import time
import sys
import os
import logging
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rookworld_rlvr.model.config import ROOKWORLD_CONFIG, GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class PerformanceTester:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Set tensor core optimization (critical for RTX 4090)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
        
        print("="*80)
        print("PERFORMANCE OPTIMIZATION TEST")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA: {torch.version.cuda}")
            print(f"Tensor Core Precision: {torch.get_float32_matmul_precision()}")
        print("")
        
    def create_test_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Create a test batch for performance testing"""
        tokenizer = TokenizerBridge()
        
        # Create multiple similar prompts
        base_text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4"
        texts = [base_text for _ in range(batch_size)]
        
        # Encode texts
        all_tokens = [tokenizer.encode(text) for text in texts]
        max_len = max(len(tokens) for tokens in all_tokens)
        
        # Create padded tensors
        input_ids = []
        attention_mask = []
        
        for tokens in all_tokens:
            padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
            input_ids.append(padded)
            
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            attention_mask.append(mask)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long, device=self.device),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long, device=self.device),
            'batch_size': batch_size,
            'seq_len': max_len
        }
    
    def benchmark_forward_pass(self, model: GPT2Model, batch: Dict[str, torch.Tensor], 
                              use_mixed_precision: bool, use_compile: bool, warmup: int = 3, 
                              iterations: int = 20) -> Dict[str, float]:
        """Benchmark forward pass with different optimizations"""
        
        model.eval()
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                else:
                    _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(iterations):
            with torch.no_grad():
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                else:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Calculate MFU
        batch_size = batch['batch_size']
        seq_len = batch['seq_len']
        model_params = sum(p.numel() for p in model.parameters())
        
        # FLOPs estimation for transformer forward pass
        # Roughly: 6 * params * batch_size * seq_len (simplified)
        flops_per_forward = 6 * model_params * batch_size * seq_len
        flops_per_second = flops_per_forward / avg_time
        
        # RTX 4090 theoretical peak (BF16 Tensor Cores)
        rtx4090_peak_flops = 1.65e14  # ~165 TFLOPs
        mfu = (flops_per_second / rtx4090_peak_flops) * 100
        
        return {
            'avg_time': avg_time,
            'flops_per_second': flops_per_second,
            'mfu_percent': mfu,
            'throughput_samples_per_sec': batch_size / avg_time
        }
    
    def test_configurations(self):
        """Test different optimization configurations"""
        
        print("Creating models and test batches...")
        
        # Test different batch sizes
        batch_sizes = [2, 8, 16, 32]
        configs = [
            {"name": "Baseline (FP32)", "mixed_precision": False, "compile": False},
            {"name": "BF16 Mixed Precision", "mixed_precision": True, "compile": False},
            {"name": "Torch Compile", "mixed_precision": False, "compile": True},
            {"name": "BF16 + Compile", "mixed_precision": True, "compile": True},
        ]
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"BATCH SIZE: {batch_size}")
            print(f"{'='*60}")
            
            # Create test batch
            batch = self.create_test_batch(batch_size)
            print(f"Batch shape: {batch['input_ids'].shape}")
            print(f"Memory before model: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            for config in configs:
                print(f"\nTesting: {config['name']}")
                print("-" * 40)
                
                try:
                    # Create fresh model for each config
                    model = GPT2Model(ROOKWORLD_CONFIG).to(self.device)
                    
                    # Apply torch.compile if requested
                    if config['compile']:
                        print("Compiling model...")
                        model = torch.compile(model, mode='reduce-overhead')
                    
                    # Benchmark
                    metrics = self.benchmark_forward_pass(
                        model, batch, 
                        use_mixed_precision=config['mixed_precision'],
                        use_compile=config['compile']
                    )
                    
                    # Store results
                    result = {
                        'batch_size': batch_size,
                        'config': config['name'],
                        'mixed_precision': config['mixed_precision'],
                        'compile': config['compile'],
                        **metrics
                    }
                    results.append(result)
                    
                    # Print results
                    print(f"Time per forward pass: {metrics['avg_time']:.4f}s")
                    print(f"Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/s")
                    print(f"FLOPs/s: {metrics['flops_per_second']:.2e}")
                    print(f"MFU: {metrics['mfu_percent']:.2f}%")
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                        print(f"Peak memory: {memory_gb:.2f}GB")
                        torch.cuda.reset_peak_memory_stats()
                    
                    # Clean up model to free memory
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Summary analysis
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Find best configuration for each batch size
        for batch_size in batch_sizes:
            batch_results = [r for r in results if r['batch_size'] == batch_size]
            if batch_results:
                best = max(batch_results, key=lambda x: x['mfu_percent'])
                print(f"\nBatch Size {batch_size}:")
                print(f"  Best: {best['config']} - {best['mfu_percent']:.2f}% MFU")
                
                # Show speedups
                baseline = next((r for r in batch_results if r['config'] == 'Baseline (FP32)'), None)
                if baseline:
                    speedup = best['throughput_samples_per_sec'] / baseline['throughput_samples_per_sec']
                    print(f"  Speedup vs baseline: {speedup:.2f}x")
        
        # Overall recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        best_overall = max(results, key=lambda x: x['mfu_percent'])
        print(f"Best overall configuration:")
        print(f"  Config: {best_overall['config']}")
        print(f"  Batch Size: {best_overall['batch_size']}")
        print(f"  MFU: {best_overall['mfu_percent']:.2f}%")
        print(f"  Throughput: {best_overall['throughput_samples_per_sec']:.1f} samples/s")
        
        if best_overall['mfu_percent'] >= 15.0:
            print(f"\n✅ SUCCESS: Achieved target MFU >15%!")
        else:
            print(f"\n⚠️  Still below target: {best_overall['mfu_percent']:.2f}% < 15%")
            print("   Consider: Larger batch sizes, Flash Attention, or model sharding")
        
        return results

def main():
    """Run performance optimization tests"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Performance tests require GPU.")
        return False
    
    tester = PerformanceTester()
    results = tester.test_configurations()
    
    # Check if we achieved target MFU
    best_mfu = max(r['mfu_percent'] for r in results)
    success = best_mfu >= 15.0
    
    print(f"\n{'='*80}")
    print(f"TEST {'PASSED' if success else 'FAILED'}: Best MFU = {best_mfu:.2f}%")
    print(f"{'='*80}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)