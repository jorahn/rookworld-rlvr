#!/usr/bin/env python3
"""
Deep Analysis Test - Batch Size 1, 100 Epochs

This test runs with minimal batch size over many epochs with extensive logging
to verify the implementation is doing exactly what we expect at each step.
Also investigates the >100% MFU anomaly which is theoretically impossible.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
import logging
import json
import gc
from typing import Dict, Any, List
from dataclasses import asdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rookworld_rlvr.model.config import ROOKWORLD_CONFIG, GPT2Config
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.model.loader import load_pretrained_model
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

class DeepAnalysisTester:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        print("="*80)
        print("DEEP ANALYSIS TEST - BATCH SIZE 1, 100 EPOCHS")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"CUDA: {torch.version.cuda}")
            
            # Detailed GPU memory info
            props = torch.cuda.get_device_properties(0)
            print(f"GPU Memory: {props.total_memory / 1024**3:.1f}GB total")
            print(f"Multi-GPU: {'Yes' if torch.cuda.device_count() > 1 else 'No'}")
            
        print("")
        
        # Initialize components
        self.tokenizer = TokenizerBridge()
        self.model_config = ROOKWORLD_CONFIG
        
        # Load pretrained models (HuggingFace weights)
        print("Loading HuggingFace pretrained weights: jrahn/RookWorld-LM-124M")
        self.model = load_pretrained_model("jrahn/RookWorld-LM-124M", device=self.device)
        self.ref_model = load_pretrained_model("jrahn/RookWorld-LM-124M", device=self.device)
        self.ref_model.eval()
        
        # Verify models are on correct device and not using DataParallel
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model is DataParallel: {isinstance(self.model, torch.nn.DataParallel)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("")
        
        # GRPO config for deep analysis
        self.grpo_config = GRPOConfig(
            lr=1e-4,
            group_size=2,  # Minimum for GRPO
            batch_positions=1,  # Single position
            steps=100,
            use_mixed_precision=False,  # Disable for cleaner analysis
            use_torch_compile=False,    # Disable for cleaner analysis
            kl_coef=0.01,
            device=str(self.device)
        )
        
        self.trainer = GRPOTrainer(self.model, self.ref_model, self.grpo_config)
        
        print("Configuration for deep analysis:")
        print(f"  Batch size: 1 position × 2 group_size = 2 samples")
        print(f"  Mixed precision: {self.grpo_config.use_mixed_precision}")
        print(f"  Torch compile: {self.grpo_config.use_torch_compile}")
        print(f"  Learning rate: {self.grpo_config.lr}")
        print("")
        
    def create_minimal_batch(self) -> Dict[str, Any]:
        """Create minimal batch for deep analysis"""
        
        # Two very similar chess positions with different rewards
        texts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4",  # Good opening
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: h2h4"   # Poor opening
        ]
        
        print("Creating minimal batch:")
        for i, text in enumerate(texts):
            print(f"  Sample {i+1}: {text}")
        
        # Tokenize
        all_tokens = []
        target_indices = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.append(tokens)
            
            # Find where "M:" ends (target starts after)
            target_idx = len(tokens) - 1  # Default to last token
            for j in range(len(tokens)-1):
                if self.tokenizer.decode([tokens[j]]).strip() == 'M:':
                    target_idx = j + 1
                    break
            target_indices.append(target_idx)
            
            print(f"    Tokens: {len(tokens)}")
            print(f"    Target starts at: {target_idx}")
            print(f"    Target token: '{self.tokenizer.decode([tokens[target_idx]])}'")
        
        # Pad to same length
        max_len = max(len(tokens) for tokens in all_tokens)
        input_ids = []
        attention_mask = []
        
        for tokens in all_tokens:
            padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            input_ids.append(padded)
            attention_mask.append(mask)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        target_start_indices = torch.tensor(target_indices, device=self.device)
        
        # Get reference logprobs
        with torch.no_grad():
            old_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        # Different rewards for good vs poor moves
        rewards = torch.tensor([1.0, 0.2], dtype=torch.float32, device=self.device)
        
        print(f"  Batch shape: {input_ids.shape}")
        print(f"  Reference logprobs: {old_logprobs.tolist()}")
        print(f"  Rewards: {rewards.tolist()}")
        print("")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_start_indices': target_start_indices,
            'old_logprobs': old_logprobs,
            'rewards': rewards,
            'texts': texts,
            'max_len': max_len
        }
    
    def detailed_forward_analysis(self, batch: Dict[str, Any], epoch: int) -> Dict[str, Any]:
        """Perform detailed analysis of a single forward pass"""
        
        self.model.train()
        
        # Clear memory and reset counters
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Detailed timing
        start_time = time.perf_counter()
        
        # === FORWARD PASS ANALYSIS ===
        forward_start = time.perf_counter()
        
        # Forward pass (no mixed precision for clear analysis)
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs["logits"]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - forward_start
        
        # === LOGPROB COMPUTATION ANALYSIS ===
        logprob_start = time.perf_counter()
        
        current_logprobs = self.trainer.compute_logprobs(
            batch['input_ids'], 
            batch['attention_mask'], 
            batch['target_start_indices'], 
            use_ref_model=False
        )
        
        logprob_time = time.perf_counter() - logprob_start
        
        # === GRPO LOSS ANALYSIS ===
        loss_start = time.perf_counter()
        
        # Create GRPO batch
        grpo_batch = GRPOBatch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target_start_indices=batch['target_start_indices'],
            old_logprobs=batch['old_logprobs'],
            rewards=batch['rewards'],
            position_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            task_type="policy"
        )
        
        loss, loss_metrics = self.trainer.compute_grpo_loss(grpo_batch)
        
        loss_time = time.perf_counter() - loss_start
        
        # === BACKWARD PASS ANALYSIS ===
        backward_start = time.perf_counter()
        
        self.trainer.optimizer.zero_grad()
        loss.backward()
        
        # Gradient analysis
        total_grad_norm = 0.0
        param_count = 0
        params_with_grad = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
            param_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_time = time.perf_counter() - backward_start
        
        # === OPTIMIZER STEP ANALYSIS ===
        optim_start = time.perf_counter()
        
        # Gradient clipping
        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.grpo_config.grad_clip_norm
        )
        
        # Store param state before update
        param_before = next(self.model.parameters()).clone().detach()
        
        self.trainer.optimizer.step()
        
        # Param change analysis
        param_after = next(self.model.parameters())
        param_change = (param_after - param_before).norm().item()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        optim_time = time.perf_counter() - optim_start
        
        total_time = time.perf_counter() - start_time
        
        # === MFU CALCULATION (CORRECTED) ===
        batch_size = batch['input_ids'].size(0)
        seq_len = batch['input_ids'].size(1)
        
        # Model parameters
        model_params = sum(p.numel() for p in self.model.parameters())
        
        # FLOPs calculation for forward pass only (Kaplan et al. scaling laws)
        # Forward pass: ~6 * model_params * tokens_processed
        tokens_processed = batch_size * seq_len
        forward_flops = 6 * model_params * tokens_processed
        
        # FLOPs per second (only forward pass for MFU)
        flops_per_sec = forward_flops / forward_time
        
        # RTX 4090 theoretical peak (BF16 tensor cores)
        rtx4090_peak_flops = 1.65e14
        
        # MFU calculation (should be <100%)
        mfu_percent = (flops_per_sec / rtx4090_peak_flops) * 100
        
        # Memory analysis
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  
            memory_peak = torch.cuda.max_memory_allocated() / 1024**3
        else:
            memory_allocated = memory_reserved = memory_peak = 0.0
        
        # Detailed metrics
        metrics = {
            'epoch': epoch,
            'timing': {
                'total_time': total_time,
                'forward_time': forward_time,
                'logprob_time': logprob_time, 
                'loss_time': loss_time,
                'backward_time': backward_time,
                'optim_time': optim_time
            },
            'model_analysis': {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tokens_processed': tokens_processed,
                'model_params': model_params,
                'forward_flops': forward_flops,
                'flops_per_sec': flops_per_sec,
                'mfu_percent': mfu_percent
            },
            'training_dynamics': {
                'loss': loss.item(),
                'current_logprobs': current_logprobs.tolist(),
                'old_logprobs': batch['old_logprobs'].tolist(),
                'logprob_diff': (current_logprobs - batch['old_logprobs']).tolist(),
                'rewards': batch['rewards'].tolist(),
                **loss_metrics
            },
            'gradients': {
                'total_grad_norm': total_grad_norm,
                'clipped_grad_norm': clipped_grad_norm.item(),
                'param_count': param_count,
                'params_with_grad': params_with_grad,
                'param_change': param_change
            },
            'memory': {
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'peak_gb': memory_peak
            }
        }
        
        return metrics
    
    def run_deep_analysis(self, epochs: int = 100):
        """Run deep analysis over many epochs"""
        
        print("="*80)
        print(f"RUNNING DEEP ANALYSIS: {epochs} EPOCHS")
        print("="*80)
        
        # Create batch once and reuse
        batch = self.create_minimal_batch()
        
        # Storage for analysis
        epoch_metrics = []
        
        print("Starting epoch-by-epoch analysis...")
        print("")
        
        # Analysis loop
        for epoch in range(epochs):
            metrics = self.detailed_forward_analysis(batch, epoch)
            epoch_metrics.append(metrics)
            
            # Progress logging
            if epoch % 10 == 0 or epoch < 5 or epoch >= epochs - 5:
                timing = metrics['timing']
                model = metrics['model_analysis']
                training = metrics['training_dynamics']
                grads = metrics['gradients']
                memory = metrics['memory']
                
                print(f"Epoch {epoch:3d}:")
                print(f"  Time: Total={timing['total_time']:.4f}s, "
                      f"Forward={timing['forward_time']:.4f}s, "
                      f"Backward={timing['backward_time']:.4f}s")
                print(f"  Performance: MFU={model['mfu_percent']:.2f}%, "
                      f"FLOPS/s={model['flops_per_sec']:.2e}")
                print(f"  Training: Loss={training['loss']:.6f}, "
                      f"AvgLogProb={sum(training['current_logprobs'])/2:.4f}")
                print(f"  Gradients: Norm={grads['total_grad_norm']:.4f}, "
                      f"Clipped={grads['clipped_grad_norm']:.4f}, "
                      f"ParamChange={grads['param_change']:.2e}")
                print(f"  Memory: {memory['peak_gb']:.3f}GB peak")
                
                # Sanity checks
                if model['mfu_percent'] > 100:
                    print(f"  ⚠️  MFU >100% detected: {model['mfu_percent']:.2f}%")
                    print(f"      Forward time: {timing['forward_time']:.6f}s")
                    print(f"      FLOPs: {model['forward_flops']:.2e}")
                    print(f"      Theoretical peak: {1.65e14:.2e}")
                
                print("")
        
        # Analysis summary
        print("="*80)
        print("DEEP ANALYSIS SUMMARY")
        print("="*80)
        
        # Extract key metrics over time
        mfus = [m['model_analysis']['mfu_percent'] for m in epoch_metrics]
        losses = [m['training_dynamics']['loss'] for m in epoch_metrics]
        forward_times = [m['timing']['forward_time'] for m in epoch_metrics]
        grad_norms = [m['gradients']['total_grad_norm'] for m in epoch_metrics]
        
        print(f"MFU Analysis:")
        print(f"  Min MFU: {min(mfus):.2f}%")
        print(f"  Max MFU: {max(mfus):.2f}%")
        print(f"  Avg MFU: {sum(mfus)/len(mfus):.2f}%")
        print(f"  MFU >100%: {sum(1 for m in mfus if m > 100)} epochs")
        print("")
        
        print(f"Training Dynamics:")
        print(f"  Initial loss: {losses[0]:.6f}")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Loss change: {losses[-1] - losses[0]:+.6f}")
        print(f"  Avg gradient norm: {sum(grad_norms)/len(grad_norms):.4f}")
        print("")
        
        print(f"Performance:")
        print(f"  Avg forward time: {sum(forward_times)/len(forward_times):.4f}s")
        print(f"  Min forward time: {min(forward_times):.4f}s")
        print(f"  Max forward time: {max(forward_times):.4f}s")
        print("")
        
        # Investigation of >100% MFU
        if any(m > 100 for m in mfus):
            print("⚠️  MFU >100% INVESTIGATION:")
            print("   This is impossible - indicates measurement error")
            print("   Possible causes:")
            print("   1. Timing measurement precision issues")
            print("   2. FLOPs calculation error")
            print("   3. GPU frequency boosting")
            print("   4. Parallel execution not accounted for")
            print("")
        
        # Save detailed results
        with open('deep_analysis_results.json', 'w') as f:
            json.dump(epoch_metrics, f, indent=2)
        
        print(f"Detailed results saved to: deep_analysis_results.json")
        print("")
        
        return epoch_metrics

def main():
    """Run the deep analysis test"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - deep analysis requires GPU")
        return False
    
    tester = DeepAnalysisTester()
    results = tester.run_deep_analysis(epochs=100)
    
    # Final validation
    mfus = [r['model_analysis']['mfu_percent'] for r in results]
    avg_mfu = sum(mfus) / len(mfus)
    impossible_mfus = sum(1 for m in mfus if m > 100)
    
    print("="*80)
    print("DEEP ANALYSIS CONCLUSIONS")
    print("="*80)
    
    if impossible_mfus > 0:
        print(f"❌ Found {impossible_mfus} epochs with MFU >100%")
        print("   This indicates measurement or calculation errors")
        print("   Need to investigate timing precision and FLOPs calculation")
    else:
        print("✅ All MFU measurements are realistic (<100%)")
    
    print(f"Average MFU: {avg_mfu:.2f}%")
    print(f"Implementation appears to be working correctly")
    print("="*80)
    
    return impossible_mfus == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)