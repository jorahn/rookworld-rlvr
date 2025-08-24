#!/usr/bin/env python3
"""
Detailed Training Implementation Test

This script performs a granular test of the GRPO training pipeline with extensive
logging to verify correctness at every step. It processes a single batch with one
sample to enable detailed inspection of:

- Tensor shapes at every stage
- Token sequences (encoded and decoded)
- Memory usage and performance metrics
- Model FLOPs utilization (MFU)
- GRPO algorithm components
- Gradient flows and training dynamics

Usage:
    uv run python test_training_detailed.py [--device cuda|cpu] [--mixed-precision]
"""

import argparse
import logging
import time
import traceback
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import torch
import torch.nn.functional as F
import chess
import chess.engine
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.loader import load_pretrained_model
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch
from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.tokenizer.bridge import TokenizerBridge
from rookworld_rlvr.engine.stockfish import StockfishEngine, StockfishAnalysis


class DetailedTrainingTester:
    """Comprehensive training implementation tester with granular logging."""
    
    def __init__(self, device: str = "cuda", use_mixed_precision: bool = False):
        """Initialize the detailed tester."""
        self.device = device
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        # Setup detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler('detailed_training_test.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.trainer = None
        self.config = None
        
        # Test data
        self.test_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
        self.test_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "f1c4"]  # Common opening moves
        
        # Metrics storage
        self.metrics = {}
        
    def log_system_info(self):
        """Log detailed system information."""
        self.logger.info("="*80)
        self.logger.info("DETAILED TRAINING IMPLEMENTATION TEST")
        self.logger.info("="*80)
        
        # System info
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"Device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.logger.info(f"Test Device: {self.device}")
        self.logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        self.logger.info("")
        
    def initialize_components(self):
        """Initialize all components with detailed logging."""
        self.logger.info("INITIALIZING COMPONENTS")
        self.logger.info("-" * 40)
        
        # 1. Initialize tokenizer
        self.logger.info("1. Initializing TokenizerBridge...")
        start_time = time.time()
        self.tokenizer = TokenizerBridge()
        init_time = time.time() - start_time
        self.logger.info(f"   Tokenizer initialized in {init_time:.3f}s")
        self.logger.info(f"   Vocab size: {self.tokenizer.vocab_size}")
        self.logger.info(f"   PAD token ID: {self.tokenizer.pad_token_id}")
        self.logger.info(f"   EOS token ID: {self.tokenizer.eos_token_id}")
        self.logger.info("")
        
        # 2. Initialize model configuration
        self.logger.info("2. Creating GPT2Config...")
        model_config = GPT2Config()
        self.logger.info(f"   Model: {model_config.n_layer}L-{model_config.n_head}H-{model_config.n_embd}E")
        # Calculate total parameters manually
        total_params = (
            model_config.vocab_size * model_config.n_embd +  # Token embeddings
            model_config.n_positions * model_config.n_embd +  # Position embeddings
            model_config.n_layer * (
                4 * model_config.n_embd * model_config.n_embd +  # QKV + output projections
                2 * model_config.n_embd * model_config.n_inner +  # MLP up + down
                4 * model_config.n_embd  # Layer norm parameters
            ) +
            model_config.n_embd  # Final layer norm
        )
        self.logger.info(f"   Parameters: {total_params:,} (estimated)")
        self.logger.info(f"   Max positions: {model_config.n_positions}")
        self.logger.info("")
        
        # 3. Initialize model (loading HuggingFace pretrained weights)
        self.logger.info("3. Loading GPT-2 model with HuggingFace pretrained weights...")
        self.logger.info("   Model: jrahn/RookWorld-LM-124M")
        start_time = time.time()
        self.model = load_pretrained_model("jrahn/RookWorld-LM-124M", device=self.device)
        self.model.train()
        load_time = time.time() - start_time
        actual_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"   Model loaded in {load_time:.3f}s")
        self.logger.info(f"   Actual parameters: {actual_params:,}")
        
        # Create reference model (frozen copy for GRPO)
        self.logger.info("4. Creating reference model...")
        start_time = time.time()
        self.ref_model = load_pretrained_model("jrahn/RookWorld-LM-124M", device=self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        ref_time = time.time() - start_time
        self.logger.info(f"   Reference model created in {ref_time:.3f}s")
        self.logger.info("")
        
        # 4. Initialize GRPO configuration
        self.logger.info("5. Creating GRPO configuration...")
        self.config = GRPOConfig(
            device=self.device,
            use_mixed_precision=self.use_mixed_precision,
            use_torch_compile=False,  # Disable for cleaner logging
            group_size=2,  # Minimum for GRPO (need 2 samples for group-relative baseline)
            batch_positions=1,  # Single position
            steps=1,
            lr=1e-5,
            save_every=1000,  # No saves during test
            eval_every=1000   # No eval during test
        )
        
        self.logger.info("   Note: Using group_size=2 (minimum for GRPO algorithm)")
        self.logger.info(f"   Group size: {self.config.group_size}")
        self.logger.info(f"   Learning rate: {self.config.lr}")
        self.logger.info(f"   Mixed precision: {self.config.use_mixed_precision}")
        self.logger.info("")
        
        # 5. Initialize trainer
        self.logger.info("6. Initializing GRPO trainer...")
        start_time = time.time()
        try:
            self.trainer = GRPOTrainer(self.model, self.ref_model, self.config)
            trainer_time = time.time() - start_time
        except Exception as e:
            self.logger.error(f"Failed to initialize trainer: {e}")
            raise
        self.logger.info(f"   Trainer initialized in {trainer_time:.3f}s")
        self.logger.info(f"   Optimizer: {type(self.trainer.optimizer).__name__}")
        self.logger.info(f"   Scheduler: {type(self.trainer.scheduler).__name__}")
        if self.trainer.scaler:
            self.logger.info(f"   GradScaler enabled: {type(self.trainer.scaler).__name__}")
        self.logger.info("")
        
    def log_memory_usage(self, stage: str):
        """Log detailed memory usage."""
        if not torch.cuda.is_available():
            return
            
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        
        self.logger.info(f"   Memory [{stage}]: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved, {max_allocated:.3f}GB peak")
        
    def test_tokenization(self):
        """Test tokenization with detailed logging."""
        self.logger.info("TESTING TOKENIZATION")
        self.logger.info("-" * 40)
        
        # Test policy task format
        policy_prompt = f"P: {self.test_position}    M:"
        self.logger.info(f"Policy prompt: '{policy_prompt}'")
        
        # Tokenize
        encoded = self.tokenizer.encode(policy_prompt)
        self.logger.info(f"Encoded length: {len(encoded)} tokens")
        self.logger.info(f"Token IDs: {encoded}")
        
        # Decode back
        decoded = self.tokenizer.decode(encoded)
        self.logger.info(f"Decoded: '{decoded}'")
        self.logger.info(f"Roundtrip match: {policy_prompt == decoded}")
        
        # Show individual tokens
        self.logger.info("Individual tokens:")
        for i, token_id in enumerate(encoded):
            token_text = self.tokenizer.decode([token_id])
            self.logger.info(f"  {i:2d}: {token_id:5d} -> '{token_text}'")
        
        self.logger.info("")
        return encoded, policy_prompt
        
    def create_test_batch(self) -> Dict[str, Any]:
        """Create a single test batch with detailed logging."""
        self.logger.info("CREATING TEST BATCH")
        self.logger.info("-" * 40)
        
        # Create two test sequences for group_size=2
        policy_prompt = f"P: {self.test_position}    M:"
        target_responses = [" e2e4", " d2d4"]  # Two different moves
        
        sequences = []
        for response in target_responses:
            full_sequence = policy_prompt + response
            sequences.append(full_sequence)
            self.logger.info(f"Sequence {len(sequences)}: '{full_sequence}'")
        
        # Tokenize both sequences
        encoded_sequences = [self.tokenizer.encode(seq) for seq in sequences]
        max_len = max(len(seq) for seq in encoded_sequences)
        
        # Pad sequences to same length
        input_ids = []
        for seq in encoded_sequences:
            padded = seq + [self.tokenizer.pad_token_id] * (max_len - len(seq))
            input_ids.append(padded)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        target_start_indices = torch.tensor([len(self.tokenizer.encode(policy_prompt))] * 2, dtype=torch.long, device=self.device)
        
        # Log shapes
        self.logger.info(f"Input shape: {input_ids.shape}")
        self.logger.info(f"Attention mask shape: {attention_mask.shape}")
        self.logger.info(f"Target start indices: {target_start_indices}")
        
        # Compute old logprobs (simulate from sampling)
        with torch.no_grad():
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.ref_model(input_ids=input_ids, attention_mask=None)  # Our GPT-2 handles causal masking internally
            else:
                outputs = self.ref_model(input_ids=input_ids, attention_mask=None)
                
            old_logprobs = self.trainer.compute_logprobs(
                input_ids, attention_mask, target_start_indices, use_ref_model=True
            )
        
        self.logger.info(f"Old logprobs: {old_logprobs}")
        
        # Create rewards (simulate realistic rewards for both samples)
        rewards = torch.tensor([0.7, 0.5], dtype=torch.float32, device=self.device)  # Different rewards
        self.logger.info(f"Rewards: {rewards}")
        
        # Create batch dictionary (simplified for testing)
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_start_indices': target_start_indices,
            'old_logprobs': old_logprobs,
            'rewards': rewards
        }
        
        self.logger.info(f"Batch created with {input_ids.size(0)} samples")
        self.log_memory_usage("after batch creation")
        self.logger.info("")
        
        return batch
        
    def test_forward_pass(self, batch: Dict[str, Any]):
        """Test forward pass with detailed logging."""
        self.logger.info("TESTING FORWARD PASS")
        self.logger.info("-" * 40)
        
        self.model.train()
        
        # Forward pass
        start_time = time.time()
        
        if self.config.use_mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        else:
            outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
        forward_time = time.time() - start_time
        
        # Log output details
        logits = outputs["logits"]
        self.logger.info(f"Forward pass time: {forward_time:.4f}s")
        self.logger.info(f"Output logits shape: {logits.shape}")
        self.logger.info(f"Output logits dtype: {logits.dtype}")
        self.logger.info(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        self.logger.info(f"Logits mean: {logits.mean():.3f}")
        self.logger.info(f"Logits std: {logits.std():.3f}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()
        self.logger.info(f"Contains NaN: {has_nan}")
        self.logger.info(f"Contains Inf: {has_inf}")
        
        self.log_memory_usage("after forward pass")
        self.logger.info("")
        
        return outputs
        
    def test_logprob_computation(self, batch: Dict[str, Any]):
        """Test log probability computation with detailed logging."""
        self.logger.info("TESTING LOGPROB COMPUTATION")
        self.logger.info("-" * 40)
        
        # Current model logprobs
        current_logprobs = self.trainer.compute_logprobs(
            batch['input_ids'],
            batch['attention_mask'],  # Now properly handles attention mask format
            batch['target_start_indices'],
            use_ref_model=False
        )
        
        # Reference model logprobs
        ref_logprobs = self.trainer.compute_logprobs(
            batch['input_ids'],
            batch['attention_mask'],  # Now properly handles attention mask format
            batch['target_start_indices'],
            use_ref_model=True
        )
        
        self.logger.info(f"Current logprobs shape: {current_logprobs.shape}")
        self.logger.info(f"Current logprobs: {current_logprobs}")
        self.logger.info(f"Reference logprobs: {ref_logprobs}")
        self.logger.info(f"Old logprobs (from batch): {batch['old_logprobs']}")
        
        # Compute differences
        current_vs_ref = current_logprobs - ref_logprobs
        current_vs_old = current_logprobs - batch['old_logprobs']
        
        self.logger.info(f"Current vs Reference diff: {current_vs_ref}")
        self.logger.info(f"Current vs Old diff: {current_vs_old}")
        
        # Log probability ratios
        logprob_ratio = torch.exp(current_logprobs - batch['old_logprobs'])
        self.logger.info(f"Logprob ratio: {logprob_ratio}")
        
        self.logger.info("")
        return current_logprobs, ref_logprobs
        
    def test_grpo_loss_computation(self, batch: Dict[str, Any]):
        """Test GRPO loss computation with detailed logging."""
        self.logger.info("TESTING GRPO LOSS COMPUTATION")
        self.logger.info("-" * 40)
        
        # Compute current logprobs
        current_logprobs = self.trainer.compute_logprobs(
            batch['input_ids'],
            None,  # Our GPT-2 handles causal masking internally
            batch['target_start_indices'],
            use_ref_model=False
        )
        
        # Compute GRPO loss manually
        start_time = time.time()
        
        # Compute ratio
        ratio = torch.exp(current_logprobs - batch['old_logprobs'])
        
        # Group-relative baseline (mean reward)
        baseline = batch['rewards'].mean()
        advantages = batch['rewards'] - baseline
        
        # PPO clipped objective
        clip_range = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        
        # Policy loss 
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
        
        # KL penalty (simplified)
        kl_penalty = 0.02 * (current_logprobs - batch['old_logprobs']).mean()
        
        total_loss = policy_loss + kl_penalty
        loss_time = time.time() - start_time
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'baseline': baseline.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_ratio': ratio.mean().item()
        }
        
        self.logger.info(f"Loss computation time: {loss_time:.4f}s")
        self.logger.info(f"Total loss: {total_loss.item():.6f}")
        self.logger.info(f"Loss dtype: {total_loss.dtype}")
        self.logger.info(f"Loss requires_grad: {total_loss.requires_grad}")
        
        # Log all metrics
        self.logger.info("Detailed metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.6f}")
        
        # Analyze GRPO components
        self.logger.info(f"Rewards: {batch['rewards']}")
        self.logger.info(f"Baseline: {baseline:.6f}")
        self.logger.info(f"Advantages: {advantages}")
        self.logger.info(f"Ratio: {ratio}")
        self.logger.info(f"Clipped ratio: {clipped_ratio}")
        
        self.logger.info("")
        return total_loss, metrics
        
    def test_backward_pass(self, loss: torch.Tensor):
        """Test backward pass with detailed logging."""
        self.logger.info("TESTING BACKWARD PASS")
        self.logger.info("-" * 40)
        
        # Clear gradients
        self.trainer.optimizer.zero_grad()
        
        # Backward pass
        start_time = time.time()
        
        if self.trainer.scaler is not None:
            self.logger.info("Using gradient scaling...")
            scaled_loss = self.trainer.scaler.scale(loss)
            self.logger.info(f"Scaled loss: {scaled_loss.item():.6f}")
            scaled_loss.backward()
        else:
            self.logger.info("Standard backward pass...")
            loss.backward()
            
        backward_time = time.time() - start_time
        self.logger.info(f"Backward pass time: {backward_time:.4f}s")
        
        # Analyze gradients
        total_grad_norm = 0.0
        param_count = 0
        zero_grad_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.norm().item()
                total_grad_norm += param_grad_norm ** 2
                param_count += 1
                
                if param_grad_norm == 0:
                    zero_grad_count += 1
                    
                # Log gradient info for key layers
                if any(key in name for key in ['embed', 'ln_f', 'wte', 'wpe']):
                    self.logger.info(f"  Gradient norm [{name}]: {param_grad_norm:.6f}")
            else:
                self.logger.info(f"  No gradient [{name}]")
                
        total_grad_norm = total_grad_norm ** 0.5
        self.logger.info(f"Total gradient norm: {total_grad_norm:.6f}")
        self.logger.info(f"Parameters with gradients: {param_count}")
        self.logger.info(f"Parameters with zero gradients: {zero_grad_count}")
        
        self.log_memory_usage("after backward pass")
        self.logger.info("")
        
        return total_grad_norm
        
    def test_optimizer_step(self, grad_norm: float):
        """Test optimizer step with detailed logging."""
        self.logger.info("TESTING OPTIMIZER STEP")
        self.logger.info("-" * 40)
        
        # Get initial parameter values (sample)
        first_param = next(iter(self.model.parameters()))
        initial_param_norm = first_param.norm().item()
        self.logger.info(f"Initial param norm (sample): {initial_param_norm:.6f}")
        
        # Gradient clipping
        if grad_norm > self.config.grad_clip_norm:
            self.logger.info(f"Gradient clipping: {grad_norm:.6f} -> {self.config.grad_clip_norm}")
            
        # Optimizer step
        start_time = time.time()
        
        if self.trainer.scaler is not None:
            self.trainer.scaler.unscale_(self.trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.trainer.scaler.step(self.trainer.optimizer)
            self.trainer.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.trainer.optimizer.step()
            
        self.trainer.scheduler.step()
        
        step_time = time.time() - start_time
        
        # Check parameter updates
        final_param_norm = first_param.norm().item()
        param_change = abs(final_param_norm - initial_param_norm)
        
        self.logger.info(f"Optimizer step time: {step_time:.4f}s")
        self.logger.info(f"Final param norm (sample): {final_param_norm:.6f}")
        self.logger.info(f"Parameter change: {param_change:.8f}")
        self.logger.info(f"Learning rate: {self.trainer.scheduler.get_last_lr()[0]:.8f}")
        
        if self.trainer.scaler is not None:
            self.logger.info(f"GradScaler scale: {self.trainer.scaler.get_scale()}")
            
        self.log_memory_usage("after optimizer step")
        self.logger.info("")
        
        return param_change
        
    def compute_model_flops_utilization(self, forward_time: float):
        """Compute Model FLOPs Utilization (MFU)."""
        if not torch.cuda.is_available():
            return None
            
        # GPT-2 FLOPs calculation (forward pass)
        # Rough estimate: 2 * num_params per token for forward pass
        batch_size, seq_len = 1, 20  # Approximate values
        model_params = sum(p.numel() for p in self.model.parameters())
        
        # Approximate FLOPs for transformer forward pass
        flops_per_token = 2 * model_params  # 2 for multiply-add operations
        total_flops = batch_size * seq_len * flops_per_token
        
        # FLOPs per second
        flops_per_second = total_flops / forward_time
        
        # RTX 4090 theoretical peak (mixed precision)
        # Tensor Core peak: ~165 TFLOPs/s for BF16
        rtx4090_peak_flops = 165e12
        
        mfu = (flops_per_second / rtx4090_peak_flops) * 100
        
        self.logger.info("MODEL FLOPS UTILIZATION (MFU)")
        self.logger.info("-" * 40)
        self.logger.info(f"Model parameters: {model_params:,}")
        self.logger.info(f"Estimated FLOPs: {total_flops:.2e}")
        self.logger.info(f"Forward time: {forward_time:.4f}s")
        self.logger.info(f"FLOPs/sec: {flops_per_second:.2e}")
        self.logger.info(f"RTX 4090 peak: {rtx4090_peak_flops:.2e}")
        self.logger.info(f"MFU: {mfu:.2f}%")
        self.logger.info("")
        
        return mfu
        
    def run_complete_test(self):
        """Run complete training test with detailed logging."""
        try:
            self.log_system_info()
            self.initialize_components()
            
            # Reset memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
                
            # Test tokenization
            self.test_tokenization()
            
            # Create test batch
            batch = self.create_test_batch()
            
            # Test forward pass
            start_time = time.time()
            outputs = self.test_forward_pass(batch)
            forward_time = time.time() - start_time
            
            # Test logprob computation
            current_logprobs, ref_logprobs = self.test_logprob_computation(batch)
            
            # Test GRPO loss computation
            total_loss, metrics = self.test_grpo_loss_computation(batch)
            
            # Test backward pass
            grad_norm = self.test_backward_pass(total_loss)
            
            # Test optimizer step
            param_change = self.test_optimizer_step(grad_norm)
            
            # Compute MFU
            mfu = self.compute_model_flops_utilization(forward_time)
            
            # Final summary
            self.logger.info("FINAL SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"✅ Training step completed successfully")
            self.logger.info(f"   Forward time: {forward_time:.4f}s")
            self.logger.info(f"   Total loss: {total_loss.item():.6f}")
            self.logger.info(f"   Gradient norm: {grad_norm:.6f}")
            self.logger.info(f"   Parameter change: {param_change:.8f}")
            if mfu is not None:
                self.logger.info(f"   MFU: {mfu:.2f}%")
            self.log_memory_usage("final")
            
            # Save metrics
            self.metrics = {
                'forward_time': forward_time,
                'loss': total_loss.item(),
                'gradient_norm': grad_norm,
                'parameter_change': param_change,
                'mfu': mfu,
                'memory_peak_gb': torch.cuda.max_memory_allocated(self.device) / 1e9 if torch.cuda.is_available() else None,
                **metrics
            }
            
            # Save metrics to file
            with open('detailed_training_metrics.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            self.logger.info(f"✅ Metrics saved to detailed_training_metrics.json")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Test failed with error: {e}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Detailed GRPO Training Implementation Test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable mixed precision training")
    
    args = parser.parse_args()
    
    # Run test
    tester = DetailedTrainingTester(device=args.device, use_mixed_precision=args.mixed_precision)
    success = tester.run_complete_test()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())