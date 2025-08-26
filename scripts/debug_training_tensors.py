#!/usr/bin/env python3
"""
Debug the actual tensor shapes during training to find memory culprit
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_tensor_info(name, tensor):
    if tensor is not None:
        shape = tuple(tensor.shape)
        numel = tensor.numel()
        memory_mb = numel * tensor.element_size() / 1024**2
        print(f"  {name}: {shape} = {numel:,} elements = {memory_mb:.1f}MB")
    else:
        print(f"  {name}: None")

def main():
    print("=== Training Tensor Analysis ===")
    
    torch.cuda.empty_cache()
    
    # Load minimal setup
    from rookworld_rlvr.train.config import GRPOConfig
    from rookworld_rlvr.model.loader import load_rookworld_model
    from rookworld_rlvr.train.policy import CausalLMPolicy
    from rookworld_rlvr.data.collector import GRPODataCollector
    
    config = GRPOConfig(
        batch_positions=2,
        group_size=4,
        steps=5,
        max_new_tokens=144,
        max_new_tokens_env=150,
        max_positions=200
    )
    
    # Load models
    training_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
    training_model = training_model.to("cuda:0")
    
    if torch.cuda.device_count() > 1:
        reference_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
        reference_model = reference_model.to("cuda:1")
    else:
        reference_model = training_model
    
    policy = CausalLMPolicy(training_model, reference_model, config)
    collector = GRPODataCollector(policy, config)
    
    print(f"Memory after setup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    
    # Try to collect one training batch
    print("Collecting training data...")
    try:
        training_data = collector.collect_mixed_batch(batch_size=1)  # Just 1 group
        
        if training_data:
            batch = training_data[0]
            print("Collected batch tensors:")
            print_tensor_info("input_ids", batch.input_ids)
            print_tensor_info("attention_mask", batch.attention_mask)
            print_tensor_info("target_start_indices", batch.target_start_indices)
            print_tensor_info("old_logprobs", batch.old_logprobs)
            print_tensor_info("rewards", batch.rewards)
            
            print(f"Memory after collection: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            # Try to compute current logprobs (this is where OOM happens)
            print("Computing current logprobs...")
            
            # Move tensors to GPU if needed
            input_ids = batch.input_ids.to("cuda:0")
            attention_mask = batch.attention_mask.to("cuda:0")
            target_start_indices = batch.target_start_indices.to("cuda:0")
            
            print("Batch tensors on GPU:")
            print_tensor_info("input_ids", input_ids)
            print_tensor_info("attention_mask", attention_mask)
            
            print(f"Memory after moving to GPU: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            # Actual shapes that cause OOM
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]
            print(f"Sequence length: {seq_len}")
            print(f"Batch size: {batch_size}")
            
            # Calculate expected logits memory
            vocab_size = training_model.config.vocab_size
            expected_logits_size = batch_size * seq_len * vocab_size
            expected_memory_gb = expected_logits_size * 4 / 1024**3  # FP32
            print(f"Expected logits: [{batch_size}, {seq_len}, {vocab_size}] = {expected_memory_gb:.2f}GB")
            
            # Try forward pass
            try:
                print("Attempting forward pass...")
                with torch.no_grad():
                    outputs = training_model(input_ids, attention_mask=attention_mask)
                    logits = outputs["logits"]
                    print(f"Actual logits shape: {logits.shape}")
                    actual_memory = logits.numel() * 4 / 1024**3
                    print(f"Actual logits memory: {actual_memory:.2f}GB")
                    print(f"Memory after forward: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            except Exception as e:
                print(f"Forward pass failed: {e}")
                print(f"Memory at failure: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        else:
            print("No training data collected")
            
    except Exception as e:
        print(f"Collection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()