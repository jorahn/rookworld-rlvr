#!/usr/bin/env python3
"""
Profile memory usage step-by-step to find the 23GB leak
"""

import torch
import logging
import sys
from pathlib import Path
import os
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def detailed_memory_report():
    """Print detailed memory breakdown"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    
    print(f"GPU {device}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
    
    # Get memory summary
    summary = torch.cuda.memory_summary(device)
    lines = summary.split('\n')
    for line in lines:
        if 'allocated' in line.lower() or 'reserved' in line.lower():
            print(f"  {line.strip()}")

def main():
    print("=== Memory Leak Profiling ===")
    
    # Start clean
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    detailed_memory_report()
    
    # Import without config changes
    from rookworld_rlvr.train.config import GRPOConfig
    
    # Create minimal config that matches sweep
    config = GRPOConfig(
        batch_positions=2,
        group_size=4,
        steps=5,  # Just 5 steps for profiling
        max_new_tokens=144,
        max_new_tokens_env=150,
        max_positions=200,  # Already reduced
        use_mixed_precision=True,
        use_torch_compile=False  # Disable for cleaner profiling
    )
    print(f"Config created - max_positions={config.max_positions}")
    detailed_memory_report()
    
    # Load training model only
    from rookworld_rlvr.model.loader import load_rookworld_model
    print("Loading training model...")
    training_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
    print(f"Model parameters: {training_model.get_num_params():,}")
    detailed_memory_report()
    
    print("Moving to GPU...")
    training_model = training_model.to("cuda:0")
    detailed_memory_report()
    
    # Check model config
    print(f"Model vocab size: {training_model.config.vocab_size}")
    print(f"Model n_embd: {training_model.config.n_embd}")
    print(f"Model n_positions: {training_model.config.n_positions}")
    
    # Load reference model to second GPU if available
    if torch.cuda.device_count() > 1:
        print("Loading reference model to GPU 1...")
        reference_model = load_rookworld_model("jrahn/RookWorld-LM-124M") 
        reference_model = reference_model.to("cuda:1")
        print("GPU 0:")
        torch.cuda.set_device(0)
        detailed_memory_report()
        print("GPU 1:")
        torch.cuda.set_device(1)
        detailed_memory_report()
        torch.cuda.set_device(0)  # Back to GPU 0
    else:
        print("Single GPU - loading reference to same device")
        reference_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
        reference_model = reference_model.to("cuda:0")
        detailed_memory_report()
    
    # Create policy wrapper 
    from rookworld_rlvr.train.policy import CausalLMPolicy
    print("Creating policy...")
    policy = CausalLMPolicy(training_model, reference_model, config)
    detailed_memory_report()
    
    # Test single forward pass with actual batch
    print("Testing forward pass...")
    batch_size = config.batch_positions * config.group_size  # 2 * 4 = 8
    seq_len = 50  # Reasonable prompt length
    
    # Create realistic input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda:0")
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = training_model.forward(input_ids, return_dict=True)
        logits = outputs["logits"]
        print(f"Logits shape: {logits.shape}")
        print(f"Logits memory: {logits.numel() * 4 / 1024**3:.2f}GB (FP32)")
        detailed_memory_report()
    
    # Test generation (this is where OOM happens)
    print("Testing generation...")
    try:
        with torch.no_grad():
            generated = training_model.generate(
                input_ids[:1],  # Single sample
                max_new_tokens=50,  # Short generation
                temperature=0.7
            )
            print(f"Generated shape: {generated.shape}")
            detailed_memory_report()
    except Exception as e:
        print(f"Generation failed: {e}")
        detailed_memory_report()
    
    # Check for any large tensors
    print("\nLooking for large tensors...")
    total_params = 0
    for name, param in training_model.named_parameters():
        param_size = param.numel() * 4 / 1024**3  # Assume FP32
        if param_size > 0.1:  # > 100MB
            print(f"  {name}: {param.shape} = {param_size:.3f}GB")
        total_params += param.numel()
    
    print(f"Total parameters: {total_params:,} = {total_params * 4 / 1024**3:.3f}GB")
    
    # Final memory check
    print("\nFinal memory state:")
    detailed_memory_report()

if __name__ == "__main__":
    main()