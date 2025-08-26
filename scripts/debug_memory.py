#!/usr/bin/env python3
"""
Memory profiling script to identify where the massive memory allocation happens
"""

import torch
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_memory_usage(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"{stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
    else:
        print(f"{stage}: CUDA not available")

def main():
    print("=== Memory Profiling Analysis ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print_memory_usage("1. Initial state")
    
    # Import modules
    from rookworld_rlvr.model.loader import load_rookworld_model
    print_memory_usage("2. After imports")
    
    # Create and load first model
    print("Creating and loading first model...")
    model1 = load_rookworld_model("jrahn/RookWorld-LM-124M")
    print_memory_usage("3. After loading first model")
    
    # Move to device
    print("Moving to GPU 0...")
    model1 = model1.to("cuda:0")
    print_memory_usage("4. After moving model1 to cuda:0")
    
    # Create and load second model
    print("Creating and loading second model...")
    model2 = load_rookworld_model("jrahn/RookWorld-LM-124M")
    print_memory_usage("5. After loading second model")
    
    # Move to second GPU
    print("Moving to GPU 1...")
    model2 = model2.to("cuda:1")
    print_memory_usage("6. After moving model2 to cuda:1")
    
    # Check memory on both GPUs
    print("\n=== Per-GPU Memory Usage ===")
    for gpu in [0, 1]:
        torch.cuda.set_device(gpu)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU {gpu}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    # Test a small forward pass
    torch.cuda.set_device(0)
    print("\n=== Testing Forward Pass ===")
    with torch.no_grad():
        # Create small batch
        input_ids = torch.randint(0, 50256, (4, 100)).to("cuda:0")  # BS=4, seq_len=100
        attention_mask = torch.ones_like(input_ids).to("cuda:0")
        
        print_memory_usage("8. Before forward pass")
        outputs = model1(input_ids=input_ids, attention_mask=attention_mask)
        print_memory_usage("9. After forward pass")
        
        # Check the actual size of outputs
        logits = outputs["logits"]
        print(f"Output logits shape: {logits.shape}")
        print(f"Output logits memory: {logits.element_size() * logits.numel() / 1024**2:.2f}MB")

if __name__ == "__main__":
    main()