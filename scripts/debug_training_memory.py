#!/usr/bin/env python3
"""
Trace memory consumption during training initialization
"""

import torch
import logging
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_memory_usage(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        total_allocated = 0
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(gpu_id)
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            total_allocated += gpu_allocated
            
        print(f"{stage}: GPU0={allocated:.2f}GB, Total={total_allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        print(f"{stage}: CUDA not available")

def main():
    print("=== Training Memory Tracing ===")
    
    # Set environment like training script
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print_memory_usage("1. Start")
    
    # Import training modules
    from rookworld_rlvr.train.config import GRPOConfig
    from rookworld_rlvr.model.loader import load_rookworld_model
    print_memory_usage("2. After imports")
    
    # Create config
    config = GRPOConfig(
        batch_positions=8,
        group_size=8,
        steps=50,
        max_new_tokens=144,
        max_new_tokens_env=150
    )
    print_memory_usage("3. After config")
    
    # Load models
    print("Loading training model...")
    training_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
    training_model = training_model.to("cuda:0")
    print_memory_usage("4. After training model")
    
    print("Loading reference model...")
    reference_model = load_rookworld_model("jrahn/RookWorld-LM-124M")
    reference_model = reference_model.to("cuda:1")
    print_memory_usage("5. After reference model")
    
    # Initialize policy
    from rookworld_rlvr.train.policy import CausalLMPolicy
    print("Creating policy...")
    policy = CausalLMPolicy(training_model, reference_model, config)
    print_memory_usage("6. After policy creation")
    
    # Initialize trainer components
    from rookworld_rlvr.train.grpo_trainer import GRPOTrainer
    print("Creating GRPO trainer...")
    trainer = GRPOTrainer(training_model, reference_model, config)
    print_memory_usage("7. After GRPO trainer")
    
    # Initialize data collector
    from rookworld_rlvr.data.collector import GRPODataCollector
    print("Creating data collector...")
    collector = GRPODataCollector(policy, config)
    print_memory_usage("8. After data collector")
    
    # Try collecting one batch
    print("Collecting first training batch...")
    try:
        training_data = collector.collect_training_data()
        print_memory_usage("9. After first data collection")
    except Exception as e:
        print(f"Error during data collection: {e}")
        print_memory_usage("9. After failed data collection")

if __name__ == "__main__":
    main()