"""
Lean GRPO training script using HuggingFace transformers and TRL.
"""

import argparse
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import OnlineDPOTrainer, OnlineDPOConfig
from datasets import Dataset
import chess
import chess.pgn
from pathlib import Path
import random

from .rewards import create_reward_function


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    model_name: str = "jrahn/RookWorld-LM-124M"
    output_dir: str = "./grpo_output"
    
    # Training parameters
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 256
    
    # GRPO specific
    beta: float = 0.1  # KL penalty coefficient
    
    # Hardware optimizations
    bf16: bool = True
    use_torch_compile: bool = False
    
    # Stockfish path (optional)
    stockfish_path: str = None


class ChessDataGenerator:
    """Generate chess training data on the fly."""
    
    def __init__(self, max_length: int = 256):
        self.max_length = max_length
        self.positions = self._load_positions()
    
    def _load_positions(self) -> List[str]:
        """Load a set of chess positions (FENs)."""
        positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # Some middle game positions
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5",
        ]
        return positions
    
    def generate_batch(self, batch_size: int) -> List[str]:
        """Generate a batch of chess prompts."""
        prompts = []
        for _ in range(batch_size):
            fen = random.choice(self.positions)
            prompt = f"P:{fen} M:"
            prompts.append(prompt)
        return prompts


def create_dataset(data_generator: ChessDataGenerator, size: int = 1000) -> Dataset:
    """Create a dataset for training."""
    prompts = []
    for _ in range(size):
        prompts.extend(data_generator.generate_batch(4))
    
    # Remove duplicates and limit size
    prompts = list(set(prompts))[:size]
    
    return Dataset.from_dict({"prompt": prompts})


def main():
    parser = argparse.ArgumentParser(description="Train GRPO on chess tasks")
    parser.add_argument("--model_name", default="jrahn/RookWorld-LM-124M", help="Model to fine-tune")
    parser.add_argument("--output_dir", default="./grpo_output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--stockfish_path", default=None, help="Path to Stockfish binary")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    
    args = parser.parse_args()
    
    config = GRPOConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        stockfish_path=args.stockfish_path,
        bf16=args.bf16,
        use_torch_compile=args.compile,
    )
    
    print(f"Loading model: {config.model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading with optimizations
    model_kwargs = {}
    if config.bf16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    if config.use_torch_compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    # Create reward function
    reward_function = create_reward_function(config.stockfish_path)
    
    # Generate training data
    data_generator = ChessDataGenerator(config.max_length)
    train_dataset = create_dataset(data_generator, size=200)
    
    print(f"Created dataset with {len(train_dataset)} samples")
    
    # Training configuration
    training_args = OnlineDPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        warmup_steps=50,
        max_length=config.max_length,
        beta=config.beta,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = OnlineDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_function=reward_function,
    )
    
    print("Starting GRPO training...")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"Training completed. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()