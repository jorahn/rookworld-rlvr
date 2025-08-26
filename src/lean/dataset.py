"""
Lean Dataset Implementation for RookWorld GRPO Training

Simple dataset loading from jrahn/rookworld_7m without complex preprocessing.
"""

import logging
from typing import List, Tuple, Iterator
from datasets import load_dataset
import random

logger = logging.getLogger(__name__)


class LeanRookWorldDataset:
    """Minimal dataset loader for RookWorld prompts"""
    
    def __init__(self, dataset_name: str = "jrahn/rookworld_7m"):
        self.dataset_name = dataset_name
        self.dataset = None
        logger.info(f"Initializing dataset: {dataset_name}")
    
    def load(self) -> None:
        """Load the dataset from HuggingFace"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)
        
        train_size = len(self.dataset['train'])
        logger.info(f"Dataset loaded - train samples: {train_size}")
        
        # Log some examples
        sample = self.dataset['train'][0]
        logger.info(f"Sample keys: {list(sample.keys())}")
        logger.info(f"Sample text preview: {sample.get('text', str(sample))[:200]}...")
    
    def get_samples(self, batch_size: int, split: str = "train") -> List[str]:
        """Get a batch of random samples"""
        if self.dataset is None:
            self.load()
        
        dataset_split = self.dataset[split]
        indices = random.sample(range(len(dataset_split)), batch_size)
        
        samples = []
        for idx in indices:
            sample = dataset_split[idx]
            text = sample.get('text', str(sample))
            samples.append(text)
        
        logger.debug(f"Retrieved {len(samples)} samples from {split}")
        return samples
    
    def parse_task_prompt(self, text: str) -> Tuple[str, str, str]:
        """
        Parse a sample into task type, prompt, and expected completion
        
        Returns:
            task_type: "P" or "A" 
            prompt: The input prompt up to the task prefix
            completion: The expected completion (for validation)
        """
        
        # Look for task prefixes
        if "P:" in text:
            task_type = "P"
            # Split on the P: to get prompt and completion parts
            if "P:" in text:
                parts = text.split("P:", 1)
                if len(parts) == 2:
                    prefix = parts[0].strip()
                    remainder = "P:" + parts[1]
                    
                    # Find where the completion starts (after M:)
                    if "M:" in remainder:
                        prompt_part = remainder.split("M:", 1)[0].strip()
                        completion_part = remainder[len(prompt_part):].strip()
                        
                        logger.debug(f"P: task - prompt: {prompt_part[:50]}..., "
                                   f"completion: {completion_part[:50]}...")
                        
                        return task_type, prompt_part, completion_part
        
        elif "A:" in text:
            task_type = "A"
            # Split on A: to get prompt and completion
            if "A:" in text:
                parts = text.split("A:", 1)
                if len(parts) == 2:
                    prefix = parts[0].strip()
                    remainder = "A:" + parts[1]
                    
                    # For A: tasks, the completion is everything after A:
                    prompt_part = remainder.split()[0] + " " + remainder.split()[1]  # "A: <FEN>+<UCI>+"
                    completion_part = " ".join(remainder.split()[2:])  # The rest
                    
                    logger.debug(f"A: task - prompt: {prompt_part[:50]}..., "
                               f"completion: {completion_part[:50]}...")
                    
                    return task_type, prompt_part, completion_part
        
        # Fallback - treat as raw text
        logger.warning(f"Could not parse task from text: {text[:100]}...")
        return "unknown", text[:50], text[50:]
    
    def get_training_batch(self, batch_size: int) -> List[Tuple[str, str, str]]:
        """Get a batch of parsed training samples"""
        raw_samples = self.get_samples(batch_size)
        
        parsed_batch = []
        for sample in raw_samples:
            task_type, prompt, completion = self.parse_task_prompt(sample)
            parsed_batch.append((task_type, prompt, completion))
        
        # Log task distribution
        p_count = sum(1 for t, _, _ in parsed_batch if t == "P")
        a_count = sum(1 for t, _, _ in parsed_batch if t == "A")
        logger.info(f"Batch composition - P: {p_count}, A: {a_count}, other: {len(parsed_batch) - p_count - a_count}")
        
        return parsed_batch