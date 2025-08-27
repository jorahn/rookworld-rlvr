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
            
            # CRITICAL: Preprocess the text
            # If it doesn't start with "P: ", it's an A: task and needs the prefix
            if not text.startswith("P: "):
                text = "A: " + text
                logger.debug(f"Added 'A: ' prefix to sample {idx}")
            
            samples.append(text)
        
        logger.debug(f"Retrieved {len(samples)} samples from {split}")
        return samples
    
    def parse_task_prompt(self, text: str) -> Tuple[str, str, str]:
        """
        Parse a sample into task type, prompt, and expected completion
        
        New format requirements:
        - P: tasks: prompt="P: [FEN]" → completion="M: [top-5-moves in UCI] E: [centipawn eval after top-5-moves] B: [best-move in UCI]"
        - A: tasks: prompt="A: [FEN]+[move in UCI]+[comma separated move history]+" → completion="[new FEN]+[reward]+[terminated]+[truncated]"
        
        Returns:
            task_type: "P" or "A" 
            prompt: The input prompt according to new spec
            completion: The expected completion (for validation)
        """
        
        # Clean up text
        text = text.strip()
        
        # Look for task prefixes
        if text.startswith("P: ") or text.startswith("P:"):
            task_type = "P"
            # Split on the P: to get prompt and completion parts
            if text.startswith("P: "):
                remainder = text  # Already has proper format
            else:
                parts = text.split("P:", 1)
                if len(parts) == 2:
                    remainder = "P: " + parts[1].strip()  # Add space after P:
                else:
                    remainder = text  # Fallback to original text
            
            # Find where the completion starts (after FEN)
            if "M:" in remainder:
                # Extract FEN part only for the prompt
                fen_part = remainder.split("M:", 1)[0].strip()
                prompt_part = fen_part  # Just "P: [FEN]"
                completion_part = "M:" + remainder.split("M:", 1)[1].strip()
                
                logger.debug(f"P: task - prompt: {prompt_part[:50]}..., "
                           f"completion: {completion_part[:50]}...")
                
                return task_type, prompt_part, completion_part
            else:
                # No M: found, treat P: <FEN> as prompt, expect M: E: B: completion
                prompt_part = remainder.strip()
                return task_type, prompt_part, ""
        
        elif text.startswith("A: ") or text.startswith("A:"):
            task_type = "A"
            # Handle A: tasks with proper spacing
            if text.startswith("A: "):
                remainder = text  # Already has proper format
            else:
                parts = text.split("A:", 1)
                if len(parts) == 2:
                    remainder = "A: " + parts[1].strip()  # Add space after A:
                else:
                    remainder = text  # Fallback
            
            # For A: tasks, look for the + delimiter pattern
            # CORRECT FORMAT per spec:
            # Prompt: "A: [FEN]+[move in UCI]+[comma separated move history]+"
            # Completion: "[new FEN]+[reward]+[terminated]+[truncated]"
            if "+" in remainder:
                # Split by + to separate components
                components = remainder.split("+")
                
                # Dataset format for A: tasks (with prefix):
                # A: FEN+move+history+new_FEN+reward+terminated+truncated
                # First 3 components = prompt, rest = completion
                
                if len(components) >= 7:
                    # Full format with all components
                    # Prompt: A: FEN+move+history+
                    prompt_part = components[0] + "+" + components[1] + "+" + components[2] + "+"
                    # Completion: new_FEN+reward+terminated+truncated
                    completion_part = "+".join(components[3:])
                elif len(components) >= 3:
                    # Might have partial data, assume first 3 are prompt parts
                    prompt_part = components[0] + "+" + components[1] + "+" + components[2] + "+"
                    completion_part = "+".join(components[3:]) if len(components) > 3 else ""
                elif len(components) == 2:
                    # Old format: FEN+move only (no history)
                    prompt_part = components[0] + "+" + components[1] + "+,"  # Add empty history
                    completion_part = ""
                else:
                    # Malformed, but try to handle
                    prompt_part = remainder
                    completion_part = ""
            else:
                # Space-separated format (older style) - convert to new format
                tokens = remainder.split()
                if len(tokens) >= 2:
                    # Try to identify FEN (has / chars) and move
                    fen_end = 1
                    for i, token in enumerate(tokens[1:], 1):
                        if "/" not in token and i > 6:  # FEN typically has 6+ parts
                            break
                        fen_end = i + 1
                    
                    if fen_end + 1 < len(tokens):
                        fen_part = " ".join(tokens[:fen_end])
                        move_part = tokens[fen_end]
                        # No history available in old format
                        prompt_part = f"A: {fen_part}+{move_part}+,"  # Empty history
                        completion_part = " ".join(tokens[fen_end+1:]) if fen_end+1 < len(tokens) else ""
                    else:
                        prompt_part = remainder
                        completion_part = ""
                else:
                    prompt_part = remainder
                    completion_part = ""
            
            logger.debug(f"A: task - prompt: {prompt_part[:50]}..., "
                       f"completion: {completion_part[:50]}...")
            
            return task_type, prompt_part, completion_part
        
        # Check for texts that don't start with P: or A: - these should be A: tasks
        else:
            # Per user instruction: if text doesn't start with P: or A:, it's an A: task
            task_type = "A"
            # Add A: prefix
            remainder = "A: " + text
            
            # Now parse as A: task
            # For A: tasks, look for the + delimiter pattern
            # CORRECT FORMAT per spec:
            # Prompt: "A: [FEN]+[move in UCI]+[comma separated move history]+"
            # Completion: "[new FEN]+[reward]+[terminated]+[truncated]"
            if "+" in remainder:
                # Split by + to separate components
                components = remainder.split("+")
                
                # Original dataset format (without prefix):
                # FEN+move+history+new_FEN+reward+terminated+truncated
                # After adding "A: ": A: FEN+move+history+new_FEN+reward+terminated+truncated
                # First 3 components = prompt, rest = completion
                
                if len(components) >= 7:
                    # Full format with all components
                    # Prompt: A: FEN+move+history+
                    prompt_part = components[0] + "+" + components[1] + "+" + components[2] + "+"
                    # Completion: new_FEN+reward+terminated+truncated
                    completion_part = "+".join(components[3:])
                elif len(components) >= 3:
                    # Might have partial data, assume first 3 are prompt parts
                    prompt_part = components[0] + "+" + components[1] + "+" + components[2] + "+"
                    completion_part = "+".join(components[3:]) if len(components) > 3 else ""
                elif len(components) == 2:
                    # Old format: FEN+move only (no history)
                    prompt_part = components[0] + "+" + components[1] + "+,"  # Add empty history
                    completion_part = ""
                else:
                    # Malformed, but try to handle
                    prompt_part = remainder
                    completion_part = ""
            else:
                # No + delimiter, might be a different format
                prompt_part = remainder
                completion_part = ""
            
            logger.debug(f"Inferred A: task (no prefix) - prompt: {prompt_part[:50]}...")
            return task_type, prompt_part, completion_part
    
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