"""
RookWorld Dataset Integration for GRPO Training

This module provides functionality to load and preprocess the jrahn/rookworld_7m 
dataset for GRPO training, including adding proper task prefixes and splitting 
samples into prompt and generation parts.
"""

from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import re
import random
from datasets import load_dataset, Dataset


@dataclass
class GRPOSample:
    """A single GRPO training sample with prompt and target"""
    prompt: str
    target: str
    task_type: str  # 'policy' or 'environment'
    position: str   # FEN position for reference


class RookWorldDatasetProcessor:
    """Processes the jrahn/rookworld_7m dataset for GRPO training"""
    
    def __init__(self, dataset_name: str = "jrahn/rookworld_7m"):
        """
        Initialize the dataset processor
        
        Args:
            dataset_name: HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self._processed_cache = {}
        
    def load_dataset(self, split: str = "train") -> Dataset:
        """Load the dataset from HuggingFace"""
        print(f"📥 Loading {self.dataset_name} dataset...")
        
        if self.dataset is None:
            self.dataset = load_dataset(self.dataset_name)
            
        return self.dataset[split]
    
    def identify_task_type(self, text: str) -> str:
        """
        Identify whether a sample is policy or environment task
        
        Args:
            text: Raw text from dataset
            
        Returns:
            'policy' if P: prefix, 'environment' if environment format
        """
        if text.startswith('P:'):
            return 'policy'
        elif not text.startswith('P:') and '+' in text:
            return 'environment'
        else:
            return 'unknown'
    
    def add_environment_prefix(self, text: str) -> str:
        """
        Add 'A: ' prefix to environment samples that lack it
        
        Args:
            text: Raw text sample
            
        Returns:
            Text with proper A: prefix for environment tasks
        """
        if self.identify_task_type(text) == 'environment' and not text.startswith('A: '):
            return f"A: {text}"
        return text
    
    def extract_position_from_text(self, text: str) -> Optional[str]:
        """
        Extract FEN position from the text
        
        Args:
            text: Sample text (P: or A: format)
            
        Returns:
            FEN position string if found
        """
        # Remove task prefix
        content = text
        if content.startswith('P: ') or content.startswith('A: '):
            content = content[3:]
            
        # Extract FEN (first part before any moves or analysis)
        parts = content.split()
        if len(parts) >= 6:
            # Standard FEN has 6 parts: position castling enpassant halfmove fullmove
            fen_parts = []
            for i, part in enumerate(parts):
                if i == 0:  # Position
                    fen_parts.append(part)
                elif i == 1 and part in ['w', 'b']:  # Active color
                    fen_parts.append(part)
                elif i == 2:  # Castling
                    fen_parts.append(part)
                elif i == 3:  # En passant
                    fen_parts.append(part)
                elif i == 4 and part.isdigit():  # Half-move clock
                    fen_parts.append(part)
                elif i == 5 and part.isdigit():  # Full-move number
                    fen_parts.append(part)
                    break
                else:
                    break
                    
            if len(fen_parts) == 6:
                return ' '.join(fen_parts)
        
        return None
    
    def split_policy_sample(self, text: str) -> Optional[GRPOSample]:
        """
        Split a policy sample into prompt and generation parts
        
        Policy format: P: [FEN] M: [moves] E: [evals] B: [best_move]
        Prompt: P: [FEN]
        Target: M: [moves] E: [evals] B: [best_move]
        
        Args:
            text: Policy sample text
            
        Returns:
            GRPOSample with prompt and target split
        """
        if not text.startswith('P: '):
            return None
            
        content = text[3:]  # Remove "P: " prefix
        
        # Find where the analysis starts (M:, E:, or B:)
        analysis_match = re.search(r'\s+(M:|E:|B:)', content)
        if not analysis_match:
            return None
            
        # Split at the analysis start
        position_part = content[:analysis_match.start()].strip()
        analysis_part = content[analysis_match.start():].strip()
        
        # Extract FEN position
        fen_position = self.extract_position_from_text(f"P: {position_part}")
        
        return GRPOSample(
            prompt=f"P: {position_part}",
            target=analysis_part,
            task_type='policy',
            position=fen_position or position_part
        )
    
    def split_environment_sample(self, text: str) -> Optional[GRPOSample]:
        """
        Split an environment sample into prompt and generation parts
        
        Environment format: A: [FEN]+[move]+[game_continuation]+[result_fen]+[reward]+[done_flag]+[termination]
        Prompt: A: [FEN]+[move]+
        Target: [game_continuation]+[result_fen]+[reward]+[done_flag]+[termination]
        
        Args:
            text: Environment sample text (with A: prefix)
            
        Returns:
            GRPOSample with prompt and target split
        """
        if not text.startswith('A: '):
            return None
            
        content = text[3:]  # Remove "A: " prefix
        
        # Find the pattern: FEN + move + result_analysis
        # Look for the second '+' which separates move from the rest
        plus_positions = [i for i, c in enumerate(content) if c == '+']
        
        if len(plus_positions) < 2:
            return None
            
        # Split after the second '+'  
        # Include the '+' in the prompt to match generation format
        split_position = plus_positions[1] + 1  # After the second '+'
        prompt_part = content[:split_position]  # Includes the second '+'
        target_part = content[split_position:]   # Everything after second '+'
        
        # Extract base FEN position (before the first '+')
        fen_position = self.extract_position_from_text(content[:plus_positions[0]])
        
        return GRPOSample(
            prompt=f"A: {prompt_part}",  # Already includes the '+' at the end
            target=target_part,
            task_type='environment',
            position=fen_position or prompt_part.split('+')[0]
        )
    
    def process_sample(self, raw_text: str) -> Optional[GRPOSample]:
        """
        Process a raw dataset sample into a GRPO sample
        
        Args:
            raw_text: Raw text from dataset
            
        Returns:
            Processed GRPOSample or None if processing fails
        """
        # Add environment prefix if needed
        text = self.add_environment_prefix(raw_text)
        
        # Split based on task type
        if text.startswith('P: '):
            return self.split_policy_sample(text)
        elif text.startswith('A: '):
            return self.split_environment_sample(text)
        else:
            return None
    
    def process_dataset(self, split: str = "train", max_samples: Optional[int] = None) -> Iterator[GRPOSample]:
        """
        Process the entire dataset split into GRPO samples
        
        Args:
            split: Dataset split to process
            max_samples: Maximum number of samples to process (None for all)
            
        Yields:
            Processed GRPOSample instances
        """
        dataset_split = self.load_dataset(split)
        
        total_samples = len(dataset_split)
        if max_samples:
            total_samples = min(total_samples, max_samples)
            
        print(f"🔄 Processing {total_samples:,} samples from {split} split...")
        
        processed_count = 0
        failed_count = 0
        
        for i in range(total_samples):
            raw_text = dataset_split[i]['text']
            
            try:
                sample = self.process_sample(raw_text)
                if sample:
                    processed_count += 1
                    yield sample
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 10:  # Only log first 10 failures
                    print(f"⚠️ Failed to process sample {i}: {e}")
        
        print(f"✅ Processing complete:")
        print(f"   📊 Processed: {processed_count:,} samples")
        print(f"   ❌ Failed: {failed_count:,} samples") 
        print(f"   📈 Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
    
    def create_balanced_batch(self, samples: List[GRPOSample], batch_size: int, 
                             policy_ratio: float = 0.8) -> List[GRPOSample]:
        """
        Create a balanced batch with specified policy/environment ratio
        
        Args:
            samples: Pool of samples to choose from
            batch_size: Target batch size
            policy_ratio: Ratio of policy samples (0.8 = 80% policy, 20% environment)
            
        Returns:
            Balanced batch of samples
        """
        # Separate samples by type
        policy_samples = [s for s in samples if s.task_type == 'policy']
        env_samples = [s for s in samples if s.task_type == 'environment']
        
        # Calculate target counts
        target_policy = int(batch_size * policy_ratio)
        target_env = batch_size - target_policy
        
        # Sample with replacement if needed
        batch = []
        
        if len(policy_samples) >= target_policy:
            batch.extend(random.sample(policy_samples, target_policy))
        else:
            batch.extend(policy_samples)
            # Fill remainder with replacement
            remaining = target_policy - len(policy_samples)
            batch.extend(random.choices(policy_samples, k=remaining))
        
        if len(env_samples) >= target_env:
            batch.extend(random.sample(env_samples, target_env))
        else:
            batch.extend(env_samples)
            # Fill remainder with replacement
            remaining = target_env - len(env_samples)
            batch.extend(random.choices(env_samples, k=remaining))
        
        # Shuffle the final batch
        random.shuffle(batch)
        return batch


def test_dataset_processing():
    """Test the dataset processing pipeline"""
    print("🧪 Testing RookWorld dataset processing...")
    
    processor = RookWorldDatasetProcessor()
    
    # Process a small sample for testing
    samples = list(processor.process_dataset(split="train", max_samples=100))
    
    print(f"\n📊 Test Results:")
    print(f"   Processed samples: {len(samples)}")
    
    # Count by task type
    policy_count = sum(1 for s in samples if s.task_type == 'policy')
    env_count = sum(1 for s in samples if s.task_type == 'environment')
    
    print(f"   Policy samples: {policy_count}")
    print(f"   Environment samples: {env_count}")
    
    # Show examples
    if samples:
        print(f"\n📝 Example samples:")
        for i, sample in enumerate(samples[:3]):
            print(f"\n   Sample {i+1} ({sample.task_type}):")
            print(f"      Prompt: {sample.prompt[:80]}...")
            print(f"      Target: {sample.target[:80]}...")
    
    # Test balanced batch creation
    if samples:
        balanced_batch = processor.create_balanced_batch(samples, batch_size=10, policy_ratio=0.8)
        policy_in_batch = sum(1 for s in balanced_batch if s.task_type == 'policy')
        env_in_batch = sum(1 for s in balanced_batch if s.task_type == 'environment')
        
        print(f"\n🎯 Balanced batch test (batch_size=10, policy_ratio=0.8):")
        print(f"   Policy in batch: {policy_in_batch} (target: 8)")
        print(f"   Environment in batch: {env_in_batch} (target: 2)")


if __name__ == "__main__":
    test_dataset_processing()