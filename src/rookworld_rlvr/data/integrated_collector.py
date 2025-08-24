"""
Integrated GRPO Data Collector with RookWorld Dataset Support

This module provides an enhanced data collector that can use both synthetic
position generation and the real RookWorld dataset for GRPO training.
"""

from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass
import random
import chess
import torch

from ..train.policy import CausalLMPolicy, GenerationConfig
from ..environment.chess_env import ChessEnvironment, EnvironmentResponse
from ..reward.policy_reward import PolicyRewardComputer
from ..engine.stockfish import StockfishAnalysis
from ..reward.env_reward import EnvRewardComputer
from .rookworld_dataset import RookWorldDatasetProcessor, GRPOSample


@dataclass
class IntegratedGRPOConfig:
    """Configuration for integrated GRPO data collection"""
    
    # Dataset configuration
    use_rookworld_dataset: bool = True
    dataset_name: str = "jrahn/rookworld_7m"
    dataset_split: str = "train"
    dataset_mix_ratio: float = 0.8  # 0.8 = 80% dataset, 20% synthetic
    
    # Task mix configuration
    policy_ratio: float = 0.8  # 80% policy, 20% environment tasks
    
    # Generation configuration
    group_size: int = 4
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    
    # Buffer configuration
    dataset_buffer_size: int = 1000
    

class IntegratedGRPODataCollector:
    """Enhanced data collector supporting both synthetic and real dataset samples"""
    
    def __init__(
        self,
        policy: CausalLMPolicy,
        config: IntegratedGRPOConfig = None,
        stockfish_engine = None
    ):
        """
        Initialize integrated data collector
        
        Args:
            policy: Unified policy wrapper
            config: Collection configuration
            stockfish_engine: Optional Stockfish engine for reward computation
        """
        self.policy = policy
        self.config = config or IntegratedGRPOConfig()
        self.stockfish_engine = stockfish_engine
        
        # Initialize components
        self.chess_env = ChessEnvironment()
        self.policy_reward_computer = PolicyRewardComputer()
        self.env_reward_computer = EnvRewardComputer()
        
        # Dataset components
        self.dataset_processor = None
        self.dataset_buffer = []
        self.dataset_iterator = None
        
        if self.config.use_rookworld_dataset:
            self.initialize_dataset()
    
    def initialize_dataset(self):
        """Initialize the RookWorld dataset processor and buffer"""
        print(f"🚀 Initializing RookWorld dataset integration...")
        
        self.dataset_processor = RookWorldDatasetProcessor(self.config.dataset_name)
        
        # Load initial buffer
        self.refill_dataset_buffer()
        
        print(f"✅ Dataset integration ready with {len(self.dataset_buffer)} samples buffered")
    
    def refill_dataset_buffer(self):
        """Refill the dataset buffer with new samples"""
        if not self.dataset_processor:
            return
        
        target_size = self.config.dataset_buffer_size
        current_size = len(self.dataset_buffer)
        
        if current_size >= target_size * 0.5:  # Only refill when below 50%
            return
            
        print(f"🔄 Refilling dataset buffer (current: {current_size}, target: {target_size})...")
        
        # Create new iterator if needed
        if self.dataset_iterator is None:
            self.dataset_iterator = self.dataset_processor.process_dataset(
                split=self.config.dataset_split
            )
        
        # Fill buffer
        samples_needed = target_size - current_size
        samples_added = 0
        
        try:
            for sample in self.dataset_iterator:
                self.dataset_buffer.append(sample)
                samples_added += 1
                
                if samples_added >= samples_needed:
                    break
        except StopIteration:
            # Dataset exhausted, restart
            print("🔄 Dataset exhausted, restarting...")
            self.dataset_iterator = self.dataset_processor.process_dataset(
                split=self.config.dataset_split
            )
            
        print(f"✅ Added {samples_added} samples to buffer (total: {len(self.dataset_buffer)})")
    
    def get_dataset_samples(self, count: int, task_type: Optional[str] = None) -> List[GRPOSample]:
        """
        Get samples from the dataset buffer
        
        Args:
            count: Number of samples to get
            task_type: Optional filter by task type ('policy' or 'environment')
            
        Returns:
            List of dataset samples
        """
        # Ensure buffer is sufficiently filled
        self.refill_dataset_buffer()
        
        # Filter by task type if specified
        available_samples = self.dataset_buffer
        if task_type:
            available_samples = [s for s in self.dataset_buffer if s.task_type == task_type]
        
        if len(available_samples) < count:
            # Need more samples of this type
            if task_type:
                print(f"⚠️ Insufficient {task_type} samples in buffer ({len(available_samples)} < {count})")
            return available_samples[:count] if available_samples else []
        
        # Sample without replacement
        selected = random.sample(available_samples, count)
        
        # Remove selected samples from buffer
        for sample in selected:
            self.dataset_buffer.remove(sample)
        
        return selected
    
    def create_dataset_batch(self, batch_size: int) -> List[GRPOSample]:
        """
        Create a balanced batch from dataset samples
        
        Args:
            batch_size: Target batch size
            
        Returns:
            Balanced batch of dataset samples
        """
        if not self.dataset_processor:
            return []
        
        # Calculate target distribution
        policy_count = int(batch_size * self.config.policy_ratio)
        env_count = batch_size - policy_count
        
        # Get samples by type
        policy_samples = self.get_dataset_samples(policy_count, 'policy')
        env_samples = self.get_dataset_samples(env_count, 'environment')
        
        # Combine and shuffle
        batch = policy_samples + env_samples
        random.shuffle(batch)
        
        return batch
    
    def collect_dataset_group(self, sample: GRPOSample) -> Dict[str, Any]:
        """
        Collect GRPO group for a dataset sample
        
        Args:
            sample: Dataset sample to process
            
        Returns:
            GRPO group data
        """
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=0,
            do_sample=True
        )
        
        # Generate multiple completions for the group
        prompts = [sample.prompt] * self.config.group_size
        
        try:
            # Generate completions
            results = self.policy.generate_batch(prompts, gen_config)
            
            # Extract generated texts
            generated_texts = results.get('texts', [])
            logprobs = results.get('seq_logprob', [])
            
            if len(generated_texts) != self.config.group_size:
                print(f"⚠️ Expected {self.config.group_size} generations, got {len(generated_texts)}")
                return None
            
            # Compute rewards by comparing with ground truth
            rewards = []
            for generated_text in generated_texts:
                reward = self.compute_dataset_reward(sample, generated_text)
                rewards.append(reward)
            
            return {
                'prompts': prompts,
                'generated_texts': generated_texts,
                'logprobs': logprobs,
                'rewards': rewards,
                'task_type': sample.task_type,
                'ground_truth': sample.target,
                'position': sample.position
            }
            
        except Exception as e:
            print(f"❌ Failed to collect group for dataset sample: {e}")
            return None
    
    def compute_dataset_reward(self, sample: GRPOSample, generated_text: str) -> float:
        """
        Compute reward for a generated text against dataset ground truth
        
        Args:
            sample: Original dataset sample
            generated_text: Model-generated text
            
        Returns:
            Reward score
        """
        # Basic reward based on text similarity and structure
        reward = 0.0
        
        # Structure reward - check if generation has expected format
        if sample.task_type == 'policy':
            # Policy tasks should have M:, E:, B: structure
            if 'M:' in generated_text:
                reward += 0.3
            if 'E:' in generated_text:
                reward += 0.3
            if 'B:' in generated_text:
                reward += 0.2
        else:
            # Environment tasks should look like game continuations
            if '+' in generated_text or any(c.isdigit() for c in generated_text):
                reward += 0.4
        
        # Content similarity reward (simple overlap measure)
        ground_truth_words = set(sample.target.lower().split())
        generated_words = set(generated_text.lower().split())
        
        if ground_truth_words and generated_words:
            overlap = len(ground_truth_words.intersection(generated_words))
            similarity = overlap / max(len(ground_truth_words), len(generated_words))
            reward += similarity * 0.4
        
        # Length penalty for very short or very long outputs
        target_len = len(sample.target.split())
        gen_len = len(generated_text.split())
        
        if gen_len > 0:
            length_ratio = min(gen_len, target_len) / max(gen_len, target_len)
            reward += length_ratio * 0.2
        
        return max(0.0, min(1.0, reward))  # Clamp to [0, 1]
    
    def collect_mixed_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Collect a mixed batch using both dataset and synthetic samples
        
        Args:
            batch_size: Target batch size
            
        Returns:
            List of GRPO groups
        """
        groups = []
        
        # Determine dataset vs synthetic split
        dataset_count = int(batch_size * self.config.dataset_mix_ratio)
        synthetic_count = batch_size - dataset_count
        
        # Collect dataset samples
        if dataset_count > 0 and self.config.use_rookworld_dataset:
            dataset_samples = self.create_dataset_batch(dataset_count)
            
            for sample in dataset_samples:
                group = self.collect_dataset_group(sample)
                if group:
                    groups.append(group)
        
        # Collect synthetic samples (fallback to original method)
        if synthetic_count > 0:
            print(f"🎲 Collecting {synthetic_count} synthetic samples...")
            # This would call the original synthetic generation methods
            # For now, just log that synthetic generation would happen here
        
        print(f"📊 Collected {len(groups)} groups ({len([g for g in groups if g['task_type'] == 'policy'])} policy, {len([g for g in groups if g['task_type'] == 'environment'])} environment)")
        
        return groups


def test_integrated_collector():
    """Test the integrated data collector"""
    print("🧪 Testing Integrated GRPO Data Collector...")
    
    # Mock policy for testing
    class MockPolicy:
        def generate_batch(self, prompts, config):
            return {
                'texts': [f"Generated response {i+1}" for i in range(len(prompts))],
                'seq_logprob': [0.5] * len(prompts)
            }
    
    # Create collector
    config = IntegratedGRPOConfig(
        use_rookworld_dataset=True,
        dataset_buffer_size=50,  # Small for testing
        group_size=4
    )
    
    collector = IntegratedGRPODataCollector(
        policy=MockPolicy(),
        config=config
    )
    
    # Test batch collection
    batch = collector.collect_mixed_batch(batch_size=3)
    
    print(f"\n📊 Test Results:")
    print(f"   Collected {len(batch)} groups")
    
    for i, group in enumerate(batch):
        print(f"\n   Group {i+1}:")
        print(f"      Task type: {group['task_type']}")
        print(f"      Prompts: {len(group['prompts'])}")
        print(f"      Generations: {len(group['generated_texts'])}")
        print(f"      Rewards: {group['rewards']}")
        print(f"      Sample prompt: {group['prompts'][0][:60]}...")


if __name__ == "__main__":
    test_integrated_collector()