#!/usr/bin/env python3
"""
Test Integrated GRPO Data Collector

Test script for the integrated data collector that uses RookWorld dataset.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.data.integrated_collector import (
    IntegratedGRPODataCollector,
    IntegratedGRPOConfig
)


def test_integrated_collector():
    """Test the integrated data collector"""
    print("🧪 Testing Integrated GRPO Data Collector...")
    
    # Mock policy for testing
    class MockPolicy:
        def generate_batch(self, prompts, config):
            return {
                'texts': [f"M: e2e4 d7d5 g1f3 E: 0.2 0.3 0.4 B: e2e4" for i in range(len(prompts))],
                'seq_logprob': [-0.5] * len(prompts)
            }
    
    # Create collector
    config = IntegratedGRPOConfig(
        use_rookworld_dataset=True,
        dataset_buffer_size=50,  # Small for testing
        group_size=4,
        dataset_mix_ratio=1.0  # 100% dataset for this test
    )
    
    collector = IntegratedGRPODataCollector(
        policy=MockPolicy(),
        config=config
    )
    
    # Test batch collection
    print("\n🚀 Testing mixed batch collection...")
    batch = collector.collect_mixed_batch(batch_size=3)
    
    print(f"\n📊 Test Results:")
    print(f"   Collected {len(batch)} groups")
    
    for i, group in enumerate(batch):
        print(f"\n   Group {i+1}:")
        print(f"      Task type: {group['task_type']}")
        print(f"      Prompts: {len(group['prompts'])}")
        print(f"      Generations: {len(group['generated_texts'])}")
        print(f"      Rewards: {[f'{r:.2f}' for r in group['rewards']]}")
        print(f"      Sample prompt: {group['prompts'][0][:60]}...")
        print(f"      Ground truth: {group['ground_truth'][:60]}...")
    
    print(f"\n✅ Integrated collector test completed successfully!")


if __name__ == "__main__":
    test_integrated_collector()