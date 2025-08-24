#!/usr/bin/env python3
"""
Examine RookWorld Dataset Structure

This script loads and examines the jrahn/rookworld_7m dataset to understand
its structure and prepare for GRPO training integration.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
import json


def examine_dataset():
    """Load and examine the rookworld_7m dataset structure"""
    
    print("="*80)
    print("EXAMINING ROOKWORLD_7M DATASET")
    print("="*80)
    
    try:
        # Load the dataset
        print("📥 Loading dataset jrahn/rookworld_7m...")
        dataset = load_dataset("jrahn/rookworld_7m")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset structure: {dataset}")
        print()
        
        # Examine splits
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"📋 Split: {split_name}")
            print(f"   Size: {len(split):,} samples")
            print(f"   Features: {list(split.features.keys())}")
            print()
        
        # Examine samples from the main split
        main_split = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        print(f"🔍 Examining samples from '{list(dataset.keys())[0]}' split:")
        print(f"   Total samples: {len(main_split):,}")
        print()
        
        # Show first few samples
        print("📝 First 5 samples:")
        for i in range(min(5, len(main_split))):
            sample = main_split[i]
            print(f"\n--- Sample {i+1} ---")
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}...")
                else:
                    print(f"{key}: {value}")
        
        # Analyze text patterns
        print("\n" + "="*50)
        print("ANALYZING TEXT PATTERNS")
        print("="*50)
        
        # Get text column name
        text_column = None
        for col in main_split.features.keys():
            if 'text' in col.lower() or col in ['content', 'sample', 'data']:
                text_column = col
                break
        
        if text_column:
            print(f"📄 Analyzing text column: '{text_column}'")
            
            # Sample analysis
            policy_count = 0
            env_count = 0
            other_count = 0
            
            samples_to_check = min(1000, len(main_split))
            
            for i in range(samples_to_check):
                text = main_split[i][text_column]
                if text.startswith('P:'):
                    policy_count += 1
                elif '+' in text and not text.startswith('A:'):
                    env_count += 1
                else:
                    other_count += 1
            
            print(f"\n📊 Pattern analysis (first {samples_to_check} samples):")
            print(f"   P: (Policy) samples: {policy_count}")
            print(f"   Environment samples (no A: prefix): {env_count}")
            print(f"   Other samples: {other_count}")
            print(f"   Policy percentage: {policy_count/samples_to_check*100:.1f}%")
            print(f"   Environment percentage: {env_count/samples_to_check*100:.1f}%")
            
            # Show example patterns
            print(f"\n📋 Example patterns:")
            
            # Find policy example
            for i in range(min(100, len(main_split))):
                text = main_split[i][text_column]
                if text.startswith('P:'):
                    print(f"   Policy example: {text[:150]}...")
                    break
            
            # Find environment example
            for i in range(min(100, len(main_split))):
                text = main_split[i][text_column]
                if '+' in text and not text.startswith('A:'):
                    print(f"   Environment example: {text[:150]}...")
                    break
        
        else:
            print("❓ Could not identify text column")
            
        return dataset, main_split, text_column
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def analyze_grpo_compatibility(dataset_split, text_column):
    """Analyze how the dataset can be adapted for GRPO training"""
    
    print("\n" + "="*50)
    print("GRPO COMPATIBILITY ANALYSIS")
    print("="*50)
    
    if not dataset_split or not text_column:
        print("❌ Cannot analyze - dataset not loaded")
        return
    
    # Sample some texts for analysis
    sample_size = min(100, len(dataset_split))
    
    print(f"🔍 Analyzing {sample_size} samples for GRPO compatibility...")
    
    compatible_samples = []
    needs_prefix_samples = []
    other_samples = []
    
    for i in range(sample_size):
        text = dataset_split[i][text_column]
        
        if text.startswith('P:'):
            compatible_samples.append(('policy', text))
        elif '+' in text and not text.startswith('A:'):
            needs_prefix_samples.append(('environment', text))
        else:
            other_samples.append(('other', text))
    
    print(f"\n📊 GRPO Compatibility Results:")
    print(f"   ✅ Already compatible (P: prefix): {len(compatible_samples)}")
    print(f"   🔄 Needs A: prefix (environment): {len(needs_prefix_samples)}")
    print(f"   ❓ Other/unknown format: {len(other_samples)}")
    
    # Show examples of what needs to be fixed
    if needs_prefix_samples:
        print(f"\n📝 Examples needing A: prefix:")
        for i, (sample_type, text) in enumerate(needs_prefix_samples[:3]):
            print(f"   Example {i+1}:")
            print(f"      Original: {text[:100]}...")
            fixed_text = f"A: {text}"
            print(f"      Fixed:    {fixed_text[:100]}...")
    
    return {
        'compatible': len(compatible_samples),
        'needs_prefix': len(needs_prefix_samples), 
        'other': len(other_samples),
        'total': sample_size,
        'examples': {
            'compatible': compatible_samples[:3],
            'needs_prefix': needs_prefix_samples[:3],
            'other': other_samples[:3]
        }
    }


if __name__ == "__main__":
    dataset, main_split, text_column = examine_dataset()
    
    if dataset:
        compatibility = analyze_grpo_compatibility(main_split, text_column)
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print("="*80)
        print(f"✅ Dataset loaded: jrahn/rookworld_7m")
        print(f"📊 Size: {len(main_split):,} samples")
        print(f"📄 Text column: '{text_column}'")
        
        if compatibility:
            print(f"🎯 GRPO Ready: {compatibility['compatible']} samples")
            print(f"🔄 Need A: prefix: {compatibility['needs_prefix']} samples")
            print(f"❓ Other: {compatibility['other']} samples")
            print(f"📈 Usability: {(compatibility['compatible'] + compatibility['needs_prefix'])/compatibility['total']*100:.1f}%")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Add 'A: ' prefix to environment samples")
        print(f"   2. Split samples into prompt and generation parts")
        print(f"   3. Integrate with GRPO data collector")