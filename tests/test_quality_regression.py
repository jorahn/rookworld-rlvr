#!/usr/bin/env python3
"""
Quality Regression Detection Test

Validates that optimizations maintain training quality by comparing
reward distributions, format validity, and training dynamics before/after changes.
"""

import torch
import time
import json
import sys
import numpy as np
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.dataset import load_and_prepare_samples
from rookworld_rlvr.reward_scorer import RewardScorer


class QualityRegressionTester:
    """Detects quality regressions in training optimizations."""
    
    def __init__(self):
        self.baseline_file = Path(__file__).parent / "quality_baseline.json"
        
    def analyze_training_run(self, steps: int = 5, **train_kwargs) -> Dict[str, Any]:
        """Run training and extract quality metrics."""
        
        # Run training with specified parameters
        log_dir = f"logs_quality_test_{int(time.time())}"
        
        cmd = [
            "uv", "run", "python", "scripts/train_logged.py",
            "--steps", str(steps),
            "--batch_size", "8",
            "--k_samples", "8", 
            "--lr", "1e-5",
            "--log_dir", log_dir,
            "--n_train_samples", "100"  # Small for faster testing
        ]
        
        # Add any additional flags
        for key, value in train_kwargs.items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr[-500:]}
            
            # Parse training history
            history_file = Path(__file__).parent.parent / log_dir / "training_history.jsonl"
            if not history_file.exists():
                return {'success': False, 'error': 'No history file'}
            
            # Read training records
            with open(history_file, 'r') as f:
                records = [json.loads(line) for line in f]
            
            if not records:
                return {'success': False, 'error': 'No training records'}
            
            # Extract quality metrics
            mean_rewards = [r['mean_reward'] for r in records]
            losses = [r['loss'] for r in records]
            step_times = [r['elapsed_time'] for r in records]
            
            # Parse log file for reward distributions and completion quality
            log_file = Path(__file__).parent.parent / log_dir / f"grpo_training_*.log"
            log_files = list(log_file.parent.glob(log_file.name))
            
            completion_analysis = {}
            if log_files:
                with open(log_files[0], 'r') as f:
                    log_content = f.read()
                
                # Count reward distribution patterns
                reward_counts = {}
                for line in log_content.split('\n'):
                    if 'Reward distribution:' in line:
                        # Extract reward distribution
                        import re
                        match = re.search(r"Reward distribution: ({.*})", line)
                        if match:
                            dist_str = match.group(1)
                            try:
                                dist = eval(dist_str)  # Parse the dict string
                                for reward_key, count in dist.items():
                                    reward_counts[reward_key] = reward_counts.get(reward_key, 0) + count
                            except:
                                pass
                
                completion_analysis['reward_distribution'] = reward_counts
                
                # Count format validity indicators
                format_valid_count = log_content.count('format_valid": true')
                format_invalid_count = log_content.count('format_valid": false')
                total_format_checks = format_valid_count + format_invalid_count
                
                completion_analysis['format_validity'] = {
                    'valid_count': format_valid_count,
                    'invalid_count': format_invalid_count, 
                    'validity_ratio': format_valid_count / max(1, total_format_checks)
                }
            
            return {
                'success': True,
                'steps_completed': len(records),
                'mean_rewards': mean_rewards,
                'final_mean_reward': mean_rewards[-1],
                'reward_progression': mean_rewards[-1] - mean_rewards[0] if len(mean_rewards) > 1 else 0,
                'losses': losses,
                'step_times': step_times,
                'avg_step_time': np.mean(step_times),
                'completion_analysis': completion_analysis,
                'cmd': ' '.join(cmd)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_quality_baseline(self) -> Dict[str, Any]:
        """Create baseline quality metrics from current known-good code."""
        print("📊 Creating Quality Baseline")
        print("=" * 50)
        
        baseline = self.analyze_training_run(steps=3)
        
        if not baseline['success']:
            raise Exception(f"Baseline creation failed: {baseline['error']}")
        
        print(f"✅ Baseline created:")
        print(f"  Steps: {baseline['steps_completed']}")
        print(f"  Final mean reward: {baseline['final_mean_reward']:.3f}")
        print(f"  Avg step time: {baseline['avg_step_time']:.2f}s")
        print(f"  Reward distribution: {baseline['completion_analysis'].get('reward_distribution', {})}")
        
        # Save baseline
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        return baseline
    
    def load_quality_baseline(self) -> Dict[str, Any]:
        """Load existing quality baseline."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def validate_against_baseline(self, test_results: Dict[str, Any], tolerance: Dict[str, float] = None) -> Dict[str, Any]:
        """Validate test results against quality baseline."""
        
        if tolerance is None:
            tolerance = {
                'mean_reward_drop': 0.10,  # Max 10% drop in mean reward
                'step_time_increase': 0.05,  # Max 5% increase in step time (for optimizations)
                'format_validity_drop': 0.05  # Max 5% drop in format validity
            }
        
        baseline = self.load_quality_baseline()
        if not baseline:
            return {'validated': False, 'error': 'No baseline found'}
        
        validation_results = {
            'validated': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        # Validate mean reward
        baseline_reward = baseline['final_mean_reward']
        test_reward = test_results['final_mean_reward']
        reward_change = (test_reward - baseline_reward) / baseline_reward
        
        validation_results['metrics']['reward_change'] = reward_change
        
        if reward_change < -tolerance['mean_reward_drop']:
            validation_results['errors'].append(
                f"Mean reward regression: {baseline_reward:.3f} → {test_reward:.3f} ({reward_change:.1%})"
            )
            validation_results['validated'] = False
        elif reward_change < -0.02:  # 2% warning threshold
            validation_results['warnings'].append(
                f"Mean reward decrease: {baseline_reward:.3f} → {test_reward:.3f} ({reward_change:.1%})"
            )
        
        # Validate step time (for optimizations, faster is good)
        baseline_time = baseline['avg_step_time']
        test_time = test_results['avg_step_time'] 
        time_change = (test_time - baseline_time) / baseline_time
        
        validation_results['metrics']['time_change'] = time_change
        
        if time_change > tolerance['step_time_increase']:
            validation_results['errors'].append(
                f"Step time regression: {baseline_time:.2f}s → {test_time:.2f}s ({time_change:.1%})"
            )
            validation_results['validated'] = False
        
        # Validate reward distribution similarity
        baseline_dist = baseline['completion_analysis'].get('reward_distribution', {})
        test_dist = test_results['completion_analysis'].get('reward_distribution', {})
        
        # Check if high-value rewards (0.8, 0.9, 1.0) are maintained
        high_value_keys = ['0.8', '0.9', '1.0']
        baseline_high_count = sum(baseline_dist.get(key, 0) for key in high_value_keys)
        test_high_count = sum(test_dist.get(key, 0) for key in high_value_keys)
        
        baseline_total = sum(baseline_dist.values()) or 1
        test_total = sum(test_dist.values()) or 1
        
        baseline_high_ratio = baseline_high_count / baseline_total
        test_high_ratio = test_high_count / test_total
        
        high_value_change = (test_high_ratio - baseline_high_ratio) / max(baseline_high_ratio, 0.01)
        validation_results['metrics']['high_value_reward_change'] = high_value_change
        
        if high_value_change < -0.2:  # 20% drop in high-value rewards
            validation_results['errors'].append(
                f"High-value reward ratio drop: {baseline_high_ratio:.1%} → {test_high_ratio:.1%}"
            )
            validation_results['validated'] = False
        
        return validation_results
    
    def test_optimization_quality(self, optimization_name: str, **train_kwargs) -> bool:
        """Test if an optimization maintains quality standards."""
        print(f"\n🧪 Testing Quality Impact: {optimization_name}")
        print("=" * 60)
        
        # Ensure baseline exists
        baseline = self.load_quality_baseline()
        if not baseline:
            print("Creating baseline first...")
            baseline = self.create_quality_baseline()
        
        # Test optimization
        print(f"Running test with: {train_kwargs}")
        test_results = self.analyze_training_run(steps=3, **train_kwargs)
        
        if not test_results['success']:
            print(f"❌ Test failed: {test_results['error']}")
            return False
        
        # Validate against baseline
        validation = self.validate_against_baseline(test_results)
        
        print(f"\n📊 Quality Analysis:")
        print(f"  Baseline mean reward: {baseline['final_mean_reward']:.3f}")
        print(f"  Test mean reward: {test_results['final_mean_reward']:.3f}")
        print(f"  Change: {validation['metrics']['reward_change']:.1%}")
        
        print(f"  Baseline step time: {baseline['avg_step_time']:.2f}s")
        print(f"  Test step time: {test_results['avg_step_time']:.2f}s") 
        print(f"  Change: {validation['metrics']['time_change']:.1%}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"  ⚠️ {warning}")
        
        if validation['errors']:
            for error in validation['errors']:
                print(f"  ❌ {error}")
            print(f"\n❌ Quality regression detected!")
            return False
        else:
            print(f"\n✅ Quality maintained!")
            return True


def test_current_baseline_quality():
    """Validate current code produces expected quality levels."""
    print("🎯 Testing Current Baseline Quality")
    print("=" * 60)
    
    tester = QualityRegressionTester()
    baseline = tester.create_quality_baseline()
    
    # Validate quality expectations
    final_reward = baseline['final_mean_reward']
    reward_dist = baseline['completion_analysis'].get('reward_distribution', {})
    
    print(f"\n📊 Quality Standards Check:")
    print(f"  Final mean reward: {final_reward:.3f}")
    
    # Check reward distribution health
    high_rewards = sum(reward_dist.get(key, 0) for key in ['0.8', '0.9', '1.0'])
    total_rewards = sum(reward_dist.values()) or 1
    high_ratio = high_rewards / total_rewards
    
    print(f"  High-value reward ratio: {high_ratio:.1%}")
    print(f"  Reward distribution: {reward_dist}")
    
    # Quality thresholds
    assert final_reward > 0.4, f"Mean reward too low: {final_reward:.3f}"
    assert high_ratio > 0.2, f"High-value reward ratio too low: {high_ratio:.1%}"
    
    print(f"✅ Baseline quality meets standards")
    return baseline


def test_batch_scoring_quality():
    """Test if batch scoring maintains quality compared to sequential."""
    print("\n🔬 Testing Batch Scoring Quality Impact")
    print("=" * 60)
    
    # Load test data
    samples = load_and_prepare_samples(n_samples=20, seed=42)
    prompts = [s[1] for s in samples]
    completions = [s[2] for s in samples]  # Use ground truth
    
    scorer = RewardScorer(
        reward_shaping="graduated",
        continuous_components={"fen_similarity": "exponential", "evaluations": "linear"}
    )
    
    print(f"Testing {len(samples)} samples...")
    
    # Sequential scoring
    sequential_rewards = []
    for prompt, completion in zip(prompts, completions):
        reward, _ = scorer.score_single(prompt, completion, log_details=False)
        sequential_rewards.append(reward)
    
    # Batch scoring  
    batch_rewards, batch_details = scorer.score_batch(prompts, completions, compute_advantages=False)
    
    # Analyze quality
    seq_rewards = np.array(sequential_rewards)
    
    print(f"\n📊 Scoring Method Comparison:")
    print(f"  Sequential mean: {seq_rewards.mean():.3f}")
    print(f"  Batch mean: {batch_rewards.mean():.3f}")
    print(f"  Max difference: {np.abs(seq_rewards - batch_rewards).max():.6f}")
    
    # Count high-quality results
    seq_high = (seq_rewards > 0.5).sum()
    batch_high = (batch_rewards > 0.5).sum()
    
    print(f"  Sequential high rewards (>0.5): {seq_high}/{len(samples)}")
    print(f"  Batch high rewards (>0.5): {batch_high}/{len(samples)}")
    
    # Validate identical results
    max_diff = np.abs(seq_rewards - batch_rewards).max()
    assert max_diff < 1e-6, f"Batch scoring differs from sequential: {max_diff}"
    
    print("✅ Batch scoring maintains identical quality")
    return True


if __name__ == "__main__":
    print("🚀 Quality Regression Detection System")
    print("=" * 70)
    
    try:
        # Test 1: Establish baseline quality standards
        baseline_results = test_current_baseline_quality()
        
        # Test 2: Validate batch scoring doesn't affect quality
        test_batch_scoring_quality()
        
        # Test 3: Create testing framework for future optimizations
        tester = QualityRegressionTester()
        
        print(f"\n🔧 Quality Testing Framework Ready:")
        print(f"  ✅ Baseline established: {baseline_results['final_mean_reward']:.3f} mean reward")
        print(f"  ✅ Batch scoring validated: Identical results to sequential")
        print(f"  ✅ Regression detection: Ready for optimization testing")
        
        print(f"\n💡 Usage for future optimizations:")
        print(f"  1. Run: python tests/test_quality_regression.py")
        print(f"  2. Test optimizations with: tester.test_optimization_quality()")
        print(f"  3. Automatic validation against quality standards")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Quality test failed: {e}")
        raise