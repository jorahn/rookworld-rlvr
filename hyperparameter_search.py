#!/usr/bin/env python3
"""
Principled Hyperparameter Grid Search for RookWorld GRPO Training

This script implements a sparse grid search over critical hyperparameters
to find stable training configurations that avoid KL divergence.
"""

import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

@dataclass
class ExperimentConfig:
    """Configuration for a single hyperparameter experiment."""
    lr: float
    kl_coef: float
    clip_range: float
    temperature: float         # Sampling temperature
    mix_env_ratio: float      # Environment task ratio
    batch_positions: int
    group_size: int
    kl_estimator: str = "kl3" # KL estimator type
    steps: int = 20           # Short runs for grid search
    exp_id: str = ""
    
    def to_args(self) -> List[str]:
        """Convert to command line arguments."""
        return [
            "--steps", str(self.steps),
            "--lr", str(self.lr),
            "--kl-coef", str(self.kl_coef),
            "--clip-range", str(self.clip_range),
            "--temperature", str(self.temperature),
            "--mix-env-ratio", str(self.mix_env_ratio),
            "--batch-positions", str(self.batch_positions),
            "--group-size", str(self.group_size),
            "--kl-estimator", str(self.kl_estimator),
            "--stockfish-path", "/usr/games/stockfish",
            "--new-run"  # Force new run for each experiment
        ]

@dataclass
class ExperimentResult:
    """Results from a hyperparameter experiment."""
    config: ExperimentConfig
    success: bool
    kl_divergence: float
    policy_reward: float
    env_reward: float
    steps_completed: int
    training_time: float
    error_message: str = ""

class HyperparameterSearch:
    """Manages hyperparameter grid search experiments."""
    
    def __init__(self, output_dir: str = "hyperparameter_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "grid_search.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def define_search_grid(self) -> List[ExperimentConfig]:
        """Define sparse hyperparameter grid focusing on stability."""
        
        # Learning rates: cover several orders of magnitude
        learning_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
        
        # KL coefficients: from very conservative to standard  
        kl_coefficients = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        # Clip ranges: from very conservative to standard
        clip_ranges = [0.05, 0.1, 0.2]
        
        # Temperature: affects sampling randomness
        temperatures = [0.3, 0.7, 1.0]  # Conservative to standard
        
        # Environment task ratio: affects task mix
        mix_env_ratios = [0.0, 0.2, 0.5]  # Policy-only to mixed
        
        # KL estimators: different stability characteristics  
        kl_estimators = ["kl1", "kl3"]  # Simple vs quadratic
        
        # Batch configurations: smaller for stability
        batch_configs = [
            (1, 2),  # Very small for stability testing
            (2, 4),  # Small
            (4, 8),  # Medium
        ]  # (batch_positions, group_size)
        
        configs = []
        exp_id = 0
        
        # Generate sparse grid - not full combinatorial to keep reasonable size
        for lr in learning_rates:
            for kl_coef in kl_coefficients:
                for clip_range in clip_ranges:
                    for temp in temperatures:
                        for mix_ratio in mix_env_ratios:
                            for kl_est in kl_estimators:
                                for batch_pos, group_size in batch_configs:
                                    # Skip some combinations to keep grid sparse
                                    if (lr >= 1e-5 and kl_coef >= 0.02):  # Skip high lr + high kl
                                        continue
                                    if (clip_range >= 0.2 and temp >= 1.0):  # Skip high clip + high temp
                                        continue
                                    if (batch_pos >= 4 and group_size >= 8):  # Skip large batches for now
                                        continue
                                        
                                    config = ExperimentConfig(
                                        lr=lr,
                                        kl_coef=kl_coef, 
                                        clip_range=clip_range,
                                        temperature=temp,
                                        mix_env_ratio=mix_ratio,
                                        kl_estimator=kl_est,
                                        batch_positions=batch_pos,
                                        group_size=group_size,
                                        exp_id=f"exp_{exp_id:03d}"
                                    )
                                    configs.append(config)
                                    exp_id += 1
        
        self.logger.info(f"Generated {len(configs)} experiment configurations")
        return configs
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single hyperparameter experiment."""
        self.logger.info(f"Running experiment {config.exp_id}: lr={config.lr}, kl={config.kl_coef}, clip={config.clip_range}")
        
        start_time = time.time()
        
        try:
            # Run training with timeout
            cmd = ["uv", "run", "python", "train_rookworld_grpo.py"] + config.to_args()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per experiment
            )
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse success metrics from output
                kl_div, policy_reward, env_reward, steps = self._parse_output(result.stdout)
                
                return ExperimentResult(
                    config=config,
                    success=True,
                    kl_divergence=kl_div,
                    policy_reward=policy_reward,
                    env_reward=env_reward,
                    steps_completed=steps,
                    training_time=training_time
                )
            else:
                # Parse failure info
                kl_div, policy_reward, env_reward, steps = self._parse_output(result.stdout)
                error_msg = self._extract_error_message(result.stdout, result.stderr)
                
                return ExperimentResult(
                    config=config,
                    success=False,
                    kl_divergence=kl_div,
                    policy_reward=policy_reward,
                    env_reward=env_reward,
                    steps_completed=steps,
                    training_time=training_time,
                    error_message=error_msg
                )
                
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                config=config,
                success=False,
                kl_divergence=float('inf'),
                policy_reward=-1.0,
                env_reward=-1.0,
                steps_completed=0,
                training_time=300.0,
                error_message="Timeout after 5 minutes"
            )
        except Exception as e:
            return ExperimentResult(
                config=config,
                success=False,
                kl_divergence=float('inf'),
                policy_reward=-1.0,
                env_reward=-1.0,
                steps_completed=0,
                training_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _parse_output(self, stdout: str) -> Tuple[float, float, float, int]:
        """Parse metrics from training output."""
        kl_divergence = float('inf')
        policy_reward = -1.0
        env_reward = -1.0
        steps_completed = 0
        
        lines = stdout.split('\n')
        for line in lines:
            if "KL divergence mean=" in line:
                try:
                    kl_str = line.split("mean=")[1].split()[0]
                    kl_divergence = float(kl_str)
                except:
                    pass
            elif "Average Reward:" in line and "Policy" in line:
                try:
                    reward_str = line.split("Average Reward:")[1].strip()
                    policy_reward = float(reward_str)
                except:
                    pass
            elif "Average Reward:" in line and "Environment" in line:
                try:
                    reward_str = line.split("Average Reward:")[1].strip()  
                    env_reward = float(reward_str)
                except:
                    pass
            elif "Steps Completed:" in line:
                try:
                    steps_str = line.split("Steps Completed:")[1].strip()
                    steps_completed = int(steps_str)
                except:
                    pass
        
        return kl_divergence, policy_reward, env_reward, steps_completed
    
    def _extract_error_message(self, stdout: str, stderr: str) -> str:
        """Extract relevant error message."""
        if "Training diverged" in stdout:
            return "KL divergence too high"
        elif "RuntimeError" in stderr:
            lines = stderr.split('\n')
            for line in lines:
                if "RuntimeError:" in line:
                    return line.split("RuntimeError: ")[1]
        elif stderr:
            return stderr.split('\n')[-2] if stderr.split('\n')[-2] else "Unknown error"
        return "Training failed"
    
    def run_grid_search(self) -> List[ExperimentResult]:
        """Run complete hyperparameter grid search."""
        configs = self.define_search_grid()
        results = []
        
        self.logger.info(f"Starting grid search with {len(configs)} experiments")
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"Progress: {i}/{len(configs)} experiments")
            
            result = self.run_experiment(config)
            results.append(result)
            
            # Log result
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            self.logger.info(
                f"{status} {config.exp_id}: KL={result.kl_divergence:.3f}, "
                f"P_reward={result.policy_reward:.3f}, steps={result.steps_completed}"
            )
            
            # Save intermediate results
            self._save_results(results)
            
            # Brief pause to avoid system overload
            time.sleep(5)
        
        return results
    
    def _save_results(self, results: List[ExperimentResult]):
        """Save results to JSON file."""
        results_data = []
        for result in results:
            results_data.append({
                "config": {
                    "exp_id": result.config.exp_id,
                    "lr": result.config.lr,
                    "kl_coef": result.config.kl_coef,
                    "clip_range": result.config.clip_range,
                    "temperature": result.config.temperature,
                    "mix_env_ratio": result.config.mix_env_ratio,
                    "kl_estimator": result.config.kl_estimator,
                    "batch_positions": result.config.batch_positions,
                    "group_size": result.config.group_size,
                    "steps": result.config.steps,
                },
                "results": {
                    "success": result.success,
                    "kl_divergence": result.kl_divergence,
                    "policy_reward": result.policy_reward,
                    "env_reward": result.env_reward,
                    "steps_completed": result.steps_completed,
                    "training_time": result.training_time,
                    "error_message": result.error_message,
                }
            })
        
        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def analyze_results(self, results: List[ExperimentResult]):
        """Analyze and report results."""
        successful_runs = [r for r in results if r.success]
        failed_runs = [r for r in results if not r.success]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("HYPERPARAMETER SEARCH RESULTS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total experiments: {len(results)}")
        self.logger.info(f"Successful runs: {len(successful_runs)}")
        self.logger.info(f"Failed runs: {len(failed_runs)}")
        
        if successful_runs:
            # Find best configurations
            best_kl = min(successful_runs, key=lambda x: x.kl_divergence)
            best_reward = max(successful_runs, key=lambda x: (x.policy_reward + x.env_reward))
            
            self.logger.info(f"\n🏆 BEST CONFIGURATIONS:")
            self.logger.info(f"\nLowest KL Divergence ({best_kl.kl_divergence:.3f}):")
            self._log_config(best_kl.config)
            
            self.logger.info(f"\nBest Combined Reward ({best_reward.policy_reward + best_reward.env_reward:.3f}):")
            self._log_config(best_reward.config)
            
            # Analysis by parameter
            self._analyze_parameter_effects(successful_runs)
        
        if failed_runs:
            self.logger.info(f"\n❌ FAILURE ANALYSIS:")
            error_counts = {}
            for run in failed_runs:
                error = run.error_message
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {error}: {count} occurrences")
    
    def _log_config(self, config: ExperimentConfig):
        """Log configuration details."""
        self.logger.info(f"  lr={config.lr}, kl_coef={config.kl_coef}, clip={config.clip_range}")
        self.logger.info(f"  temp={config.temperature}, mix_env={config.mix_env_ratio}, estimator={config.kl_estimator}")
        self.logger.info(f"  batch={config.batch_positions}x{config.group_size}")
    
    def _analyze_parameter_effects(self, successful_runs: List[ExperimentResult]):
        """Analyze which parameters lead to better outcomes."""
        self.logger.info(f"\n📊 PARAMETER ANALYSIS:")
        
        # Group by learning rate
        lr_groups = {}
        for run in successful_runs:
            lr = run.config.lr
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(run.kl_divergence)
        
        self.logger.info("\nLearning Rate Analysis:")
        for lr, kl_values in sorted(lr_groups.items()):
            avg_kl = sum(kl_values) / len(kl_values)
            self.logger.info(f"  lr={lr}: avg_kl={avg_kl:.3f} ({len(kl_values)} runs)")

def main():
    """Run hyperparameter grid search."""
    search = HyperparameterSearch()
    results = search.run_grid_search()
    search.analyze_results(results)

if __name__ == "__main__":
    main()