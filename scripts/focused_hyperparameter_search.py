#!/usr/bin/env python3
"""
Focused Hyperparameter Search - Target Most Promising Ranges

Based on initial analysis, focus on the most likely successful combinations
with a smaller, more manageable grid size.
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
    temperature: float
    mix_env_ratio: float
    batch_positions: int
    group_size: int
    kl_estimator: str = "kl3"
    steps: int = 30           # Slightly longer runs
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
            # Add our improved parameters
            "--kl-divergence-threshold", "10.0",  # Much higher threshold
            "--kl-warmup-steps", "10",           # Warmup for first 10 steps
            "--kl-warmup-factor", "0.0",         # No KL penalty during warmup
            "--reward-warmup-steps", "10",       # Reward curriculum warmup
            "--new-run"
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

class FocusedHyperparameterSearch:
    """Focused hyperparameter search targeting promising ranges."""
    
    def __init__(self, output_dir: str = "focused_hyperparameter_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "focused_search.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def define_focused_grid(self) -> List[ExperimentConfig]:
        """Define focused grid on most promising parameter ranges."""
        
        # Aggressive learning rates for improved system with warmup
        learning_rates = [1e-5, 2e-5, 5e-5]
        
        # Much higher KL coefficients that work with our warmup and increased threshold
        kl_coefficients = [0.05, 0.1, 0.2]
        
        # Standard clip ranges
        clip_ranges = [0.1, 0.2]
        
        # Normal temperatures
        temperatures = [0.5, 0.7, 1.0]
        
        # Mix ratios: policy-only and light mixing
        mix_env_ratios = [0.0, 0.2]  # Focus on policy-only and light mixing
        
        # Small batch sizes for stability
        batch_configs = [
            (1, 2),  # Very small
            (2, 4),  # Small
        ]
        
        # Both KL estimators
        kl_estimators = ["kl1", "kl3"]
        
        configs = []
        exp_id = 0
        
        # Generate all combinations - smaller grid so we can afford full combinatorial
        for lr in learning_rates:
            for kl_coef in kl_coefficients:
                for clip_range in clip_ranges:
                    for temp in temperatures:
                        for mix_ratio in mix_env_ratios:
                            for kl_est in kl_estimators:
                                for batch_pos, group_size in batch_configs:
                                    config = ExperimentConfig(
                                        lr=lr,
                                        kl_coef=kl_coef,
                                        clip_range=clip_range,
                                        temperature=temp,
                                        mix_env_ratio=mix_ratio,
                                        kl_estimator=kl_est,
                                        batch_positions=batch_pos,
                                        group_size=group_size,
                                        exp_id=f"focus_{exp_id:03d}"
                                    )
                                    configs.append(config)
                                    exp_id += 1
        
        self.logger.info(f"Generated {len(configs)} focused experiment configurations")
        return configs
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with extended monitoring."""
        self.logger.info(f"Running {config.exp_id}: lr={config.lr:.0e}, kl={config.kl_coef}, "
                        f"clip={config.clip_range}, temp={config.temperature}")
        
        start_time = time.time()
        
        try:
            cmd = ["uv", "run", "python", "train_rookworld_grpo.py"] + config.to_args()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes per experiment
            )
            
            training_time = time.time() - start_time
            
            # Parse results regardless of success/failure
            kl_div, policy_reward, env_reward, steps = self._parse_output(result.stdout)
            
            if result.returncode == 0:
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
                training_time=600.0,
                error_message="Timeout after 10 minutes"
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
            elif "Average Reward:" in line:
                try:
                    reward_str = line.split("Average Reward:")[1].strip()
                    if "Policy" in line:
                        policy_reward = float(reward_str)
                    elif "Environment" in line:
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
    
    def run_focused_search(self) -> List[ExperimentResult]:
        """Run focused hyperparameter search."""
        configs = self.define_focused_grid()
        results = []
        
        self.logger.info(f"Starting focused search with {len(configs)} experiments")
        self.logger.info(f"Expected completion time: ~{len(configs) * 5} minutes")
        
        successful_runs = 0
        
        for i, config in enumerate(configs, 1):
            self.logger.info(f"Progress: {i}/{len(configs)} experiments ({i/len(configs)*100:.1f}%)")
            
            result = self.run_experiment(config)
            results.append(result)
            
            if result.success:
                successful_runs += 1
                status = "✅ SUCCESS"
                self.logger.info(f"{status} KL={result.kl_divergence:.3f}, "
                                f"Steps={result.steps_completed}, Time={result.training_time:.1f}s")
            else:
                status = "❌ FAILED"
                self.logger.info(f"{status} Error: {result.error_message}")
            
            # Log running success rate
            success_rate = successful_runs / i * 100
            self.logger.info(f"Running success rate: {success_rate:.1f}% ({successful_runs}/{i})")
            
            # Save intermediate results
            self._save_results(results)
            
            # Brief pause
            time.sleep(2)
        
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
        
        with open(self.output_dir / "focused_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def analyze_results(self, results: List[ExperimentResult]):
        """Analyze and report focused search results."""
        successful_runs = [r for r in results if r.success]
        failed_runs = [r for r in results if not r.success]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("FOCUSED HYPERPARAMETER SEARCH RESULTS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total experiments: {len(results)}")
        self.logger.info(f"Successful runs: {len(successful_runs)} ({len(successful_runs)/len(results)*100:.1f}%)")
        self.logger.info(f"Failed runs: {len(failed_runs)} ({len(failed_runs)/len(results)*100:.1f}%)")
        
        if successful_runs:
            # Find best configurations
            best_kl = min(successful_runs, key=lambda x: abs(x.kl_divergence) if x.kl_divergence != float('inf') else float('inf'))
            
            self.logger.info(f"\n🏆 BEST STABLE CONFIGURATION:")
            self.logger.info(f"KL Divergence: {best_kl.kl_divergence:.3f}")
            self.logger.info(f"Steps Completed: {best_kl.steps_completed}")
            self.logger.info(f"Policy Reward: {best_kl.policy_reward:.3f}")
            self.logger.info(f"Config: lr={best_kl.config.lr:.0e}, kl_coef={best_kl.config.kl_coef}, "
                           f"clip={best_kl.config.clip_range}, temp={best_kl.config.temperature}")
            
            # Parameter analysis for successful runs
            self._analyze_successful_parameters(successful_runs)
        
        if failed_runs:
            self.logger.info(f"\n❌ FAILURE ANALYSIS:")
            error_counts = {}
            for run in failed_runs:
                error = run.error_message
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {error}: {count} occurrences ({count/len(failed_runs)*100:.1f}%)")
    
    def _analyze_successful_parameters(self, successful_runs: List[ExperimentResult]):
        """Analyze parameter patterns in successful runs."""
        if not successful_runs:
            return
            
        self.logger.info(f"\n📊 SUCCESSFUL PARAMETER ANALYSIS:")
        
        # Learning rate analysis
        lr_success = {}
        for run in successful_runs:
            lr = run.config.lr
            lr_success[lr] = lr_success.get(lr, 0) + 1
        
        self.logger.info("\nLearning Rate Success:")
        total_successful = len(successful_runs)
        for lr, count in sorted(lr_success.items()):
            self.logger.info(f"  lr={lr:.0e}: {count} successes ({count/total_successful*100:.1f}%)")
        
        # KL coefficient analysis
        kl_success = {}
        for run in successful_runs:
            kl_coef = run.config.kl_coef
            kl_success[kl_coef] = kl_success.get(kl_coef, 0) + 1
        
        self.logger.info("\nKL Coefficient Success:")
        for kl_coef, count in sorted(kl_success.items()):
            self.logger.info(f"  kl_coef={kl_coef}: {count} successes ({count/total_successful*100:.1f}%)")

def main():
    """Run focused hyperparameter search."""
    search = FocusedHyperparameterSearch()
    results = search.run_focused_search()
    search.analyze_results(results)

if __name__ == "__main__":
    main()