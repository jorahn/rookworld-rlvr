"""
Single Batch Training Test with Comprehensive Logging

This test runs the GRPO training pipeline on a minimal configuration:
- 2 samples (1 policy P:, 1 environment A:)  
- Group size 2
- 1 training step
- Full logging of all training components

Logs the following for each component:
- Prompt
- Completion  
- Expected completion (ground truth)
- Format validation result
- Individual component rewards
- Total reward
- Final loss value
"""

import pytest
import torch
import chess
import logging
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rookworld_rlvr.train.config import GRPOConfig
from rookworld_rlvr.train.grpo_trainer import GRPOTrainer, GRPOBatch, GRPOTrainingStep
from rookworld_rlvr.train.policy import CausalLMPolicy, GenerationConfig
from rookworld_rlvr.data.collector import GRPODataCollector, GRPOCollectionConfig
from rookworld_rlvr.engine.stockfish import StockfishEngine, StockfishAnalysis
from rookworld_rlvr.reward.policy_reward import PolicyRewardComputer, PolicyRewardConfig
from rookworld_rlvr.reward.env_reward import EnvRewardComputer, EnvRewardConfig
from rookworld_rlvr.model.gpt2 import GPT2Model
from rookworld_rlvr.model.config import GPT2Config
from rookworld_rlvr.model.loader import load_pretrained_model
from rookworld_rlvr.environment.chess_env import ChessEnvironment


class TestSingleBatchTraining:
    """Test single batch training with comprehensive logging."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal GRPO configuration for testing."""
        return GRPOConfig(
            # Training configuration 
            steps=1,                      # Only 1 step
            batch_positions=2,            # 2 samples total
            group_size=2,                 # Group size 2 
            
            # Task mixing (1 policy + 1 env)
            mix_env_ratio=0.5,            # 50% env tasks = 1 env, 1 policy
            
            # Model and generation  
            model_name="gpt2",            # Use base GPT-2 for testing
            max_new_tokens_policy=50,
            max_new_tokens_env=80,
            temperature=0.3,
            
            # Training hyperparameters
            lr=1e-6,
            kl_coef=0.001,
            clip_range=0.05,
            
            # Optimizations (disabled for testing)
            mixed_precision=False,
            torch_compile=False,
            
            # Logging
            log_level="DEBUG",
            verbose_logging=True
        )

    @pytest.fixture  
    def test_positions(self):
        """Create test chess positions for training."""
        return [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"  # After 1.e4
        ]

    def setup_method(self):
        """Set up test environment with comprehensive logging."""
        # Configure detailed logging for test
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
        
        # Create logger for detailed test output
        self.logger = logging.getLogger("SingleBatchTest")
        self.logger.setLevel(logging.DEBUG)
        
        # Storage for logged data
        self.training_log = {
            "samples": [],
            "rewards": [],
            "losses": [],
            "step_data": {}
        }

    def log_sample_data(self, sample_idx: int, task_type: str, prompt: str, 
                       completion: str, expected: str, format_valid: bool,
                       reward_components: Dict[str, float], total_reward: float):
        """Log comprehensive data for a single training sample."""
        sample_data = {
            "sample_idx": sample_idx,
            "task_type": task_type,
            "prompt": prompt,
            "completion": completion,
            "expected_completion": expected,
            "format_validation_passed": format_valid,
            "reward_components": reward_components,
            "total_reward": total_reward
        }
        
        self.training_log["samples"].append(sample_data)
        
        self.logger.info(f"=== SAMPLE {sample_idx} ({task_type.upper()}) ===")
        self.logger.info(f"Prompt: {prompt}")
        self.logger.info(f"Completion: {completion}")
        self.logger.info(f"Expected: {expected}")
        self.logger.info(f"Format Valid: {format_valid}")
        self.logger.info(f"Reward Components: {json.dumps(reward_components, indent=2)}")
        self.logger.info(f"Total Reward: {total_reward:.4f}")
        self.logger.info("=" * 50)

    def test_single_batch_training(self, minimal_config, test_positions):
        """
        Test single batch training with comprehensive logging.
        
        This test runs the full GRPO training pipeline on a minimal batch
        and logs every component for inspection.
        """
        self.logger.info("🚀 Starting Single Batch Training Test")
        self.logger.info(f"Config: steps={minimal_config.steps}, batch_size={minimal_config.batch_positions}, group_size={minimal_config.group_size}")
        
        # Initialize Stockfish engine for ground truth
        try:
            stockfish = StockfishEngine()
            self.logger.info("✅ Stockfish engine initialized")
        except Exception as e:
            # For testing, create a mock stockfish if real one not available
            self.logger.warning(f"⚠️ Stockfish not available ({e}), using mock")
            stockfish = self._create_mock_stockfish()
        
        # Load model (use small model for testing)
        try:
            model_config = GPT2Config(
                vocab_size=50257,
                n_positions=1024,
                n_embd=768,
                n_head=12,
                n_layer=12
            )
            model = GPT2Model(model_config)
            
            # Create reference model (copy of original)
            ref_model = GPT2Model(model_config)
            ref_model.load_state_dict(model.state_dict())
            
            # Move models to the specified device
            device = minimal_config.device
            model = model.to(device)
            ref_model = ref_model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"✅ Models loaded: {total_params:,} parameters")
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            # Try to continue with a simpler setup for testing
            if hasattr(self, '_pytest_running'):
                pytest.fail(f"Model loading failed: {e}")
            else:
                raise Exception(f"Model loading failed: {e}")

        # Initialize policy wrapper
        policy = CausalLMPolicy(
            model=model,
            ref_model=ref_model,
            config=minimal_config,
            device=device
        )
        
        # Initialize reward computers
        policy_reward_config = PolicyRewardConfig()
        env_reward_config = EnvRewardConfig()
        policy_reward_computer = PolicyRewardComputer(policy_reward_config)
        env_reward_computer = EnvRewardComputer(env_reward_config)
        
        # Initialize data collector
        collection_config = GRPOCollectionConfig(
            group_size=minimal_config.group_size,
            max_new_tokens_policy=minimal_config.max_new_tokens,
            max_new_tokens_env=minimal_config.max_new_tokens_env,
            temperature=minimal_config.temperature,
            mix_env_ratio=minimal_config.mix_env_ratio
        )
        
        data_collector = GRPODataCollector(
            policy=policy,
            config=collection_config
        )
        
        # Initialize GRPO trainer
        trainer = GRPOTrainer(model, ref_model, minimal_config)
        
        self.logger.info("✅ All components initialized")
        
        # === STEP 1: Data Collection ===
        self.logger.info("📊 Starting data collection...")
        
        try:
            # Collect batch data
            batch_list = data_collector.collect_mixed_batch(minimal_config.batch_positions)
            
            self.logger.info(f"✅ Collected {len(batch_list)} batches")
            
            # Log batch details
            for i, batch in enumerate(batch_list):
                self.logger.info(f"  📋 Batch {i+1}:")
                self.logger.info(f"    Task: {batch.task_type}")
                self.logger.info(f"    Position: {batch.position_fen}")
                self.logger.info(f"    Group size: {batch.input_ids.shape[0]}")
                self.logger.info(f"    Sequence length: {batch.input_ids.shape[1]}")
                self.logger.info(f"    Rewards shape: {batch.rewards.shape}")
                self.logger.info(f"    Mean reward: {batch.rewards.mean().item():.3f}")
                
                # Basic validation
                assert batch.input_ids.shape[0] == minimal_config.group_size, f"Wrong group size: {batch.input_ids.shape[0]}"
                assert batch.rewards.shape[0] == minimal_config.group_size, f"Wrong reward count: {batch.rewards.shape[0]}"
                assert batch.task_type in ["policy", "environment"], f"Invalid task type: {batch.task_type}"
            
            self.logger.info("✅ All batches validated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            if hasattr(self, '_pytest_running'):
                pytest.fail(f"Data collection failed: {e}")
            else:
                raise Exception(f"Data collection failed: {e}")
                
        batch_list = batch_list  # Make batch_list available for training
        
        # === STEP 2: Training Step ===
        self.logger.info("🏋️ Running training step...")
        
        try:
            # Run single training step
            # Create training step from batch list
            step_data = GRPOTrainingStep(groups=batch_list)
            training_metrics = trainer.training_step(step_data)
            
            # Log training results
            self.training_log["step_data"] = {
                "loss": float(training_metrics.get("loss", 0.0)),
                "kl_divergence": float(training_metrics.get("kl_div", 0.0)),
                "entropy": float(training_metrics.get("entropy", 0.0)),
                "lr": float(training_metrics.get("lr", 0.0))
            }
            
            total_loss = training_metrics.get("loss", 0.0)
            
            self.logger.info("=== TRAINING STEP RESULTS ===")
            self.logger.info(f"Loss: {training_metrics.get('loss', 0.0):.6f}")
            self.logger.info(f"KL Divergence: {training_metrics.get('kl_div', 0.0):.6f}")
            self.logger.info(f"Entropy: {training_metrics.get('entropy', 0.0):.6f}")
            self.logger.info(f"Learning Rate: {training_metrics.get('lr', 0.0):.8f}")
            self.logger.info(f"Total Loss: {total_loss:.6f}")
            self.logger.info("=" * 30)
            
        except Exception as e:
            self.logger.error(f"❌ Training step failed: {e}")
            pytest.fail(f"Training step failed: {e}")
        
        # === Validation ===
        self.logger.info("✅ Single batch training completed successfully!")
        
        # Validate that we have the expected data
        assert len(self.training_log["samples"]) == minimal_config.batch_positions
        assert len([s for s in self.training_log["samples"] if s["task_type"] == "policy"]) >= 1
        assert len([s for s in self.training_log["samples"] if s["task_type"] == "environment"]) >= 1
        assert "policy_loss" in self.training_log["step_data"]
        
        # Log summary
        self.logger.info("=== TEST SUMMARY ===")
        self.logger.info(f"✅ Processed {len(self.training_log['samples'])} samples")
        self.logger.info(f"✅ Policy samples: {len([s for s in self.training_log['samples'] if s['task_type'] == 'policy'])}")
        self.logger.info(f"✅ Environment samples: {len([s for s in self.training_log['samples'] if s['task_type'] == 'environment'])}")
        self.logger.info(f"✅ Average reward: {sum(s['total_reward'] for s in self.training_log['samples']) / len(self.training_log['samples']):.4f}")
        self.logger.info(f"✅ Final loss: {total_loss:.6f}")
        self.logger.info("=" * 20)
        
        return self.training_log

    def _format_expected_policy(self, analysis: StockfishAnalysis) -> str:
        """Format expected policy completion from Stockfish analysis."""
        moves = [pv.move for pv in analysis.principal_variations[:5]]
        evals = [pv.evaluation for pv in analysis.principal_variations[:5]]
        best_move = moves[0] if moves else "e2e4"
        
        # Pad to 5 moves/evals if needed
        while len(moves) < 5:
            moves.append("e2e4")
        while len(evals) < 5:
            evals.append(0.0)
            
        moves_str = " ".join(moves[:5])
        evals_str = " ".join(f"{eval:.1f}" for eval in evals[:5])
        
        return f"M: {moves_str}    E: {evals_str}    B: {best_move}"

    def _format_expected_environment(self, prompt: str) -> str:
        """Format expected environment completion from prompt."""
        # Parse A: prompt to extract FEN and move
        # A: <prev_fen>+<uci_move>+<history>+
        parts = prompt.replace("A: ", "").split("+")
        if len(parts) >= 2:
            prev_fen = parts[0]
            uci_move = parts[1]
            
            # Simulate the move to get expected result
            try:
                board = chess.Board(prev_fen)
                move = chess.Move.from_uci(uci_move)
                board.push(move)
                new_fen = board.fen()
                
                # Simple reward calculation (0 for normal moves)
                reward = 0.0
                terminated = board.is_game_over()
                truncated = False
                
                return f"<new_fen>+{reward}+{terminated}+{truncated}"
            except:
                return "<invalid_fen>+0.0+false+false"
        
        return "<missing_data>+0.0+false+false"

    def _create_mock_stockfish(self):
        """Create a mock Stockfish engine for testing when real one is not available."""
        class MockStockfish:
            def analyze_position(self, board, depth=10, multipv=5):
                # Create mock analysis with reasonable chess moves
                class MockPV:
                    def __init__(self, move_str, eval_score):
                        self.move = move_str
                        self.evaluation = eval_score
                
                class MockAnalysis:
                    def __init__(self):
                        # Generate some reasonable opening moves for testing
                        if board.fen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
                            # Starting position
                            self.principal_variations = [
                                MockPV("e2e4", 0.3),
                                MockPV("d2d4", 0.2),
                                MockPV("g1f3", 0.1),
                                MockPV("c2c4", 0.0),
                                MockPV("b1c3", -0.1)
                            ]
                        else:
                            # Generic moves for other positions
                            legal_moves = list(board.legal_moves)
                            self.principal_variations = []
                            for i, move in enumerate(legal_moves[:5]):
                                eval_score = 0.1 * (5 - i)  # Decreasing evaluation
                                self.principal_variations.append(MockPV(str(move), eval_score))
                            
                            # Pad to 5 if fewer moves available
                            while len(self.principal_variations) < 5:
                                self.principal_variations.append(MockPV("e2e4", 0.0))
                
                return MockAnalysis()
        
        return MockStockfish()


if __name__ == "__main__":
    """Run single batch test directly."""
    test = TestSingleBatchTraining()
    test.setup_method()
    
    # Create test config and positions
    config = GRPOConfig(
        steps=1,
        batch_positions=2, 
        group_size=2,
        mix_env_ratio=0.5,
        model_name="gpt2",
        max_new_tokens_policy=50,
        max_new_tokens_env=80,
        temperature=0.3,
        lr=1e-6,
        mixed_precision=False,
        torch_compile=False,
        log_level="DEBUG"
    )
    
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    ]
    
    # Run the test
    try:
        results = test.test_single_batch_training(config, positions)
        print("🎉 Single batch training test completed successfully!")
        print(f"📊 Logged {len(results['samples'])} samples with full details")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise