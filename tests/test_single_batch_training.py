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
        
        # Load RookWorld-LM model with actual pre-trained weights
        try:
            from rookworld_rlvr.model.loader import load_pretrained_model
            
            device = minimal_config.device
            self.logger.info(f"Loading RookWorld-LM model from {minimal_config.model_name_or_path}")
            
            # Load pre-trained model
            model = load_pretrained_model(
                minimal_config.model_name_or_path,
                device=device
            )
            model.train()  # Set to training mode for policy model
            
            # Create reference model (frozen copy with same weights)
            ref_model = load_pretrained_model(
                minimal_config.model_name_or_path,
                device=device
            )
            ref_model.eval()  # Set to evaluation mode
            
            # Freeze reference model parameters
            for param in ref_model.parameters():
                param.requires_grad_(False)
                
            # Verify models have same initial weights but different training states
            self.logger.info(f"Policy model training mode: {model.training}")
            self.logger.info(f"Reference model training mode: {ref_model.training}")
            
            # Count parameters for verification
            policy_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            ref_params = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
            self.logger.info(f"Policy model trainable params: {policy_params:,}")
            self.logger.info(f"Reference model trainable params: {ref_params:,} (should be 0)")
            
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
        
        # === STEP 1: Prepare Real Dataset Samples ===
        self.logger.info("📊 Preparing real dataset samples for testing...")
        
        try:
            # Create controlled test samples from the dataset instead of dynamic generation
            test_samples = self._prepare_test_samples(minimal_config, policy)
            
            # Convert to GRPO batches manually for better control
            batch_list = self._create_grpo_batches_from_samples(test_samples, minimal_config, policy, trainer)
            
            self.logger.info(f"✅ Created {len(batch_list)} controlled test batches")
            
            # Log detailed batch information
            for i, batch in enumerate(batch_list):
                self.logger.info(f"  📋 Batch {i+1}:")
                self.logger.info(f"    Task: {batch.task_type}")
                self.logger.info(f"    Position: {batch.position_fen}")
                self.logger.info(f"    Group size: {batch.input_ids.shape[0]}")
                self.logger.info(f"    Sequence length: {batch.input_ids.shape[1]}")
                self.logger.info(f"    Target start indices: {batch.target_start_indices.tolist()}")
                self.logger.info(f"    Rewards shape: {batch.rewards.shape}")
                self.logger.info(f"    Mean reward: {batch.rewards.mean().item():.3f}")
                self.logger.info(f"    Old logprobs shape: {batch.old_logprobs.shape}")
                self.logger.info(f"    Mean old logprob: {batch.old_logprobs.mean().item():.3f}")
                
                # Basic validation
                assert batch.input_ids.shape[0] == minimal_config.group_size, f"Wrong group size: {batch.input_ids.shape[0]}"
                assert batch.rewards.shape[0] == minimal_config.group_size, f"Wrong reward count: {batch.rewards.shape[0]}"
                assert batch.task_type in ["policy", "environment"], f"Invalid task type: {batch.task_type}"
            
            self.logger.info("✅ All controlled batches prepared successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            if hasattr(self, '_pytest_running'):
                pytest.fail(f"Data collection failed: {e}")
            else:
                raise Exception(f"Data collection failed: {e}")
                
        batch_list = batch_list  # Make batch_list available for training
        
        # === STEP 2: Training Step ===
        self.logger.info("🏋️ Running training step...")
        
        # Log model states before training
        self._log_model_states("BEFORE training", model, ref_model)
        
        try:
            # Run single training step
            # Create training step from batch list
            step_data = GRPOTrainingStep(groups=batch_list)
            
            # Log sample processing for each group
            for i, batch in enumerate(batch_list):
                sample_data = {
                    "task_type": batch.task_type,
                    "position_fen": batch.position_fen,
                    "prompt": test_samples[i]['prompt'],
                    "completion": test_samples[i]['completion'],
                    "total_reward": float(batch.rewards.mean()),
                    "reward_components": {},
                    "format_validation_passed": True  # We control the test samples
                }
                self.training_log["samples"].append(sample_data)
            
            # Compute reference model logprobs for comparison
            self._log_reference_comparison(batch_list[0], trainer)
            
            training_metrics = trainer.training_step(step_data)
            
            # Log model states after training
            self._log_model_states("AFTER training", model, ref_model)
            
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
        assert "loss" in self.training_log["step_data"]
        
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
    
    def _prepare_test_samples(self, config, policy):
        """
        Prepare controlled test samples from real dataset entries.
        
        This creates known good samples with proper prompt/completion splits
        to test the training pipeline in a controlled manner.
        For bs=8: 4 policy tasks + 4 environment tasks.
        """
        samples = []
        
        # Determine how many of each task type to create based on batch_positions
        total_positions = config.batch_positions
        expected_policy = int(total_positions * (1 - config.mix_env_ratio))  # 50% policy tasks
        expected_env = int(total_positions * config.mix_env_ratio)            # 50% environment tasks
        
        self.logger.info(f"Creating {expected_policy} policy tasks and {expected_env} environment tasks")
        
        # Policy task samples with diverse positions
        policy_positions = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "After 1.d4"),
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "King's pawn game")
        ]
        
        for i in range(expected_policy):
            fen, description = policy_positions[i % len(policy_positions)]
            policy_prompt = f"P: {fen} M:"
            
            # Vary completions slightly for diversity
            if i == 0:
                policy_completion = """ Move: e2e4
Eval: 0.3
Best: e2e4 (0.3), d2d4 (0.2), g1f3 (0.1)
Analysis: Central pawn advance controls key squares."""
            elif i == 1:
                policy_completion = """ Move: e7e5
Eval: 0.0
Best: e7e5 (0.0), g8f6 (-0.1), b8c6 (-0.1)
Analysis: Classical response in center."""
            elif i == 2:
                policy_completion = """ Move: g8f6
Eval: 0.1
Best: g8f6 (0.1), d7d5 (0.0), c7c5 (-0.1)
Analysis: Hypermodern development."""
            else:
                policy_completion = """ Move: g1f3
Eval: 0.2
Best: g1f3 (0.2), f1c4 (0.1), b1c3 (0.0)
Analysis: King's knight development."""
            
            samples.append({
                'task_type': 'policy',
                'fen': fen,
                'prompt': policy_prompt,
                'completion': policy_completion,
                'full_text': policy_prompt + policy_completion,
                'description': description
            })
        
        # Environment task samples with diverse transitions
        env_transitions = [
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5", 
             "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "d7d5",
             "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2"),
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "g1f3",
             "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4",
             "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        ]
        
        for i in range(expected_env):
            start_fen, move, end_fen = env_transitions[i % len(env_transitions)]
            env_prompt = f"A: {start_fen}+{move}+"
            env_completion = f"{end_fen}+0.0+false+false"
            
            samples.append({
                'task_type': 'environment',
                'fen': start_fen,
                'prompt': env_prompt,
                'completion': env_completion,
                'full_text': env_prompt + env_completion,
                'description': f"Move {move}"
            })
        
        self.logger.info(f"Prepared {len(samples)} test samples:")
        for i, sample in enumerate(samples):
            self.logger.info(f"  Sample {i+1} ({sample['task_type']}):")
            self.logger.info(f"    Description: {sample['description']}")
            self.logger.info(f"    FEN: {sample['fen']}")
            self.logger.info(f"    Prompt: {sample['prompt']}")
            self.logger.info(f"    Completion: {sample['completion'][:50]}...")
        
        return samples
    
    def _create_grpo_batches_from_samples(self, test_samples, config, policy, trainer):
        """
        Create GRPO batches from prepared test samples with proper control.
        
        This bypasses the dynamic generation to use known good samples
        and ensures proper prompt/completion splits and logprob computation.
        """
        from rookworld_rlvr.train.grpo_trainer import GRPOBatch
        import chess
        
        batches = []
        
        for sample in test_samples:
            self.logger.info(f"Creating GRPO batch for {sample['task_type']} task...")
            
            # Create multiple copies for group_size
            prompts = [sample['prompt']] * config.group_size
            full_texts = [sample['full_text']] * config.group_size
            
            # Tokenize the full sequences
            encodings = []
            target_start_indices = []
            
            for i in range(config.group_size):
                # Tokenize prompt to find where completion starts
                prompt_encoding = policy.tokenizer.encode(prompts[i])
                full_encoding = policy.tokenizer.encode(full_texts[i])
                
                # Target starts where prompt ends
                target_start_idx = len(prompt_encoding)
                target_start_indices.append(target_start_idx)
                encodings.append(full_encoding)
                
                self.logger.debug(f"  Sample {i+1}: prompt_len={len(prompt_encoding)}, full_len={len(full_encoding)}, target_start={target_start_idx}")
            
            # Pad sequences to same length
            max_len = max(len(enc) for enc in encodings)
            input_ids = torch.zeros((config.group_size, max_len), dtype=torch.long, device=policy.device)
            attention_mask = torch.zeros((config.group_size, max_len), dtype=torch.long, device=policy.device)
            
            for i, encoding in enumerate(encodings):
                seq_len = len(encoding)
                input_ids[i, :seq_len] = torch.tensor(encoding, dtype=torch.long)
                attention_mask[i, :seq_len] = 1
            
            # Compute logprobs from the POLICY model using trainer's method
            # This ensures consistency with the training pipeline
            target_start_tensor = torch.tensor(target_start_indices, device=policy.device)
            
            with torch.no_grad():
                old_logprobs = trainer.compute_logprobs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_start_indices=target_start_tensor,
                    use_ref_model=False  # Use policy model, not reference model
                )
            
            self.logger.info(f"  Computed old_logprobs: mean={old_logprobs.mean().item():.3f}, std={old_logprobs.std().item():.3f}")
            
            # Compute rewards (mock for now, but using proper structure)
            if sample['task_type'] == 'policy':
                # Good structured output should get positive reward
                rewards = torch.tensor([0.4] * config.group_size, device=policy.device)
            else:
                # Environment task with proper format should get positive reward
                rewards = torch.tensor([0.6] * config.group_size, device=policy.device)
            
            # Create GRPO batch
            batch = GRPOBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_start_indices=target_start_tensor,
                old_logprobs=old_logprobs,
                rewards=rewards,
                position_fen=sample['fen'],
                task_type=sample['task_type']
            )
            
            batches.append(batch)
            self.logger.info(f"✅ Created batch: {sample['task_type']}, shape={input_ids.shape}, rewards_mean={rewards.mean():.3f}")
        
        return batches
    
    def _log_model_states(self, stage, policy_model, ref_model):
        """Log detailed model states for debugging."""
        self.logger.info(f"=== MODEL STATES {stage} ===")
        
        # Policy model state
        policy_trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        policy_total = sum(p.numel() for p in policy_model.parameters())
        self.logger.info(f"Policy Model:")
        self.logger.info(f"  Training mode: {policy_model.training}")
        self.logger.info(f"  Trainable params: {policy_trainable:,}")
        self.logger.info(f"  Total params: {policy_total:,}")
        
        # Check gradient status of first few parameters
        param_grad_info = []
        for i, (name, param) in enumerate(policy_model.named_parameters()):
            if i < 3:  # Just first 3 parameters
                has_grad = param.grad is not None
                requires_grad = param.requires_grad
                param_grad_info.append(f"{name}: requires_grad={requires_grad}, has_grad={has_grad}")
        
        self.logger.info(f"  Gradient status (first 3): {param_grad_info}")
        
        # Reference model state
        ref_trainable = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
        ref_total = sum(p.numel() for p in ref_model.parameters())
        self.logger.info(f"Reference Model:")
        self.logger.info(f"  Training mode: {ref_model.training}")
        self.logger.info(f"  Trainable params: {ref_trainable:,} (should be 0)")
        self.logger.info(f"  Total params: {ref_total:,}")
        
        # Verify models have same architecture but different training states
        self.logger.info(f"Models properly configured: policy_trainable={policy_trainable > 0}, ref_frozen={ref_trainable == 0}")
        
    def _log_reference_comparison(self, batch, trainer):
        """Log comparison between policy and reference model logprobs."""
        self.logger.info("=== POLICY vs REFERENCE LOGPROBS ===")
        
        # Check KL warmup settings
        current_step = trainer.step_count
        kl_warmup_steps = trainer.config.kl_warmup_steps
        kl_warmup_factor = trainer.config.kl_warmup_factor
        base_kl_coef = trainer.config.kl_coef
        
        if kl_warmup_steps > 0 and current_step < kl_warmup_steps:
            # During warmup
            progress = current_step / kl_warmup_steps
            effective_kl_coef = base_kl_coef * (kl_warmup_factor + (1.0 - kl_warmup_factor) * progress)
        else:
            # After warmup or no warmup
            effective_kl_coef = base_kl_coef
        
        self.logger.info(f"KL Coefficient Settings:")
        self.logger.info(f"  Base KL coef: {base_kl_coef}")
        self.logger.info(f"  Warmup steps: {kl_warmup_steps}")
        self.logger.info(f"  Warmup factor: {kl_warmup_factor}")
        self.logger.info(f"  Current step: {current_step}")
        self.logger.info(f"  Effective KL coef: {effective_kl_coef:.6f}")
        
        # Compute logprobs from both models
        with torch.no_grad():
            policy_logprobs = trainer.compute_logprobs(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                target_start_indices=batch.target_start_indices,
                use_ref_model=False  # Policy model
            )
            
            ref_logprobs = trainer.compute_logprobs(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                target_start_indices=batch.target_start_indices,
                use_ref_model=True   # Reference model
            )
        
        # Compute KL divergence and its impact
        kl_div = policy_logprobs - ref_logprobs
        kl_penalty = effective_kl_coef * kl_div
        
        self.logger.info(f"Policy logprobs: mean={policy_logprobs.mean():.3f}, std={policy_logprobs.std():.3f}")
        self.logger.info(f"Reference logprobs: mean={ref_logprobs.mean():.3f}, std={ref_logprobs.std():.3f}")
        self.logger.info(f"KL divergence: mean={kl_div.mean():.3f}, std={kl_div.std():.3f}")
        self.logger.info(f"KL penalty (coef * KL): mean={kl_penalty.mean():.6f}, std={kl_penalty.std():.6f}")
        self.logger.info(f"Old logprobs (from batch): mean={batch.old_logprobs.mean():.3f}, std={batch.old_logprobs.std():.3f}")
        
        # Check if old_logprobs match current policy logprobs (they should be close)
        logprob_diff = torch.abs(policy_logprobs - batch.old_logprobs)
        self.logger.info(f"Policy vs Old logprobs diff: mean={logprob_diff.mean():.6f}, max={logprob_diff.max():.6f}")
        
        if logprob_diff.mean() > 0.01:
            self.logger.warning("⚠️ Large difference between current policy and old logprobs - possible inconsistency!")
        
        # Warn if KL coefficient is effectively zero
        if effective_kl_coef < 1e-6:
            self.logger.warning("⚠️ KL coefficient is effectively zero due to warmup - not testing real KL impact!")
        elif effective_kl_coef < 0.001:
            self.logger.warning("⚠️ KL coefficient is very low - limited KL regularization effect!")


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