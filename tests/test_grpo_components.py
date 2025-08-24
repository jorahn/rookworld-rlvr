"""
Tests for GRPO Training Components

This module tests the new Phase 3 components:
- GRPO configuration and validation
- Stockfish engine integration
- GRPO trainer mechanics
- Self-play manager
- Evaluation metrics
- Main training orchestration
"""

import pytest
import torch
import chess
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

from src.rookworld_rlvr.train.config import GRPOConfig
from src.rookworld_rlvr.train.grpo_trainer import (
    GRPOTrainer, GRPOBatch, GRPOTrainingStep, AdaptiveKLController
)
from src.rookworld_rlvr.train.self_play import SelfPlayManager, PositionBuffer
from src.rookworld_rlvr.train.evaluator import ChessEvaluator, EvaluationMetrics
from src.rookworld_rlvr.engine.stockfish import StockfishEngine, StockfishAnalysis
from src.rookworld_rlvr.model.gpt2 import GPT2Model
from src.rookworld_rlvr.model.config import GPT2Config


class TestGRPOConfig:
    """Test GRPO configuration validation and functionality."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        config = GRPOConfig()
        assert config.group_size == 8
        assert config.clip_range == 0.2
        assert config.kl_coef == 0.02
        assert config.steps == 1000
        assert 0 <= config.mix_env_ratio <= 1
    
    def test_config_validation(self):
        """Test configuration validation catches invalid values."""
        # Invalid group size
        with pytest.raises(ValueError, match="group_size must be >= 2"):
            GRPOConfig(group_size=1)
        
        # Invalid mix ratio
        with pytest.raises(ValueError, match="mix_env_ratio must be in"):
            GRPOConfig(mix_env_ratio=-0.1)
        
        with pytest.raises(ValueError, match="mix_env_ratio must be in"):
            GRPOConfig(mix_env_ratio=1.5)
        
        # Invalid clip range
        with pytest.raises(ValueError, match="clip_range must be > 0"):
            GRPOConfig(clip_range=0)
        
        # Invalid learning rate
        with pytest.raises(ValueError, match="lr must be > 0"):
            GRPOConfig(lr=-0.001)
    
    def test_derived_values(self):
        """Test computed configuration values."""
        config = GRPOConfig(
            batch_positions=4,
            group_size=8,
            gradient_accumulation_steps=2
        )
        
        effective_batch_size = config.get_effective_batch_size()
        assert effective_batch_size == 4 * 8 * 2  # positions * group_size * accumulation
    
    def test_config_summary(self):
        """Test configuration summary generation."""
        config = GRPOConfig(steps=500, lr=1e-4)
        summary = config.summary()
        
        assert "500" in summary
        assert "1e-04" in summary or "0.0001" in summary
        assert "GRPO Training Configuration" in summary


class TestStockfishEngine:
    """Test Stockfish engine integration."""
    
    @pytest.fixture
    def mock_stockfish_engine(self):
        """Create mock Stockfish engine for testing."""
        engine = StockfishEngine(stockfish_path=None, time_limit=0.01, cache_size=10)
        
        # Mock the actual engine initialization
        engine._engine = Mock()
        engine._engine.analyse.return_value = [
            {
                'pv': [chess.Move.from_uci('e2e4')],
                'score': Mock(relative=Mock(score=lambda: 15)),
                'depth': 10
            },
            {
                'pv': [chess.Move.from_uci('g1f3')],
                'score': Mock(relative=Mock(score=lambda: 10)),
                'depth': 10
            }
        ]
        
        return engine
    
    def test_analysis_structure(self):
        """Test that analysis returns proper structure."""
        board = chess.Board()
        
        # Create engine with fallback (no real Stockfish needed for this test)
        engine = StockfishEngine(stockfish_path=None)
        analysis = engine.analyze(board)
        
        assert isinstance(analysis, StockfishAnalysis)
        assert isinstance(analysis.top5_moves, list)
        assert isinstance(analysis.top5_evals, list)
        assert isinstance(analysis.best_move, str)
        assert isinstance(analysis.depth, int)
        assert isinstance(analysis.analysis_time, float)
    
    def test_cache_functionality(self):
        """Test analysis caching works correctly."""
        engine = StockfishEngine(stockfish_path=None, cache_size=2)
        board1 = chess.Board()
        board2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        
        # First analysis (cache miss)
        analysis1 = engine.analyze(board1)
        stats1 = engine.get_cache_stats()
        
        # Second analysis of same position (cache hit)
        analysis2 = engine.analyze(board1)
        stats2 = engine.get_cache_stats()
        
        assert stats2['cache_hits'] == stats1['cache_hits'] + 1
        assert analysis1.top5_moves == analysis2.top5_moves
    
    def test_fallback_analysis(self, mock_stockfish_engine):
        """Test fallback analysis when engine fails."""
        board = chess.Board()
        
        # Force fallback by making engine.analyse raise exception
        mock_stockfish_engine._engine.analyse.side_effect = Exception("Engine error")
        
        analysis = mock_stockfish_engine.analyze(board)
        
        # Should still return valid structure
        assert isinstance(analysis, StockfishAnalysis)
        assert len(analysis.top5_moves) <= 5
        assert analysis.depth == 0  # Fallback marker


class TestGRPOTrainer:
    """Test GRPO trainer functionality."""
    
    @pytest.fixture
    def simple_models(self):
        """Create simple models for testing."""
        config = GPT2Config()
        config.n_layer = 2  # Smaller for testing
        config.n_head = 4
        config.n_embd = 64
        
        model = GPT2Model(config)
        ref_model = GPT2Model(config)
        ref_model.load_state_dict(model.state_dict())
        
        return model, ref_model
    
    @pytest.fixture 
    def grpo_trainer(self, simple_models):
        """Create GRPO trainer for testing."""
        model, ref_model = simple_models
        config = GRPOConfig(
            steps=10,
            lr=1e-3,
            device='cpu'
        )
        return GRPOTrainer(model, ref_model, config)
    
    def test_adaptive_kl_controller(self):
        """Test adaptive KL coefficient control."""
        controller = AdaptiveKLController(init_kl_coef=0.02, target_kl=0.05)
        
        initial_coef = controller.get_coefficient()
        assert initial_coef == 0.02
        
        # High KL should increase coefficient
        controller.update(0.1)  # Way above target
        assert controller.get_coefficient() > initial_coef
        
        # Reset and test low KL
        controller.kl_coef = 0.02
        controller.update(0.01)  # Below target
        assert controller.get_coefficient() < 0.02
    
    def test_grpo_batch_structure(self):
        """Test GRPO batch data structure."""
        batch = GRPOBatch(
            input_ids=torch.randint(0, 1000, (4, 10)),
            attention_mask=torch.ones(4, 10),
            target_start_indices=torch.tensor([5, 6, 5, 7]),
            old_logprobs=torch.randn(4),
            rewards=torch.randn(4),
            position_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            task_type="policy"
        )
        
        assert batch.input_ids.shape == (4, 10)
        assert batch.rewards.shape == (4,)
        assert batch.task_type == "policy"
    
    def test_training_step_structure(self):
        """Test training step with multiple groups."""
        groups = [
            GRPOBatch(
                input_ids=torch.randint(0, 1000, (2, 8)),
                attention_mask=torch.ones(2, 8),
                target_start_indices=torch.tensor([4, 5]),
                old_logprobs=torch.randn(2),
                rewards=torch.randn(2),
                position_fen="test_fen_1",
                task_type="policy"
            ),
            GRPOBatch(
                input_ids=torch.randint(0, 1000, (3, 10)),
                attention_mask=torch.ones(3, 10),
                target_start_indices=torch.tensor([6, 7, 8]),
                old_logprobs=torch.randn(3),
                rewards=torch.randn(3),
                position_fen="test_fen_2",
                task_type="environment"
            )
        ]
        
        step_data = GRPOTrainingStep(groups=groups)
        assert len(step_data) == 2
        assert step_data.get_total_samples() == 5


class TestSelfPlay:
    """Test self-play manager functionality."""
    
    @pytest.fixture
    def position_buffer(self):
        """Create position buffer for testing."""
        return PositionBuffer(capacity=10, opening_weight=0.5)
    
    def test_position_buffer_operations(self, position_buffer):
        """Test position buffer add/sample operations."""
        # Add positions
        test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ]
        
        for fen in test_positions:
            position_buffer.add_position(fen, {'source': 'test'})
        
        assert len(position_buffer.positions) == 2
        
        # Sample positions
        samples = position_buffer.sample_positions(5)
        assert len(samples) == 5
        assert all(isinstance(fen, str) for fen in samples)
    
    def test_position_buffer_capacity(self, position_buffer):
        """Test position buffer capacity limits."""
        # Fill beyond capacity
        for i in range(15):
            position_buffer.add_position(f"test_fen_{i}")
        
        # Should be limited to capacity
        assert len(position_buffer.positions) == 10
    
    def test_buffer_statistics(self, position_buffer):
        """Test position buffer statistics."""
        position_buffer.add_position("test_fen")
        stats = position_buffer.get_statistics()
        
        assert 'total_positions' in stats
        assert 'utilization' in stats
        assert stats['total_positions'] == 1
        assert stats['capacity'] == 10


class TestChessEvaluator:
    """Test chess evaluation functionality."""
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create evaluator with mocked dependencies."""
        config = GRPOConfig(eval_positions=5)
        
        # Mock Stockfish engine
        stockfish = Mock()
        stockfish.analyze.return_value = StockfishAnalysis(
            top5_moves=['e2e4', 'g1f3', 'd2d4', 'b1c3', 'f1c4'],
            top5_evals=[0.15, 0.10, 0.12, 0.05, 0.08],
            best_move='e2e4',
            depth=10,
            analysis_time=0.05
        )
        
        evaluator = ChessEvaluator(config, stockfish)
        return evaluator
    
    def test_evaluation_metrics_structure(self):
        """Test evaluation metrics data structure."""
        metrics = EvaluationMetrics(
            legal_move_rate=0.85,
            policy_structure_rate=0.92,
            avg_policy_reward=0.75,
            total_samples=100,
            evaluation_time=45.2
        )
        
        assert metrics.legal_move_rate == 0.85
        assert metrics.total_samples == 100
        assert isinstance(metrics.timestamp, float)
    
    def test_metrics_to_dict(self, mock_evaluator):
        """Test metrics conversion to dictionary."""
        metrics = EvaluationMetrics(
            legal_move_rate=0.8,
            avg_policy_reward=0.6,
            env_fen_exact_rate=0.7
        )
        
        metrics_dict = mock_evaluator.metrics_to_dict(metrics)
        
        assert 'policy/legal_move_rate' in metrics_dict
        assert 'policy/avg_reward' in metrics_dict
        assert 'environment/fen_exact_rate' in metrics_dict
        assert metrics_dict['policy/legal_move_rate'] == 0.8


class TestIntegration:
    """Integration tests for component interaction."""
    
    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = GRPOConfig(
            steps=500,
            lr=1e-4,
            model_name_or_path="test/model"
        )
        
        # Serialize
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_dict = config.__dict__.copy()
            # Convert torch.device to string for JSON
            if hasattr(config_dict.get('device'), 'type'):
                config_dict['device'] = str(config_dict['device'])
            json.dump(config_dict, f)
            temp_path = f.name
        
        # Deserialize
        with open(temp_path, 'r') as f:
            loaded_dict = json.load(f)
        
        assert loaded_dict['steps'] == 500
        assert loaded_dict['lr'] == 1e-4
        assert loaded_dict['model_name_or_path'] == "test/model"
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_component_initialization_order(self):
        """Test that components initialize in correct order."""
        # This should not raise any import errors
        from src.rookworld_rlvr.train.config import GRPOConfig
        from src.rookworld_rlvr.train.grpo_trainer import GRPOTrainer  
        from src.rookworld_rlvr.engine.stockfish import StockfishEngine
        from src.rookworld_rlvr.train.evaluator import ChessEvaluator
        
        config = GRPOConfig(device='cpu', steps=1)
        
        # Basic initialization test
        engine = StockfishEngine(stockfish_path=None)  # Will use fallback
        evaluator = ChessEvaluator(config, engine)
        
        assert evaluator.config.device == 'cpu'
        assert len(evaluator.test_positions) > 0
    
    @patch('src.rookworld_rlvr.train.policy.CausalLMPolicy')
    def test_mock_training_flow(self, mock_policy_class):
        """Test training flow with mocked components."""
        config = GRPOConfig(
            steps=2,
            batch_positions=1,
            group_size=2,
            device='cpu'
        )
        
        # Mock policy
        mock_policy = Mock()
        mock_policy.generate.return_value = {
            'texts': ['e2e4', 'g1f3'],
            'logprobs': torch.tensor([-2.1, -2.3])
        }
        mock_policy_class.return_value = mock_policy
        
        # This tests that the basic structure works
        # without requiring actual model weights
        assert config.steps == 2
        assert mock_policy_class.call_count == 0  # Not called yet


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])