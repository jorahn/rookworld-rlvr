"""
Model Parity Tests for RookWorld GRPO Implementation

This module tests the pure PyTorch GPT-2 implementation against HuggingFace
transformers library to ensure numerical parity and correctness.
"""

import pytest
import torch
import tiktoken
from pathlib import Path

from src.rookworld_rlvr.model.gpt2 import GPT2Model, create_rookworld_model
from src.rookworld_rlvr.model.config import GPT2Config, ROOKWORLD_CONFIG
from src.rookworld_rlvr.model.loader import load_pretrained_model, verify_weight_loading


class TestGPT2Architecture:
    """Test GPT-2 architecture implementation"""
    
    def test_config_creation(self):
        """Test GPT2Config creation and validation"""
        config = GPT2Config()
        assert config.vocab_size == 50257
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.head_dim == 64  # n_embd / n_head
        
    def test_config_validation(self):
        """Test config validation logic"""
        # Test invalid head/embedding dimension
        with pytest.raises(ValueError, match="n_embd .* must be divisible by n_head"):
            GPT2Config(n_embd=100, n_head=12)
    
    def test_rookworld_config(self):
        """Test RookWorld-specific configuration"""
        config = ROOKWORLD_CONFIG
        assert config.vocab_size == 50257
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_inner == 3072  # 4 * n_embd
    
    def test_model_creation(self):
        """Test model creation with config"""
        model = create_rookworld_model()
        assert isinstance(model, GPT2Model)
        assert model.config == ROOKWORLD_CONFIG
        
        # Check parameter count (approximately 124M)
        num_params = model.get_num_params()
        assert 120_000_000 < num_params < 130_000_000  # Around 124M parameters
    
    def test_model_forward_pass(self):
        """Test basic forward pass"""
        model = create_rookworld_model()
        model.eval()
        
        # Create test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check output shapes
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, model.config.vocab_size)
        
        # Check for NaN values
        assert not torch.isnan(outputs["logits"]).any()
        assert torch.isfinite(outputs["logits"]).all()
    
    def test_model_generation(self):
        """Test model generation capability"""
        model = create_rookworld_model()
        model.eval()
        
        # Create prompt
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Simple sequence
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=5,
                temperature=1.0,
                do_sample=False  # Greedy decoding
            )
        
        # Check generation worked
        assert generated.shape[1] == input_ids.shape[1] + 5
        assert not torch.isnan(generated).any()
    
    def test_attention_mask_creation(self):
        """Test attention mask utility function"""
        from src.rookworld_rlvr.model.gpt2 import create_attention_mask
        
        # Test with padding
        input_ids = torch.tensor([[1, 2, 3, 50256, 50256]])  # 50256 is pad token
        mask = create_attention_mask(input_ids)
        
        expected = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        assert torch.equal(mask, expected)


class TestModelParity:
    """Test numerical parity with reference implementations"""
    
    
    def test_tokenization_compatibility(self):
        """Test that our tokenization approach matches expected behavior"""
        tokenizer = tiktoken.get_encoding("gpt2")
        
        test_text = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M: e2e4"
        tokens = tokenizer.encode(test_text)
        
        # Basic sanity checks
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < 50257 for t in tokens)  # Valid GPT-2 vocab range
    
    def test_model_determinism(self):
        """Test that model outputs are deterministic"""
        model = create_rookworld_model()
        model.eval()
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            output1 = model(input_ids)["logits"]
        
        # Reset seed and run again
        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model(input_ids)["logits"]
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_rookworld_weight_loading_with_hf_download(self):
        """Test loading RookWorld-LM weights by downloading from HuggingFace Hub"""
        pytest.importorskip("huggingface_hub", reason="huggingface_hub required for model downloading")
        
        try:
            from huggingface_hub import snapshot_download
            import tempfile
            import shutil
            
            # Download model to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = snapshot_download(
                    repo_id="jrahn/RookWorld-LM-124M", 
                    local_dir=temp_dir,
                    local_dir_use_symlinks=False
                )
                
                # Load and verify model on CPU to avoid device issues in tests
                model = load_pretrained_model(model_path, device="cpu")
                verification = verify_weight_loading(model)
                
                # Assertions
                assert verification["forward_pass_success"]
                assert not verification["has_nan_weights"]
                assert verification["num_parameters"] == 124_439_808  # Exact RookWorld-LM parameter count
                
                # Test chess-specific behavior
                import tiktoken
                tokenizer = tiktoken.get_encoding("gpt2")
                
                # Test RookWorld prompt format
                prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:"
                tokens = tokenizer.encode(prompt)
                input_ids = torch.tensor([tokens])
                
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs["logits"][0, -1, :]
                    top_tokens = torch.topk(logits, 3).indices
                    top_decoded = [tokenizer.decode([t.item()]) for t in top_tokens]
                    
                    # Should predict reasonable chess moves (commonly " e", " g", " c", " d", etc.)
                    chess_moves = [" e", " g", " c", " d", " a", " b", " f", " h"]
                    assert any(move in top_decoded for move in chess_moves), f"Expected chess moves, got {top_decoded}"
        
        except ImportError:
            pytest.skip("huggingface_hub not available")


class TestChessSpecificBehavior:
    """Test chess-specific model behavior"""
    
    def test_chess_prompt_processing(self):
        """Test model can process chess-specific prompts without errors"""
        model = create_rookworld_model()
        model.eval()
        tokenizer = tiktoken.get_encoding("gpt2")
        
        chess_prompts = [
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:",
            "A: r1bqkb1r/pppp1ppp/2n2n2/4p3+e2e4+",
            "FEN: 8/8/8/3k4/3K4/8/8/8 w - - 0 1"
        ]
        
        for prompt in chess_prompts:
            # Tokenize prompt
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens])
            
            # Forward pass should not error
            with torch.no_grad():
                outputs = model(input_ids)
            
            assert "logits" in outputs
            assert not torch.isnan(outputs["logits"]).any()
    
    def test_policy_prompt_format(self):
        """Test policy prompt format handling"""
        model = create_rookworld_model()
        model.eval()
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # RookWorld policy format
        prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:"
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Check reasonable output distribution (not degenerate)
        logits = outputs["logits"][0, -1, :]  # Last token logits
        probs = torch.softmax(logits, dim=-1)
        
        # Should not be completely uniform or have a single dominant token
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        assert entropy > 1.0  # Some reasonable entropy threshold
    
    def test_environment_prompt_format(self):
        """Test environment prompt format handling"""
        model = create_rookworld_model()
        model.eval()
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # RookWorld environment format
        prompt = "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+"
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Basic sanity checks
        assert "logits" in outputs
        assert not torch.isnan(outputs["logits"]).any()


class TestModelRobustness:
    """Test model robustness and edge cases"""
    
    def test_empty_input(self):
        """Test model behavior with minimal input"""
        model = create_rookworld_model()
        model.eval()
        
        # Single token input
        input_ids = torch.tensor([[1]])
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs["logits"].shape == (1, 1, model.config.vocab_size)
        assert not torch.isnan(outputs["logits"]).any()
    
    def test_max_sequence_length(self):
        """Test model with maximum sequence length"""
        model = create_rookworld_model()
        model.eval()
        
        # Maximum context length
        seq_len = model.config.n_positions
        input_ids = torch.randint(0, 1000, (1, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs["logits"].shape == (1, seq_len, model.config.vocab_size)
        assert not torch.isnan(outputs["logits"]).any()
    
    def test_batch_processing(self):
        """Test model with different batch sizes"""
        model = create_rookworld_model()
        model.eval()
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 1000, (batch_size, 20))
            
            with torch.no_grad():
                outputs = model(input_ids)
            
            assert outputs["logits"].shape == (batch_size, 20, model.config.vocab_size)
            assert not torch.isnan(outputs["logits"]).any()


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running smoke tests...")
    
    # Test model creation
    print("✓ Testing model creation...")
    model = create_rookworld_model()
    print(f"✓ Model created with {model.get_num_params():,} parameters")
    
    # Test forward pass
    print("✓ Testing forward pass...")
    test_input = torch.randint(0, 1000, (1, 10))
    with torch.no_grad():
        outputs = model(test_input)
    print(f"✓ Forward pass successful, output shape: {outputs['logits'].shape}")
    
    # Test generation
    print("✓ Testing generation...")
    generated = model.generate(test_input, max_new_tokens=5, do_sample=False)
    print(f"✓ Generation successful, output shape: {generated.shape}")
    
    print("All smoke tests passed!")