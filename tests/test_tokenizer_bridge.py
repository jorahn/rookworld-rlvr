"""
Tests for TokenizerBridge

Verify pure PyTorch tokenization functionality for RookWorld-LM training.
"""

import pytest
import torch
from src.rookworld_rlvr.tokenizer.bridge import TokenizerBridge, create_tokenizer_bridge


class TestTokenizerBridge:
    """Test TokenizerBridge functionality"""
    
    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer bridge instance"""
        return TokenizerBridge()
    
    def test_initialization(self, tokenizer):
        """Test tokenizer initialization"""
        assert tokenizer.vocab_size == 50257  # GPT-2 vocab size
        assert tokenizer.eos_token_id == 50256
        assert tokenizer.pad_token_id == 50256
    
    def test_encode_decode_single(self, tokenizer):
        """Test single text encoding and decoding"""
        text = "Hello, world!"
        
        # Test encoding
        tokens = tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < tokenizer.vocab_size for t in tokens)
        
        # Test decoding
        decoded = tokenizer.decode(tokens)
        assert decoded == text
        
        # Test with EOS token
        tokens_eos = tokenizer.encode(text, add_eos=True)
        assert len(tokens_eos) == len(tokens) + 1
        assert tokens_eos[-1] == tokenizer.eos_token_id
    
    def test_encode_batch(self, tokenizer):
        """Test batch encoding"""
        texts = [
            "Hello, world!",
            "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:",
            "Short text"
        ]
        
        result = tokenizer.encode_batch(texts)
        
        # Check result structure
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "lengths" in result
        
        # Check tensor shapes
        batch_size = len(texts)
        assert result["input_ids"].shape[0] == batch_size
        assert result["attention_mask"].shape[0] == batch_size
        assert result["lengths"].shape == (batch_size,)
        
        # Check attention mask corresponds to actual content
        for i in range(batch_size):
            length = result["lengths"][i].item()
            assert result["attention_mask"][i, :length].sum().item() == length
            assert result["attention_mask"][i, length:].sum().item() == 0
    
    def test_decode_batch(self, tokenizer):
        """Test batch decoding"""
        texts = [
            "Hello, world!",
            "This is a test.",
            "Short"
        ]
        
        # Encode then decode
        encoded = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode_batch(encoded["input_ids"])
        
        assert len(decoded) == len(texts)
        for original, recovered in zip(texts, decoded):
            assert recovered == original
    
    def test_chess_prompts(self, tokenizer):
        """Test chess prompt creation"""
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5"
        ]
        
        # Test policy prompts
        policy_prompts = tokenizer.create_chess_prompts(fens, "policy")
        assert len(policy_prompts) == len(fens)
        for prompt, fen in zip(policy_prompts, fens):
            assert prompt == f"P: {fen}    M:"
        
        # Test environment prompt creation (partial)
        env_prompts = tokenizer.create_chess_prompts(fens, "environment")
        assert len(env_prompts) == len(fens)
        for prompt, fen in zip(env_prompts, fens):
            assert prompt == f"A: {fen}+"
    
    def test_env_prompts_with_moves(self, tokenizer):
        """Test environment prompts with UCI moves"""
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5"
        ]
        uci_moves = ["e2e4", "c4c5"]
        
        env_prompts = tokenizer.create_env_prompts(fens, uci_moves)
        assert len(env_prompts) == len(fens)
        
        for prompt, fen, uci in zip(env_prompts, fens, uci_moves):
            expected = f"A: {fen}+{uci}+"
            assert prompt == expected
        
        # Test length mismatch
        with pytest.raises(ValueError):
            tokenizer.create_env_prompts(fens, ["e2e4"])  # Mismatched lengths
    
    def test_prompt_length(self, tokenizer):
        """Test prompt length calculation"""
        prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:"
        
        length = tokenizer.get_prompt_length(prompt)
        manual_length = len(tokenizer.encode(prompt))
        
        assert length == manual_length
        assert length > 0
    
    def test_split_generated_text(self, tokenizer):
        """Test extracting generated text from full sequence"""
        prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1    M:"
        generated = " e2e4 d2d4 g1f3 b1c3 f2f4    E: 0.25 0.18 0.12 0.08 0.15    B: e2e4"
        full_text = prompt + generated
        
        extracted = tokenizer.split_generated_text(full_text, prompt)
        assert extracted == generated
        
        # Test with non-matching prompt (fallback)
        extracted = tokenizer.split_generated_text("Some random text", prompt)
        assert extracted == "Some random text"
    
    def test_device_handling(self, tokenizer):
        """Test tensor device placement"""
        texts = ["Hello", "World"]
        
        # Test CPU (default)
        result_cpu = tokenizer.encode_batch(texts, device="cpu")
        assert result_cpu["input_ids"].device.type == "cpu"
        assert result_cpu["attention_mask"].device.type == "cpu"
        
        # Test CUDA if available
        if torch.cuda.is_available():
            result_cuda = tokenizer.encode_batch(texts, device="cuda")
            assert result_cuda["input_ids"].device.type == "cuda"
            assert result_cuda["attention_mask"].device.type == "cuda"
    
    def test_max_length_handling(self, tokenizer):
        """Test max length truncation and padding"""
        texts = ["Short", "This is a much longer text that should be truncated"]
        max_length = 5
        
        result = tokenizer.encode_batch(texts, max_length=max_length)
        
        # Check all sequences are max_length
        assert result["input_ids"].shape[1] == max_length
        assert result["attention_mask"].shape[1] == max_length
        
        # Check truncation worked
        assert result["lengths"][1].item() <= max_length
    
    def test_convenience_function(self):
        """Test convenience function"""
        tokenizer = create_tokenizer_bridge()
        assert isinstance(tokenizer, TokenizerBridge)
        assert tokenizer.vocab_size == 50257


class TestChessSpecificBehavior:
    """Test chess-specific tokenization behavior"""
    
    @pytest.fixture
    def tokenizer(self):
        return TokenizerBridge()
    
    def test_policy_task_format(self, tokenizer):
        """Test policy task prompt format matches RookWorld expectations"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        prompt = tokenizer.create_chess_prompts([fen], "policy")[0]
        
        # Verify exact format with proper spacing
        expected = f"P: {fen}    M:"
        assert prompt == expected
        
        # Verify it tokenizes without issues
        tokens = tokenizer.encode(prompt)
        assert len(tokens) > 0
        
        # Verify round-trip
        decoded = tokenizer.decode(tokens)
        assert decoded == prompt
    
    def test_environment_task_format(self, tokenizer):
        """Test environment task prompt format matches RookWorld expectations"""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        uci = "e2e4"
        
        prompt = tokenizer.create_env_prompts([fen], [uci])[0]
        
        # Verify exact format
        expected = f"A: {fen}+{uci}+"
        assert prompt == expected
        
        # Verify it tokenizes without issues
        tokens = tokenizer.encode(prompt)
        assert len(tokens) > 0
        
        # Verify round-trip
        decoded = tokenizer.decode(tokens)
        assert decoded == prompt
    
    def test_structured_output_parsing(self, tokenizer):
        """Test handling of structured outputs"""
        # Policy task structured output
        policy_output = " e2e4 d2d4 g1f3 b1c3 f2f4    E: 0.25 0.18 0.12 0.08 0.15    B: e2e4"
        
        tokens = tokenizer.encode(policy_output)
        decoded = tokenizer.decode(tokens)
        assert decoded == policy_output
        
        # Environment task structured output
        env_output = "+rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+0+0"
        
        tokens = tokenizer.encode(env_output)
        decoded = tokenizer.decode(tokens)
        assert decoded == env_output