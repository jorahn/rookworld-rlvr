"""
Minimal weight loader for HuggingFace models

Loads RookWorld-LM-124M weights from HuggingFace into our minimal GPT-2 model.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import torch

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed. Install with: pip install safetensors")

try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface-hub")

from model import GPT2Model, GPT2Config


def create_weight_mapping() -> Dict[str, str]:
    """Create mapping from HuggingFace naming to our model naming"""
    mappings = {
        # Embeddings
        "transformer.wte.weight": "wte.weight",
        "transformer.wpe.weight": "wpe.weight",
        
        # Final layer norm
        "transformer.ln_f.weight": "ln_f.weight",
        "transformer.ln_f.bias": "ln_f.bias",
    }
    
    # Add mappings for each transformer block
    for i in range(12):  # 12 layers for GPT-2 124M
        block_mappings = {
            # Layer norms
            f"transformer.h.{i}.ln_1.weight": f"h.{i}.ln_1.weight",
            f"transformer.h.{i}.ln_1.bias": f"h.{i}.ln_1.bias",
            f"transformer.h.{i}.ln_2.weight": f"h.{i}.ln_2.weight",
            f"transformer.h.{i}.ln_2.bias": f"h.{i}.ln_2.bias",
            
            # Attention
            f"transformer.h.{i}.attn.c_attn.weight": f"h.{i}.attn.c_attn.weight",
            f"transformer.h.{i}.attn.c_attn.bias": f"h.{i}.attn.c_attn.bias",
            f"transformer.h.{i}.attn.c_proj.weight": f"h.{i}.attn.c_proj.weight",
            f"transformer.h.{i}.attn.c_proj.bias": f"h.{i}.attn.c_proj.bias",
            
            # MLP
            f"transformer.h.{i}.mlp.c_fc.weight": f"h.{i}.mlp.c_fc.weight",
            f"transformer.h.{i}.mlp.c_fc.bias": f"h.{i}.mlp.c_fc.bias",
            f"transformer.h.{i}.mlp.c_proj.weight": f"h.{i}.mlp.c_proj.weight",
            f"transformer.h.{i}.mlp.c_proj.bias": f"h.{i}.mlp.c_proj.bias",
        }
        mappings.update(block_mappings)
    
    return mappings


def load_safetensors_weights(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load weights from safetensors file"""
    safetensors_path = model_path / "model.safetensors"
    
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No safetensors file found at {safetensors_path}")
    
    weights = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    
    return weights


def load_pytorch_weights(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load weights from pytorch_model.bin file"""
    pytorch_path = model_path / "pytorch_model.bin"
    
    if not pytorch_path.exists():
        raise FileNotFoundError(f"No pytorch_model.bin found at {pytorch_path}")
    
    return torch.load(pytorch_path, map_location="cpu")


def convert_hf_to_mini(hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert HuggingFace weight dict to our mini model format"""
    weight_mapping = create_weight_mapping()
    converted_weights = {}
    
    for hf_key, mini_key in weight_mapping.items():
        if hf_key in hf_weights:
            weight = hf_weights[hf_key].clone()
            
            # Handle weight transposition for linear layers
            # HF stores as [in_features, out_features], PyTorch expects [out_features, in_features]
            if any(layer_type in hf_key for layer_type in ['.c_attn.weight', '.c_proj.weight', '.c_fc.weight']):
                weight = weight.t()
            
            converted_weights[mini_key] = weight
    
    return converted_weights


def load_rookworld_model(
    model_name: str = "jrahn/RookWorld-LM-124M",
    device: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> GPT2Model:
    """
    Load RookWorld-LM-124M model with weights from HuggingFace
    
    Args:
        model_name: HuggingFace model name or local path
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        cache_dir: Directory to cache downloaded models
        
    Returns:
        Loaded GPT2Model ready for inference
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if it's a local path or HF model name
    model_path = Path(model_name)
    
    if not model_path.exists() and "/" in model_name:
        # Download from HuggingFace
        if not HAS_HF_HUB:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface-hub")
        
        print(f"Downloading {model_name} from HuggingFace...")
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.bin", "*.json"]
        )
        model_path = Path(local_path)
    
    # Load weights
    print(f"Loading weights from {model_path}")
    
    # Try safetensors first, fall back to pytorch_model.bin
    if (model_path / "model.safetensors").exists() and HAS_SAFETENSORS:
        hf_weights = load_safetensors_weights(model_path)
    elif (model_path / "pytorch_model.bin").exists():
        hf_weights = load_pytorch_weights(model_path)
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    # Convert weights to our format
    converted_weights = convert_hf_to_mini(hf_weights)
    
    # Create model and load weights
    config = GPT2Config()
    model = GPT2Model(config)
    
    # Load state dict
    missing, unexpected = model.load_state_dict(converted_weights, strict=False)
    
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    # Verify loading
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully with {num_params:,} parameters on {device}")
    
    return model


def quick_test(model: GPT2Model, device: str = 'cuda'):
    """Quick test to verify model works"""
    import tiktoken
    
    # Get tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Test prompt
    test_prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    input_ids = torch.tensor([enc.encode(test_prompt)], device=device)
    
    # Generate
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits']
        print(f"Test forward pass successful! Logits shape: {logits.shape}")
        
        # Try generation
        generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
        generated_text = enc.decode(generated[0].cpu().tolist())
        print(f"Generated: {generated_text}")
    
    return True


if __name__ == "__main__":
    # Test loading
    print("Testing RookWorld model loading...")
    model = load_rookworld_model()
    quick_test(model)