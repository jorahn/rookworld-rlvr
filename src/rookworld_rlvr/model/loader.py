"""
Model Weight Loader for RookWorld GRPO Implementation

This module handles loading HuggingFace safetensors weights into the pure PyTorch
GPT-2 implementation. Provides mapping between HF naming conventions and our
PyTorch model structure.
"""

import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")

from .gpt2 import GPT2Model
from .config import GPT2Config


def load_hf_config(model_path: Union[str, Path]) -> GPT2Config:
    """
    Load HuggingFace config.json and convert to GPT2Config
    
    Args:
        model_path: Path to model directory containing config.json
        
    Returns:
        GPT2Config instance
    """
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        hf_config = json.load(f)
    
    # Map HF config keys to our config keys
    config_mapping = {
        'vocab_size': 'vocab_size',
        'n_positions': 'n_positions', 
        'n_embd': 'n_embd',
        'n_layer': 'n_layer',
        'n_head': 'n_head',
        'n_inner': 'n_inner',
        'activation_function': 'activation_function',
        'resid_pdrop': 'resid_pdrop',
        'embd_pdrop': 'embd_pdrop', 
        'attn_pdrop': 'attn_pdrop',
        'layer_norm_epsilon': 'layer_norm_epsilon',
        'initializer_range': 'initializer_range',
        'bos_token_id': 'bos_token_id',
        'eos_token_id': 'eos_token_id',
        'use_cache': 'use_cache',
        'tie_word_embeddings': 'tie_word_embeddings'
    }
    
    # Extract relevant config values
    config_dict = {}
    for hf_key, our_key in config_mapping.items():
        if hf_key in hf_config:
            config_dict[our_key] = hf_config[hf_key]
    
    return GPT2Config(**config_dict)


def create_weight_mapping() -> Dict[str, str]:
    """
    Create mapping from HuggingFace weight keys to PyTorch model keys
    
    Returns:
        Dictionary mapping HF keys to our model keys
    """
    # Base mappings
    mappings = {
        # Embeddings
        "transformer.wte.weight": "wte.weight",
        "transformer.wpe.weight": "wpe.weight",
        
        # Final layer norm
        "transformer.ln_f.weight": "ln_f.weight",
        "transformer.ln_f.bias": "ln_f.bias",
        
        # Language model head (if not tied)
        "lm_head.weight": "lm_head.weight",
    }
    
    # Add transformer block mappings
    for i in range(48):  # Support up to 48 layers (more than we need)
        layer_mappings = {
            # Layer norm 1
            f"transformer.h.{i}.ln_1.weight": f"h.{i}.ln_1.weight",
            f"transformer.h.{i}.ln_1.bias": f"h.{i}.ln_1.bias",
            
            # Attention
            f"transformer.h.{i}.attn.c_attn.weight": f"h.{i}.attn.c_attn.weight",
            f"transformer.h.{i}.attn.c_attn.bias": f"h.{i}.attn.c_attn.bias",
            f"transformer.h.{i}.attn.c_proj.weight": f"h.{i}.attn.c_proj.weight",
            f"transformer.h.{i}.attn.c_proj.bias": f"h.{i}.attn.c_proj.bias",
            
            # Layer norm 2  
            f"transformer.h.{i}.ln_2.weight": f"h.{i}.ln_2.weight",
            f"transformer.h.{i}.ln_2.bias": f"h.{i}.ln_2.bias",
            
            # MLP
            f"transformer.h.{i}.mlp.c_fc.weight": f"h.{i}.mlp.c_fc.weight",
            f"transformer.h.{i}.mlp.c_fc.bias": f"h.{i}.mlp.c_fc.bias",
            f"transformer.h.{i}.mlp.c_proj.weight": f"h.{i}.mlp.c_proj.weight",
            f"transformer.h.{i}.mlp.c_proj.bias": f"h.{i}.mlp.c_proj.bias",
        }
        mappings.update(layer_mappings)
    
    return mappings


def load_safetensors_weights(model_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load weights from safetensors file
    
    Args:
        model_path: Path to model directory containing model.safetensors
        
    Returns:
        Dictionary of weight tensors
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors library is required. Install with: pip install safetensors")
    
    safetensors_path = Path(model_path) / "model.safetensors"
    
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")
    
    weights = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    
    return weights


def load_pytorch_weights(model_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load weights from pytorch_model.bin file (fallback)
    
    Args:
        model_path: Path to model directory containing pytorch_model.bin
        
    Returns:
        Dictionary of weight tensors
    """
    pytorch_path = Path(model_path) / "pytorch_model.bin"
    
    if not pytorch_path.exists():
        raise FileNotFoundError(f"PyTorch weights file not found: {pytorch_path}")
    
    return torch.load(pytorch_path, map_location="cpu")


def load_weights_from_hf(model_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load weights from HuggingFace model directory
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary of weight tensors
    """
    model_path = Path(model_path)
    
    # Try safetensors first (preferred), then pytorch_model.bin
    if (model_path / "model.safetensors").exists() and SAFETENSORS_AVAILABLE:
        return load_safetensors_weights(model_path)
    elif (model_path / "pytorch_model.bin").exists():
        return load_pytorch_weights(model_path)
    else:
        raise FileNotFoundError(
            f"No weight files found in {model_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )


def convert_hf_weights_to_pytorch(hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace weights to PyTorch model format
    
    Args:
        hf_weights: Dictionary of HF weight tensors
        
    Returns:
        Dictionary of converted weight tensors
    """
    weight_mapping = create_weight_mapping()
    pytorch_weights = {}
    
    for hf_key, pytorch_key in weight_mapping.items():
        if hf_key in hf_weights:
            weight = hf_weights[hf_key].clone()
            
            # Handle weight transposition for linear layers
            # HuggingFace stores linear weights as [in_features, out_features]
            # PyTorch nn.Linear expects [out_features, in_features]
            if any(layer_type in hf_key for layer_type in ['.c_attn.weight', '.c_proj.weight', '.c_fc.weight']):
                weight = weight.t()  # Transpose
            
            pytorch_weights[pytorch_key] = weight
        # Don't error on missing keys - some might be optional
    
    return pytorch_weights


def load_pretrained_model(
    model_path: Union[str, Path], 
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> GPT2Model:
    """
    Load a pretrained GPT-2 model from HuggingFace format
    
    Args:
        model_path: Path to model directory or HF model name
        device: Device to load model on
        torch_dtype: Data type for model weights
        
    Returns:
        Loaded GPT2Model instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch_dtype is None:
        torch_dtype = torch.float32
    
    # Handle HuggingFace model names (for future hub integration)
    if isinstance(model_path, str) and "/" in model_path and not Path(model_path).exists():
        # This would be a HF hub model name - for now just error
        raise NotImplementedError("HuggingFace Hub loading not yet implemented. Use local path.")
    
    model_path = Path(model_path)
    
    # Load config and create model
    config = load_hf_config(model_path)
    model = GPT2Model(config)
    
    # Load and convert weights
    print(f"Loading weights from {model_path}")
    hf_weights = load_weights_from_hf(model_path)
    pytorch_weights = convert_hf_weights_to_pytorch(hf_weights)
    
    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(pytorch_weights, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
    
    # Move to device and set dtype
    model = model.to(device=device, dtype=torch_dtype)
    
    print(f"Model loaded successfully with {model.get_num_params():,} parameters")
    return model


def verify_weight_loading(model: GPT2Model, test_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Verify that weights loaded correctly by checking basic properties
    
    Args:
        model: Loaded model to verify
        test_input: Optional test input for forward pass verification
        
    Returns:
        Dictionary with verification results
    """
    model.eval()
    
    results = {
        "num_parameters": model.get_num_params(),
        "embedding_weights_shape": model.wte.weight.shape,
        "position_weights_shape": model.wpe.weight.shape,
        "forward_pass_success": False,
        "output_shape": None,
        "has_nan_weights": False
    }
    
    # Check for NaN weights
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            results["has_nan_weights"] = True
            print(f"Warning: NaN detected in {name}")
            break
    
    # Test forward pass
    if test_input is None:
        # Create dummy input
        test_input = torch.randint(0, model.config.vocab_size, (1, 10))
    
    try:
        with torch.no_grad():
            outputs = model(test_input)
            results["forward_pass_success"] = True
            results["output_shape"] = outputs["logits"].shape
            
            # Check for NaN in outputs
            if torch.isnan(outputs["logits"]).any():
                print("Warning: NaN detected in model outputs")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    
    return results


# Convenience function for RookWorld model loading
def load_rookworld_model(
    model_path: str = "jrahn/RookWorld-LM-124M",
    device: Optional[Union[str, torch.device]] = None
) -> GPT2Model:
    """
    Load the RookWorld-LM-124M model specifically
    
    Args:
        model_path: Path to RookWorld model (default: HF repo name)
        device: Device to load on
        
    Returns:
        Loaded RookWorld GPT2Model
    """
    return load_pretrained_model(model_path, device=device, torch_dtype=torch.float32)