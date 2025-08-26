"""
Lean GRPO Implementation for RookWorld-LM Training

This package provides a minimal, focused implementation of GRPO training
for RookWorld-LM without the complexity and memory leaks of the original codebase.

Key principles:
- Minimal dependencies
- No dead code
- Extensive logging 
- Clear GPU placement (cuda:0 for training, cuda:1 for frozen model)
- Simple reward computation with validation
"""

__version__ = "0.1.0"