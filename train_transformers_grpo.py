#!/usr/bin/env python3
"""
Simple runner for transformers/TRL GRPO training.
"""

import sys
import os
sys.path.insert(0, 'src')

from transformers_grpo.train import main

if __name__ == "__main__":
    main()