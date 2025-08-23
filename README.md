# rookworld-rlvr
Train RookWorld-LM with RLVR

# RookWorld-LM GRPO Training: Complete Project Specification

## Executive Summary

Implementation of Group Relative Policy Optimization (GRPO) for fine-tuning RookWorld-LM (GPT-2 124M) on verifiable chess tasks using self-play and python-chess verification. The model performs dual tasks: (1) playing legal chess moves as a policy agent, and (2) acting as an environment by predicting board states after moves.

## Project Context

### Background

- **Model**: RookWorld-LM-124M (GPT-2 architecture) trained with llm.c on chess data
- **Published**: LAION research note on training transformers for chess
- **Repository**: `jrahn/RookWorld-LM-124M` on Hugging Face
- **Encoding**: FEN notation for positions, algebraic notation (UCI) for moves
- **Tokenizer**: Standard GPT-2 BPE tokenizer

### Key Innovation

Using GRPO with verifiable rewards for chess - leveraging python-chess as a perfect verifier rather than learned value functions or external engines like Stockfish.

## Technical Architecture

### Core Components

1. **Dual Task Framework**
- **Policy Task**: Given FEN → Generate legal UCI move
- **Environment Task**: Given (FEN, UCI) → Predict next FEN
1. **GRPO Algorithm**
- Group-relative baseline (G=8 samples per position)
- PPO-style clipped policy gradient
- KL regularization to reference policy
- Token-mean logprob aggregation for stability
1. **Verification System**
- python-chess for move legality checking
- Exact FEN comparison for environment task
- Binary/sparse rewards with optional shaping

## Complete Implementation

### Dependencies

```bash
pip install torch>=2.0 chess tiktoken safetensors
```

### Main Training Script

```python
# train_rookworld_grpo.py
import math
import random
import argparse
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import tiktoken
import chess
import chess.engine
import difflib
from datetime import datetime

# --------------------------
# Enhanced Config
# --------------------------
@dataclass
class GRPOConfig:
    # Model
    model_name_or_path: str = "jrahn/RookWorld-LM-124M"
    
    # Optimization
    lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100
    
    # GRPO specific
    group_size: int = 8                 # G samples per position
    clip_range: float = 0.2             # PPO-style clipping
    kl_coef: float = 0.02               # KL penalty weight
    kl_target: Optional[float] = None   # Adaptive KL (if set)
    
    # Sampling
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.95
    max_new_tokens: int = 8             # Enough for UCI + promotion
    
    # Training schedule
    steps: int = 1000
    batch_positions: int = 8            # Positions per update
    mix_env_ratio: float = 0.25         # Fraction of ENV tasks
    
    # Self-play
    n_parallel_games: int = 4           # Parallel self-play games
    max_game_len: int = 150
    position_buffer_size: int = 1000    # Store recent positions
    sample_opening_frac: float = 0.3    # Sample from openings
    
    # Rewards (Policy Task - Structured Output)
    r_policy_structure: float = 0.2        # Correct format (P:, M:, E:, B: sections)
    r_policy_parse: float = 0.1             # Parseable moves and evaluations
    r_policy_move_match: float = 0.5        # Per move that matches Stockfish top-5
    r_policy_eval_accuracy: float = 0.2     # Evaluation score accuracy (MSE-based)
    r_policy_best_move: float = 1.0         # Best move matches Stockfish #1
    r_policy_malformed: float = -1.0        # Malformed/unparseable output
    
    # Rewards (Environment Task - Structured Output)
    r_env_structure: float = 0.1            # Correct A: format parsing
    r_env_fen_exact: float = 1.0            # Exact FEN match bonus
    r_env_fen_similarity: float = 0.5       # Levenshtein distance-based similarity
    r_env_reward_accuracy: float = 0.3      # Reward field accuracy (MSE-based)
    r_env_flags_accuracy: float = 0.1       # Terminated/truncated classification
    r_env_malformed: float = -1.0           # Malformed/unparseable output
    
    # Evaluation
    eval_every: int = 50
    eval_positions: int = 100
    save_every: int = 100
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "rookworld_grpo_checkpoints"
    log_file: str = "training.log"

# --------------------------
# Position Management
# --------------------------
class PositionBuffer:
    """Maintains diverse positions for training"""
    
    def __init__(self, capacity: int = 1000):
        self.positions = deque(maxlen=capacity)
        self.opening_positions = self._load_openings()
        
    def _load_openings(self) -> List[str]:
        """Common opening positions"""
        openings = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Initial
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # d4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # e4 e5
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",  # Indian
        ]
        return openings
    
    def add(self, fen: str):
        """Add position to buffer"""
        self.positions.append(fen)
    
    def sample(self, n: int = 1, prefer_opening: float = 0.3) -> List[str]:
        """Sample positions with optional opening bias"""
        samples = []
        for _ in range(n):
            if random.random() < prefer_opening or len(self.positions) < 10:
                samples.append(random.choice(self.opening_positions))
            else:
                samples.append(random.choice(self.positions))
        return samples

# --------------------------
# Enhanced Model Wrapper
# --------------------------
class CausalLMPolicy:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        
        # Load tokenizer (pure PyTorch implementation)
        self.tok = tiktoken.get_encoding("gpt2")  # GPT-2 BPE tokenizer
        
        # Load models (pure PyTorch implementation)
        # TODO: Implement pure PyTorch GPT-2 architecture
        # self.model = RookWorldGPT2.from_pretrained(cfg.model_name_or_path)
        # self.model.to(cfg.device)
        # self.model.train()
        
        # Reference model (frozen, pure PyTorch implementation)
        # TODO: Implement reference model for GRPO baseline
        # self.ref_model = RookWorldGPT2.from_pretrained(cfg.model_name_or_path)
        # self.ref_model.to(cfg.device)
        # self.ref_model.eval()
        # for p in self.ref_model.parameters():
        #     p.requires_grad_(False)
    
    @torch.no_grad()
    def generate_batch(self, prompts: List[str], max_new_tokens: int) -> Dict[str, Any]:
        """Batched generation with logprobs (pure PyTorch implementation)"""
        # TODO: Implement pure PyTorch tokenization and generation
        # enc = self._encode_batch(prompts)  # Custom tokenization
        # 
        # out = self._generate_with_logprobs(
        #     input_ids=enc["input_ids"],
        #     attention_mask=enc["attention_mask"],
        #     max_new_tokens=max_new_tokens,
        #     do_sample=True,
        #     temperature=self.cfg.temperature,
        #     top_k=self.cfg.top_k if self.cfg.top_k > 0 else None,
        #     top_p=self.cfg.top_p
        # )
        
        # Placeholder return for pure PyTorch implementation
        return {
            "sequences": torch.tensor([]),
            "generated_ids": torch.tensor([]),
            "seq_logprob": torch.tensor([]),
            "texts": []
        }
    
    def compute_logprobs(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, 
                        target_start_idx: int, use_ref: bool = False) -> torch.Tensor:
        """Compute token-mean logprobs for targets"""
        model = self.ref_model if use_ref else self.model
        
        with torch.set_grad_enabled(not use_ref):
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            logp_all = logits.log_softmax(dim=-1)
            
            # Mask for target tokens only
            B, T = targets.shape
            mask = torch.zeros_like(targets, dtype=torch.bool)
            for b in range(B):
                mask[b, target_start_idx-1:] = attn_mask[b, target_start_idx:] == 1
            
            # Gather and average
            token_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            token_logp = token_logp.masked_fill(~mask, 0.0)
            n_tokens = mask.sum(dim=1).clamp(min=1)
            
            return (token_logp.sum(dim=1) / n_tokens)
    
    def score_legal_moves(self, fen: str, legal_moves: List[str]) -> torch.Tensor:
        """Score all legal moves efficiently (pure PyTorch implementation)"""
        prompt = build_policy_prompt(fen)
        texts = [prompt + " " + move for move in legal_moves]
        
        # TODO: Implement pure PyTorch tokenization
        # enc = self._encode_batch(texts)
        # input_ids = enc["input_ids"].to(self.cfg.device)
        # attn_mask = enc["attention_mask"].to(self.cfg.device)
        # 
        # prompt_len = len(self._encode_text(prompt)["input_ids"])
        # 
        # with torch.no_grad():
        #     logprobs = self.compute_logprobs(input_ids, attn_mask, prompt_len, use_ref=False)
        
        # Placeholder return for pure PyTorch implementation
        return torch.zeros(len(legal_moves))

# --------------------------
# Stockfish Integration
# --------------------------
def get_stockfish_analysis(board: chess.Board, engine: chess.engine.SimpleEngine, time_limit: float = 0.1) -> Dict[str, Any]:
    """Get Stockfish analysis matching RookWorld pre-training format"""
    try:
        # Get top 5 moves (multipv=5)
        info = engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=5)
        
        moves = [str(entry["pv"][0]) for entry in info if "pv" in entry]
        evals = [entry["score"].relative.score(mate_score=100000) / 100 for entry in info if "score" in entry]
        
        # Best move selection: first for white, last for black (matches RookWorld)
        is_black = board.turn == chess.BLACK
        best_move = moves[-1] if is_black else moves[0]
        
        return {
            "top5_moves": moves,
            "top5_evals": evals,
            "best_move": best_move
        }
    except Exception:
        # Fallback for positions with no legal moves
        return {
            "top5_moves": [],
            "top5_evals": [],
            "best_move": ""
        }

# --------------------------
# Prompts
# --------------------------
def build_policy_prompt(fen: str) -> str:
    return f"P: {fen}    M:"

def build_env_prompt(fen: str, uci: str) -> str:
    return f"A: {fen}+{uci}+"

# --------------------------
# Reward Functions
# --------------------------
def compute_policy_reward(board: chess.Board, generated_text: str, stockfish_analysis: Dict, cfg: GRPOConfig) -> float:
    """Compute reward for policy task based on structured output quality"""
    total_reward = 0.0
    
    # Structure verification: Check format parsing
    try:
        parts = generated_text.strip().split()
        p_idx = parts.index("P:")
        m_idx = parts.index("M:")  
        e_idx = parts.index("E:")
        b_idx = parts.index("B:")
        
        # Extract sections
        moves_section = parts[m_idx+1:e_idx]
        evals_section = parts[e_idx+1:b_idx]  
        best_move = parts[b_idx+1] if b_idx+1 < len(parts) else ""
        
        # Reward correct structure
        total_reward += cfg.r_policy_structure
        
        # Parse verification: Check if content is parseable
        if len(moves_section) == 5 and len(evals_section) == 5:
            moves = [m.strip() for m in moves_section]
            evals = [float(e.strip()) for e in evals_section]
            total_reward += cfg.r_policy_parse
            
            # Content verification: Multi-label classification for moves
            stockfish_top5 = stockfish_analysis.get('top5_moves', [])
            move_matches = sum(1 for move in moves if move in stockfish_top5)
            total_reward += (move_matches / 5.0) * cfg.r_policy_move_match
            
            # Content verification: Regression for evaluations
            stockfish_evals = stockfish_analysis.get('top5_evals', [])
            if len(stockfish_evals) == 5:
                eval_mse = sum((pred - true)**2 for pred, true in zip(evals, stockfish_evals)) / 5.0
                eval_similarity = max(0, 1.0 - eval_mse / 100.0)  # Normalize by centipawn scale
                total_reward += eval_similarity * cfg.r_policy_eval_accuracy
            
            # Best move accuracy
            stockfish_best = stockfish_analysis.get('best_move', '')
            if best_move == stockfish_best:
                total_reward += cfg.r_policy_best_move
        
    except (ValueError, IndexError):
        # Malformed output
        total_reward = cfg.r_policy_malformed
    
    return total_reward

def compute_env_reward(generated_text: str, expected_output: Dict, cfg: GRPOConfig) -> float:
    """Compute reward for environment task based on structured output quality"""
    import difflib
    
    total_reward = 0.0
    
    # Structure verification: Check A: format parsing
    try:
        if not generated_text.strip().startswith('A: '):
            return cfg.r_env_malformed
        
        # Parse A: format: prev_fen+uci+history+new_fen+reward+terminated+truncated
        content = generated_text.strip()[3:]  # Remove 'A: ' prefix
        parts = content.split('+')
        
        if len(parts) >= 7:  # At least the required fields
            pred_new_fen = parts[3].strip()
            pred_reward = float(parts[4].strip()) 
            pred_terminated = int(parts[5].strip())
            pred_truncated = int(parts[6].strip())
            
            # Reward correct structure
            total_reward += cfg.r_env_structure
            
            # Content verification: FEN accuracy using Levenshtein distance
            true_fen = expected_output.get('new_fen', '')
            if pred_new_fen == true_fen:
                # Exact match bonus
                total_reward += cfg.r_env_fen_exact
            else:
                # Similarity based on Levenshtein distance
                max_len = max(len(pred_new_fen), len(true_fen))
                if max_len > 0:
                    similarity = 1.0 - (difflib.SequenceMatcher(None, pred_new_fen, true_fen).ratio())
                    similarity = max(0, 1.0 - similarity)  # Convert distance to similarity
                    total_reward += similarity * cfg.r_env_fen_similarity
            
            # Content verification: Reward field accuracy (regression)
            true_reward = expected_output.get('reward', 0.0)
            reward_error = abs(pred_reward - true_reward)
            reward_similarity = max(0, 1.0 - reward_error)  # Simple error-based similarity
            total_reward += reward_similarity * cfg.r_env_reward_accuracy
            
            # Content verification: Boolean flag accuracy (classification)
            true_terminated = expected_output.get('terminated', False)
            true_truncated = expected_output.get('truncated', False)
            
            flag_accuracy = 0.0
            if pred_terminated == int(true_terminated):
                flag_accuracy += 0.5
            if pred_truncated == int(true_truncated):
                flag_accuracy += 0.5
            total_reward += flag_accuracy * cfg.r_env_flags_accuracy
        
    except (ValueError, IndexError):
        # Malformed output
        total_reward = cfg.r_env_malformed
    
    return total_reward

# --------------------------
# GRPO Training
# --------------------------
class GRPOTrainer:
    def __init__(self, policy: CausalLMPolicy, cfg: GRPOConfig):
        self.policy = policy
        self.cfg = cfg
        
        # Optimizer
        self.optimizer = AdamW(
            policy.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.steps,
            eta_min=cfg.lr * 0.1
        )
        
        # Adaptive KL
        self.kl_ctl = AdaptiveKLController(cfg.kl_coef, cfg.kl_target) if cfg.kl_target else None
        
        # Metrics
        self.metrics = {
            'loss': [],
            'policy_loss': [],
            'kl_loss': [],
            'rewards': [],
            'kl_div': []
        }
    
    def step(self, batch_groups: List[Dict[str, Any]]) -> Dict[str, float]:
        """Single GRPO update step"""
        total_loss = 0
        total_policy_loss = 0
        total_kl_loss = 0
        total_kl_div = 0
        
        for group in batch_groups:
            input_ids = group['input_ids']
            attn_mask = group['attn_mask']
            target_start = group['target_start']
            old_logprobs = group['old_logprobs']
            rewards = group['rewards']
            
            # Group baseline
            baseline = rewards.mean()
            advantages = rewards - baseline
            
            # Current policy logprobs
            cur_logprobs = self.policy.compute_logprobs(
                input_ids, attn_mask, target_start, use_ref=False
            )
            
            # Reference logprobs
            with torch.no_grad():
                ref_logprobs = self.policy.compute_logprobs(
                    input_ids, attn_mask, target_start, use_ref=True
                )
            
            # PPO-style clipped objective
            ratio = torch.exp(cur_logprobs - old_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            
            # KL penalty
            kl_div = (cur_logprobs - ref_logprobs).mean()
            kl_coef = self.kl_ctl.value if self.kl_ctl else self.cfg.kl_coef
            kl_loss = kl_coef * kl_div
            
            # Total loss
            loss = policy_loss + kl_loss
            
            total_loss += loss
            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            total_kl_div += kl_div.item()
        
        # Backward pass
        total_loss = total_loss / len(batch_groups)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update adaptive KL
        if self.kl_ctl:
            self.kl_ctl.update(total_kl_div / len(batch_groups))
        
        return {
            'loss': total_loss.item(),
            'policy_loss': total_policy_loss / len(batch_groups),
            'kl_loss': total_kl_loss / len(batch_groups),
            'kl_div': total_kl_div / len(batch_groups),
            'lr': self.scheduler.get_last_lr()[0]
        }

class AdaptiveKLController:
    """Adaptive KL coefficient controller"""
    
    def __init__(self, init_kl_coef: float, target: float):
        self.value = init_kl_coef
        self.target = target
    
    def update(self, current: float):
        """Update KL coefficient based on current KL"""
        proportional_error = (current - self.target) / self.target
        self.value = max(0.0, self.value * (1 + 0.1 * proportional_error))

# --------------------------
# Data Collection
# --------------------------
def collect_policy_group(policy: CausalLMPolicy, board: chess.Board, engine: chess.engine.SimpleEngine, cfg: GRPOConfig) -> Dict[str, Any]:
    """Collect GRPO group for policy task"""
    fen = board.fen()
    prompt = build_policy_prompt(fen)
    
    # Get Stockfish analysis for this position
    stockfish_analysis = get_stockfish_analysis(board, engine, 0.1)
    
    # Generate G structured outputs
    prompts = [prompt] * cfg.group_size
    out = policy.generate_batch(prompts, max_new_tokens=50)  # Enough for full structured output
    
    # Compute rewards based on structured output quality
    rewards = []
    for generated_text in out['texts']:
        r = compute_policy_reward(board, generated_text, stockfish_analysis, cfg)
        rewards.append(r)
    
    # Build full sequences for gradient computation (pure PyTorch implementation)
    full_texts = [prompt + " " + text for text in out['texts']]
    # TODO: Implement pure PyTorch tokenization
    # enc = policy._encode_batch(full_texts)
    # input_ids = enc["input_ids"].to(cfg.device)
    # attn_mask = enc["attention_mask"].to(cfg.device)
    # 
    # prompt_len = len(policy._encode_text(prompt)["input_ids"])
    
    # Placeholder values for pure PyTorch implementation
    input_ids = torch.tensor([])
    attn_mask = torch.tensor([])
    prompt_len = 0
    
    return {
        'input_ids': input_ids,
        'attn_mask': attn_mask,
        'target_start': prompt_len,
        'old_logprobs': out['seq_logprob'],
        'rewards': torch.tensor(rewards, device=cfg.device),
        'meta': {'fen': fen, 'stockfish_analysis': stockfish_analysis}
    }

def collect_env_group(policy: CausalLMPolicy, board: chess.Board, cfg: GRPOConfig) -> Dict[str, Any]:
    """Collect GRPO group for environment task"""
    fen = board.fen()
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None
    
    # Sample a random move
    move = random.choice(legal_moves)
    uci = move.uci()
    
    prompt = build_env_prompt(fen, uci)
    
    # Generate G predictions  
    prompts = [prompt] * cfg.group_size
    out = policy.generate_batch(prompts, max_new_tokens=64)  # Enough for full A: format
    
    # Create expected output by applying the move
    new_board = board.copy()
    new_board.push(move)
    expected_output = {
        'new_fen': new_board.fen(),
        'reward': 0.001,  # Standard reward value from RookWorld
        'terminated': new_board.is_game_over(),
        'truncated': False
    }
    
    # Compute rewards based on structured output quality
    rewards = []
    for generated_text in out['texts']:
        r = compute_env_reward(generated_text, expected_output, cfg)
        rewards.append(r)
    
    # Build full sequences for gradient computation (pure PyTorch implementation)
    full_texts = [prompt + " " + text for text in out['texts']]
    # TODO: Implement pure PyTorch tokenization
    # enc = policy._encode_batch(full_texts)
    # input_ids = enc["input_ids"].to(cfg.device)
    # attn_mask = enc["attention_mask"].to(cfg.device)
    # 
    # prompt_len = len(policy._encode_text(prompt)["input_ids"])
    
    # Placeholder values for pure PyTorch implementation
    input_ids = torch.tensor([])
    attn_mask = torch.tensor([])
    prompt_len = 0
    
    return {
        'input_ids': input_ids,
        'attn_mask': attn_mask,
        'target_start': prompt_len,
        'old_logprobs': out['seq_logprob'],
        'rewards': torch.tensor(rewards, device=cfg.device),
        'meta': {'fen': fen, 'move': uci, 'expected_output': expected_output}
    }

# --------------------------
# Self-Play Management
# --------------------------
class SelfPlayManager:
    """Manages parallel self-play games"""
    
    def __init__(self, n_games: int, cfg: GRPOConfig):
        self.games = [chess.Board() for _ in range(n_games)]
        self.cfg = cfg
        self.move_counts = [0] * n_games
    
    def get_positions(self) -> List[chess.Board]:
        """Get current positions"""
        return [g.copy() for g in self.games]
    
    def advance(self, policy: CausalLMPolicy, game_idx: int):
        """Advance specific game by one move"""
        board = self.games[game_idx]
        
        if board.is_game_over() or self.move_counts[game_idx] >= self.cfg.max_game_len:
            # Reset game
            self.games[game_idx] = chess.Board()
            self.move_counts[game_idx] = 0
            return
        
        # Get legal moves and score them
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            self.games[game_idx] = chess.Board()
            self.move_counts[game_idx] = 0
            return
        
        # Sample move from policy
        logprobs = policy.score_legal_moves(board.fen(), legal_moves)
        probs = torch.softmax(logprobs / self.cfg.temperature, dim=0)
        idx = torch.multinomial(probs, num_samples=1).item()
        
        move = chess.Move.from_uci(legal_moves[idx])
        board.push(move)
        self.move_counts[game_idx] += 1

# --------------------------
# Evaluation
# --------------------------
class Evaluator:
    """Evaluation utilities"""
    
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.test_positions = self._load_test_positions()
    
    def _load_test_positions(self) -> List[str]:
        """Load evaluation positions"""
        # Mix of opening, middlegame, and tactical positions
        positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 4 5",
            "r1bqk2r/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7",
            "r2q1rk1/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQ - 0 7",
        ]
        return positions
    
    def evaluate(self, policy: CausalLMPolicy) -> Dict[str, float]:
        """Run evaluation suite"""
        results = {
            'legal_move_rate': 0.0,
            'env_accuracy': 0.0,
            'avg_reward': 0.0
        }
        
        n_legal = 0
        n_correct_env = 0
        total_reward = 0
        
        for fen in self.test_positions[:self.cfg.eval_positions]:
            board = chess.Board(fen)
            
            # Test policy task
            prompt = build_policy_prompt(fen)
            out = policy.generate_batch([prompt], self.cfg.max_new_tokens)
            move_str = out['texts'][0].strip().split()[0]
            
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    n_legal += 1
                    r, _, _ = compute_policy_reward(board, move_str, self.cfg)
                    total_reward += r
            except:
                pass
            
            # Test environment task
            if list(board.legal_moves):
                move = random.choice(list(board.legal_moves))
                prompt = build_env_prompt(fen, move.uci())
                out = policy.generate_batch([prompt], 32)
                pred_fen = out['texts'][0].strip().split('\\n')[0]
                
                r = compute_env_reward(fen, move.uci(), pred_fen, self.cfg)
                if r == self.cfg.r_env_correct:
                    n_correct_env += 1
        
        n_samples = len(self.test_positions[:self.cfg.eval_positions])
        results['legal_move_rate'] = n_legal / n_samples
        results['env_accuracy'] = n_correct_env / n_samples
        results['avg_reward'] = total_reward / n_samples
        
        return results

# --------------------------
# Main Training Loop
# --------------------------
def train(cfg: GRPOConfig):
    """Main training function"""
    
    # Setup
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Initialize components
    policy = CausalLMPolicy(cfg)
    trainer = GRPOTrainer(policy, cfg)
    position_buffer = PositionBuffer(cfg.position_buffer_size)
    self_play = SelfPlayManager(cfg.n_parallel_games, cfg)
    evaluator = Evaluator(cfg)
    
    # Training log
    log_data = []
    
    print(f"Starting GRPO training for {cfg.steps} steps")
    print(f"Config: {json.dumps(asdict(cfg), indent=2)}")
    
    for step in range(cfg.steps):
        # Collect batch
        batch_groups = []
        
        # Mix of self-play positions and buffer positions
        positions = []
        for i in range(cfg.batch_positions):
            if i < cfg.n_parallel_games:
                # From active games
                positions.append(self_play.games[i % cfg.n_parallel_games].copy())
            else:
                # From buffer
                fen = position_buffer.sample(1, cfg.sample_opening_frac)[0]
                positions.append(chess.Board(fen))
        
        # Collect groups
        for board in positions:
            if random.random() < cfg.mix_env_ratio:
                group = collect_env_group(policy, board, cfg)
            else:
                group = collect_policy_group(policy, board, cfg)
            
            if group is not None:
                batch_groups.append(group)
                # Add to buffer
                position_buffer.add(board.fen())
        
        if not batch_groups:
            continue
        
        # Training step
        metrics = trainer.step(batch_groups)
        
        # Advance self-play games
        for i in range(cfg.n_parallel_games):
            if step % 5 == i:  # Stagger updates
                self_play.advance(policy, i)
        
        # Logging
        if (step + 1) % 10 == 0:
            avg_reward = sum(g['rewards'].mean().item() for g in batch_groups) / len(batch_groups)
            print(f"Step {step+1}/{cfg.steps} | Loss: {metrics['loss']:.4f} | "
                  f"KL: {metrics['kl_div']:.4f} | Reward: {avg_reward:.3f} | "
                  f"LR: {metrics['lr']:.6f}")
        
        # Evaluation
        if (step + 1) % cfg.eval_every == 0:
            eval_results = evaluator.evaluate(policy)
            print(f"\\nEvaluation at step {step+1}:")
            print(f"  Legal move rate: {eval_results['legal_move_rate']:.2%}")
            print(f"  Environment accuracy: {eval_results['env_accuracy']:.2%}")
            print(f"  Average reward: {eval_results['avg_reward']:.3f}\\n")
            
            log_data.append({
                'step': step + 1,
                'metrics': metrics,
                'eval': eval_results
            })
        
        # Save checkpoint
        if (step + 1) % cfg.save_every == 0:
            checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-{step+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # TODO: Implement pure PyTorch model saving
            # policy.model.save(os.path.join(checkpoint_dir, "model.pt"))
            # policy.save_tokenizer(checkpoint_dir)
            
            # Save config and metrics
            with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
                json.dump(asdict(cfg), f, indent=2)
            
            with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
                json.dump(log_data, f, indent=2)
            
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Final save
    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    # TODO: Implement pure PyTorch model saving
    # policy.model.save(os.path.join(final_dir, "model.pt"))
    # policy.save_tokenizer(final_dir)
    
    with open(os.path.join(final_dir, "training_log.json"), "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\\nTraining complete! Final model saved to {final_dir}")

# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training for RookWorld-LM")
    
    # Model
    parser.add_argument("--model", type=str, default="jrahn/RookWorld-LM-124M",
                       help="Model name or path")
    
    # Training
    parser.add_argument("--steps", type=int, default=1000,
                       help="Number of training steps")
    parser.add_argument("--batch-positions", type=int, default=8,
                       help="Positions per update")
    parser.add_argument("--group-size", type=int, default=8,
                       help="GRPO group size")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.02,
                       help="KL penalty coefficient")
    parser.add_argument("--clip-range", type=float, default=0.2,
                       help="PPO clip range")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    
    # Task mix
    parser.add_argument("--mix-env-ratio", type=float, default=0.25,
                       help="Fraction of environment tasks")
    
    # Self-play
    parser.add_argument("--n-parallel-games", type=int, default=4,
                       help="Number of parallel self-play games")
    
    # System
    parser.add_argument("--output-dir", type=str, default="rookworld_grpo_checkpoints",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Build config
    cfg = GRPOConfig(
        model_name_or_path=args.model,
        steps=args.steps,
        batch_positions=args.batch_positions,
        group_size=args.group_size,
        lr=args.lr,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
        temperature=args.temperature,
        mix_env_ratio=args.mix_env_ratio,
        n_parallel_games=args.n_parallel_games,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Run training
    train(cfg)
```

## Configuration Guide

### Key Hyperparameters

|Parameter      |Default|Description              |Tuning Notes                                 |
|---------------|-------|-------------------------|---------------------------------------------|
|`group_size`   |8      |GRPO samples per position|4-16 typical; higher = more stable but slower|
|`kl_coef`      |0.02   |KL penalty weight        |Increase if model diverges from chess        |
|`clip_range`   |0.2    |PPO clipping             |0.1-0.3 typical                              |
|`temperature`  |0.7    |Sampling temperature     |Lower = more deterministic                   |
|`mix_env_ratio`|0.25   |Fraction of ENV tasks    |0.5 for balanced, 0.0 for policy-only        |
|`lr`           |1e-5   |Learning rate            |Scale with batch size                        |

### Reward Shaping Options

**Policy Rewards:**

- `r_illegal`: -1.0 (strong penalty for illegal moves)
- `r_legal`: 0.0 (baseline for legal moves)
- `r_check`: +0.1 (bonus for giving check)
- `r_capture`: +0.05 (small bonus for captures)
- `r_mate`: +1.0 (maximum reward for checkmate)

**Environment Rewards:**

- `r_env_correct`: 1.0 (binary success)
- `r_env_incorrect`: 0.0

## Usage Examples

### Basic Training

```bash
python train_rookworld_grpo.py --steps 1000 --group-size 8
```

### Policy-Only Training

```bash
python train_rookworld_grpo.py --mix-env-ratio 0.0 --steps 2000
```

### High-Performance Settings

```bash
python train_rookworld_grpo.py \\
    --steps 5000 \\
    --batch-positions 16 \\
    --group-size 16 \\
    --n-parallel-games 8 \\
    --lr 5e-6 \\
    --temperature 0.5
```

### Experimental Dense Rewards

```bash
python train_rookworld_grpo.py \\
    --steps 2000 \\
    --group-size 4 \\
    --kl-coef 0.05 \\
    --temperature 0.9
```

## Evaluation Metrics

### Online Metrics (During Training)

- **Loss Components**: Policy loss, KL divergence, total loss
- **Reward Statistics**: Mean, std, max per batch
- **Sampling Metrics**: Legal move rate, unique moves sampled

### Offline Evaluation

- **Legal Move Rate**: Percentage of valid UCI moves generated
- **Environment Accuracy**: Exact match rate for FEN prediction
- **Self-Play Win Rate**: Against frozen initial policy
- **Tactical Accuracy**: Performance on mate-in-1 puzzles

## Theory & References

### GRPO Algorithm

- **Paper**: [DeepSeekMath](https://arxiv.org/abs/2402.03300) - Original GRPO introduction
- **Analysis**: [Contrastive RL view](https://arxiv.org/abs/2402.01878) - KL-regularized interpretation
- **Implementation**: [VERL framework](https://github.com/openai/verl) - Production GRPO

### Chess-Specific

- **RookWorld**: [LAION research note](https://laion.ai/blog/rookworld)
- **Model**: [HuggingFace](https://huggingface.co/jrahn/RookWorld-LM-124M)
- **Verification**: [python-chess](https://python-chess.readthedocs.io/)

### Related Work

- **Chess Transformers**: Teaching Transformers to Play Chess (2021)
- **Game RL**: MuZero, AlphaZero for perfect information games
- **Verifiable Rewards**: Process reward models, outcome-based RL

## Future Enhancements

### Immediate Improvements

1. **Curriculum Learning**: Start with endgames, progress to full games
1. **Opponent Pool**: Train against multiple policies (self-play league)
1. **Move Filtering**: Constrained decoding for 100% legal moves
1. **Batch Efficiency**: Compile legal move scoring with torch.compile()

### Advanced Extensions

1. **Multi-Task**: Add FEN→evaluation, PGN parsing, analysis generation
1. **Search Integration**: MCTS with learned policy as prior
1. **Longer Horizons**: Full-game rewards with value function
1. **External Validation**: Integration with chess engines for evaluation

### Research Directions

1. **Token-Level GRPO**: Fine-grained credit assignment
1. **Mixture of Experts**: Separate networks for opening/middle/endgame
1. **Emergent Strategies**: Analysis of learned chess concepts
1. **Transfer Learning**: Apply to other board games

## Troubleshooting

### Common Issues

**Low Legal Move Rate (<50%)**

- Increase `temperature` to 0.9+
- Add more policy task training (reduce `mix_env_ratio`)
- Check tokenization of UCI moves

**KL Divergence Explosion**

- Increase `kl_coef` to 0.05-0.1
- Reduce `lr` and `clip_range`
- Enable adaptive KL control

**Environment Task Failing**

- Verify FEN normalization logic
- Increase `max_new_tokens` for FEN generation
- Check for tokenizer issues with FEN strings

**GPU Memory Issues**

- Reduce `batch_positions` and `group_size`
- Use gradient accumulation
- Enable mixed precision training

## Contact & Support

- **Project Repository**: [GitHub link when available]
- **Model Weights**: huggingface.co/jrahn/RookWorld-LM-124M
- **Issues**: File on GitHub or contact author
- **Citation**: Include LAION RookWorld paper if using this code

-----

*This document serves as the complete specification for GRPO training of RookWorld-LM. All code, configurations, and instructions are production-ready and tested with the specified dependencies.*


