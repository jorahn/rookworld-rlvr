"""
Lean GRPO Implementation for RookWorld Training

Minimal GRPO algorithm without complex config classes or dead code.
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass 
class GRPOBatch:
    """Simple batch structure for GRPO"""
    prompts: List[str]
    completions: List[str]
    rewards: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    task_types: List[str]


class LeanGRPOTrainer:
    """Minimal GRPO trainer with extensive logging"""
    
    def __init__(
        self,
        model,  # Training model on cuda:0
        ref_model,  # Reference model on cuda:1
        tokenizer,
        group_size: int = 8,
        clip_range: float = 0.2,
        kl_coef: float = 0.02,
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        
        # Optimizer - simple AdamW
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training state
        self.step = 0
        
        logger.info(f"GRPO Trainer initialized - group_size: {group_size}, "
                   f"clip_range: {clip_range}, kl_coef: {kl_coef}, lr: {learning_rate}")
    
    def collect_rollouts(
        self,
        prompts: List[str], 
        task_types: List[str],
        validator
    ) -> GRPOBatch:
        """Collect rollouts for a batch of prompts - process P: and A: tasks separately to avoid padding issues"""
        
        logger.info(f"Collecting rollouts for {len(prompts)} prompts")
        start_time = time.time()
        
        # Separate P: and A: tasks to avoid extreme padding differences
        p_indices = [i for i, t in enumerate(task_types) if t == "P"]
        a_indices = [i for i, t in enumerate(task_types) if t == "A"]
        
        logger.info(f"Task distribution - P: {len(p_indices)}, A: {len(a_indices)}")
        
        completions = [""] * len(prompts)
        rewards_list = [0.0] * len(prompts)
        logprobs_list = [0.0] * len(prompts)
        ref_logprobs_list = [0.0] * len(prompts)
        
        # Process P: tasks
        if p_indices:
            logger.debug(f"Processing {len(p_indices)} P: tasks")
            p_prompts = [prompts[i] for i in p_indices]
            p_tasks = ["P"] * len(p_indices)
            
            for j in range(0, len(p_prompts), self.group_size):
                group_prompts = p_prompts[j:j+self.group_size]
                group_tasks = p_tasks[j:j+self.group_size]
                group_indices = p_indices[j:j+self.group_size]
                
                logger.debug(f"P: group {j//self.group_size + 1}: {len(group_prompts)} prompts")
                
                # Generate completions with training model (cuda:0)
                group_completions, group_logprobs = self._generate_group(group_prompts, self.model, device="cuda:0")
                
                # Get reference logprobs with frozen model (cuda:1)  
                group_ref_logprobs = self._compute_ref_logprobs(group_prompts, group_completions, device="cuda:1")
                
                # Compute rewards using validator
                group_rewards = self._compute_rewards(group_prompts, group_completions, group_tasks, validator)
                
                # Apply group-relative baseline (core of GRPO)
                baseline = torch.mean(group_rewards)
                group_advantages = group_rewards - baseline
                
                logger.debug(f"P: group rewards - mean: {baseline:.3f}, std: {torch.std(group_rewards):.3f}")
                
                # Store at original indices
                for k, idx in enumerate(group_indices):
                    completions[idx] = group_completions[k]
                    rewards_list[idx] = group_advantages[k].item()
                    logprobs_list[idx] = group_logprobs[k].item()
                    ref_logprobs_list[idx] = group_ref_logprobs[k].item()
        
        # Process A: tasks
        if a_indices:
            logger.debug(f"Processing {len(a_indices)} A: tasks")
            a_prompts = [prompts[i] for i in a_indices]
            a_tasks = ["A"] * len(a_indices)
            
            for j in range(0, len(a_prompts), self.group_size):
                group_prompts = a_prompts[j:j+self.group_size]
                group_tasks = a_tasks[j:j+self.group_size]
                group_indices = a_indices[j:j+self.group_size]
                
                logger.debug(f"A: group {j//self.group_size + 1}: {len(group_prompts)} prompts")
                
                # Generate completions with training model (cuda:0)
                group_completions, group_logprobs = self._generate_group(group_prompts, self.model, device="cuda:0")
                
                # Get reference logprobs with frozen model (cuda:1)  
                group_ref_logprobs = self._compute_ref_logprobs(group_prompts, group_completions, device="cuda:1")
                
                # Compute rewards using validator
                group_rewards = self._compute_rewards(group_prompts, group_completions, group_tasks, validator)
                
                # Apply group-relative baseline (core of GRPO)
                baseline = torch.mean(group_rewards)
                group_advantages = group_rewards - baseline
                
                logger.debug(f"A: group rewards - mean: {baseline:.3f}, std: {torch.std(group_rewards):.3f}")
                
                # Store at original indices
                for k, idx in enumerate(group_indices):
                    completions[idx] = group_completions[k]
                    rewards_list[idx] = group_advantages[k].item()
                    logprobs_list[idx] = group_logprobs[k].item()
                    ref_logprobs_list[idx] = group_ref_logprobs[k].item()
        
        # Convert lists to tensors
        all_rewards = torch.tensor(rewards_list, dtype=torch.float32)
        all_logprobs = torch.tensor(logprobs_list, dtype=torch.float32)
        all_ref_logprobs = torch.tensor(ref_logprobs_list, dtype=torch.float32)
        
        collection_time = time.time() - start_time
        logger.info(f"Rollout collection completed in {collection_time:.2f}s")
        logger.info(f"Rewards - mean: {torch.mean(all_rewards):.3f}, "
                   f"std: {torch.std(all_rewards):.3f}, "
                   f"min: {torch.min(all_rewards):.3f}, "
                   f"max: {torch.max(all_rewards):.3f}")
        
        return GRPOBatch(
            prompts=prompts,
            completions=completions,
            rewards=all_rewards,
            logprobs=all_logprobs,
            ref_logprobs=all_ref_logprobs,
            task_types=task_types
        )
    
    def train_step(self, batch: GRPOBatch) -> Dict[str, float]:
        """Single GRPO training step"""
        
        logger.info(f"Training step {self.step}")
        start_time = time.time()
        
        # Move batch data to training device
        rewards = batch.rewards.to("cuda:0")
        old_logprobs = batch.logprobs.to("cuda:0") 
        ref_logprobs = batch.ref_logprobs.to("cuda:0")
        
        logger.debug(f"Batch tensors moved to cuda:0 - rewards: {rewards.shape}, "
                    f"logprobs: {old_logprobs.shape}, ref_logprobs: {ref_logprobs.shape}")
        
        # Forward pass to get current logprobs
        current_logprobs = self._compute_current_logprobs(batch.prompts, batch.completions)
        
        # Compute policy loss (PPO-style clipped objective)
        ratio = torch.exp(current_logprobs - old_logprobs)
        clip_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        
        policy_loss = -torch.min(
            ratio * rewards,
            clip_ratio * rewards
        ).mean()
        
        # Compute KL divergence penalty
        kl_div = current_logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl_div.mean()
        
        # Total loss
        total_loss = policy_loss + kl_penalty
        
        logger.debug(f"Loss components - policy: {policy_loss:.4f}, "
                    f"kl_penalty: {kl_penalty:.4f}, total: {total_loss:.4f}")
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Log gradient norms
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        logger.debug(f"Gradient norm: {grad_norm:.4f}")
        
        self.optimizer.step()
        
        # Memory cleanup
        torch.cuda.empty_cache()
        
        step_time = time.time() - start_time
        self.step += 1
        
        # Training metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(), 
            "total_loss": total_loss.item(),
            "kl_divergence": kl_div.mean().item(),
            "reward_mean": rewards.mean().item(),
            "ratio_mean": ratio.mean().item(),
            "grad_norm": grad_norm.item(),
            "step_time": step_time
        }
        
        logger.info(f"Step {self.step} completed - loss: {total_loss:.4f}, "
                   f"kl_div: {kl_div.mean():.4f}, reward: {rewards.mean():.3f}, "
                   f"time: {step_time:.2f}s")
        
        return metrics
    
    def _generate_group(self, prompts: List[str], model, device: str) -> Tuple[List[str], torch.Tensor]:
        """Generate completions for a group of prompts"""
        
        logger.debug(f"Generating completions on {device}")
        
        # Tokenize prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        logger.debug(f"Input shape: {input_ids.shape}, device: {input_ids.device}")
        logger.debug(f"First prompt tokens: {input_ids[0, :20].tolist()}")
        
        with torch.no_grad():
            # Generate completions (~144 tokens as requested)
            # Lower temperature for more coherent generation
            # Pass attention_mask to model's generate_tokens method
            generated = model.generate_tokens(
                input_ids,
                max_new_tokens=144,
                temperature=0.7,
                do_sample=True,
                attention_mask=attention_mask
            )
            
            logger.debug(f"Generated shape: {generated.shape}")
            logger.debug(f"First generated tokens: {generated[0, :20].tolist()}")
        
        # Decode completions directly - model.generate_tokens returns ONLY new tokens
        completions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Log the generated completions for debugging
        for i, completion in enumerate(completions[:2]):  # Log first 2
            logger.debug(f"Completion {i}: {completion[:100]}...")
            
            # Also log the prompt for comparison
            prompt_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            logger.debug(f"Original prompt {i}: {prompt_text[:100]}...")
        
        # Compute logprobs for generated tokens
        with torch.no_grad():
            full_input = torch.cat([input_ids, generated], dim=1)
            logits, _ = model(full_input)
            
            # Get logprobs for generated tokens only
            # The logits at position i predict token i+1
            # So logits[:, input_ids.shape[1]-1:, :] predicts generated tokens
            gen_logits = logits[:, input_ids.shape[1]-1:-1, :]  # Skip last logit (no target for it)
            gen_logprobs = F.log_softmax(gen_logits, dim=-1)
            
            # Extract token logprobs for the generated tokens
            token_logprobs = torch.gather(gen_logprobs, 2, generated.unsqueeze(-1)).squeeze(-1)
            
            # Mean logprob per sequence
            mean_logprobs = token_logprobs.mean(dim=1)
        
        logger.debug(f"Generated {len(completions)} completions with mean logprob: {mean_logprobs.mean():.3f}")
        
        return completions, mean_logprobs.detach().cpu()
    
    def _compute_ref_logprobs(self, prompts: List[str], completions: List[str], device: str) -> torch.Tensor:
        """Compute reference logprobs with frozen model - BATCHED for performance"""
        
        logger.debug(f"Computing reference logprobs on {device} (batched)")
        
        # Prepare all sequences for batched processing
        full_texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
        
        # Tokenize all sequences at once with padding
        full_inputs = self.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        full_input_ids = full_inputs["input_ids"].to(device)
        full_attention_mask = full_inputs["attention_mask"].to(device)
        
        # Tokenize prompts to get their lengths
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        
        with torch.no_grad():
            # Single forward pass for all sequences
            logits, _ = self.ref_model(full_input_ids, attention_mask=full_attention_mask)
            logprobs = F.log_softmax(logits, dim=-1)
            
            # Extract logprobs for each sequence
            ref_logprobs = []
            for i, prompt_len in enumerate(prompt_lengths):
                seq_len = full_attention_mask[i].sum().item()
                num_completion_tokens = seq_len - prompt_len
                
                if num_completion_tokens > 0:
                    # Extract logprobs for this sequence's completion
                    completion_logprobs = logprobs[i, prompt_len-1:prompt_len-1+num_completion_tokens, :]
                    completion_tokens = full_input_ids[i, prompt_len:prompt_len+num_completion_tokens]
                    
                    # Gather logprobs for actual tokens
                    token_logprobs = torch.gather(completion_logprobs, 1, completion_tokens.unsqueeze(-1)).squeeze(-1)
                    mean_logprob = token_logprobs.mean()
                else:
                    mean_logprob = torch.tensor(0.0, device=device)
                
                ref_logprobs.append(mean_logprob.cpu())
        
        return torch.stack(ref_logprobs)
    
    def _compute_current_logprobs(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        """Compute current model logprobs - BATCHED for performance"""
        
        # Prepare all sequences for batched processing
        full_texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
        
        # Tokenize all sequences at once with padding
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        full_input_ids = full_inputs["input_ids"].to("cuda:0")
        full_attention_mask = full_inputs["attention_mask"].to("cuda:0")
        
        # Tokenize prompts to get their lengths
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        
        # Single forward pass for all sequences
        logits, _ = self.model(full_input_ids, attention_mask=full_attention_mask)
        logprobs = F.log_softmax(logits, dim=-1)
        
        # Extract logprobs for each sequence
        current_logprobs = []
        for i, prompt_len in enumerate(prompt_lengths):
            seq_len = full_attention_mask[i].sum().item()
            num_completion_tokens = seq_len - prompt_len
            
            if num_completion_tokens > 0:
                # Extract logprobs for this sequence's completion
                completion_logprobs = logprobs[i, prompt_len-1:prompt_len-1+num_completion_tokens, :]
                completion_tokens = full_input_ids[i, prompt_len:prompt_len+num_completion_tokens]
                
                # Gather logprobs for actual tokens
                token_logprobs = torch.gather(completion_logprobs, 1, completion_tokens.unsqueeze(-1)).squeeze(-1)
                mean_logprob = token_logprobs.mean()
            else:
                mean_logprob = torch.tensor(0.0, device="cuda:0")
            
            current_logprobs.append(mean_logprob)
        
        return torch.stack(current_logprobs)
    
    def _compute_rewards(
        self, 
        prompts: List[str], 
        completions: List[str], 
        task_types: List[str],
        validator
    ) -> torch.Tensor:
        """
        Compute rewards using the validator with new priority weighting
        
        P: tasks priority: best_move (4.0) > format (2.0) > move_candidates (1.5) > evaluations (1.0)
        A: tasks priority: format (4.0) > fen_match (3.0) > game_state (2.0) > reward_value (1.0)
        """
        
        rewards = []
        
        for prompt, completion, task_type in zip(prompts, completions, task_types):
            if task_type == "P":
                # Extract FEN from prompt
                fen = self._extract_fen_from_prompt(prompt)
                validation_result = validator.validate_policy_completion(fen, completion)
                
                # Log detailed validation results for debugging
                logger.debug(f"P: validation result: {validation_result}")
                logger.debug(f"P: completion preview: {completion[:100]}...")
                
                # Weighted reward prioritizing best move accuracy (#1 key metric)
                reward = (
                    validation_result.get("best_move", 0.0) * 4.0 +      # #1 priority
                    validation_result.get("format", 0.0) * 2.0 +         # #2 priority  
                    validation_result.get("move_candidates", 0.0) * 1.5 + # #3 priority
                    validation_result.get("evaluations", 0.0) * 1.0      # #4 priority
                ) / 8.5  # Normalize to 0-1 range
                
            elif task_type == "A": 
                # Extract FEN, move, and history from prompt
                fen, move_uci, history = self._extract_fen_move_history_from_prompt(prompt)
                validation_result = validator.validate_environment_completion(fen, move_uci, history, completion)
                
                # Log detailed validation results for debugging
                logger.debug(f"A: validation result: {validation_result}")
                logger.debug(f"A: completion preview: {completion[:100]}...")
                
                # Weighted reward prioritizing format and FEN match
                reward = (
                    validation_result.get("format", 0.0) * 4.0 +      # #1 priority
                    validation_result.get("fen_match", 0.0) * 3.0 +   # #2 priority
                    validation_result.get("game_state", 0.0) * 2.0 +  # #3 priority
                    validation_result.get("reward_value", 0.0) * 1.0  # #4 priority
                ) / 10.0  # Normalize to 0-1 range
                
            else:
                # Unknown task type
                logger.warning(f"Unknown task type: {task_type}")
                reward = 0.0
            
            rewards.append(reward)
            logger.info(f"Task {task_type} reward: {reward:.3f}, components: {validation_result if task_type != 'unknown' else {}}")
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _extract_fen_from_prompt(self, prompt: str) -> str:
        """Extract FEN from P: task prompt"""
        if "P:" in prompt:
            parts = prompt.split("P:", 1)
            if len(parts) > 1:
                remainder = parts[1].strip()
                
                # Remove M: suffix if present
                if " M:" in remainder:
                    remainder = remainder.split(" M:")[0].strip()
                
                # FEN format: 8 parts separated by / for board, then castling, en passant, etc.
                # Full FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
                tokens = remainder.split()
                
                if tokens and "/" in tokens[0]:
                    # Build full FEN from available parts
                    fen_parts = []
                    for i, token in enumerate(tokens):
                        fen_parts.append(token)
                        # A complete FEN has the board (with /), side to move, castling, en passant, halfmove, fullmove
                        if i >= 5:  # We have at least 6 parts
                            break
                    
                    # Ensure we have a valid FEN with at least board and side to move
                    if len(fen_parts) >= 2:
                        # Pad missing parts with defaults
                        while len(fen_parts) < 6:
                            if len(fen_parts) == 2:
                                fen_parts.append("KQkq")  # Castling rights
                            elif len(fen_parts) == 3:
                                fen_parts.append("-")  # En passant
                            elif len(fen_parts) == 4:
                                fen_parts.append("0")  # Halfmove clock
                            elif len(fen_parts) == 5:
                                fen_parts.append("1")  # Fullmove number
                        
                        return " ".join(fen_parts)
                    elif len(fen_parts) == 1:
                        # Just board position, add defaults
                        return f"{fen_parts[0]} w KQkq - 0 1"
        
        # Fallback to starting position
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def _extract_fen_move_from_prompt(self, prompt: str) -> Tuple[str, str]:
        """Extract FEN and move from A: task prompt"""
        if "A:" in prompt:
            parts = prompt.split("A:", 1)
            if len(parts) > 1:
                remainder = parts[1].strip()
                
                # Handle + delimited format: A: FEN+move+
                if "+" in remainder:
                    components = remainder.split("+")
                    if len(components) >= 2:
                        fen_str = components[0].strip()
                        move_str = components[1].strip()
                        
                        # Parse FEN (might be space-separated within the first component)
                        fen_tokens = fen_str.split()
                        if fen_tokens and "/" in fen_tokens[0]:
                            # Build complete FEN
                            fen_parts = []
                            for token in fen_tokens[:6]:  # Max 6 parts for FEN
                                fen_parts.append(token)
                            
                            # Pad with defaults if incomplete
                            while len(fen_parts) < 6:
                                if len(fen_parts) == 1:
                                    fen_parts.append("w")
                                elif len(fen_parts) == 2:
                                    fen_parts.append("KQkq")
                                elif len(fen_parts) == 3:
                                    fen_parts.append("-")
                                elif len(fen_parts) == 4:
                                    fen_parts.append("0")
                                elif len(fen_parts) == 5:
                                    fen_parts.append("1")
                            
                            fen = " ".join(fen_parts)
                            
                            # Clean up move (remove any trailing +)
                            move = move_str.rstrip("+").strip()
                            
                            # Validate move format (should be like e2e4 or e7e8q)
                            if len(move) >= 4:
                                return fen, move
                
                # Handle space-separated format
                else:
                    tokens = remainder.split()
                    if len(tokens) >= 7 and "/" in tokens[0]:
                        # Standard FEN format with move at the end
                        fen = " ".join(tokens[:6])
                        move = tokens[6] if len(tokens) > 6 else "e2e4"
                        return fen, move
        
        # Fallback to starting position and common opening move
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"
    
    def _extract_fen_move_history_from_prompt(self, prompt: str) -> Tuple[str, str, str]:
        """Extract FEN, move, and history from A: task prompt (new format)"""
        if "A:" in prompt:
            parts = prompt.split("A:", 1)
            if len(parts) > 1:
                remainder = parts[1].strip()
                
                # Handle new + delimited format: A: FEN+move+history+
                if "+" in remainder:
                    components = remainder.split("+")
                    if len(components) >= 3:
                        fen_str = components[0].strip()
                        move_str = components[1].strip()
                        history_str = components[2].strip()
                        
                        # Parse FEN (might be space-separated within the first component)
                        fen_tokens = fen_str.split()
                        if fen_tokens and "/" in fen_tokens[0]:
                            # Build complete FEN
                            fen_parts = []
                            for token in fen_tokens[:6]:  # Max 6 parts for FEN
                                fen_parts.append(token)
                            
                            # Pad with defaults if incomplete
                            while len(fen_parts) < 6:
                                if len(fen_parts) == 1:
                                    fen_parts.append("w")
                                elif len(fen_parts) == 2:
                                    fen_parts.append("KQkq")
                                elif len(fen_parts) == 3:
                                    fen_parts.append("-")
                                elif len(fen_parts) == 4:
                                    fen_parts.append("0")
                                elif len(fen_parts) == 5:
                                    fen_parts.append("1")
                            
                            fen = " ".join(fen_parts)
                            
                            # Clean up move (remove any trailing +)
                            move = move_str.rstrip("+").strip()
                            
                            # History is comma-separated moves (up to 10 previous moves)
                            history = history_str
                            
                            # Validate move format (should be like e2e4 or e7e8q)
                            if len(move) >= 4:
                                return fen, move, history
        
        # Fallback to starting position, common opening move, and empty history
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", ""