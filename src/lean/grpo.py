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
        """Collect rollouts for a batch of prompts"""
        
        logger.info(f"Collecting rollouts for {len(prompts)} prompts")
        start_time = time.time()
        
        completions = []
        rewards = []
        logprobs = []
        ref_logprobs = []
        
        # Process prompts in groups
        for i in range(0, len(prompts), self.group_size):
            group_prompts = prompts[i:i+self.group_size]
            group_tasks = task_types[i:i+self.group_size]
            
            logger.debug(f"Processing group {i//self.group_size + 1}: {len(group_prompts)} prompts")
            
            # Generate completions with training model (cuda:0)
            group_completions, group_logprobs = self._generate_group(group_prompts, self.model, device="cuda:0")
            
            # Get reference logprobs with frozen model (cuda:1)  
            group_ref_logprobs = self._compute_ref_logprobs(group_prompts, group_completions, device="cuda:1")
            
            # Compute rewards using validator
            group_rewards = self._compute_rewards(group_prompts, group_completions, group_tasks, validator)
            
            # Apply group-relative baseline (core of GRPO)
            baseline = torch.mean(group_rewards)
            group_advantages = group_rewards - baseline
            
            logger.debug(f"Group rewards - mean: {baseline:.3f}, std: {torch.std(group_rewards):.3f}")
            logger.debug(f"Group advantages - mean: {torch.mean(group_advantages):.3f}")
            
            completions.extend(group_completions)
            rewards.append(group_advantages)  # Use advantages, not raw rewards
            logprobs.append(group_logprobs)
            ref_logprobs.append(group_ref_logprobs)
        
        # Combine all groups
        all_rewards = torch.cat(rewards)
        all_logprobs = torch.cat(logprobs)
        all_ref_logprobs = torch.cat(ref_logprobs)
        
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
        
        with torch.no_grad():
            # Generate completions (~144 tokens as requested)
            generated = model.generate_tokens(
                input_ids,
                max_new_tokens=144,
                temperature=0.8,
                do_sample=True
            )
            
            logger.debug(f"Generated shape: {generated.shape}")
        
        # Decode completions
        completions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Compute logprobs for generated tokens
        with torch.no_grad():
            full_input = torch.cat([input_ids, generated], dim=1)
            logits, _ = model(full_input)
            
            # Get logprobs for generated tokens only
            gen_logits = logits[:, input_ids.shape[1]:, :]
            gen_logprobs = F.log_softmax(gen_logits, dim=-1)
            
            # Extract token logprobs
            token_logprobs = torch.gather(gen_logprobs, 2, generated.unsqueeze(-1)).squeeze(-1)
            
            # Mean logprob per sequence
            mean_logprobs = token_logprobs.mean(dim=1)
        
        logger.debug(f"Generated {len(completions)} completions with mean logprob: {mean_logprobs.mean():.3f}")
        
        return completions, mean_logprobs.detach().cpu()
    
    def _compute_ref_logprobs(self, prompts: List[str], completions: List[str], device: str) -> torch.Tensor:
        """Compute reference logprobs with frozen model"""
        
        logger.debug(f"Computing reference logprobs on {device}")
        
        ref_logprobs = []
        
        for prompt, completion in zip(prompts, completions):
            # Tokenize full sequence
            full_text = prompt + completion
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to(device)
            
            # Get prompt length
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                logits, _ = self.ref_model(input_ids)
                logprobs = F.log_softmax(logits, dim=-1)
                
                # Extract logprobs for completion tokens only
                completion_logprobs = logprobs[0, prompt_len:, :]
                completion_tokens = input_ids[0, prompt_len+1:]  # Shift for next-token prediction
                
                if len(completion_tokens) > 0:
                    token_logprobs = torch.gather(completion_logprobs[:-1], 0, completion_tokens.unsqueeze(-1)).squeeze(-1)
                    mean_logprob = token_logprobs.mean()
                else:
                    mean_logprob = torch.tensor(0.0, device=device)
                
                ref_logprobs.append(mean_logprob.cpu())
        
        return torch.stack(ref_logprobs)
    
    def _compute_current_logprobs(self, prompts: List[str], completions: List[str]) -> torch.Tensor:
        """Compute current model logprobs"""
        
        current_logprobs = []
        
        for prompt, completion in zip(prompts, completions):
            # Tokenize full sequence
            full_text = prompt + completion
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to("cuda:0")
            
            # Get prompt length
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            logits, _ = self.model(input_ids)
            logprobs = F.log_softmax(logits, dim=-1)
            
            # Extract logprobs for completion tokens only
            completion_logprobs = logprobs[0, prompt_len:, :]
            completion_tokens = input_ids[0, prompt_len+1:]  # Shift for next-token prediction
            
            if len(completion_tokens) > 0:
                token_logprobs = torch.gather(completion_logprobs[:-1], 0, completion_tokens.unsqueeze(-1)).squeeze(-1)
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
        """Compute rewards using the validator"""
        
        rewards = []
        
        for prompt, completion, task_type in zip(prompts, completions, task_types):
            if task_type == "P":
                # Extract FEN from prompt
                fen = self._extract_fen_from_prompt(prompt)
                validation_result = validator.validate_policy_completion(fen, completion)
                reward = sum(validation_result.values())
                
            elif task_type == "A": 
                # Extract FEN and move from prompt
                fen, move_uci = self._extract_fen_move_from_prompt(prompt)
                validation_result = validator.validate_environment_completion(fen, move_uci, completion)
                reward = sum(validation_result.values())
                
            else:
                # Unknown task type
                reward = 0.0
            
            rewards.append(reward)
            logger.debug(f"Task {task_type} reward: {reward:.3f}")
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _extract_fen_from_prompt(self, prompt: str) -> str:
        """Extract FEN from P: task prompt"""
        # Simple extraction - assumes FEN follows P:
        if "P:" in prompt:
            parts = prompt.split("P:")
            if len(parts) > 1:
                fen_part = parts[1].strip().split()[0] if parts[1].strip() else ""
                return fen_part
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position fallback
    
    def _extract_fen_move_from_prompt(self, prompt: str) -> Tuple[str, str]:
        """Extract FEN and move from A: task prompt"""
        # Simple extraction - assumes format A: FEN+move+
        if "A:" in prompt:
            parts = prompt.split("A:")
            if len(parts) > 1:
                remainder = parts[1].strip()
                if "+" in remainder:
                    fen_move = remainder.split("+")[0]
                    # Split FEN and move (last part is move)
                    fen_parts = fen_move.split()
                    if len(fen_parts) >= 6:
                        move_uci = fen_parts[-1] if len(fen_parts) > 6 else "e2e4"
                        fen = " ".join(fen_parts[:-1]) if len(fen_parts) > 6 else fen_move
                        return fen, move_uci
        
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"  # Fallbacks