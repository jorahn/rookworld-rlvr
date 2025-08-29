"""
Inspect generations during training to verify quality
"""

import torch
import tiktoken
import numpy as np
from pathlib import Path

from config import GRPOConfig
from loader import load_rookworld_model
from dataset import load_and_prepare_samples
from reward_scorer import compute_grpo_rewards


def inspect_generations():
    """Generate and inspect completions with the model."""
    
    print("=" * 80)
    print("INSPECTING MODEL GENERATIONS")
    print("=" * 80)
    
    # Setup
    config = GRPOConfig()
    config.n_train_samples = 10  # Small sample for inspection
    config.k_samples = 4
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load model
    print(f"\nLoading model from {config.model_path}")
    model = load_rookworld_model(config.model_path, device=device)
    model.eval()
    
    # Load samples
    print(f"Loading {config.n_train_samples} samples...")
    samples = load_and_prepare_samples(
        n_samples=config.n_train_samples,
        seed=config.data_seed
    )
    
    # Separate by task type
    p_samples = [s for s in samples if s[0] == 'P'][:2]  # 2 P tasks
    a_samples = [s for s in samples if s[0] == 'A'][:2]  # 2 A tasks
    test_samples = p_samples + a_samples
    
    print(f"\nTesting on {len(p_samples)} P: tasks and {len(a_samples)} A: tasks")
    print("=" * 80)
    
    for idx, (task_type, prompt, ground_truth, _) in enumerate(test_samples):
        print(f"\n[Sample {idx+1}] {task_type} Task")
        print("-" * 80)
        print(f"PROMPT:\n{prompt}\n")
        print(f"GROUND TRUTH:\n{ground_truth}\n")
        
        # Generate K completions
        all_completions = []
        all_rewards = []
        
        print(f"Generating {config.k_samples} completions...")
        
        for k in range(config.k_samples):
            # Tokenize and generate
            prompt_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([prompt_ids], device=device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    pad_token_id=50256
                )
            
            # Decode completion
            generated_ids = generated[0].cpu().tolist()
            completion = tokenizer.decode(generated_ids[len(prompt_ids):])
            
            # Clean up
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            all_completions.append(completion)
        
        # Score all completions
        prompts_batch = [prompt] * config.k_samples
        _, reward_details = compute_grpo_rewards(
            prompts_batch,
            all_completions,
            group_size=config.k_samples,
            reward_shaping=config.reward_shaping,
            verbose=False
        )
        
        # Display results
        print("\nGENERATED COMPLETIONS:")
        for k, (comp, detail) in enumerate(zip(all_completions, reward_details)):
            print(f"\n  Completion {k+1}:")
            print(f"  Reward: {detail.shaped_reward:.3f} (Format Valid: {detail.format_valid})")
            
            # Show first 200 chars or full if shorter
            if len(comp) > 200:
                print(f"  Text: {comp[:200]}...")
            else:
                print(f"  Text: {comp}")
        
        # Statistics
        rewards = [d.shaped_reward for d in reward_details]
        print(f"\n  Statistics:")
        print(f"    Mean Reward: {np.mean(rewards):.3f}")
        print(f"    Std Reward: {np.std(rewards):.3f}")
        print(f"    Min/Max: [{min(rewards):.3f}, {max(rewards):.3f}]")
        print(f"    Format Valid: {sum(d.format_valid for d in reward_details)}/{len(reward_details)}")
    
    # Check if there's a trained checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / "final_model.pt"
    if checkpoint_path.exists():
        print("\n" + "=" * 80)
        print("LOADING TRAINED MODEL")
        print("=" * 80)
        
        # Load trained weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        print("\nGenerating with TRAINED model on same samples...")
        
        for idx, (task_type, prompt, ground_truth, _) in enumerate(test_samples[:2]):  # Just 2 samples
            print(f"\n[Trained Sample {idx+1}] {task_type} Task")
            print("-" * 40)
            
            # Generate single completion
            prompt_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([prompt_ids], device=device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=config.max_new_tokens,
                    temperature=0.7,  # Lower temp for trained model
                    top_k=config.top_k,
                    top_p=config.top_p,
                    pad_token_id=50256
                )
            
            # Decode
            generated_ids = generated[0].cpu().tolist()
            completion = tokenizer.decode(generated_ids[len(prompt_ids):])
            
            if '<|endoftext|>' in completion:
                completion = completion.replace('<|endoftext|>', '').strip()
            
            # Score
            _, details = compute_grpo_rewards(
                [prompt],
                [completion],
                group_size=1,
                reward_shaping=config.reward_shaping,
                verbose=False
            )
            
            print(f"Reward: {details[0].shaped_reward:.3f} (Format Valid: {details[0].format_valid})")
            if len(completion) > 200:
                print(f"Text: {completion[:200]}...")
            else:
                print(f"Text: {completion}")
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    inspect_generations()