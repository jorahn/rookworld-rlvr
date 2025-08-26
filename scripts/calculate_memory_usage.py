#!/usr/bin/env python3
"""
Calculate expected memory usage with the constraint of 144+ tokens
"""

def calculate_memory():
    print("=== Memory Calculation for RookWorld GRPO ===")
    print()
    
    # Constants from current configuration
    batch_size = 8  # batch_positions=2 × group_size=4
    vocab_size = 50257  # GPT-2 vocabulary
    max_new_tokens = 144  # User constraint: minimum 144 tokens
    typical_prompt = 30  # Estimated chess prompt length
    max_seq_len = typical_prompt + max_new_tokens  # ~174 tokens
    
    print(f"Batch size: {batch_size}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Max new tokens: {max_new_tokens} (user constraint)")
    print(f"Typical prompt: {typical_prompt} tokens")
    print(f"Max sequence length: {max_seq_len} tokens")
    print()
    
    # Memory calculations
    print("=== Memory Usage Analysis ===")
    
    # Model parameters (124M model)
    model_params = 124_439_808
    model_memory_fp32 = model_params * 4 / 1024**3  # FP32
    model_memory_bf16 = model_params * 2 / 1024**3  # BF16
    print(f"Model memory (FP32): {model_memory_fp32:.2f}GB")
    print(f"Model memory (BF16): {model_memory_bf16:.2f}GB")
    
    # Two models (training + reference)
    dual_model_memory = model_memory_bf16 * 2
    print(f"Dual models (training + ref): {dual_model_memory:.2f}GB")
    print()
    
    # Logits tensor - this is the critical one
    logits_elements = batch_size * max_seq_len * vocab_size
    logits_memory_fp32 = logits_elements * 4 / 1024**3
    logits_memory_bf16 = logits_elements * 2 / 1024**3
    
    print(f"Logits tensor elements: {logits_elements:,}")
    print(f"Logits memory (FP32): {logits_memory_fp32:.2f}GB ⚠️")
    print(f"Logits memory (BF16): {logits_memory_bf16:.2f}GB ⚠️")
    print()
    
    # Hidden states
    hidden_dim = 768  # GPT-2 124M
    hidden_elements = batch_size * max_seq_len * hidden_dim
    hidden_memory = hidden_elements * 2 / 1024**3  # BF16
    print(f"Hidden states memory: {hidden_memory:.3f}GB")
    
    # Key-Value cache (if using past_key_values)
    n_layers = 12
    n_heads = 12
    head_dim = 64
    kv_elements = 2 * batch_size * n_layers * n_heads * max_seq_len * head_dim  # 2 for K,V
    kv_memory = kv_elements * 2 / 1024**3  # BF16
    print(f"KV cache memory: {kv_memory:.3f}GB")
    print()
    
    # Total estimated memory
    total_memory = dual_model_memory + logits_memory_bf16 + hidden_memory + kv_memory
    print(f"=== TOTAL ESTIMATED MEMORY ===")
    print(f"Total: {total_memory:.2f}GB")
    print()
    
    # GPU capacity check
    gpu_memory = 24  # RTX 4090 has ~24GB
    if total_memory > gpu_memory * 0.9:  # 90% threshold
        print("❌ PROBLEM: Estimated memory exceeds GPU capacity!")
        print(f"Need: {total_memory:.2f}GB, Have: {gpu_memory}GB")
        print()
        print("🔍 ROOT CAUSE ANALYSIS:")
        print(f"  Logits tensor is consuming {logits_memory_bf16:.2f}GB ({logits_memory_bf16/total_memory*100:.1f}%)")
        print(f"  This is {batch_size} × {max_seq_len} × {vocab_size} = {logits_elements:,} elements")
        print()
        print("💡 SOLUTIONS:")
        print("1. Reduce batch size (but you said min 8)")
        print("2. Reduce sequence length (but you need 144+ tokens)")
        print("3. Use vocabulary splitting/chunking")
        print("4. Use gradient checkpointing to trade compute for memory")
        print("5. Use CPU optimizer offloading")
    else:
        print("✅ Memory usage should be acceptable")
    
    print()
    print("=== COMPARISON WITH ACTUAL USAGE ===")
    actual_usage = 22.62  # From the OOM error
    print(f"Actual usage observed: {actual_usage}GB")
    print(f"Calculated estimate: {total_memory:.2f}GB")
    print(f"Difference: {actual_usage - total_memory:.2f}GB")
    
    if actual_usage > total_memory * 1.2:
        print("⚠️  Actual usage is significantly higher than estimate!")
        print("This suggests additional memory leaks or hidden allocations.")

if __name__ == "__main__":
    calculate_memory()