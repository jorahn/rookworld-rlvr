# Production Optimization Plan for Mini Implementation

## Executive Summary
Optimize the mini GRPO implementation for production with RTX 4090 GPUs (compute capability 8.9), implementing mixed precision training, parallel computation, and multi-GPU support.

## 1. Mixed Precision Training (AMP with BF16)

### Implementation
- **Add automatic mixed precision with BFloat16** (better for RTX 4090 than FP16)
  - Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` 
  - Implement GradScaler for stable gradient scaling
  - BF16 maintains FP32 range, avoiding overflow issues

- **Enable TF32 for matmul operations**
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```
  - Provides ~2-3x speedup on Ampere/Ada GPUs

### Benefits
- 50-70% memory reduction
- 1.5-2x training speedup
- Better numerical stability than FP16

## 2. Parallel Reward Computation

### Implementation
- **Parallelize reward scoring across samples**
  - Use `torch.multiprocessing` or `concurrent.futures` 
  - Process K samples per prompt in parallel
  - Expected 2-4x speedup for reward computation

- **Batch reward validation**
  - Vectorize FEN validation operations
  - Batch Levenshtein distance calculations

### Example Structure
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_score_rewards(samples, scorer, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for prompt, completion, ground_truth in samples:
            future = executor.submit(scorer.score_single, 
                                    prompt, completion, ground_truth)
            futures.append(future)
        
        rewards = [f.result()[0] for f in futures]
    return rewards
```

## 3. Multi-GPU Support (Data Parallel)

### DistributedDataParallel (DDP)
- **Implementation**
  - Split batch across 2 RTX 4090s
  - Synchronize gradients with NCCL backend
  - Scale effective batch size (2x throughput)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def create_ddp_model(model, rank):
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model
```

### Alternative: Model Parallel
- Split model layers across GPUs if needed
- Useful for future scaling beyond 124M params

## 4. Memory Optimizations

### Gradient Checkpointing
```python
def enable_gradient_checkpointing(model):
    for block in model.h:
        block.use_checkpoint = True
```
- Trade compute for memory in transformer blocks
- Enable for longer sequences or larger batches

### CPU Offloading
- Keep reference model on CPU, move to GPU only when needed
- Saves ~500MB VRAM

### Optimize Tensor Allocations
- Pre-allocate buffers for recurring operations
- Use in-place operations where possible

## 5. Computation Optimizations

### Torch.compile
```python
model = torch.compile(model, mode='reduce-overhead')
```
- 10-30% speedup on RTX 4090
- Best for Ada Lovelace architecture

### Flash Attention (Optional)
```python
# Requires: pip install flash-attn
from flash_attn import flash_attn_func
```
- 2-3x attention speedup
- Memory efficient attention computation

### Fused Optimizers
```python
optimizer = torch.optim._multi_tensor.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    foreach=True  # Fused operations
)
```

## 6. Implementation Structure

### Config Additions
```python
@dataclass
class GRPOConfig:
    # Precision settings
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # or "float16"
    use_tf32: bool = True
    
    # Parallel settings  
    parallel_rewards: bool = True
    num_reward_workers: int = 4
    
    # Multi-GPU
    use_ddp: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Memory optimization
    gradient_checkpointing: bool = False
    offload_reference: bool = True
    
    # Compilation
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
```

### Training Loop Modifications
```python
def train_with_optimizations():
    # Enable TF32
    if config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup AMP
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    # Compile model
    if config.compile_model:
        model = torch.compile(model, mode=config.compile_mode)
    
    # Training step with AMP
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=config.use_amp):
        loss, metrics = grpo_loss(...)
    
    # Scaled backward
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
```

## 7. Performance Targets

| Configuration | Time per Step | Speedup |
|--------------|---------------|---------|
| **Current** (baseline) | ~18-20s | 1.0x |
| + AMP/BF16 | ~10-12s | 1.5-2x |
| + Parallel rewards | ~8-10s | 2-2.5x |
| + Torch.compile | ~7-9s | 2.2-2.8x |
| + 2x GPU (DDP) | ~4-5s | 4-5x |

### Memory Usage
- Current: ~4.8GB VRAM
- With BF16: ~2.5-3GB VRAM
- With optimizations: <3GB per GPU

## 8. Testing & Validation Plan

### Phase 1: AMP Implementation
1. Implement BF16 autocast
2. Validate numerical stability
3. Compare loss curves with FP32
4. Benchmark speedup

### Phase 2: Parallel Rewards
1. Implement ThreadPoolExecutor for scoring
2. Verify reward values match serial computation
3. Measure speedup for different K values

### Phase 3: Multi-GPU
1. Implement DDP wrapper
2. Test gradient synchronization
3. Validate convergence matches single GPU
4. Benchmark linear scaling

### Phase 4: Full Integration
1. Enable all optimizations
2. Profile with `torch.profiler`
3. Run extended training (1000+ steps)
4. Compare final metrics with baseline

## 9. Files to Modify

| File | Changes |
|------|---------|
| `train_logged.py` | Add AMP context, DDP setup, parallel rewards |
| `grpo.py` | Optimize loss computation, add AMP decorators |
| `reward_scorer.py` | Parallelize scoring, batch operations |
| `config.py` | Add production optimization flags |
| `model.py` | Optional gradient checkpointing |
| `distributed_train.py` | **New** - Multi-GPU launcher script |
| `benchmark.py` | **New** - Performance profiling script |

## 10. Risk Mitigation

### Stability Safeguards
- Keep original implementation as fallback
- Add flags to disable each optimization individually
- Implement gradient clipping safeguards
- Monitor for NaN/Inf in losses

### Monitoring
```python
# Add to training loop
if torch.isnan(loss) or torch.isinf(loss):
    logger.warning(f"Unstable loss detected: {loss.item()}")
    # Fallback to FP32 or reduce learning rate
```

### Checkpointing
- Save checkpoints every 100 steps (vs 1000)
- Include optimizer state for AMP scaler
- Version checkpoints with optimization config

## 11. Implementation Priority

1. **High Priority (Week 1)**
   - BF16 mixed precision
   - TF32 enablement
   - Parallel reward computation

2. **Medium Priority (Week 2)**
   - Torch.compile integration
   - Memory optimizations
   - Enhanced profiling

3. **Lower Priority (Week 3+)**
   - Multi-GPU support
   - Flash Attention
   - Advanced optimizations

## 12. Expected Outcomes

### Performance Gains
- **Training speed**: 4-5x faster
- **Memory efficiency**: 50% reduction
- **Batch size capacity**: 2-3x larger
- **Energy efficiency**: ~60% power reduction with BF16

### Quality Maintenance
- Loss convergence: ≤5% deviation from FP32
- Final accuracy: Maintained or improved
- Training stability: No divergence issues

## Appendix: Quick Start Commands

```bash
# Test BF16 training
USE_AMP=true AMP_DTYPE=bfloat16 ./train.sh

# Enable all CPU optimizations
USE_TF32=true COMPILE_MODEL=true ./train.sh

# Run with parallel rewards
PARALLEL_REWARDS=true NUM_WORKERS=4 ./train.sh

# Multi-GPU training (requires new launcher)
torchrun --nproc_per_node=2 distributed_train.py

# Benchmark performance
python benchmark.py --config production.yaml
```

## References

- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [RTX 4090 Optimization Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [BFloat16 vs Float16 Comparison](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)