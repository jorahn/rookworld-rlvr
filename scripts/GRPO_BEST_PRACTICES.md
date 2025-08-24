[DOCUMENT_1]

1) Algorithmic correctness checklist (GRPO vs. TRL & Tülu3)

A. Grouping semantics and objective
	•	Exactly K completions per prompt (group) are sampled; advantages are computed per group as A_i = r_i - \bar r with \bar r the group mean (not batch mean). If you accidentally center over the whole batch, you’ll bias updates when prompts have different reward scales. Cross-check your collation + sampler produce group_size contiguous items and your loss computes groupwise means. TRL’s GRPOTrainer does this, and Tülu3 reproduces it.  ￼ ￼
	•	No value function in vanilla GRPO (unlike PPO). The policy loss is advantage-weighted log-prob ratio plus KL regularization to the reference model. Ensure you’re not initializing/stepping a critic head.  ￼

B. Reference policy & KL term
	•	Verify you compute KL wrt a frozen reference model. TRL exposes KL estimators kl1|kl2|kl3; make sure your code sets one explicitly and logs it. Tülu3’s public recipes emphasize KL control during post-training; their GRPO page shows the same trends (IFEval/GSM8K up) when KL is handled correctly. If you roll your own KL, it should be token-wise with the same masking as the loss.  ￼ ￼

C. Importance sampling
	•	If you enable importance sampling, your per-token weight from the ratio \pi_\theta / \pi_{\text{ref}} must be detached appropriately to avoid high-variance second-order effects; TRL’s importance_sampling_level handles this—confirm your code paths mirror the TRL semantics if you implemented your own.  ￼

D. Reward normalization & clipping
	•	Best practice is per-group centering (always) and optional per-group scaling/clipping to control gradient spikes; Tülu3 + RLVR use verifiable rewards that can be multi-constraint and spiky—safest is z-score or robust scaling inside each group before the policy loss. Also log reward histograms by constraint type if you train on IFBench/RLVR-style data.  ￼

E. Generation pipeline invariants
	•	Generation config used for rollouts (stop sequences, max_new_tokens, temperature, top_p, repetition_penalty) must be identical across samples within a group. If you integrate vLLM via TRL, confirm stop strings & chat templates are passed consistently; TRL’s vLLM integration outlines the split of training vs generation devices and arguments.  ￼

F. Reward functions
	•	For instruction-following, prefer verifiable programmatic rewards (IFEval / IFBench constraints). Ensure reward fns are pure w.r.t. inputs (no randomness), vectorized, and side-effect-free; compute rewards on decoded text after stop-sequence truncation. Tülu3’s public docs emphasize RLVR (RL with verifiable rewards) for constraint following; IFBench/IFEval provide ready verifiers.  ￼ ￼ ￼

Sanity probe: If you set reward to “length of completion”, GRPO should strictly increase length at fixed max_new_tokens until it saturates—your curves should replicate TRL’s toy examples. If it doesn’t, your grouping, masking, or advantage centering is off.  ￼

⸻

2) PyTorch & kernel-level efficiency on RTX 4090 (Ada)

Precision & math
	•	bf16 training is stable on 4090 (Ada has BF16 Tensor Cores). Prefer bf16 parameters/grad; reserve fp16 only if you must.  ￼
	•	Enable TF32 for fp32 matmuls to accelerate lingering fp32 codepaths:

torch.set_float32_matmul_precision("high")
# equivalently: torch.backends.cuda.matmul.allow_tf32 = True

This is off by default in modern PyTorch; turning it on is a free win on Ampere/Ada.  ￼

Attention
	•	Use PyTorch SDPA (Flash/Efficient kernels) by default; don’t force xFormers unless you’re on older PyTorch or exotic shapes. SDPA auto-selects the best kernel; you can assert via:

from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    ...

Expect lower memory and faster attention vs. eager math.  ￼ ￼

Optimizers
	•	Prefer AdamW with foreach or fused; on CUDA the default will pick foreach when safe. Test both; fused is theoretically fastest, but on some shapes foreach wins. Don’t set both flags simultaneously unless your version explicitly supports it. Example:

torch.optim.AdamW(model.parameters(), lr=..., betas=(0.9,0.95),
                  eps=1e-8, weight_decay=0.1, foreach=True)

￼

Compiler
	•	torch.compile can help throughput in trainer steps (not generation). Try fullgraph=False, mode="max-autotune", then capture a short warm-up. Verify kernels are graph-safe (no data-dependent control flow in Python). PyTorch’s tuning guide discusses expected fusions.  ￼

Allocator hygiene
	•	Avoid fragmentation on 24 GB cards with:

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

(Measure 64–256 MB.) This is a recommended knob in PyTorch CUDA notes.  ￼

Gradient checkpointing
	•	If you use it (you probably will on 24 GB), set use_reentrant=False explicitly; that’s the recommended variant with DDP and avoids the noisy warning. TRL/Transformers allow gradient_checkpointing_kwargs={"use_reentrant": False}.  ￼ ￼

Liger-Kernel
	•	For extra headroom, apply Liger-Kernel (Triton fused ops: RMSNorm, RoPE, SwiGLU, CE). Expect ~20% throughput and ~60% memory reduction on LLaMA-like blocks (varies by model). TRL documents integration.  ￼ ￼

Multi-GPU & vLLM
	•	With consumer Ada (4090), P2P over NVLink is absent, and NCCL P2P over PCIe can be flaky; for 2+ GPUs consider setting NCCL_P2P_DISABLE=1 if you see hangs, at the cost of bandwidth. For GRPO + vLLM, prefer dedicating one GPU to vLLM and the other to training; colocating with PEFT may hang (known TRL issues).  ￼ ￼ ￼

⸻

3) vLLM generation path (if you use it)
	•	TRL’s vLLM integration exposes a server that handles generation while training runs elsewhere. Pass --tensor-parallel-size/--data-parallel-size to scale. If single-GPU only, be aware of TRL issues sharing the device and PEFT/LoRA edge cases.  ￼ ￼

⸻

4) Data & rewards (Tülu3 + RLVR style)
	•	For instruction-following improvements, RLVR datasets (verifiable constraints) + IFEval/IFBench coverage are the current open best practice. Make sure your verifier suite spans formatting, keyword counts, structural constraints, and length/case requirements; unit-test each verifier.  ￼ ￼

⸻

5) What to change in your code (drop-in patches)

Paths below assume a typical layout. Apply analogous edits in your repo (excluding scripts/).

5.1 Trainer init: policy/ref models, precision, TF32, SDPA, checkpointing

*** a/src/training/grpo_trainer.py
--- b/src/training/grpo_trainer.py
@@
 class GRPOTrainer(...):
     def __init__(self, model, tokenizer, ref_model=None, cfg: GRPOConfig = ..., **kwargs):
         super().__init__(...)
+        # Precision & math on Ada/4090
+        import torch
+        torch.set_float32_matmul_precision("high")  # enable TF32 matmul on Ampere/Ada
+        self._amp_dtype = torch.bfloat16  # prefer bf16 on RTX 4090
+
+        # Gradient checkpointing (recommended variant)
+        if getattr(cfg, "use_gradient_checkpointing", True):
+            try:
+                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
+            except TypeError:
+                model.gradient_checkpointing_enable()
+                self.logger.warning("Set use_reentrant=False where supported.")
+
+        # Ensure PyTorch SDPA is used (Flash/Efficient attention)
+        from torch.nn.attention import sdpa_kernel, SDPBackend
+        self._sdpa_ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
+
         # Reference model must be frozen
         self.ref_model = ref_model or self._load_reference_model(model, tokenizer)
         for p in self.ref_model.parameters():
             p.requires_grad_(False)
@@
     def _load_reference_model(...):
         ...

(Why: TF32 improves matmul throughput; bf16 is stable on 4090; SDPA lowers memory/latency; checkpointing with use_reentrant=False plays nicer with DDP.)  ￼

5.2 Loss: strict group semantics, KL estimator, masking

*** a/src/training/grpo_loss.py
--- b/src/training/grpo_loss.py
@@
 def grpo_loss(logprobs, ref_logprobs, rewards, attn_mask, group_indices, cfg):
-    # TODO: implement
+    """
+    logprobs:  (B, T) token logp of sampled completions
+    ref_logprobs: (B, T) token logp under frozen reference
+    rewards:  (B,) scalar reward per completion (post stop-truncation)
+    attn_mask: (B, T) 1 for valid tokens
+    group_indices: (B,) int group id per sample (same id for K samples of each prompt)
+    """
+    import torch, torch.nn.functional as F
+    device = logprobs.device
+
+    # Compute per-sample return; per-group baseline
+    B = rewards.shape[0]
+    # group_mean: mean reward within each group id
+    group_ids = group_indices
+    group_mean = torch.scatter_reduce(
+        torch.zeros_like(rewards).index_copy(0, torch.arange(B, device=device), rewards),
+        0, group_ids, reduce="mean", include_self=False
+    )  # works if group_ids are compact [0..G-1]; otherwise do segment_mean
+    advantages = rewards - group_mean
+
+    # Token-level KL between policy and ref on valid tokens
+    kl_t = (logprobs - ref_logprobs) * attn_mask
+    if   cfg.kl_estimator == "kl1": kl = kl_t.sum() / attn_mask.sum().clamp_min(1)
+    elif cfg.kl_estimator == "kl2": kl = (kl_t.exp() - 1 - kl_t) .mul(attn_mask).sum() / attn_mask.sum().clamp_min(1)
+    else:                           kl = (kl_t**2 * 0.5).sum() / attn_mask.sum().clamp_min(1)  # "kl3"
+
+    # Sequence log-prob (masked sum) → scalar per sample
+    seq_logp = (logprobs * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp_min(1)
+    # Advantage-weighted objective
+    loss_policy = -(advantages.detach() * seq_logp).mean()
+    loss = loss_policy + cfg.kl_coef * kl
+    return loss, {"loss_policy": loss_policy.detach(), "kl": kl.detach()}

	•	Check your real code uses segment means for groups if IDs aren’t compact. Ensure attn mask trims only generated tokens (exclude prompt). Map this onto TRL’s shapes if you use their trainer.  ￼

5.3 Optimizer & scheduler

*** a/src/optim/optim.py
--- b/src/optim/optim.py
@@
 def build_optimizer(model, cfg):
-    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
+    return torch.optim.AdamW(
+        model.parameters(),
+        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.wd,
+        foreach=True  # try fused=True on your build; benchmark both
+    )

(Prefer foreach/fused; 0.95 beta2 is common in modern LLM recipes.)  ￼

5.4 Liger-Kernel (optional but recommended)

*** a/src/models/build.py
--- b/src/models/build.py
@@
 def load_llm(name, **kwargs):
     model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, **kwargs)
+    try:
+        from liger_kernel.transformers import apply_liger_kernel_to_llama
+        apply_liger_kernel_to_llama(rope=True, swiglu=True, cross_entropy=True, rms_norm=True)
+    except Exception as e:
+        logger.info(f"Liger-Kernel not applied: {e}")
     return model

(Expect notable memory savings and throughput gains on LLaMA-like blocks.)  ￼

5.5 vLLM client (if using TRL’s integration)

If you call TRL’s vLLM server, ensure your client passes identical gen config per group and logs prompt→completions with the same stop handling used by reward functions. Align with TRL’s vllm-serve guide.  ￼

⸻

6) What to grep for in your repo (fast audit)
	•	Grouping: group_size, num_generations, any place that shuffles batch—flag cross-group reduction.
	•	KL estimator: look for kl_estimator, kl_coef, and that the reference model params are requires_grad=False.
	•	Masking: ensure prompt tokens are excluded from loss/kl; check attention_mask and any labels preprocessing.
	•	Precision: any autocast set to fp16 only → move to dtype=bf16.
	•	Checkpointing: any calls to torch.utils.checkpoint.checkpoint w/o use_reentrant → set to False.  ￼
	•	xFormers hard-requirement: if present, consider removing; prefer SDPA.  ￼
	•	Allocator / env: check your launchers (outside scripts/ if applicable) for PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128.  ￼

⸻

7) Tests that catch the sneaky bugs
	1.	Group invariants

# test_grpo_groups.py
# Construct 2 prompts × group_size=4 with rewards [1, 1, 1, 9] and [1, 1, 1, 9]
# Assert advantages sum to zero **per group** and loss decreases when 9→10 for any single sample

	2.	Masking

# test_masks.py
# Prompt length varies; ensure loss/kl identical when adding prompt-only padding

	3.	KL estimator parity

# test_kl_estimators.py
# On small vocab and fixed logits, check kl1/kl2/kl3 values against closed-form expectations

	4.	Toy reward monotonicity

# test_len_reward.py
# Reward = min(len, 20); check expected monotonic increase until cap

(TRL’s GRPO tutorial demonstrates the length-reward toy; reproducing it is a quick signal you’re aligned.)  ￼
	5.	Liger correctness (if applied)

	•	Compare logits/grad norms ±1e-3 vs. vanilla ops on a fixed seed minibatch.

	6.	vLLM E2E

	•	If using vLLM, run one batch with use_vllm=False vs. server path; verify identical decoded text with the same seed/config.

⸻

8) Micro-benchmarks (RTX 4090)

Run each for 200–500 trainer steps after 50 warm-up steps; report tokens/s and VRAM.

A. SDPA kernels
	•	Baseline (no context manager)
	•	Force FLASH_ATTENTION vs. EFFICIENT_ATTENTION (seq-len × head-dim grid). Expect SDPA > eager math; Flash wins when head-dim ≤ 128–160 at common lengths.  ￼ ￼

B. Precision toggles
	•	bf16 + TF32 on vs fp16 + GradScaler vs fp32 + TF32. Expect bf16 stable and fast on 4090; TF32 helps residual fp32 paths.  ￼ ￼

C. Optimizer
	•	AdamW foreach=True vs fused=True. Keep the faster. (In recent PyTorch, foreach is often the safe bet; measure.)  ￼

D. Liger-Kernel
	•	On LLaMA-family models: measure peak VRAM and step time with/without Liger kernels. Expect noticeable savings.  ￼

E. torch.compile
	•	On the trainer step (not generation). Try mode="max-autotune" and record speedup.  ￼

⸻

9) Config exemplar (fits 24 GB, single 4090)

# configs/grpo_4090.yaml
model_name: meta-llama/Llama-3.1-8B-Instruct
dtype: bf16
group_size: 4
max_prompt_tokens: 512
max_new_tokens: 256
temperature: 0.8
top_p: 0.95
reward:
  suite: [ifeval_basic, ifbench_easy]   # verifiable
  normalize: group_center
  clip: null
optimization:
  lr: 1.5e-5
  wd: 0.1
  betas: [0.9, 0.95]
  optimizer: adamw
  foreach: true
  grad_accum_steps: 4
  max_steps: 20_000
stability:
  gradient_checkpointing: true
  use_reentrant: false
  grad_clip_norm: 1.0
regularization:
  kl_estimator: kl3
  kl_coef: 0.02
  entropy_bonus: 0.0
system:
  torch_compile: true
  tf32: true
  sdpa: flash_or_efficient
  env:
    PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:128"
logging:
  log_rewards_by_constraint: true
  eval: [ifeval, ifbench]

(IFEval/IFBench are contemporary instruction-following verifiers; RLVR-style training has been effective in open recipes like Tülu3.)  ￼ ￼

⸻

10) Known foot-guns to avoid
	•	Cross-group mixing during loss computation due to sampler reshuffles. Keep groups contiguous or carry explicit group_id tensor.
	•	Masking prompt tokens inconsistently between policy and reference log-probs → wrong KL.
	•	Colocated vLLM + PEFT on one GPU: hangs/timeouts have been reported; if you must colocate, pin both processes to the same device and reduce GPU memory utilization; otherwise dedicate the GPU.  ￼
	•	Hard-requiring xFormers when SDPA suffices and is better integrated today.  ￼
	•	TF32 left at default (“highest”) → you leave perf on the table; set "high".  ￼

⸻

Quick “apply & verify” sequence
	1.	Apply the diffs above in your trainer/model/optim code.
	2.	Run the toy length-reward test → confirm monotonicity plateaus at cap.  ￼
	3.	Run SDPA micro-bench at your target seq_len/head_dim.
	4.	If using vLLM, stand up TRL’s server, run 100 steps with use_vllm=True, then False; check decoded outputs parity and throughput.  ￼
	5.	Start a 2–4 h pilot on IFEval subset; monitor reward histograms & KL. If KL runs away, increase kl_coef by ×1.5–2; if reward saturates with degenerate outputs, add constraint-specific reward shaping.  ￼

⸻

References & further reading
	•	TRL GRPO trainer & docs; vLLM integration and examples.  ￼ ￼
	•	AllenAI Open-Instruct / Tülu3: public recipes & GRPO page; RLVR emphasis and instruction-following gains.  ￼ ￼
	•	IFEval / IFBench: verifiable instruction following (datasets + papers).  ￼ ￼
	•	PyTorch TF32, SDPA, AdamW foreach/fused, checkpointing best practices.  ￼
	•	Liger-Kernel Triton fused ops for LLM training.  ￼
	•	4090 (Ada) BF16/FP8/TF32 tensor cores (whitepaper).  ￼

⸻

[DOCUMENT_2]

Short version: in GRPO (à la DeepSeek-style group-relative PPO), you usually don’t “update a KL model.” You freeze the reference policy and update the current policy every optimizer step; the KL term is computed per token against the frozen reference. You refresh rollouts and their cached reference log-probs only after you’ve finished a small number of epochs over the current batch.

Here’s a concrete, battle-tested cadence that tends to work:

What gets updated and when
	•	Reference policy: frozen for the whole run. Don’t update it.
	•	Policy (the one you’re training): updated every optimizer step.
	•	KL penalty coefficient (β): adapt every optimizer step or every epoch to hit a target per-token KL.
	•	Rollouts / cached ref log-probs: refresh after finishing 1–4 epochs over the current rollout batch (i.e., when you go collect new samples).

Typical numbers (good starting points)
	•	Prompts per rollout batch: 256–1024
	•	Candidates per prompt (GRPO): 4–8
	•	Max generated tokens per candidate: 64–256 (cap hard; truncate long tails)
	•	Effective tokens per rollout batch: ~50k–200k (prompts × candidates × gen_len)
	•	Epochs over the rollout batch: 1–2 (rarely >4 to avoid off-policy drift)
	•	Minibatch size (tokens): 2k–8k tokens/step (scale with hardware)
	•	KL target (mean per token): 0.03–0.10 nats/token (tune by task; code tends to like 0.05)
	•	β adaptation: adjust each step using a simple proportional update, e.g.
β ← β × exp(clip((KL_obs − KL_target)/KL_target, −0.2, 0.2))
or a linear update β ← β + k·(KL_obs − KL_target) with small k.

Practical loop (one “update block”)
	1.	Roll out with current policy. For each prompt, generate m candidates, get rewards, and compute group-relative advantages (e.g., reward − group mean/percentile).
	2.	Cache reference log-probs for exactly the tokens you generated.
	3.	Optimize for 1–2 epochs over minibatches: each optimizer step updates the policy; compute per-token KL vs the cached reference log-probs and include it in the loss; adapt β to keep mean KL near target.
	4.	Discard rollouts and go collect fresh ones with the now-updated policy.

Notes & gotchas
	•	Don’t recompute ref logits during epochs. Use the cached ones from rollout time; recompute only when you collect new rollouts.
	•	Early KL guardrails. If observed KL > ~2× target, either bump β immediately or shorten generation length on the next rollout.
	•	Sequence vs token KL. Use token-level KL and average over generated tokens; it’s more stable than a sequence-level single number.
	•	On-policy drift. If you crank epochs >2–3, you’ll start to see drift; better to refresh rollouts more often than to over-epoch.
	•	Entropy bonus. Optional; with a KL-to-reference you usually don’t need a separate entropy term, but a tiny one can help exploration on very peaky rewards.
	•	Logging. Track both mean and 95th-percentile KL/token; the tail often predicts collapse before the mean does.

