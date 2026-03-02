# FlashInfer-Bench Starter Kit Integration Guide
## Plugging MoE V2 Kernel into the Contest Framework

---

## Quick Start (TL;DR)

```bash
# 1. Fork starter kit
git clone https://github.com/flashinfer-ai/flashinfer-bench-starter-kit my-moe-solution
cd my-moe-solution

# 2. Install deps
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal

# 3. Clone the contest dataset
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=$(pwd)/mlsys26-contest

# 4. Copy our files into the starter kit
#    - config.toml        → repo root (replace existing)
#    - kernel.py           → solution/triton/kernel.py (replace existing)

# 5. Inspect the definition to verify parameter names
python scripts/inspect_definition.py

# 6. Pack & test locally
python scripts/pack_solution.py
python scripts/run_local.py

# 7. Test on B200 via Modal (optional)
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace ./mlsys26-contest
modal run scripts/run_modal.py
```

---

## File-by-File Breakdown

### 1. `config.toml` — Track Configuration

```toml
[solution]
name = "moe-v2-zerocopy-noatomic"    # Shows on leaderboard
definition = "fused_moe"             # MUST be "fused_moe" for MoE track
author = "prateek-team"              # Your team name

[build]
language = "triton"                  # We use Triton
entry_point = "kernel"               # Function name in kernel.py
```

**Key points:**
- `definition = "fused_moe"` — This tells the framework to match against ALL
  fused_moe definitions in the dataset (including `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`)
- `entry_point = "kernel"` — The framework imports `kernel.py` and calls `kernel()`
  with keyword arguments matching the definition JSON

### 2. `solution/triton/kernel.py` — Your Kernel Implementation

This is the **only file the framework executes**. It must contain a top-level
function named `kernel` (matching `entry_point` in config.toml).

**How the framework calls your kernel:**

```python
# The framework does roughly this:
from solution.triton.kernel import kernel

# DPS mode (default): output tensor pre-allocated and passed in
result = kernel(
    routing_logits=routing_logits_tensor,
    routing_bias=routing_bias_tensor,
    hidden_states=hidden_states_tensor,
    hidden_states_scale=hidden_states_scale_tensor,
    gemm1_weights=gemm1_weights_tensor,
    gemm1_weights_scale=gemm1_weights_scale_tensor,
    gemm2_weights=gemm2_weights_tensor,
    gemm2_weights_scale=gemm2_weights_scale_tensor,
    num_experts=256,
    top_k=8,
    n_group=8,
    topk_group=4,
    intermediate_size=2048,
    local_expert_offset=0,
    local_num_experts=32,
    routed_scaling_factor=1.0,
    output=output_tensor,      # pre-allocated [seq_len, hidden_size] bf16
)
```

**Your kernel function signature MUST accept all these parameters as keyword args.**
The exact parameter names come from the definition JSON in the dataset.

### 3. `scripts/inspect_definition.py` — Verify Parameter Names

**This is critical.** Before submitting, run this to confirm the exact parameter
names the framework will pass:

```bash
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/inspect_definition.py
```

This prints:
- All `fused_moe` definitions found in the dataset
- Exact input names, shapes, dtypes
- Axes (constant values baked in)
- The reference implementation code
- The expected function signature

**If the definition JSON uses different names than our kernel.py, you must rename
the parameters to match.** The framework uses `**kwargs` style calling.

---

## Kernel Signature Deep Dive

Based on the FlashInfer `trtllm_fp8_block_scale_moe` reference implementation,
the fused_moe definition expects:

### Tensor Inputs

| Parameter | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `routing_logits` | `[S, 256]` | float32 | Router output logits |
| `routing_bias` | `[256]` | bfloat16 | Bias for routing (can be None) |
| `hidden_states` | `[S, 7168]` | float8_e4m3fn | FP8-quantized input activations |
| `hidden_states_scale` | `[56, S]` | float32 | Per-block scales for hidden_states |
| `gemm1_weights` | `[32, 4096, 7168]` | float8_e4m3fn | Gate+Up projection weights |
| `gemm1_weights_scale` | `[32, 32, 56]` | float32 | Per-block scales for GEMM1 weights |
| `gemm2_weights` | `[32, 7168, 2048]` | float8_e4m3fn | Down projection weights |
| `gemm2_weights_scale` | `[32, 56, 16]` | float32 | Per-block scales for GEMM2 weights |

### Scalar Axes (may be passed as kwargs OR baked into definition)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_experts` | 256 | Total expert count |
| `top_k` | 8 | Experts per token |
| `n_group` | 8 | Number of expert groups |
| `topk_group` | 4 | Groups selected per token |
| `intermediate_size` | 2048 | Expert FFN intermediate dim |
| `local_expert_offset` | 0 | Offset for EP (expert parallelism) |
| `local_num_experts` | 32 | Experts on this device (256/8 EP) |
| `routed_scaling_factor` | 1.0 or None | Scaling factor for gating weights |

### DPS Output

| Parameter | Shape | Dtype |
|-----------|-------|-------|
| `output` | `[S, 7168]` | bfloat16 |

If `output` is passed, write results into it. If not, allocate and return.

---

## What Our V2 Kernel Does Inside kernel()

```
STAGE 1: Routing (PyTorch)
  routing_logits → sigmoid → group_topk → expert_select → sort_by_expert
  Produces: sorted_token_ids, sorted_expert_ids, sorted_weights, expert_offsets

STAGE 2: GEMM1 Zero-Copy (Triton)
  hidden_states[sorted_token_ids[m]] × gemm1_weights[expert] → gemm1_output
  Key: No 56MB gather copy — reads via indirection

STAGE 3: SwiGLU + FP8 Quant (Triton)
  silu(gate) * up → FP8 quantize with per-block scales

STAGE 4: GEMM2 No-Atomic (Triton)
  activated_fp8 × gemm2_weights[expert] → gemm2_output (contiguous buffer)
  Key: No atomic_add — writes to per-assignment buffer

STAGE 5: Weighted Reduce → BF16 (Triton)
  For each token: output[t] = Σ weight[a] × gemm2_output[a]
  Key: Direct BF16 output, no FP32 intermediate buffer
```

---

## DPS vs Value-Returning Style

The framework defaults to **Destination Passing Style (DPS)**, where it
pre-allocates the output tensor and passes it as a keyword argument.

Our kernel handles both modes:

```python
def kernel(..., output=None):
    if output is None:
        output = torch.zeros(seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    else:
        output.zero_()
    # ... compute into output ...
    return output
```

If you need to use **value-returning style** instead (not recommended—DPS
is faster because it avoids measuring allocation overhead), set
`destination_passing_style = false` in the solution spec. But our kernel
supports DPS natively.

---

## Common Errors and Fixes

### "Destination-passing style callable: expected XX parameters, but got XX"

**Cause:** Your kernel function has wrong number of parameters.

**Fix:** Run `inspect_definition.py` to see the exact parameters. The DPS
output adds one extra parameter vs the input count.

### "routing_bias is not None but expected None" (or vice versa)

**Cause:** Some workloads may have `routing_bias=None`.

**Fix:** Our kernel already handles this:
```python
if routing_bias is not None:
    scores = scores + routing_bias.float()
```

### Triton compilation errors on non-Blackwell GPUs

**Cause:** Some Triton features (like `tl.float8e4nv`) may not work on H100.

**Fix:** For local testing on H100, you may need to adjust FP8 types. The
official evaluation runs on B200 which supports all features.

### "No matching definition found"

**Cause:** `definition` in config.toml doesn't match any definition in dataset.

**Fix:** Ensure `definition = "fused_moe"` (exact string).

---

## Testing Workflow

### Step 1: Local Sanity Check (your GPU)

```bash
cd my-moe-solution
export FIB_DATASET_PATH=/path/to/mlsys26-contest
python scripts/run_local.py
```

This runs correctness checks (comparing against the reference) and measures
latency. You'll see output like:

```
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
Workload: ... (seq_len=1024)
Correctness: PASSED (max_rel_error=1.2e-3)
Latency: 0.85ms (reference: 2.1ms, speedup: 2.47x)
```

### Step 2: Modal B200 Benchmark (official hardware)

```bash
modal run scripts/run_modal.py
```

Note: Modal scores are reference only (clock frequency cannot be locked).
Official evaluations run on bare metal.

### Step 3: Submit

```bash
python scripts/pack_solution.py    # Creates solution.json
git add -A && git commit -m "V2 kernel submission"
git tag submission-v1
git push origin main --tags
```

---

## File Layout Summary

```
my-moe-solution/
├── config.toml                          # ← EDIT: track config
├── solution/
│   └── triton/
│       └── kernel.py                    # ← EDIT: your V2 kernel
├── scripts/
│   ├── run_local.py                     # Local benchmark runner
│   ├── run_modal.py                     # Modal cloud runner
│   ├── pack_solution.py                 # Pack for submission
│   └── inspect_definition.py            # ← NEW: verify parameter names
├── solution.json                        # Generated by pack_solution.py
└── mlsys26-contest/                     # HuggingFace dataset (git-lfs)
    ├── definitions/
    │   └── moe_fp8_block_scale_*.json   # Kernel spec + reference impl
    ├── traces/
    │   └── *.json                       # Workload inputs (safetensors)
    └── solutions/
        └── *.json                       # Other submitted solutions
```
