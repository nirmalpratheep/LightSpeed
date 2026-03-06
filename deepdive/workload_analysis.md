# MoE Workload Analysis

Analysis of `mlsys26-contest/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl`

## Workload Summary

19 test cases with the same MoE definition (`e32_h7168_i2048`, topk=8, ng=8, kg=4). Two variables change across workloads.

## Variable 1: `seq_len` (T) — PRIMARY

This is the axis that drives both memory and compute cost. It spans 4 orders of magnitude:

| seq_len | Category |
|---------|----------|
| 1 | Single-token decode |
| 7, 14–16 | Very small batch |
| 32, 52–62, 80 | Medium batch |
| 901 | Large batch |
| 11,948 / 14,107 | Prefill-scale |

### Memory impact (all scale linearly with T)

| Tensor | Shape | Dtype | Size formula |
|--------|-------|-------|-------------|
| hidden_states | [T, 7168] | FP8 | T × 7168 bytes |
| hidden_states_scale | [56, T] | FP32 | 56 × T × 4 bytes |
| routing_logits | [T, 256] | FP32 | T × 256 × 4 bytes |
| output | [T, 7168] | BF16 | T × 7168 × 2 bytes |
| Per-expert intermediates | [Tk, ...] | FP32 | Tk proportional to T × 8 / 256 |

### Compute impact (all scale linearly with T)

| Operation | Cost scales as |
|-----------|---------------|
| GEMM1/GEMM2 | Tk × H × I, where Tk proportional to T |
| Routing | T × 256 |
| Dequant activations | T × 7168 |

### Regime shift

- **Small T (1–16)**: Weight-dominated / memory-bound. Weight tensors (~1.3 GB total FP8) dwarf activations. GEMMs have tiny M dimension — poor hardware utilization.
- **Large T (11K–14K)**: Activation-dominated / compute-bound. Activation tensors grow large. GEMMs have favorable shapes for throughput.

## Variable 2: `local_expert_offset` — IMPACTS ROOFLINE

Observed values: 0, 32, 64, 96, 128, 160, 192, 224

Determines **which** 32 of the 256 global experts are local. The `routing_logits` are **real data** (loaded from safetensors, not random), so top-8 routing decisions are deterministic. Different offsets select different expert subsets, directly changing Tk_e (tokens per expert) for each local expert.

### How it shifts the roofline

For a single expert's GEMM1 (`[Tk_e, H] @ [H, 2I] -> [Tk_e, 2I]`):

```
FLOPs  = 2 * Tk_e * H * 2I                            (scales with Tk_e)
Bytes  = H * 2I (weights, fixed) + Tk_e * (H + 2I)    (activations scale with Tk_e)

Arithmetic Intensity = FLOPs / Bytes
    = 2 * Tk_e * H * 2I / (H * 2I + Tk_e * (H + 2I))
```

- **Small Tk_e**: weight term `H * 2I` dominates denominator → AI ≈ `2 * Tk_e` → low → **memory-bound** (full weight load for few tokens)
- **Large Tk_e**: activation terms dominate → AI → `2*H*2I / (H+2I)` → high → **compute-bound**

The same logic applies to GEMM2 (`[Tk_e, I] @ [I, H]`).

### Concrete effect

Since `local_expert_offset` determines which experts are local, and `routing_logits` (real data) determine how many tokens each expert receives:
- Offset selecting **popular** experts → large Tk_e per expert → higher AI → more compute-bound
- Offset selecting **unpopular** experts → small Tk_e per expert → lower AI → more memory-bound
- Within a single workload, different experts have different Tk_e, so some experts may be memory-bound while others are compute-bound simultaneously

## Fixed Constants (invariant across all 19 workloads)

| Parameter | Value |
|-----------|-------|
| H (hidden dim) | 7168 |
| I (intermediate dim) | 2048 |
| E_local (local experts) | 32 |
| E_global (global experts) | 256 |
| BLOCK (FP8 quant block) | 128 |
| TOP_K | 8 |
| N_GROUP | 8 |
| TOPK_GROUP | 4 |
| routed_scaling_factor | 2.5 |
| gemm1_weights shape | [32, 4096, 7168] FP8 |
| gemm2_weights shape | [32, 7168, 2048] FP8 |

## Implications for Scheduler Design

1. **seq_len sets the global regime** — the scheduler must handle both tiny decode (T=1) and large prefill (T=14107) efficiently.
2. **Small-T regime** is memory-bound: minimize data movement of weights (reuse across experts, retain in fast memory).
3. **Large-T regime** is compute-bound: maximize GEMM utilization, tile for parallelism.
4. **local_expert_offset shifts the roofline per-expert** — even at a given seq_len, the actual compute/memory balance depends on which experts are local and how many tokens each receives.
5. **Mixed regime within a workload** — some experts may be memory-bound (low Tk_e, weight-load dominated) while others are compute-bound (high Tk_e), requiring adaptive scheduling per expert rather than a one-size-fits-all granularity.