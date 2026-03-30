# Round 0 — Baseline (V6)

**Commit**: c150589
**Date**: 2026-03-27
**Strategy**: Baseline — FP8 GEMM1+GEMM2 for medium/large, FP32 GEMM2 for tiny/small

## Results

| Tier | seq | lat(ms) | speedup | Status |
|------|----:|--------:|--------:|--------|
| tiny | 1 | 0.541 | 20.68x | PASSED |
| small | 80 | 0.862 | 18.42x | PASSED |
| medium | 901 | 1.096 | 19.26x | PASSED |
| large | 14107 | 6.656 | 6.77x | PASSED |

**Avg speedup**: 16.27x

## Observations

- Large tier at 6.77x is the clear bottleneck vs 18–21x for other tiers.
- GEMM2 large: 6195μs, 49% SM, 1.7% L1 — deep HBM-latency bound.
- Medium/tiny/small all excellent (18–21x).
- NCU from mar27 is pre-FP8 GEMM2 for medium/large — fresh profiling needed.

## Hypothesis for Round 1

Large GEMM2: try BM=32→64 with stages=4.
smem: (64+128)×128×1×4×2 = 192KB < 232KB → 2 CTAs/SM ✓
Arithmetic intensity: 51→89 FLOPS/byte — expect ≥10% speedup.
