# reflections.md — LightSpeed Optimization Log

Auto-appended by `scripts/run_modal_limited.py` after each benchmark run.
Fill in the **What worked** and **Hypothesis** fields manually after reading the table.

Columns: UUID (first 8 chars) | Tier | seq_len | Status | latency_ms | speedup vs reference | max_abs_err | max_rel_err

---

## How to read this file

- **speedup** > 1.0 = faster than the reference implementation.  Goal: maximize.
- **PASSED** = correctness within `atol=1.0 rtol=0.3 match_ratio=0.9`.
- Each section is one `modal run scripts/run_modal_limited.py` run.
- After every FAILED or regressed run, document the revert in the next section's note.

## Workflow reminder

```
# After modifying kernel.py:
git commit -m "perf: <short description>"
modal run scripts/run_modal_limited.py --note "<what changed>"

# After every commit (to refresh NCU data for reasoning):
modal run scripts/run_profiling.py
# → saves kernel_wl0*.ncu-rep to ./NCU/  (or current dir)
```

---

<!-- run entries are appended below this line by run_modal_limited.py -->

## 2026-03-27  `c150589`  (V6 baseline — README + NCU/mar27)
**Note**: V6 baseline — FP8 GEMM1+GEMM2 for medium/large tiers committed at 76baa0d; results from run_modal.py recorded at c150589. NCU profiles saved to NCU/mar27/.

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.541 | 20.68x | 2.05e+03 | 9.31e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.862 | 18.42x | 4.10e+03 | 8.20e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.096 | 19.26x | 4.10e+05 | 4.10e+13 |
| 5e8dc11c | large | 14107 | PASSED | 6.656 | 6.77x | 5.82e+05 | 5.37e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.27x** (tiny=20.68x, small=18.42x, medium=19.26x, large=6.77x)

**NCU baseline (NCU/mar27) — key metrics:**

| Tier | Kernel | Duration (μs) | SM% | L1% | Warp% | Bottleneck |
|------|--------|-------------:|----:|----:|------:|------------|
| tiny | _gemm1_swiglu_kernel | 121 | 6.9 | ~4 | 13 | launch overhead + tiny tiles |
| tiny | _gemm2_kernel | 56 | 30.0 | ~3 | 24 | small grid, FP32 |
| medium | _gemm1_swiglu_kernel | 403 | 31.2 | 3.8 | 12.5 | HBM-latency bound |
| medium | _gemm2_kernel (pre-FP8) | 629 | 40.4 | 2.9 | 24.2 | HBM-latency bound |
| large | _gemm1_swiglu_kernel | 2652 | 34.8 | 5.0 | 12.6 | HBM-latency bound |
| large | _gemm2_kernel (pre-FP8) | 6195 | 49.0 | 1.7 | 22.2 | HBM-latency bound |
| large | _reduce_kernel | 596 | 71.8 | 19.8 | 90.3 | memory-bound (good) |

**What worked**: FP8×FP8 GEMM1+GEMM2 for medium/large delivered 19x+ on those tiers. Large is 6.77x — significantly lower than other tiers, indicating GEMM2 or routing is still the bottleneck there.

**Hypothesis for next run**: Large tier at 6.77x is the biggest gap vs other tiers (18-20x). GEMM2 large is 6195μs at only 49% SM — try increasing BM=32→64 with stages=4 (smem: 24KB×4×2=192KB < 232KB). Also run_profiling to get fresh NCU for FP8 GEMM2 path.

---

## 2026-03-29 06:01 UTC  `6f86f66`
**Note**: large GEMM2 BM=64 stages=4

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.615 | 17.98x | 2.05e+03 | 4.10e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.930 | 16.98x | 4.10e+03 | 1.14e+02 |
| 1a4c6ba1 | medium | 901 | INCORRECT_NUMERICAL | N/A | N/A | 3.26e+05 | 3.26e+13 |
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 5.28e+05 | 4.87e+13 |

**Summary**: 2/4 PASSED | avg speedup **17.48x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 06:06 UTC  `8e31f99`
**Note**: true FP32 baseline confirm

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.595 | 18.72x | 2.05e+03 | 9.86e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.911 | 17.52x | 4.10e+03 | 1.03e+03 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.133 | 18.61x | 3.44e+05 | 3.40e+13 |
| 5e8dc11c | large | 14107 | PASSED | 6.500 | 6.95x | 5.61e+05 | 5.08e+13 |

**Summary**: 4/4 PASSED | avg speedup **15.45x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 06:16 UTC  `2fe5aff`
**Note**: large GEMM1 s3->5, GEMM2 s6->7 deeper pipeline

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.594 | 18.79x | 2.05e+03 | 2.09e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.914 | 17.39x | 4.10e+03 | 1.76e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.137 | 18.58x | 5.04e+05 | 3.79e+13 |
| 5e8dc11c | large | 14107 | PASSED | 6.509 | 6.93x | 5.61e+05 | 5.61e+13 |

**Summary**: 4/4 PASSED | avg speedup **15.42x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 06:25 UTC  `fcc06bf`
**Note**: large GEMM2 BM=64 FP32, halves B reads

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.549 | 20.06x | 2.05e+03 | 1.14e+01 |
| 8f1ff9f1 | small | 80 | PASSED | 0.890 | 17.75x | 4.10e+03 | 3.03e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.105 | 18.93x | 3.50e+05 | 3.50e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.894 | 9.20x | 5.12e+05 | 5.02e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.49x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 06:37 UTC  `8ee1ca1`
**Note**: large BF16 act, GEMM2 s7

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.628 | 17.85x | 2.05e+03 | 4.02e+01 |
| 8f1ff9f1 | small | 80 | PASSED | 0.945 | 16.85x | 4.10e+03 | 1.26e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.166 | 18.12x | 3.75e+05 | 3.75e+13 |
| 5e8dc11c | large | 14107 | RUNTIME_ERROR | N/A | N/A | N/A | N/A |

**Summary**: 3/4 PASSED | avg speedup **17.61x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 06:43 UTC  `a399e7e`
**Note**: large BF16 act stages=7

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.624 | 17.82x | 2.05e+03 | 5.09e+01 |
| 8f1ff9f1 | small | 80 | PASSED | 0.927 | 17.05x | 4.10e+03 | 2.27e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.151 | 18.36x | 4.81e+05 | 4.81e+13 |
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 5.16e+05 | 4.28e+13 |

**Summary**: 3/4 PASSED | avg speedup **17.74x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 23:31 UTC  `cf98dd2`
**Note**: large GEMM1 BK=64 reduce reg pressure target 2 CTAs/SM

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 5.555 | 8.10x | 5.41e+05 | 4.89e+13 |

**Summary**: 1/1 PASSED | avg speedup **8.10x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 23:33 UTC  `7b7a583`
**Note**: large stream overlap prefetch_tm2

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.980 | 9.01x | 5.82e+05 | 4.77e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.01x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 23:35 UTC  `51b2b3c`
**Note**: large GEMM2 BK=32 stages=3 target 3 CTAs/SM

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 5.425 | 8.31x | 5.65e+05 | 4.79e+13 |

**Summary**: 1/1 PASSED | avg speedup **8.31x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 23:37 UTC  `60a390f`
**Note**: large fuse GEMM2+reduce no g2o buffer

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 5.191 | 8.66x | 5.98e+05 | 5.98e+13 |

**Summary**: 1/1 PASSED | avg speedup **8.66x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-29 23:55 UTC  `ce0f05c`
**Note**: large FP8 act GEMM1 fp8out GEMM2 fp8xfp8

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 5.10e+05 | 4.32e+13 |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:00 UTC  `943905e`
**Note**: large GEMM1 BK=64 stages=1 target 2 CTAs/SM via reduced addr regs

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 8.289 | 5.45x | 4.92e+05 | 4.65e+13 |

**Summary**: 1/1 PASSED | avg speedup **5.45x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:02 UTC  `4995b04`
**Note**: large GEMM2 BN=64 BK=64 stages=3 target 3 CTAs/SM

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 7.062 | 6.37x | 5.18e+05 | 5.18e+13 |

**Summary**: 1/1 PASSED | avg speedup **6.37x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:03 UTC  `2e7bb9c`
**Note**: medium baseline check

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.150 | 18.33x | 4.28e+05 | 2.70e+13 |

**Summary**: 1/1 PASSED | avg speedup **18.33x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:06 UTC  `2e7bb9c`
**Note**: small baseline check

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.873 | 18.13x | 4.10e+03 | 9.88e+02 |

**Summary**: 1/1 PASSED | avg speedup **18.13x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:10 UTC  `2e7bb9c`
**Note**: medium GEMM1 stages=3->4 (smem 72->96KB/CTA, 2CTAs/SM=192KB<232KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.142 | 18.46x | 4.10e+05 | 3.13e+13 |

**Summary**: 1/1 PASSED | avg speedup **18.46x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:11 UTC  `2e7bb9c`
**Note**: medium GEMM1 s4 GEMM2 stages=8->9 (smem 108KB/CTA, 2CTAs=216KB<232KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.156 | 18.27x | 4.19e+05 | 3.52e+13 |

**Summary**: 1/1 PASSED | avg speedup **18.27x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:12 UTC  `2e7bb9c`
**Note**: medium GEMM1 s4 GEMM2 BK=32->64 stages=8->4 (smem 96KB/CTA 2CTAs=192KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.114 | 19.19x | 3.89e+05 | 3.89e+13 |

**Summary**: 1/1 PASSED | avg speedup **19.19x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:15 UTC  `617b45e`
**Note**: medium GEMM2 BK=64 stages=8 grid_mult=1 (smem 192KB/CTA deeper pipeline)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.228 | 16.98x | 4.34e+05 | 3.44e+13 |

**Summary**: 1/1 PASSED | avg speedup **16.98x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:17 UTC  `617b45e`
**Note**: medium GEMM2 BK=64 stages=3 grid_mult=2 (smem 72KB/CTA)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.127 | 18.96x | 3.34e+05 | 3.34e+13 |

**Summary**: 1/1 PASSED | avg speedup **18.96x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:19 UTC  `617b45e`
**Note**: small GEMM1 stages=6->8 GEMM2 stages=8->12 (deeper pipeline)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | RUNTIME_ERROR | N/A | N/A | N/A | N/A |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:20 UTC  `617b45e`
**Note**: small GEMM1 stages=6->8 only (smem 160KB/CTA)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | RUNTIME_ERROR | N/A | N/A | N/A | N/A |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:21 UTC  `617b45e`
**Note**: small stages=6 GEMM1 baseline recheck

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.967 | 16.50x | 4.10e+03 | 6.65e+02 |

**Summary**: 1/1 PASSED | avg speedup **16.50x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:22 UTC  `617b45e`
**Note**: small variance check round 2

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 1.041 | 15.67x | 4.10e+03 | 6.32e+02 |

**Summary**: 1/1 PASSED | avg speedup **15.67x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:23 UTC  `617b45e`
**Note**: medium verify still 19.19x

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.095 | 19.58x | 4.44e+05 | 4.44e+13 |

**Summary**: 1/1 PASSED | avg speedup **19.58x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:24 UTC  `617b45e`
**Note**: small BK=128->64 for GEMM2 also (both BKs changed to 64 like medium)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.947 | 17.22x | 4.10e+03 | 8.71e+01 |

**Summary**: 1/1 PASSED | avg speedup **17.22x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:25 UTC  `617b45e`
**Note**: small GEMM2 BK=64->128 stages=8->6 (half K iters, smem 192KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.890 | 17.93x | 4.10e+03 | 9.35e+01 |

**Summary**: 1/1 PASSED | avg speedup **17.93x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:26 UTC  `617b45e`
**Note**: small GEMM2 BK=128 stages=7 (smem 224KB/CTA, max pipeline)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.901 | 17.74x | 4.10e+03 | 5.89e+02 |

**Summary**: 1/1 PASSED | avg speedup **17.74x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:27 UTC  `617b45e`
**Note**: small GEMM2 BK=64 stages=10 (160KB/CTA)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.904 | 17.63x | 4.10e+03 | 8.98e+07 |

**Summary**: 1/1 PASSED | avg speedup **17.63x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:28 UTC  `617b45e`
**Note**: small GEMM1 stages=7 GEMM2 BK=128 stages=6 combo

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.833 | 18.89x | 4.10e+03 | 3.67e+03 |

**Summary**: 1/1 PASSED | avg speedup **18.89x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:29 UTC  `617b45e`
**Note**: small GEMM1 s7 GEMM2 BK=128 s6 confirmation

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.870 | 18.18x | 4.10e+03 | 2.35e+02 |

**Summary**: 1/1 PASSED | avg speedup **18.18x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:30 UTC  `9c36d7e`
**Note**: small GEMM1 stages=8 test (160KB/CTA) with GEMM2 BK=128 s6

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | RUNTIME_ERROR | N/A | N/A | N/A | N/A |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:31 UTC  `9c36d7e`
**Note**: small GEMM1 s7 GEMM2 BK=128 stages=7 (224KB, max depth)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.911 | 17.45x | 4.10e+03 | 3.00e+02 |

**Summary**: 1/1 PASSED | avg speedup **17.45x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:32 UTC  `9c36d7e`
**Note**: small GEMM2 BK=64 stages=6 control (vs BK=128 s6)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.880 | 17.87x | 4.10e+03 | 9.67e+01 |

**Summary**: 1/1 PASSED | avg speedup **17.87x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:33 UTC  `9c36d7e`
**Note**: tiny GEMM1 stages=2->5 GEMM2 stages=2->5 (deeper pipeline)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.594 | 18.53x | 2.05e+03 | 8.18e+00 |

**Summary**: 1/1 PASSED | avg speedup **18.53x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 00:35 UTC  `9c36d7e`
**Note**: large GEMM2 BK=64 grid_mult=2->3 stages=4->3 (3CTAs/SM 216KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 5.512 | 8.19x | 6.68e+05 | 6.68e+13 |

**Summary**: 1/1 PASSED | avg speedup **8.19x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:05 UTC  `9c36d7e`
**Note**: large GEMM1 stages=3->4 (96KB/CTA, 2CTAs=192KB<232KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.946 | 9.10x | 5.73e+05 | 5.49e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.10x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:06 UTC  `9c36d7e`
**Note**: large GEMM1 stages=4 confirmation run

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.922 | 9.17x | 5.06e+05 | 5.06e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.17x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:08 UTC  `9c36d7e`
**Note**: all-tier score check after medium+small improvements

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.629 | 17.72x | 2.05e+03 | 1.51e+01 |
| 8f1ff9f1 | small | 80 | PASSED | 0.915 | 17.42x | 4.10e+03 | 1.70e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.088 | 19.41x | 3.28e+05 | 3.07e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.949 | 9.13x | 5.86e+05 | 5.86e+13 |

**Summary**: 4/4 PASSED | avg speedup **15.92x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:12 UTC  `9c36d7e`
**Note**: small GEMM2 BK=256 stages=3 (28 K-iters, 192KB/CTA)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | INCORRECT_NUMERICAL | N/A | N/A | 9.22e+05 | 8.44e+05 |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:14 UTC  `9c36d7e`
**Note**: large GEMM1 HALF_N=128->64 BN1=256->128 (halve acc regs, 2x N-tiles)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 1.72e+06 | 4.59e+13 |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:17 UTC  `9c36d7e`
**Note**: medium GEMM2 BM=64->128 stages=4->2 (3192 tiles vs 6328, 80KB/CTA)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.510 | 13.91x | 3.17e+05 | 2.93e+13 |

**Summary**: 1/1 PASSED | avg speedup **13.91x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:18 UTC  `9c36d7e`
**Note**: medium GEMM1 BK=128->64 stages=4->8 (deeper pipeline 96KB/CTA 2CTAs)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.213 | 17.47x | 5.70e+05 | 2.95e+13 |

**Summary**: 1/1 PASSED | avg speedup **17.47x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:20 UTC  `9c36d7e`
**Note**: small GEMM2 BK=128 stages=6->3 grid_mult=1->2 (2CTAs/SM, 192KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.929 | 17.15x | 4.10e+03 | 2.37e+03 |

**Summary**: 1/1 PASSED | avg speedup **17.15x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:22 UTC  `9c36d7e`
**Note**: final all-tier score after medium+small improvements

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.545 | 20.19x | 2.05e+03 | 3.78e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.864 | 18.34x | 4.10e+03 | 5.33e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.031 | 20.49x | 3.81e+05 | 3.81e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.933 | 9.14x | 5.28e+05 | 4.83e+13 |

**Summary**: 4/4 PASSED | avg speedup **17.04x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:24 UTC  `9c36d7e`
**Note**: small GEMM2 BM=32->64 stages=4 (diff BM tilemap, 560 tiles vs 1120)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 1.126 | 13.97x | 4.10e+03 | 1.68e+02 |

**Summary**: 1/1 PASSED | avg speedup **13.97x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:26 UTC  `9c36d7e`
**Note**: small GEMM2 BK=64 stages=4 grid_mult=2 (2CTAs/SM 128KB)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.850 | 18.53x | 4.10e+03 | 7.23e+02 |

**Summary**: 1/1 PASSED | avg speedup **18.53x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:27 UTC  `9c36d7e`
**Note**: small GEMM2 BK=64 stages=4 grid_mult=2 confirmation

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.875 | 18.23x | 4.10e+03 | 2.99e+02 |

**Summary**: 1/1 PASSED | avg speedup **18.23x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:28 UTC  `9c36d7e`
**Note**: small GEMM1 s6 GEMM2 BK=64 s4 gm2 combo

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.904 | 17.53x | 4.10e+03 | 2.81e+02 |

**Summary**: 1/1 PASSED | avg speedup **17.53x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:30 UTC  `8b9d15c`
**Note**: large reduce BLOCK_H=256->512 (14 blocks vs 28)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.818 | 9.35x | 4.81e+05 | 4.59e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.35x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:31 UTC  `8b9d15c`
**Note**: large reduce BLOCK_H=512 confirmation

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.802 | 9.37x | 4.69e+05 | 4.49e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.37x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:32 UTC  `8b9d15c`
**Note**: large reduce BLOCK_H=1024 (7 blocks)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.720 | 9.49x | 5.82e+05 | 5.02e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.49x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:33 UTC  `8b9d15c`
**Note**: large reduce BLOCK_H=1024 confirmation

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.502 | 9.72x | 5.18e+05 | 4.44e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.72x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:35 UTC  `8b9d15c`
**Note**: large reduce BLOCK_H=3584 (2 blocks, 7168/2)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | RUNTIME_ERROR | N/A | N/A | N/A | N/A |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:36 UTC  `8b9d15c`
**Note**: large reduce BLOCK_H=1792 (4 blocks)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | RUNTIME_ERROR | N/A | N/A | N/A | N/A |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:38 UTC  `1e4e786`
**Note**: all-tier score after large reduce BLOCK_H=1024 improvement

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.564 | 19.56x | 2.05e+03 | 5.54e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.871 | 18.15x | 4.10e+03 | 1.61e+03 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.169 | 18.05x | 3.38e+05 | 3.38e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.786 | 9.44x | 7.37e+05 | 5.18e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.30x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:39 UTC  `1e4e786`
**Note**: medium reduce scatter_add->_launch_reduce BLOCK_H=1024

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.087 | 19.64x | 3.79e+05 | 2.58e+13 |

**Summary**: 1/1 PASSED | avg speedup **19.64x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:40 UTC  `1e4e786`
**Note**: medium reduce _launch_reduce BH=1024 confirmation

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.069 | 19.61x | 4.01e+05 | 4.01e+13 |

**Summary**: 1/1 PASSED | avg speedup **19.61x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:41 UTC  `1e4e786`
**Note**: medium reduce scatter_add control (vs _launch_reduce)

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 0.865 | 22.97x | 3.83e+05 | 3.83e+13 |

**Summary**: 1/1 PASSED | avg speedup **22.97x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:42 UTC  `1e4e786`
**Note**: medium scatter_add confirmation 2

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.079 | 19.49x | 5.69e+05 | 3.42e+13 |

**Summary**: 1/1 PASSED | avg speedup **19.49x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:43 UTC  `1e4e786`
**Note**: small committed state measurement from same instance as medium

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.939 | 16.80x | 4.10e+03 | 9.59e+02 |

**Summary**: 1/1 PASSED | avg speedup **16.80x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:44 UTC  `1e4e786`
**Note**: small reduce _launch_reduce BH=1024 vs scatter_add

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.927 | 17.09x | 4.10e+03 | 3.13e+02 |

**Summary**: 1/1 PASSED | avg speedup **17.09x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:46 UTC  `1e4e786`
**Note**: large GEMM1 stages=4 with BH=1024 reduce

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.730 | 9.52x | 5.53e+05 | 5.00e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.52x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:47 UTC  `1e4e786`
**Note**: large GEMM1 stages=4 BH=1024 confirmation

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.736 | 9.51x | 5.45e+05 | 4.69e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.51x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:48 UTC  `1e4e786`
**Note**: all-tier final check: GEMM1 s4 + BH=1024 large reduce

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.610 | 18.36x | 2.05e+03 | 6.58e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.901 | 17.69x | 4.10e+03 | 4.29e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.081 | 19.53x | 3.36e+05 | 2.89e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.719 | 9.57x | 5.57e+05 | 4.85e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.29x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:55 UTC  `e354cad`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.485 | 9.75x | 5.32e+05 | 5.18e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.75x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:56 UTC  `e354cad`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.727 | 9.60x | 5.57e+05 | 4.61e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.60x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:57 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.893 | 17.62x | 4.10e+03 | 7.07e+02 |

**Summary**: 1/1 PASSED | avg speedup **17.62x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 04:58 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 8f1ff9f1 | small | 80 | PASSED | 0.887 | 17.78x | 4.10e+03 | 1.39e+02 |

**Summary**: 1/1 PASSED | avg speedup **17.78x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 05:03 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 1a4c6ba1 | medium | 901 | PASSED | 1.444 | 14.52x | 4.30e+05 | 4.30e+13 |

**Summary**: 1/1 PASSED | avg speedup **14.52x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 05:05 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.387 | 27.32x | 2.05e+03 | 3.78e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.691 | 21.95x | 4.10e+03 | 2.06e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 0.876 | 23.20x | 4.03e+05 | 4.03e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.520 | 9.75x | 5.69e+05 | 5.69e+13 |

**Summary**: 4/4 PASSED | avg speedup **20.55x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 05:07 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.554 | 19.70x | 2.05e+03 | 5.78e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.861 | 18.15x | 4.10e+03 | 2.47e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.041 | 20.02x | 3.46e+05 | 2.95e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.658 | 9.63x | 5.53e+05 | 5.04e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.88x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 05:08 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.593 | 18.82x | 2.05e+03 | 4.62e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.898 | 17.74x | 4.10e+03 | 4.23e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.066 | 19.74x | 3.70e+05 | 2.93e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.708 | 9.58x | 4.92e+05 | 4.59e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.47x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 05:10 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.569 | 19.20x | 2.05e+03 | 4.42e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.880 | 17.76x | 4.10e+03 | 6.75e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.053 | 19.77x | 3.75e+05 | 3.75e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.670 | 9.60x | 5.24e+05 | 5.24e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.58x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-30 05:11 UTC  `0224ca5`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.582 | 18.84x | 2.05e+03 | 2.12e+01 |
| 8f1ff9f1 | small | 80 | PASSED | 0.868 | 18.06x | 4.10e+03 | 1.39e+03 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.052 | 19.85x | 5.02e+05 | 5.02e+13 |
| 5e8dc11c | large | 14107 | PASSED | 4.663 | 9.63x | 6.43e+05 | 6.43e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.59x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-31 02:31 UTC  `4b83789`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 5.45e+05 | 4.65e+13 |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-31 02:32 UTC  `4b83789`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 4.85e+05 | 4.26e+13 |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-31 02:34 UTC  `4b83789`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 6.43e+05 | 4.77e+13 |

**Summary**: 0/1 PASSED | avg speedup **0.00x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-31 02:36 UTC  `4b83789`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny | 1 | PASSED | 0.606 | 18.46x | 2.05e+03 | 8.96e+00 |
| 8f1ff9f1 | small | 80 | PASSED | 0.898 | 17.74x | 4.10e+03 | 6.73e+02 |
| 1a4c6ba1 | medium | 901 | PASSED | 1.076 | 19.50x | 4.06e+05 | 4.06e+13 |
| 5e8dc11c | large | 14107 | INCORRECT_NUMERICAL | N/A | N/A | 5.28e+05 | 5.28e+13 |

**Summary**: 3/4 PASSED | avg speedup **18.56x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-31 02:38 UTC  `4b83789`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.755 | 9.49x | 6.23e+05 | 6.23e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.49x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---
## 2026-03-31 02:39 UTC  `4b83789`


| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| 5e8dc11c | large | 14107 | PASSED | 4.734 | 9.58x | 6.02e+05 | 4.89e+13 |

**Summary**: 1/1 PASSED | avg speedup **9.58x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_

---