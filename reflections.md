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