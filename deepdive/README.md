# MoE Kernel Optimization Strategies

Fusion and scheduling strategies for the DeepSeek-V3 MoE kernel on NVIDIA B200,
derived from hardware cost primitives in `cost_model_analysis.md`.

---

## B200 Theoretical Specs

| Parameter | Value |
|-----------|-------|
| Architecture | Blackwell |
| HBM3e Capacity | 192 GB |
| HBM3e Bandwidth | 8,000 GB/s (bidirectional) |
| FP4 Tensor Core | 9,000 TFLOPS |
| FP8 Tensor Core | 4,500 TFLOPS |
| FP16 Tensor Core | 4,500 TFLOPS |
| BF16 Tensor Core | 2,250 TFLOPS |
| TF32 Tensor Core | 2,250 TFLOPS |
| FP32 (non-TC) | ~1,125 TFLOPS |
| FP64 Tensor Core | 45 TFLOPS |
| SMs | 192 |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 (4th gen) |
| SRAM per SM | 256 KB |
| Total SRAM | 49,152 KB (~48 MB) |
| L2 Cache | ~128 MB |
| Interconnect | NVLink 5 (1,800 GB/s) |
| TDP | 1,000W |

### Measured Effective Values (from benchmarks)

| Parameter | Measured | % of Spec |
|-----------|----------|-----------|
| HBM Bandwidth (copy) | 6,354 GB/s | 79% |
| FP8 GEMM | 3,122 TFLOPS | 69% |
| BF16 GEMM | 1,650 TFLOPS | 73% |
| FP32 GEMM | 63.9 TFLOPS | 5.7% |
| Kernel Launch Overhead | 14.2 us | — |

---

## MoE Dimensions (DeepSeek-V3)

```
T  = 4096    (tokens per batch)
H  = 7168    (hidden size)
I  = 2048    (intermediate size)
BLOCK = 128  (quantization block size)
E_global = 256, E_local = 32
TOP_K = 8, N_GROUP = 8, TOPK_GROUP = 4
Tk = T * TOP_K / E_global = 128  (avg tokens per expert)
```

---

## Strategy A: Unfused Baseline (8 separate ops)

Every intermediate written to HBM and read back.
This is what `kernel_ref.py` does.

| Op | Name | bytes_read | bytes_write | FLOPs | mem_us | compute_us | launches | cost_us |
|----|------|-----------|-------------|-------|--------|------------|----------|---------|
| 0 | DequantA | 30 MB | 112 MB | 29 M | 22.4 | ~0 | 1 | 36.6 |
| 1 | DequantW13 | 896 MB | 3,584 MB | 940 M | 704.7 | ~0 | 1 | 718.9 |
| 2 | DequantW2 | 448 MB | 1,792 MB | 470 M | 352.8 | ~0 | 1 | 367.0 |
| 3 | Routing | 4 MB | 4 MB | 21 M | 1.3 | ~0 | 1 | 15.5 |
| 4 | GEMM1 (32 experts) | 3,696 MB | 64 MB | 241 G | 591.5 | 77.2 | 32 | 1,046 |
| 5 | SwiGLU (32 experts) | 64 MB | 32 MB | 42 M | 15.1 | ~0 | 32 | 469.5 |
| 6 | GEMM2 (32 experts) | 1,824 MB | 112 MB | 120 G | 305.0 | 38.4 | 32 | 759.4 |
| 7 | WeightedAccum (32 exp) | 172 MB | 56 MB | 59 M | 35.9 | ~0 | 32 | 490.3 |
| | **TOTAL** | **7,134 MB** | **5,756 MB** | **361 G** | | | **164** | **3,903 us** |

Per-expert ops (4-7) each launch 32 kernels: launch overhead = 32 x 14.2 = 454 us per op.
Launch overhead alone accounts for ~2,329 us (60% of total).

**Unfused total HBM traffic: ~12.6 GB. Nearly everything is memory-bound.**

---

## Strategy B: Online Dequant (fuse dequant into GEMM)

Read FP8 weights directly in GEMM, dequantize in registers. No FP32 intermediates.

| Op | bytes_read | bytes_write | FLOPs | mem_us | compute_us | launches | cost_us |
|----|-----------|-------------|-------|--------|------------|----------|---------|
| Routing | 4 MB | 4 MB | 21 M | 1.3 | ~0 | 1 | 15.5 |
| GEMM1+DequantA+DequantW13 | 926 MB (FP8) | 64 MB | 241 G | 155.9 | 77.2 | 32 | 610.3 |
| SwiGLU | 64 MB | 32 MB | 42 M | 15.1 | ~0 | 32 | 469.5 |
| GEMM2+DequantW2 | 480 MB (FP8) | 112 MB | 120 G | 93.2 | 38.4 | 32 | 547.6 |
| WeightedAccum | 172 MB | 56 MB | 59 M | 35.9 | ~0 | 32 | 490.3 |
| **TOTAL** | **1,646 MB** | **268 MB** | **361 G** | | | **129** | **2,133 us** |

**Savings vs A: 1,770 us (45%)** — from eliminating 10.7 GB of FP32 weight traffic.

---

## Strategy C: Grouped Expert GEMM (single kernel per op)

Same as B but use grouped/batched GEMM instead of Python loop over 32 experts.
Eliminates per-expert launch overhead.

| Op | bytes_read | bytes_write | FLOPs | mem_us | compute_us | launches | cost_us |
|----|-----------|-------------|-------|--------|------------|----------|---------|
| Routing | 4 MB | 4 MB | 21 M | 1.3 | ~0 | 1 | 15.5 |
| GEMM1 (grouped) | 926 MB | 64 MB | 241 G | 155.9 | 77.2 | 1 | 170.1 |
| SwiGLU (fused) | 64 MB | 32 MB | 42 M | 15.1 | ~0 | 1 | 29.3 |
| GEMM2 (grouped) | 480 MB | 112 MB | 120 G | 93.2 | 38.4 | 1 | 107.4 |
| WeightedAccum (fused) | 172 MB | 56 MB | 59 M | 35.9 | ~0 | 1 | 50.1 |
| **TOTAL** | **1,646 MB** | **268 MB** | **361 G** | | | **5** | **372 us** |

**Savings vs B: 1,761 us (83%)** — launch overhead was the dominant cost.

---

## Strategy D: Full Fusion (single kernel)

Fuse GEMM1 -> SwiGLU -> GEMM2 per expert. Intermediates (G1, C) stay in SRAM.
Only read FP8 weights + activations from HBM, write final output.

| Op | bytes_read | bytes_write | FLOPs | mem_us | compute_us | launches | cost_us |
|----|-----------|-------------|-------|--------|------------|----------|---------|
| Full fused kernel | 1,382 MB | 56 MB | 361 G | 226.5 | 115.7 | 1 | 240.7 |

```
bytes_read:
  hidden_states [T, H] fp8          =  29.4 MB
  hs_scale [56, T] fp32             =   0.9 MB
  gemm1_weights [32, 4096, 7168] fp8 = 896.0 MB
  gemm1_scale                       =   0.2 MB
  gemm2_weights [32, 7168, 2048] fp8 = 448.0 MB
  gemm2_scale                       =   0.1 MB
  routing_logits [T, 256] fp32      =   4.0 MB
  routing_bias [256] fp32           =   0.001 MB
  Total: ~1,378 MB

bytes_write:
  output [T, H] bf16 = 56 MB

compute:
  GEMM1: 32 * 2 * 128 * 7168 * 4096 = 241 GFLOPs
  GEMM2: 32 * 2 * 128 * 2048 * 7168 = 120 GFLOPs
  Total: 361 GFLOPs / 3,122 TFLOPS  = 115.7 us
```

---

## Strategy Comparison

| Strategy | HBM Traffic | Kernel Launches | Estimated Latency | vs Unfused |
|----------|-------------|-----------------|-------------------|------------|
| A: Unfused | 12.9 GB | 164 | 3,903 us | 1.0x |
| B: Online Dequant | 1.9 GB | 129 | 2,133 us | 1.8x |
| C: B + Grouped GEMM | 1.9 GB | 5 | 372 us | 10.5x |
| D: Full Fusion | 1.4 GB | 1 | 241 us | 16.2x |

### Where the savings come from

```
Unfused baseline:              3,903 us
  - Eliminate FP32 dequant:   -1,770 us  (remove 10.7 GB phantom traffic)
  - Group expert kernels:     -1,761 us  (eliminate 124 kernel launches)
  - Fuse intermediates:         -131 us  (keep G1, C in SRAM)
Full fusion:                     241 us
```

---

## Scaling with T (Sequence Length)

| T | Tk | Weight Read (MB) | Activation (MB) | GEMM FLOPs | mem_us | compute_us | Bound |
|---|----|--------------------|-----------------|------------|--------|------------|-------|
| 1 | 0.03 | 1,344 | ~0 | 0.9 G | 212 | 0.3 | Memory |
| 128 | 4 | 1,344 | 3.6 | 11.3 G | 212 | 3.6 | Memory |
| 4096 | 128 | 1,344 | 112 | 361 G | 229 | 116 | Memory |
| 16384 | 512 | 1,344 | 450 | 1,445 G | 282 | 463 | **Compute** |
| 65536 | 2048 | 1,344 | 1,800 | 5,780 G | 495 | 1,852 | **Compute** |

**Crossover**: At T ~ 14,000 (Tk ~ 437), the fused kernel transitions from
memory-bound to compute-bound. Below this, weight loading dominates.

For contest workloads (T from 1 to ~14K), **most cases are memory-bound**.
Priority is minimizing HBM traffic, not maximizing TFLOPS.

---

## SRAM Working Set Analysis

Per SM: 256 KB SRAM. For a fused GEMM1 tile:

```
granularity = [w, h, k]

GEMM1 tile working set:
  A_tile:     h * k * 1 byte   (FP8 input)
  W13_tile:   w * k * 1 byte   (FP8 weight, dequant on-the-fly)
  G1_tile:    h * w * 4 bytes  (FP32 output accumulator)

Example: h=128, w=128, k=128
  A_tile:   128 * 128 * 1 = 16 KB
  W13_tile: 128 * 128 * 1 = 16 KB
  G1_tile:  128 * 128 * 4 = 64 KB
  Total: ~96 KB  (fits easily in 256 KB)
```

For fused GEMM1 + SwiGLU + GEMM2:
```
Need simultaneously:
  G1_tile (GEMM1 output / SwiGLU input)
  C_tile  (SwiGLU output / GEMM2 input)
  W2_tile (GEMM2 weight)
  O_tile  (GEMM2 output)

This constrains tile sizes but fits within 256 KB for reasonable tile dimensions.
```

Weights (1.34 GB FP8) must stream through SRAM — they cannot be kept resident.
The question is what tile granularity allows pipelining.
