# Cost Model Analysis: MoE Kernel DAG on B200

Mapping the abstract scheduling model from `PROBLEM.md` onto the real
DeepSeek-V3 MoE kernel running on an NVIDIA B200 GPU.

---

## 1. Core Latency Formula (from PROBLEM.md)

```
per_step_latency = max(compute_time, memory_time_in + memory_time_out)
```

- **memory_time** = bytes transferred / HBM bandwidth
- **compute_time** = FLOPs / peak throughput (precision-dependent)
- If `memory_time > compute_time` the op is **memory-bound**.
- If `compute_time > memory_time` the op is **compute-bound**.

---

## 2. Hardware Mapping: PROBLEM.md to B200

| PROBLEM.md Concept | B200 Reality | Capacity | Bandwidth |
|---------------------|--------------|----------|-----------|
| **Slow Memory** | HBM3e | ~192 GB | ~8 TB/s |
| **Fast Memory** | SRAM (shared mem + registers per SM) | ~256 KB/SM x 192 SMs | on-chip (effectively infinite) |
| **L2 Cache** | Sits between -- transparent cache | ~128 MB | ~12-15 TB/s (est.) |

Key simplification: treat **HBM as slow memory** and **SRAM as fast memory**.
L2 acts as a transparent cache -- if your working set fits in L2 you get
"free" bandwidth, but this is hard to model without benchmarks.

### Peak Throughput (placeholder -- verify with benchmarks)

| Precision | Peak TFLOPS | Notes |
|-----------|-------------|-------|
| FP8 Tensor Core | 4,500 | For GEMM with FP8 inputs |
| BF16 Tensor Core | 2,250 | For GEMM with BF16 inputs |
| FP32 (non-TC) | 1,125 | For pointwise / elementwise |

---

## 3. Tensor Sizes and Data Movement

### Production Dimensions

```
T  = 4096    (tokens)
H  = 7168    (hidden size)
I  = 2048    (intermediate size)
BLOCK = 128
E_global = 256, E_local = 32
TOP_K = 8, N_GROUP = 8, TOPK_GROUP = 4
Tk = T * TOP_K / E_global = 128  (avg tokens per expert)
```

### Dtype Byte Sizes

| Dtype | Bytes |
|-------|-------|
| fp8   | 1     |
| bf16  | 2     |
| fp32  | 4     |
| int64 | 8     |

---

## 4. Per-Op Cost Breakdown

For each op we compute:
- **bytes_in**: bytes read from HBM (input tensors)
- **bytes_out**: bytes written to HBM (output tensors)
- **FLOPs**: floating-point operations
- **mem_time**: `(bytes_in + bytes_out) / 8e12` seconds
- **compute_time**: `FLOPs / peak_throughput` seconds
- **bound**: whichever is larger determines the bottleneck

### Op[0]: DequantA (Pointwise)

```
Operation:  A = hidden_states.to(fp32) * scale_expanded
bytes_in:   hidden_states [T, 7168] fp8       = T * 7168 * 1     =  29.4 MB
            hidden_states_scale [56, T] fp32   = 56 * T * 4       =   0.9 MB
            Total in                                              =  30.2 MB

bytes_out:  A [T, 7168] fp32                   = T * 7168 * 4     = 112.0 MB
            Total out                                             = 112.0 MB

FLOPs:      T * 7168 (one multiply per element)                   =  29.4 M
mem_time:   (30.2 + 112.0) MB / 8 TB/s                           =  17.8 us
compute_time: 29.4 M / 1125 TFLOPS                               =   0.03 us
Bound:      MEMORY (by ~600x)
```

### Op[1]: DequantW13 (Pointwise)

```
Operation:  W13 = gemm1_weights.to(fp32) * scale_expanded
bytes_in:   gemm1_weights [32, 4096, 7168] fp8       = 896.0 MB
            gemm1_weights_scale [32, 32, 56] fp32     =   0.2 MB
            Total in                                  = 896.2 MB

bytes_out:  W13 [32, 4096, 7168] fp32                 = 3,584.0 MB
            Total out                                 = 3,584.0 MB

FLOPs:      32 * 4096 * 7168                          = 939.5 M
mem_time:   (896.2 + 3584.0) MB / 8 TB/s              = 560.0 us
compute_time: 939.5 M / 1125 TFLOPS                   =   0.8 us
Bound:      MEMORY (by ~700x)
```

### Op[2]: DequantW2 (Pointwise)

```
Operation:  W2 = gemm2_weights.to(fp32) * scale_expanded
bytes_in:   gemm2_weights [32, 7168, 2048] fp8       = 448.0 MB
            gemm2_weights_scale [32, 56, 16] fp32     =   0.1 MB
            Total in                                  = 448.1 MB

bytes_out:  W2 [32, 7168, 2048] fp32                  = 1,792.0 MB
            Total out                                 = 1,792.0 MB

FLOPs:      32 * 7168 * 2048                          = 469.8 M
mem_time:   (448.1 + 1792.0) MB / 8 TB/s              = 280.0 us
compute_time: 469.8 M / 1125 TFLOPS                   =   0.4 us
Bound:      MEMORY (by ~700x)
```

### Op[3]: Routing (Pointwise / Mixed)

```
Operation:  sigmoid -> add_bias -> group_topk -> normalize_weights
bytes_in:   routing_logits [T, 256] fp32      = T * 256 * 4    =   4.0 MB
            routing_bias [256] fp32            = 256 * 4        =   0.001 MB
            Total in                                           =   4.0 MB

bytes_out:  topk_idx [T, 8] int64             = T * 8 * 8      =   0.25 MB
            weights [T, 256] fp32             = T * 256 * 4    =   4.0 MB
            Total out                                          =   4.25 MB

FLOPs:      ~T * 256 * 20 (sigmoid + scoring + topk + norm)   =  21.0 M
mem_time:   (4.0 + 4.25) MB / 8 TB/s                          =   1.0 us
compute_time: 21.0 M / 1125 TFLOPS                            =   0.02 us
Bound:      MEMORY (but very small -- likely latency-bound in practice)
```

### Op[4]: GEMM1 (MatMul, all 32 experts)

```
Operation:  G1 = A_e @ W13_e.T   per expert, for 32 experts
            shape: [Tk, H] @ [H, 2I] -> [Tk, 4096]

bytes_in:   A_e [Tk, 7168] fp32 x 32           = 32 * 128 * 7168 * 4  = 112.0 MB
            W13_e [4096, 7168] fp32 x 32        = 32 * 4096 * 7168 * 4 = 3,584.0 MB
            topk_idx (for token select, small)                         =   0.25 MB
            Total in                                                   = 3,696.3 MB

bytes_out:  G1 [Tk, 4096] fp32 x 32            = 32 * 128 * 4096 * 4  =  64.0 MB
            Total out                                                  =  64.0 MB

FLOPs:      32 * 2 * Tk * H * 2I = 32 * 2 * 128 * 7168 * 2048        = 241 GFLOPs
            (Note: 2x for multiply-add in matmul)

mem_time:   (3696.3 + 64.0) MB / 8 TB/s                               = 470.0 us
compute_time: 241 GFLOPs / 4500 TFLOPS (if FP8 inputs)                =  53.6 us
              241 GFLOPs / 1125 TFLOPS (if FP32 inputs)               = 214.2 us
Bound:      MEMORY (weights dominate -- 3.5 GB read)
```

**Critical observation**: W13 weights are 3.5 GB. They dominate the cost.
In the reference code, weights are dequantized to FP32 first (Op[1]), then
read again here. That is **7 GB of unnecessary HBM traffic** if not fused.

### Op[5]: SwiGLU (Pointwise, all 32 experts)

```
Operation:  C = silu(X2) * X1   where G1 is split into X1, X2
bytes_in:   G1 [Tk, 4096] fp32 x 32            = 64.0 MB
bytes_out:  C [Tk, 2048] fp32 x 32             = 32.0 MB

FLOPs:      32 * Tk * 2048 * 5                  = 41.9 M
            (silu = 3 ops + 1 mul + 1 mul)
mem_time:   (64.0 + 32.0) MB / 8 TB/s           =  12.0 us
compute_time: 41.9 M / 1125 TFLOPS              =   0.04 us
Bound:      MEMORY
```

### Op[6]: GEMM2 (MatMul, all 32 experts)

```
Operation:  O = C @ W2_e.T   per expert, for 32 experts
            shape: [Tk, I] @ [I, H] -> [Tk, 7168]

bytes_in:   C [Tk, 2048] fp32 x 32             = 32.0 MB
            W2_e [7168, 2048] fp32 x 32         = 1,792.0 MB
            Total in                            = 1,824.0 MB

bytes_out:  O [Tk, 7168] fp32 x 32             = 112.0 MB
            Total out                           = 112.0 MB

FLOPs:      32 * 2 * Tk * I * H = 32 * 2 * 128 * 2048 * 7168 = 120 GFLOPs
mem_time:   (1824.0 + 112.0) MB / 8 TB/s        = 242.0 us
compute_time: 120 GFLOPs / 4500 TFLOPS (FP8)    =  26.8 us
              120 GFLOPs / 1125 TFLOPS (FP32)   = 107.1 us
Bound:      MEMORY (weights dominate again -- 1.75 GB read)
```

### Op[7]: WeightedAccum (Pointwise, all 32 experts)

```
Operation:  output[tok] += O * w   (scatter-add with routing weights)
bytes_in:   O [Tk, 7168] fp32 x 32             = 112.0 MB
            weights (small, already computed)    =   4.0 MB
            output (read-modify-write)          =  56.0 MB
            Total in                            = 172.0 MB

bytes_out:  output [T, 7168] bf16               =  56.0 MB
            Total out                           =  56.0 MB

FLOPs:      32 * Tk * 7168 * 2                  = 58.7 M
mem_time:   (172.0 + 56.0) MB / 8 TB/s          =  28.5 us
compute_time: 58.7 M / 1125 TFLOPS              =   0.05 us
Bound:      MEMORY
```

---

## 5. Summary Table (Unfused -- All Ops Independent)

| Op | Name | bytes_in | bytes_out | FLOPs | mem_time (us) | compute_time (us) | Bound |
|----|------|----------|-----------|-------|---------------|-------------------|-------|
| 0 | DequantA | 30 MB | 112 MB | 29 M | 17.8 | 0.03 | Memory |
| 1 | DequantW13 | 896 MB | 3,584 MB | 940 M | 560.0 | 0.8 | Memory |
| 2 | DequantW2 | 448 MB | 1,792 MB | 470 M | 280.0 | 0.4 | Memory |
| 3 | Routing | 4 MB | 4 MB | 21 M | 1.0 | 0.02 | Memory |
| 4 | GEMM1 | 3,696 MB | 64 MB | 241 G | 470.0 | 53.6 | Memory |
| 5 | SwiGLU | 64 MB | 32 MB | 42 M | 12.0 | 0.04 | Memory |
| 6 | GEMM2 | 1,824 MB | 112 MB | 120 G | 242.0 | 26.8 | Memory |
| 7 | WeightedAccum | 172 MB | 56 MB | 59 M | 28.5 | 0.05 | Memory |
| | **TOTAL** | **7,134 MB** | **5,756 MB** | **361 G** | **1,611 us** | | |

**Unfused total HBM traffic: ~12.6 GB. Nearly everything is memory-bound.**

---

## 6. Fusion Impact (Subgraph Strategies)

This is where PROBLEM.md's subgraph model directly applies. Fusing ops
eliminates HBM round-trips for intermediate tensors.

### Strategy A: No Fusion (baseline, 8 separate subgraphs)

```
subgraphs: [[0], [1], [2], [3], [4], [5], [6], [7]]
Total HBM traffic: 12.6 GB
Estimated latency: ~1,611 us
```

Every intermediate (A, W13, W2, topk_idx, weights, G1, C, O) is written
to HBM and read back. This is what `kernel_ref.py` does.

### Strategy B: Fuse Dequant into GEMM (online dequant)

```
subgraphs: [[0, 4], [1 fused into 4], [2 fused into 6], [3], [5], [6], [7]]

Eliminated HBM traffic:
  - A [T, 7168] fp32: skip 112 MB write + 112 MB read = 224 MB saved
  - W13 [32, 4096, 7168] fp32: skip 3,584 MB write + 3,584 MB read = 7,168 MB saved
  - W2 [32, 7168, 2048] fp32: skip 1,792 MB write + 1,792 MB read = 3,584 MB saved

Total saved: ~10.7 GB of HBM traffic!
Remaining HBM traffic: ~1.9 GB
```

Read FP8 weights (896 + 448 = 1,344 MB) + dequant on-the-fly in SRAM.
This is the single biggest optimization -- eliminates ~85% of HBM traffic.

### Strategy C: Fuse Expert Compute Pipeline

```
subgraphs: [[0+1+4, 5, 6+2], [3], [7]]
   i.e., [DequantA + DequantW13 + GEMM1 + SwiGLU + DequantW2 + GEMM2]

Additional savings over Strategy B:
  - G1 [Tk, 4096] fp32: skip 64 MB write + 64 MB read = 128 MB saved
  - C [Tk, 2048] fp32: skip 32 MB write + 32 MB read = 64 MB saved

Remaining HBM traffic: ~1.7 GB
```

But SRAM working set constraint: can we fit the required data in ~256 KB?
- Per tile of GEMM1: need A_tile + W13_tile + G1_tile
- This determines the tile/granularity `[w, h, k]`

### Strategy D: Full Fusion (everything in one kernel)

```
subgraphs: [[0, 1, 2, 3, 4, 5, 6, 7]]

HBM traffic (minimum possible):
  bytes_in:  hidden_states (29 MB) + hs_scale (0.9 MB)
             + gemm1_weights (896 MB) + gemm1_scale (0.2 MB)
             + gemm2_weights (448 MB) + gemm2_scale (0.1 MB)
             + routing_logits (4 MB) + routing_bias (0.001 MB)
             = ~1,378 MB

  bytes_out: output [T, 7168] bf16 = 56 MB

  Total: ~1,434 MB (vs 12.6 GB unfused = 8.8x reduction!)

mem_time: 1,434 MB / 8 TB/s = 179 us
```

But: working set must fit in SRAM. The weights alone (896 + 448 = 1,344 MB)
must stream through -- they cannot be kept resident. The question becomes
what granularity allows this to fit.

---

## 7. SRAM Working Set Analysis (Fast Memory Capacity)

Per SM: ~256 KB SRAM. For a single GEMM1 tile:

```
granularity = [w, h, k]

GEMM1 tile working set:
  A_tile:     h * k * 4 bytes     (FP32 input slice)
  W13_tile:   (2*I_tile) * k * 1 byte  (FP8, dequant on-the-fly)
  W13_scale:  small (per-block)
  G1_tile:    h * w * 4 bytes     (FP32 output accumulator)

Example: h=128, w=128, k=128
  A_tile:   128 * 128 * 4 = 64 KB
  W13_tile: 256 * 128 * 1 = 32 KB  (FP8)
  G1_tile:  128 * 128 * 4 = 64 KB
  Total: ~160 KB  (fits in 256 KB)

Example: h=128, w=256, k=128
  A_tile:   128 * 128 * 4 = 64 KB
  W13_tile: 256 * 128 * 1 = 32 KB
  G1_tile:  128 * 256 * 4 = 128 KB
  Total: ~224 KB  (tight fit)
```

For fused GEMM1 + SwiGLU + GEMM2 in one subgraph:
```
Need simultaneously: G1_tile (output of GEMM1 / input to SwiGLU)
                   + C_tile (output of SwiGLU / input to GEMM2)
                   + W2_tile (weight for GEMM2)
                   + O_tile (output of GEMM2)

This is the split-K / pipeline analysis from PROBLEM.md Example 5.
Reducing k shrinks the intermediate working set.
```

---

## 8. What to Benchmark on B200

### Isolated Op Benchmarks (gives `base_costs`)

```python
import torch, time

def benchmark_op(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / iters

# Op 0: DequantA
#   A = hidden_states.to(torch.float32) * scale_expanded

# Op 1: DequantW13
#   W13 = gemm1_weights.to(torch.float32) * scale_expanded

# Op 2: DequantW2
#   W2 = gemm2_weights.to(torch.float32) * scale_expanded

# Op 3: Routing
#   full routing pipeline: sigmoid -> add_bias -> group_topk -> normalize

# Op 4: GEMM1 (single expert)
#   G1 = A_e @ W13_e.T    shape: [Tk, 7168] @ [7168, 4096]

# Op 5: SwiGLU (single expert)
#   C = silu(X2) * X1

# Op 6: GEMM2 (single expert)
#   O = C @ W2_e.T         shape: [Tk, 2048] @ [2048, 7168]

# Op 7: WeightedAccum
#   output.index_add_(0, token_idx, O * w.unsqueeze(1))
```

### Fused Op Benchmarks (gives subgraph costs)

```python
# Fused A: DequantA + GEMM1 (dequant FP8 on-the-fly per block)
# Fused B: GEMM1 + SwiGLU + GEMM2 (full expert pipe, FP32 inputs)
# Fused C: DequantA + GEMM1 + SwiGLU + GEMM2 + Accum (everything)
# Fused D: Full kernel (routing + all experts) -- this is kernel.py
```

### Memory Bandwidth Measurement

```python
# Measure actual achieved HBM bandwidth:
# 1. Large memcpy (torch.empty -> .copy_())
# 2. Compare to theoretical 8 TB/s
# 3. Use achieved BW in cost model instead of peak
```

### L2 Cache Effects

```python
# Measure with different working set sizes:
# - Small (fits in L2, ~128 MB): should see >8 TB/s effective BW
# - Large (exceeds L2): should see ~8 TB/s (HBM-bound)
# Weights (1.3 GB FP8) do NOT fit in L2 -- always HBM-bound
# Activations (112 MB FP32 for A) might fit in L2 if accessed quickly
```

---

## 9. Filling in the DAG

Once benchmarks are collected, update `moe_dag['base_costs']` in
`deepdive/kernel_dag.ipynb`:

```python
moe_dag['base_costs'] = [
    measured_dequant_a_ns,       # Op 0
    measured_dequant_w13_ns,     # Op 1
    measured_dequant_w2_ns,      # Op 2
    measured_routing_ns,         # Op 3
    measured_gemm1_all_ns,       # Op 4
    measured_swiglu_all_ns,      # Op 5
    measured_gemm2_all_ns,       # Op 6
    measured_accum_ns,           # Op 7
]
```

Then compute the scheduling strategies:

```python
# Strategy A: no fusion
latency_A = sum(max(mem_time[i], compute_time[i]) for i in range(8))

# Strategy B: fuse dequant into GEMM
# (recalculate mem_time with eliminated intermediates)

# Strategy C: fuse full expert pipeline
# Strategy D: full fusion (single kernel)
```

---

## 10. Key Takeaways

1. **Everything is memory-bound** in the unfused baseline. Arithmetic
   intensity is far below the roofline crossover for every single op.

2. **Dequantization dominates**: Writing FP32 weights to HBM and reading
   them back accounts for ~10.7 GB of the 12.6 GB total traffic.
   Fusing dequant into GEMM (online dequant from FP8) is the #1 optimization.

3. **Minimum possible HBM traffic** is ~1.4 GB (read FP8 inputs + write
   BF16 output). The fully fused kernel achieves close to this.

4. **SRAM capacity determines granularity**: The tile size `[w, h, k]` must
   keep the working set under ~256 KB/SM. This is the same constraint as
   `fast_memory_capacity` in PROBLEM.md.

5. **L2 cache is the wildcard**: 128 MB L2 can hold activations but not
   weights. Benchmark to measure actual L2 hit rates.

6. **Expert traversal order matters**: Processing experts that share
   token subsets consecutively can keep activations in L2 (analogous to
   `traversal_orders` in PROBLEM.md Example 4).
