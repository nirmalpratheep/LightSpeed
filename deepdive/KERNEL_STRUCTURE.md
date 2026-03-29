# kernel.py — Complete AST Structure & Workload-Specialization Analysis

`solution/triton/kernel.py` (2 284 lines) implements a **Fused Mixture-of-Experts (MoE) forward pass**
for DeepSeek-V3 on B200 GPUs.  Its defining architectural choice is that every
parameter — tile sizes, warp counts, pipeline depth, sort algorithm, reduction
strategy — is chosen *per workload tier* based on `seq_len`.  This document
walks the entire AST, explains each block, and connects the choices back to NCU
profiling data.

---

## Table of Contents

1. [Hardware Detection & Module Constants](#1-hardware-detection--module-constants)
2. [AST Top-Level Structure](#2-ast-top-level-structure)
3. [Section A — PyTorch Routing Functions](#3-section-a--pytorch-routing-functions)
4. [Section B — A100 PyTorch Path](#4-section-b--a100-pytorch-path-kernel_pytorch)
5. [Section C — Triton Kernel Definitions (shared JIT)](#5-section-c--triton-kernel-definitions-shared-jit)
6. [Section D — Fused Routing Launchers](#6-section-d--fused-routing-launchers)
7. [Section E — Tile Map Builders](#7-section-e--tile-map-builders)
8. [Section F — GEMM Launch Helpers](#8-section-f--gemm-launch-helpers)
9. [Section G — Workload Tiers (the core dispatch tree)](#9-section-g--workload-tiers-the-core-dispatch-tree)
10. [Entry Point `kernel()`](#10-entry-point-kernel)
11. [Workload Tier Decision Table](#11-workload-tier-decision-table)
12. [Key Optimization Techniques](#12-key-optimization-techniques)
13. [NCU Profiling Data Mapped to Code](#13-ncu-profiling-data-mapped-to-code)
14. [Where the Bottlenecks Are](#14-where-the-bottlenecks-are)

---

## 1. Hardware Detection & Module Constants

```python
# Lines 30-51
_SM_MAJOR, _SM_MINOR = torch.cuda.get_device_capability()
USE_FP8_TRITON = (_SM_MAJOR >= 10) or (_SM_MAJOR == 9 and FORCE_TRITON_SM90)
```

| Constant | Value | Role |
|---|---|---|
| `NUM_EXPERTS` | 256 | Total global experts |
| `NUM_LOCAL_EXPERTS` | 32 | Experts on this GPU |
| `HIDDEN_SIZE` (H) | 7168 | Token embedding dimension |
| `INTERMEDIATE_SIZE` (I) | 2048 | SwiGLU hidden dim |
| `GEMM1_OUT_SIZE` | 4096 | GEMM1 output = 2×I (gate+up) |
| `BLOCK` | 128 | FP8 quantization block width |
| `TOP_K` | 8 | Experts per token |
| `N_GROUP` | 8 | Expert groups for DeepSeek routing |
| `TOPK_GROUP` | 4 | Groups selected per token |
| `_SEQ_TINY` | 8 | Tier boundary: tiny → small |
| `_SEQ_SMALL` | 128 | Tier boundary: small → medium |
| `_SEQ_MEDIUM` | 2048 | Tier boundary: medium → large |

`USE_FP8_TRITON = True` on B200 (SM 10.0), `False` on A100 (SM 8.0).

---

## 2. AST Top-Level Structure

```
kernel.py
│
├── [Lines 1-51]   Module constants + hardware detection
│
├── [Lines 53-174] SECTION A — Python/PyTorch Routing
│   ├── _routing_core()           shared sigmoid+topk logic
│   ├── routing_tiny()            seq ≤ 8  →  .sort()
│   ├── routing_standard()        seq 9-2048 →  argsort
│   └── routing_large()           seq > 2048 →  bincount
│
├── [Lines 177-236] SECTION B — A100 Path
│   └── kernel_pytorch()          FP32 loop-over-experts
│
└── [Lines 239-2284] if USE_FP8_TRITON:   — B200-only block
    │
    ├── [Lines 242-248]   _NUM_SMS detection
    │
    ├── [Lines 250-433]   SECTION C — Triton JIT Kernels (routing)
    │   ├── @triton.jit _fused_routing_kernel     one CTA/token
    │   └── @triton.jit _counting_sort_kernel     atomic scatter sort
    │
    ├── [Lines 374-694]   SECTION D — Fused Routing Python Launchers
    │   ├── routing_fused()
    │   ├── routing_fused_deferred()
    │   ├── routing_and_tilemaps_onesync()
    │   ├── routing_fused_argsort()
    │   └── routing_fused_large()
    │
    ├── [Lines 696-1041]  SECTION E — Tile Map Builders
    │   ├── @triton.jit _build_tile_map_kernel
    │   ├── build_tile_map_gpu()
    │   ├── build_two_tile_maps_gpu()
    │   └── build_two_tile_maps_diff_bm_gpu()
    │
    ├── [Lines 995-1041]  Assign-Index Helper
    │   ├── @triton.jit _build_assign_idx_kernel
    │   └── _build_assign_idx()
    │
    ├── [Lines 1042-1617] SECTION C2 — Triton JIT Kernels (GEMM + Reduce)
    │   ├── @triton.jit _gemm1_swiglu_kernel         FP8→FP32 act
    │   ├── @triton.jit _gemm1_swiglu_fp8out_kernel  FP8→FP8 act
    │   ├── @triton.jit _gemm2_kernel                FP32 A × FP8→FP32 B
    │   ├── @triton.jit _gemm2_bf16_kernel           BF16 A × BF16 B
    │   ├── @triton.jit _gemm2_fp8_kernel            FP8×FP8 (native)
    │   ├── @triton.jit _gemm2_fp8_perrow_kernel     FP8×FP8 per-row scale
    │   ├── @triton.jit _gemm2_fused_reduce_kernel   GEMM2+reduce fused
    │   ├── @triton.jit _reduce_kernel               scatter-reduce
    │   └── @triton.jit _quantize_to_fp8_rowwise_kernel
    │
    ├── [Lines 1618-1942] SECTION F — Python Launch Helpers
    │   ├── _launch_gemm1()
    │   ├── _launch_gemm1_prefetch_tm2()     stream2 overlap
    │   ├── _launch_gemm2()
    │   ├── _launch_gemm2_with_tm()
    │   ├── _launch_gemm2_bf16_with_tm()
    │   ├── _launch_gemm1_fp8out_with_tm()
    │   ├── _launch_gemm2_fp8_with_tm()
    │   ├── _launch_gemm2_fp8_perrow()
    │   ├── _launch_reduce()
    │   └── _launch_gemm2_fused_reduce()
    │
    ├── [Lines 1944-2214] SECTION G — Workload Tier Functions
    │   ├── kernel_triton_tiny()    seq ≤ 8
    │   ├── kernel_triton_small()   seq 9-128
    │   ├── kernel_triton_medium()  seq 129-2048
    │   └── kernel_triton_large()   seq > 2048
    │
    └── [Lines 2216-2244] Dispatcher: kernel_triton()
        └── seq_len → branch → appropriate tier function

[Lines 2247-2284] ENTRY POINT: kernel()
    ├── USE_FP8_TRITON → kernel_triton()
    └── else           → kernel_pytorch()
```

---

## 3. Section A — PyTorch Routing Functions

**Lines 53–174**

All routing starts with `_routing_core()`, a shared function containing the
DeepSeek-V3 routing algorithm.

### `_routing_core()` (lines 58–102) — shared logic

```
sigmoid(logits) → s_biased  [S, 256]
s_grouped = s_biased.view(S, 8, 32)                # 8 groups × 32 experts/group
top2_vals = topk(s_grouped, k=2)                   # top-2 per group
group_scores = top2_vals.sum(dim=-1)               # [S, 8] group quality
top_groups = topk(group_scores, k=4)               # top-4 groups per token
score_mask = expand top_groups → [S, 256]          # zero non-selected groups
topk_idx = topk(s_biased * mask, k=8)             # top-8 experts per token
topk_weights = normalize(sigmoid scores, 8)        # [S, 8] weights
→ (flat_tids, flat_eids, flat_wts)                 # local expert assignments
```

**Critical optimisation**: Avoids allocating the full `[S, 256]` scored matrix —
computes the normalised weights by gathering only the 8 selected sigmoid values.

### Tier-specific wrappers (lines 105–174)

| Function | Seq range | Sort method | Expert offset computation | Why |
|---|---|---|---|---|
| `routing_tiny` | ≤ 8 | `.sort()` (radix) | `scatter_add` | Single-warp radix faster for ≤64 elements |
| `routing_standard` | 9–2048 | `argsort` | `scatter_add` | General case |
| `routing_large` | > 2048 | `argsort` | `bincount` | Avoids `ones_like` for 100K+ tensors |

These are the *pure-Python fallback* routing paths used only on A100 or when
`USE_FP8_TRITON=False`.

---

## 4. Section B — A100 PyTorch Path: `kernel_pytorch`

**Lines 177–236**

The reference path. Used on A100 (SM 8.0). Also proven correct on B200 and
used as the golden output for numerical validation.

```
routing_standard() → sorted_tids, sorted_eids, sorted_wts, expert_offsets
Bulk dequant:
    hidden_states  fp8 → fp32  [S, 7168]
    W13            fp8 → fp32  [32, 4096, 7168]
    W2             fp8 → fp32  [32, 7168, 2048]
    (with repeat_interleave for scale expansion)
for e in range(32):
    A_e = A[tok_ids]                    # [Tk, 7168]
    G1  = A_e @ W13[e].T               # [Tk, 4096]
    X1, X2 = split(G1, 2048)
    C   = silu(X2) * X1                 # SwiGLU
    O   = C @ W2[e].T                  # [Tk, 7168]
    output += scatter(O * weights)
```

**Performance problem**: The bulk dequantisation allocates `W13` as 32 × 4096 × 7168 × 4 bytes ≈ 15 GB of FP32 weight. This is why it is only a reference path.

---

## 5. Section C — Triton JIT Kernels (shared)

All `@triton.jit` functions are defined once but compiled into multiple
variants at launch time via `tl.constexpr` parameters. Each workload tier calls
them with different `BLOCK_M`, `BLOCK_K`, `num_warps`, `num_stages`.

### `_fused_routing_kernel` (lines 254–346)

**Grid**: `(seq_len,)` — one CTA per token.
**Registers**: 256 experts × float + scratch ≈ many registers; use `num_warps=1` to keep register file small.

```
For each token (CTA):
  Load logits[tok, 0:256]  →  sigmoid  →  score[256]
  Group scoring (static loop N_GROUP=8):
      in_g = mask for group g
      t1 = max(score where in_g)
      t2 = second-max
      group_scores[g] = t1 + max(t2, 0)
  Select top-4 groups (iterative argmax, TOPK_GROUP=4 iterations):
      sel_groups |= argmax(gs_work)
  Mask out non-selected group experts
  Select top-8 experts (iterative argmax, TOP_K=8 iterations):
      eid_arr[k] = argmax(score_pruned); wt_accum[k] = sigmoid[eid_arr[k]]
  Normalise weights: wt_arr = wt_accum / sum(wt_accum) * scaling_factor
  Write local assignments to out_tids/eids/wts (at most 8 per token)
  atomic_add(expert_hist[local_eid], 1)  ← count per expert
  Sentinel: mark remaining slots eid = NUM_LOCAL
```

Key: the histogram is built *inside* the routing kernel via atomics.
No separate reduction step needed.

### `_counting_sort_kernel` (lines 347–372)

**Grid**: `(seq_len,)` — one CTA per token.
Replaces `argsort` for small-to-medium sequences.

```
For each slot in [tok * TOP_K .. (tok+1) * TOP_K]:
    eid = in_eids[slot]
    if eid < NUM_LOCAL:
        pos = atomic_add(expert_ctr[eid], 1)   ← get write index within expert bucket
        off = expert_offsets[eid] + pos
        store sorted_tids/eids/wts at off
```

This is O(T × TOP_K) scatter instead of O(T × TOP_K × log) for argsort.

### GEMM Kernel Family

All GEMM kernels share the same **persistent tile-loop** pattern:

```python
start_pid = tl.program_id(0)
for pid in tl.range(start_pid, total_tiles, NUM_SMS):  # wave-based schedule
    eid, e_start, M_e, pm, pn = tile_map[pid]          # read tile descriptor
    offs_m = pm * BLOCK_M + arange(BLOCK_M)
    offs_n = pn * BLOCK_N + arange(BLOCK_N)
    acc = zeros((BLOCK_M, BLOCK_N))
    for ks in range(0, K, BLOCK_K):
        a_tile = load(...)     # FP8 or FP32
        b_tile = load(...)     # FP8 (always)
        acc += dot(a_tile, b_tile.T) * scale
    store(out[e_start+offs_m, offs_n], acc)
```

#### `_gemm1_swiglu_kernel` (lines 1047–1134)  — Standard path
- Input A: FP8 hidden states
- Input B: FP8 gate+up weights (W1 and W3 packed: `GEMM1_OUT_SIZE = 4096 = 2 × INTER`)
- Output: FP32 activations (after SwiGLU: `(x2 * sigmoid(x2)) * x1`)
- Scale dequant: per-block (128-element) scales for both A and B

#### `_gemm1_swiglu_fp8out_kernel` (lines 1136–1243)  — FP8 pipeline
- Same compute, but **quantises the SwiGLU output to FP8** before storing.
- Writes: `act_fp8 [T, INTER]` + `act_scale [T, INTER/128]` (per-tile amax/448)
- Enables native FP8×FP8 tensor cores in GEMM2 (`_gemm2_fp8_kernel`).

#### `_gemm2_kernel` (lines 1404–1458)  — Standard GEMM2 (deployed)
- Input A: FP32 activations from GEMM1
- Input B: FP8 weights, dequantised to FP32 inside kernel (`b_fp8.to(tl.float32)`)
- Uses **TF32 tensor cores** on B200 (`tl.dot(a_fp32, b_fp32.T)`)
- This is the kernel that runs in production (seen in NCU as `_gemm2_kernel`)

#### `_gemm2_bf16_kernel` (lines 1460–1518)
- Casts A from FP32→BF16, B from FP8→BF16 before `tl.dot`
- B200 BF16 peak: ~9000 TFLOPS vs TF32 2250 TFLOPS → potential 4× if latency limited
- Status: candidate path (BF16 error ~0.78%, tolerance 2%)

#### `_gemm2_fp8_kernel` (lines 1244–1306)
- Native FP8×FP8 tensor cores (SM 10.0 only)
- Both A and B are FP8; A comes from `_gemm1_swiglu_fp8out_kernel`
- Per-128-block A scales (one scale per HALF_N=128 columns per row)

#### `_gemm2_fp8_perrow_kernel` (lines 1340–1402)
- FP8×FP8 but with **per-row A scales** (one scale per token, vs 16 per token for per-block)
- Reduces quantisation error: `acc * A_scale[row] * B_scale[n_blk, k_blk]`
- A scales loaded once per tile (not per K-iteration)

#### `_gemm2_fused_reduce_kernel` (lines 1544–1616)
- Eliminates the separate `g2o` buffer entirely
- After GEMM2 accumulation, immediately `atomic_add` weighted result into output
- Saves 3 kernel launches: `_build_assign_idx`, `_reduce_kernel`, separate g2o allocation

#### `_reduce_kernel` (lines 1520–1542)
- Grid: `(seq_len, ceil(H/BLOCK_H))`
- For each output token, loads up to 8 assignments from the reorder index
- Weighted sum into BF16 output

---

## 6. Section D — Fused Routing Launchers

**Lines 374–694**

These are Python wrappers that orchestrate the Triton routing kernels, optimising
the number of GPU→CPU synchronisation points (`.item()` calls).

| Function | Sync points | Description |
|---|---|---|
| `routing_fused` | 1 | `_fused_routing_kernel` + cumsum + `_counting_sort_kernel`. One `eo[-1].item()` to get T. |
| `routing_fused_deferred` | 0 | Same but returns worst-case-sized arrays. Caller does the sync. |
| `routing_and_tilemaps_onesync` | 1 | Routing + counting sort + **both** tile maps in one `.tolist([T, sum_m1, sum_m2])`. |
| `routing_fused_argsort` | 1 | For large seqs: uses `argsort` on sentinel-padded `out_eids` instead of `_counting_sort_kernel`. Avoids atomic contention at high T. |
| `routing_fused_large` | 1 | For large seqs: counting sort with `num_warps=4` (4× more memory bandwidth for the routing kernel). |

**Why does sync count matter?**
Each `.item()` forces a GPU→CPU transfer and pipeline stall. At seq=1, this
adds ~5–10 μs of host latency on top of the ~13 μs kernel time.

### Design pattern: deferred sync

```python
# routing_fused_deferred returns worst-case array:
stids_full, seids_full, swts_full, eo, total_slots = routing_fused_deferred(...)
# ... launch other GPU work (tile maps, counting_sort) ...
# ONE sync that retrieves T + tile counts simultaneously:
three_vals = torch.stack([eo[-1], m_tiles1.sum(), m_tiles2.sum()]).tolist()
T, sum_m1, sum_m2 = int(three_vals[0]), int(three_vals[1]), int(three_vals[2])
stids = stids_full[:T]   # slice after sync
```

---

## 7. Section E — Tile Map Builders

**Lines 696–993**

The GEMM kernels operate in a **tile-parallel** fashion: each CTA handles one
`(expert, pm, pn)` tile. The tile map is a `[total_tiles, 5]` int64 tensor
where each row is `[expert_id, e_start, M_e, pm, pn]`.

### `_build_tile_map_kernel` (lines 700–737)

GPU kernel that populates the tile map without any CPU involvement.

```
Grid: (total_tiles,)
Each thread:
    tid = program_id(0)
    Binary search tiles_cumsum for expert owning tile tid
        (log₂(32) ≈ 5 iterations, SEARCH_DEPTH=6 for safety)
    pm = local_idx % m_tiles[eid]    ← row tile index
    pn = local_idx // m_tiles[eid]   ← column tile index
    out[tid] = [eid, e_start, M_e, pm, pn]
```

### Python tile map helpers

| Function | Syncs | When used |
|---|---|---|
| `build_tile_map_gpu` | 1 | Single GEMM tile map |
| `build_two_tile_maps_gpu` | 1 | GEMM1 + GEMM2 share same BM → single `m_tiles.sum().item()` |
| `build_two_tile_maps_diff_bm_gpu` | 1 | Different BM → `stack([sum_m1, sum_m2]).tolist()` |

Sharing `m_tiles` when both GEMMs use the same BM halves the sync count for tile maps.

---

## 8. Section F — GEMM Launch Helpers

**Lines 1618–1942**

Python functions that encapsulate the full argument marshaling for each kernel
variant. Each helper corresponds to one combination of:
- Which GEMM kernel (`_gemm2_kernel`, `_gemm2_bf16_kernel`, etc.)
- Whether the tile map is pre-built or needs building
- Grid strategy (`grid_mult=0` non-persistent, `1` = 1×SMS, `2` = 2×SMS)

**`_launch_gemm1_prefetch_tm2`** (lines 1659–1704) — the most complex helper:

```python
# Build GEMM1 tile map on default stream
tm1 = build_tile_map_gpu(eoffs, GEMM1_OUT_SIZE, BM1, BN_G1, device)

# Launch GEMM2 tile-map build on stream2 (runs CONCURRENTLY with GEMM1)
_stream2.wait_stream(current_stream)
with torch.cuda.stream(_stream2):
    tm2 = build_tile_map_gpu(eoffs, HIDDEN_SIZE, BM2, BN2, device)

# Launch GEMM1 on default stream (overlaps with stream2 tile-map work)
_gemm1_swiglu_kernel[...]

# Sync stream2 before GEMM2
current_stream.wait_stream(_stream2)
```

This overlaps the GEMM2 tile-map build with GEMM1 execution, hiding tile-map
latency behind compute.

---

## 9. Section G — Workload Tiers (the core dispatch tree)

**Lines 1944–2244**

This is the heart of the workload-specialised design. Four functions cover the
four regimes that have fundamentally different performance profiles.

### Tier 1 — TINY (`seq ≤ 8`, lines 1944–2011): `kernel_triton_tiny`

**NCU context**: At seq=1, `_gemm1_swiglu_kernel` has 6.88% SM utilisation.
The GPU is mostly idle — there simply are not enough tiles to fill the SMs.

```
Routing   : routing_fused()           [2 Triton kernels + 1 sync]
Tile maps : build_two_tile_maps_diff_bm_gpu(BM1=16, BM2=32)   [1 sync]
GEMM1     : BLOCK_M=16, BLOCK_K=128, HALF_N=128
            num_warps=4, num_stages=2
            grid = total_tiles    ← NON-PERSISTENT (one CTA per tile)
GEMM2     : BLOCK_M=32, BK=128, BN=128
            num_warps=4, num_stages=2, grid_mult=0 (non-persistent)
Reduce    : inline scatter_add (at most 8×7168×4 = 230 KB — trivially small)
```

**Why BLOCK_M=16?**  At seq=1 with 32 experts, each expert sees ≤1 token.
BM=16 means only 15 padding rows per tile instead of 63 (BM=64).
Fewer wasted FLOPs per tile. Also, 2 stages + 4 warps ≈ 120 registers vs 233
for the medium config → potential for 2 CTAs/SM occupancy.

**Why non-persistent?** With total_tiles << NUM_SMS (e.g., grid=48 < 148 SMs),
a persistent grid would create idle CTAs that loop with no work.

### Tier 2 — SMALL (`seq 9–128`, lines 2013–2076): `kernel_triton_small`

```
Routing   : routing_fused()
Tile maps : build_two_tile_maps_gpu(BM=32, BN1=256, BN2=128)   [shared BM → 1 sync]
GEMM1     : BLOCK_M=32, BLOCK_K=128, HALF_N=128
            num_warps=8, num_stages=6
            grid = min(NUM_SMS, total_tiles)    ← 1× persistent
GEMM2     : BLOCK_M=32, BK=64, BN=128
            num_warps=8, num_stages=8, grid_mult=1
Reduce    : inline scatter_add
```

Shared BM=32 for both GEMMs allows a single `m_tiles.sum().item()` call
covering both tile maps (vs two separate syncs).

At seq=32–128 the grid fills ~50–100% of SMs; 1×NUM_SMS persistent gives
each SM exactly one active CTA.

### Tier 3 — MEDIUM (`seq 129–2048`, lines 2078–2143): `kernel_triton_medium`

```
Routing   : routing_fused()
Tile maps : build_two_tile_maps_gpu(BM=64, BN1=256, BN2=128)   [shared BM → 1 sync]
GEMM1     : BLOCK_M=64, BLOCK_K=128, HALF_N=128
            num_warps=8, num_stages=3
            grid = min(2×NUM_SMS, total_tiles)  ← 2× persistent
GEMM2     : BLOCK_M=64, BK=32, BN=128
            num_warps=8, num_stages=8, grid_mult=2
Reduce    : inline scatter_add
```

BM=64 doubles the useful work per tile. 2×NUM_SMS persistent means each SM
runs 2 CTAs to fill the pipeline when tiles > SMs.

BK=32 for GEMM2 with stages=8: deep pipeline (8 stages × 2×(BM×BK + BN×BK) = 8
× 2×(64×32+128×32)×2 bytes ≈ 128 KB smem) is within B200's 232 KB shared
memory limit while hiding the K=2048 reduction latency.

### Tier 4 — LARGE (`seq > 2048`, lines 2145–2214): `kernel_triton_large`

The large tier switches routing algorithm because at T≈14K tokens the counting
sort suffers severe atomic contention (each expert counter is hit ~437 times
simultaneously).

```
Routing   : routing_fused_large()        ← counting sort, num_warps=4
Tile maps : build_two_tile_maps_diff_bm_gpu(BM1=64, BM2=32)    [stacked 1 sync]
GEMM1     : BLOCK_M=64, BLOCK_K=128, HALF_N=128
            num_warps=8, num_stages=3, grid_mult=2
GEMM2     : BLOCK_M=32, BK=64, BN=128
            num_warps=8, num_stages=6, grid_mult=2
Reduce    : _launch_reduce(block_h=256)  ← Triton scatter-reduce kernel
```

**Why BM=32 for GEMM2?**
At large seq, smem is the bottleneck:
- BM=64, BK=128, stages=7 → 224 KB → 1 CTA/SM → 12.5% occupancy
- BM=32, BK=64, stages=6 → 96 KB → 2 CTAs/SM → ~25% occupancy

The exhaustive search found BK=64/stages=6 as the sweet spot (96 KB × 2 = 192 KB < 232 KB limit).

**Why switch reduce method?**
`scatter_add` on `stids.long().unsqueeze(1).expand(T, H)` allocates a T×H×8-byte
index tensor. At T=90K: 90K × 7168 × 8 ≈ 5 GB — out of memory.
The `_reduce_kernel` uses BLOCK_H=256 (28 column blocks vs 56 for BLOCK_H=128),
halving the launch count.

---

## 10. Entry Point `kernel()`

**Lines 2247–2284**

```python
def kernel(routing_logits, routing_bias, hidden_states, hidden_states_scale,
           gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
           local_expert_offset, routed_scaling_factor, output=None, **kwargs):

    # Normalise scalar inputs (handle tensor wrapping from framework)
    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = local_expert_offset.item()

    if output is None:
        output = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, ...)

    if USE_FP8_TRITON:
        return kernel_triton(...)      # B200 path
    else:
        return kernel_pytorch(...)     # A100 reference path
```

`kernel_triton` is the 4-way dispatcher:
```python
def kernel_triton(...):
    seq_len = routing_logits.shape[0]
    if   seq_len <= 8:    return kernel_triton_tiny(...)
    elif seq_len <= 128:  return kernel_triton_small(...)
    elif seq_len <= 2048: return kernel_triton_medium(...)
    else:                 return kernel_triton_large(...)
```

---

## 11. Workload Tier Decision Table

| Axis | Tiny (≤8) | Small (9–128) | Medium (129–2048) | Large (>2048) |
|---|---|---|---|---|
| **Routing kernel** | `_fused_routing_kernel` (warps=1) | same | same | same (warps=4) |
| **Sort method** | `_counting_sort_kernel` | same | same | `_counting_sort_kernel` (or argsort) |
| **GEMM1 BLOCK_M** | 16 | 32 | 64 | 64 |
| **GEMM1 warps** | 4 | 8 | 8 | 8 |
| **GEMM1 stages** | 2 | 6 | 3 | 3 |
| **GEMM1 grid** | non-persistent | 1×SMS | 2×SMS | 2×SMS |
| **GEMM2 BLOCK_M** | 32 | 32 | 64 | 32 |
| **GEMM2 BK** | 128 | 64 | 32 | 64 |
| **GEMM2 stages** | 2 | 8 | 8 | 6 |
| **GEMM2 grid** | non-persistent | 1×SMS | 2×SMS | 2×SMS |
| **Tile map syncs** | 1 (diff BM) | 1 (shared BM) | 1 (shared BM) | 1 (diff BM) |
| **Reduce method** | `scatter_add` inline | same | same | `_reduce_kernel` BLOCK_H=256 |
| **Routing variant** | `routing_fused` | same | same | `routing_fused_large` |

---

## 12. Key Optimization Techniques

### 12.1 Minimising GPU→CPU Syncs

Every `.item()` / `.tolist()` stalls the GPU pipeline. The kernel minimises
these with two techniques:

**Batched sync via `torch.stack(...).tolist()`:**
```python
# One transfer returns T + both tile-map totals
three = torch.stack([eo[-1].to(torch.int64),
                     m_tiles1.sum(),
                     m_tiles2.sum()]).tolist()
T, sum_m1, sum_m2 = int(three[0]), int(three[1]), int(three[2])
```

**Deferred sync with worst-case allocation:**
```python
# Allocate seq_len × TOP_K slots (worst case)
out_tids = torch.empty(seq_len * TOP_K, ...)
_fused_routing_kernel[...]  # fills what it needs, uses sentinels for rest
# Launch counting_sort with worst-case output too
sorted_tids_full = torch.empty(seq_len * TOP_K, ...)
_counting_sort_kernel[...]
# Now sync once to get T, then slice
T = int(eo[-1].item())
stids = sorted_tids_full[:T]
```

### 12.2 Persistent Kernel Pattern

GEMM kernels loop over tiles instead of one CTA per tile:
```python
start_pid = tl.program_id(0)
for pid in tl.range(start_pid, total_tiles, NUM_SMS):  # stride = NUM_SMS
    # ... process tile pid ...
```

Benefits:
- Grid size = `min(NUM_SMS * mult, total_tiles)` — never launches more CTAs than SMs
- Avoids kernel re-launch overhead for each wave
- CTA stays resident, keeping L1 cache warm across iterations

### 12.3 Tile Map as Data Structure

Instead of a loop over experts in the GEMM kernel body, the tile map
pre-computes all `(expert, pm, pn)` combinations on the GPU:
```
tile_map[tile_id] = [eid, e_start, M_e, pm, pn]
```

The GEMM CTA just reads this 5-element descriptor and knows exactly which
rows/cols to compute. Expert dispatch is entirely data-driven — no branching
on expert ID inside the kernel.

### 12.4 FP8 Precision Pipeline

GEMM1 always reads FP8 input (hidden states + weights). Standard path outputs FP32.
Optional path (`_gemm1_swiglu_fp8out_kernel`) quantises the SwiGLU output
back to FP8 using per-row amax:
```python
amax = tl.max(tl.abs(act), axis=1)   # [BLOCK_M] per-row max
scale = amax / 448.0
act_fp8 = (act / scale[:, None]).to(tl.float8e4nv)
```
This enables `_gemm2_fp8_kernel` to use native FP8×FP8 tensor cores (SM10 only).

### 12.5 SwiGLU Fused in GEMM1

```python
# acc_x1 = gate,  acc_x2 = up  (both computed in the same K loop)
act = (acc_x2 * tl.sigmoid(acc_x2)) * acc_x1
```
Computing both halves of W13 in the same kernel avoids a second pass over
the 7168-element K dimension.

### 12.6 Fused GEMM2 + Reduce

`_gemm2_fused_reduce_kernel` eliminates the intermediate `g2o` buffer:
```python
# After accumulation inside GEMM2 tile:
tids = sorted_tids[e_start + offs_m]    # token ids for this tile's rows
wts  = sorted_wts[e_start + offs_m]
row_ptrs = out_fp32 + tids[:, None] * stride_o + on[None, :]
tl.atomic_add(row_ptrs, acc * wts[:, None])   # direct scatter into output
```

Saves: `T × 7168 × 4` bytes of FP32 buffer + 3 kernel launches.
Trade-off: atomic contention when multiple tiles write to the same token row.
This is acceptable for small/medium seq where each token has ≤8 experts.

### 12.7 Secondary CUDA Stream for Tile Map Prefetch

```python
_stream2 = torch.cuda.Stream()

# In _launch_gemm1_prefetch_tm2:
_stream2.wait_stream(cur_stream)         # stream2 starts after routing
with torch.cuda.stream(_stream2):
    tm2 = build_tile_map_gpu(...)        # GEMM2 tile map on stream2
# GEMM1 runs on default stream concurrently with tm2 build
current_stream.wait_stream(_stream2)     # wait before launching GEMM2
```

---

## 13. NCU Profiling Data Mapped to Code

The following table maps NCU observations from `kernel_wl01_seq1.ncu-rep`
to the specific code paths executed for `seq_len=1`:

| NCU Kernel | Duration | SM% | DRAM% | Code Location | Notes |
|---|---|---|---|---|---|
| `void at::vectorized_elementwise_kernel<4, FillFunctor<int>>` | 4.86 μs | 0.01 | 0.01 | output buffer `torch.zeros(...)` | Launch overhead dominates |
| `_fused_routing_kernel` | **13.70 μs** | 0.03 | 0.04 | `routing_fused()` → `_fused_routing_kernel[(1,)](...)` | 1 CTA for 1 token |
| `DeviceScanInitKernel` + `DeviceScanKernel` | 11.58 μs | 0.01 | 0.06 | `expert_hist.cumsum(0)` inside `routing_fused` | CUB scan overhead |
| `_counting_sort_kernel` | **12.99 μs** | 0.00 | 0.01 | `routing_fused()` → `_counting_sort_kernel[(1,)](...)` | 1 CTA, trivial work |
| Misc copy/elementwise (×15) | ~100 μs | <0.1 | <0.1 | Various `torch.zeros`, `cumsum`, slice ops | **Root cause of routing overhead** |
| `_build_tile_map_kernel` × 2 | **14.63 μs** | 0.27 | 0.01 | `build_two_tile_maps_diff_bm_gpu()` | Binary search for 48+168 tiles |
| `_gemm1_swiglu_kernel` | **121.31 μs** | 6.88 | 11.25 | `kernel_triton_tiny` GEMM1 launch | 6% utilisation — too small |
| `_gemm2_kernel` | **55.94 μs** | 30.01 | 11.88 | `kernel_triton_tiny` GEMM2 launch | Better but still small |
| `_scatter_gather_elementwise_kernel` | **14.21 μs** | 0.48 | 0.18 | `scatter_add_` in reduce | 1 token — trivial |
| `bfloat16_copy_kernel` | 5.95 μs | 0.04 | 0.09 | `output.copy_(out_fp32)` | dtype cast |

**Total GPU time (seq=1): ~411 μs**

---

## 14. Where the Bottlenecks Are

### At seq=1 (Tiny tier)

```
Routing overhead   183 μs  ████████████████████████████████  44.5%  [LAUNCH OVERHEAD]
GEMM1+SwiGLU       121 μs  █████████████████████            29.4%  [COMPUTE — 6.9% SM]
GEMM2               56 μs  █████████                        13.6%  [COMPUTE — 30% SM]
Tile maps           15 μs  ██                                3.6%
Scatter-reduce      14 μs  ██                                3.4%
Misc               22 μs  ███                                5.4%
```

**Problem 1 — Routing launch explosion (44.5% of time)**: The `routing_fused`
path requires several PyTorch GPU operations after the Triton kernel
(`expert_hist.cumsum`, `zeros`, slicing). Each is a separate kernel launch
with ~5 μs minimum overhead. At seq=1 there are 28 kernels before GEMM1 even starts.
**Fix direction**: collapse more operations into the fused routing kernel or
combine the `eo[-1].item()` sync with tile-map syncs.

**Problem 2 — GEMM1 severely underutilised (6.9% SM)**: With seq=1 and 32 experts,
at most 1 token per expert, the total tile count is tiny (≤48 tiles for BM=16).
The GPU has 148 SMs but only a handful are active.
**Fix direction**: Further reduce BLOCK_M (e.g., BM=8) to create more tiles, or
batch multiple MoE calls together.

**Problem 3 — GEMM2 better but still latency bound (30% SM)**: 168 tiles for
BM=32 × GEMM2 gives better SM coverage but still far below peak.

### At seq=901 (Medium tier, from workload table)

Expected NCU picture (not profiled here but inferred):
- Routing: ~20 μs (fewer sync artifacts at larger T)
- GEMM1: ~1000 μs (growing linearly with T × expert count)
- GEMM2: ~500 μs
- SM utilisation: 70–90% (near saturation)
- **Bottleneck shifts to compute** — this is the designed operating point.

### At seq=14107 (Large tier)

- Routing itself takes significant time (counting sort on 14K×8 = 113K assignments)
- `routing_fused_large` uses `num_warps=4` to improve bandwidth for the routing kernel
- Tile maps: 14K tokens × 64BM tiles = ~7K tiles per expert → GPU well saturated
- Reduce uses `_reduce_kernel` (scatter_add would OOM at ~5 GB index tensor)
- **Bottleneck: GEMM compute** — both GEMMs run at high SM utilisation

### Summary

| Seq range | Dominant cost | Primary bottleneck |
|---|---|---|
| ≤ 8 | Kernel launch overhead + routing | `_routing_core` / sync count |
| 9–128 | GEMM latency (small tiles) | Underutilised SM due to small BM |
| 129–2048 | GEMM compute | Near-ideal — best operating point |
| > 2048 | GEMM compute + reduce | Reduce kernel (cannot use scatter_add) |
