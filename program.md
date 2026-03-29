# program.md — LightSpeed MoE Kernel Auto-Optimizer

Autonomous optimization loop for `solution/triton/kernel.py` targeting B200 GPU.
Modelled after karpathy/autoresearch: modify → run → log → keep/revert. Never stop.

## Script cheat-sheet

| Script | When to run | Cost |
|--------|-------------|------|
| `modal run scripts/run_modal_limited.py --note "..."` | **After every commit** — benchmark 4 representative workloads, append to `reflections.md` | ~3 min |
| `modal run scripts/run_profiling.py` | **After every commit** — refresh NCU `.ncu-rep` files for reasoning | ~10 min |
| `modal run scripts/run_modal.py` | **Final validation only** — all 18 workloads for submission | ~15 min |

The optimization loop uses **run_modal_limited** (not run_modal) to keep modal cost low.

## Dashboard

Open `dashboard.html` in a browser (file:// is fine — no server needed).

- **Live-reloads every 60 s** from `reflections.md` and `results.tsv` (same directory).
- Shows: per-tier speedup tiles, speedup trend chart, latency trend chart, NCU metrics table, experiment history.
- Update the `NCU_DATA` array inside `dashboard.html` after each `run_profiling` run to keep hardware metrics current.

---

## 0. Mandatory Background Reading (do once at startup)

Read these files before touching anything:

| File | What to extract |
|---|---|
| `solution/triton/kernel.py` | Current tier configs (BM, BK, stages, grid_mult, precision path) |
| `deepdive/KERNEL_STRUCTURE.md` | NCU-derived bottleneck map for all 4 tiers |
| `NCU/mar27/kernel_wl0*.ncu-rep` | Raw hardware metrics (extract with ncu CLI) |
| `scripts/run_modal.py` | How benchmark runs, what it returns |
| `scripts/run_profiling.py` | The 4 representative workloads and their seq_len values |
| `mlsys26-contest/definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json` | Ground-truth reference implementation and tolerance spec |

Key constants to internalize:
```
NUM_SMS        = 148          # B200 SMs
SMEM_LIMIT     = 232_768      # bytes per SM (232 KB)
HBM_BW         = 6347         # GB/s
FP8_TFLOPS     = 3070         # TFLOPS peak
TF32_TFLOPS    = 2250         # TFLOPS peak
CONTEST_ATOL   = 1.0
CONTEST_RTOL   = 0.3
MATCH_RATIO    = 0.9          # 90% of elements must pass
```

---

## 1. Setup (run once)

```bash
# 1. Branch
git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')
```

> **Baseline is already recorded — do NOT re-run it.**
> - `results.tsv` row 0 = V6 results from README (commit `c150589`)
> - `reflections.md` first entry = V6 run with full NCU/mar27 metrics
> - NCU profiles: `NCU/mar27/kernel_wl01_seq1.ncu-rep` through `kernel_wl08_seq14107.ncu-rep`
>
> Start directly at Round 1 (first optimization attempt).

### Logging: reflections.md (primary) + results.tsv (compact)

**reflections.md** is auto-appended by `run_modal_limited.py` after every run.
Each entry includes the 4-row result table plus fields you fill in manually:
- "What worked / what didn't"
- "Hypothesis for next run"

**results.tsv** — one row per decision (including skips), tab-separated:
```
round  tier    seq_len  speedup  sm_pct_gemm2  max_rel_err  status              description
0      tiny    1        1.00     30.01         0.001        baseline            FP32 GEMM2
0      small   80       1.00     ?             0.001        baseline            FP32 GEMM2
0      medium  901      1.00     40.43         0.001        baseline            FP32 GEMM2
0      large   14107    1.00     49.01         0.001        baseline            FP32 GEMM2
```

`status` values: `baseline`, `keep`, `discard`, `crash`, `correctness_fail`, `skip`

---

## 2. Tier Priority Order

Always tackle tiers from **highest potential gain to lowest**. Re-evaluate after each change.

```
Priority 1 → LARGE   (seq=14107)  GEMM2 = 6195 μs, 49% SM — biggest wall-clock gain
Priority 2 → MEDIUM  (seq=901)    GEMM2 = 629 μs,  40% SM
Priority 3 → SMALL   (seq=80)     routing + GEMM both suboptimal
Priority 4 → TINY    (seq=1)      routing overhead 44% — hard to fix without redesign
```

---

## 3. Reasoning Protocol (MANDATORY before every modal run)

**Modal runs cost money and time. Never run speculatively.**
Complete ALL of the following reasoning steps before calling `run_modal.py`:

### 3.1 Roofline Classification

For each affected kernel, compute:
```
arithmetic_intensity = total_FLOPS / total_HBM_bytes

ridge_point_fp8   = FP8_TFLOPS  × 10^12 / (HBM_BW × 10^9) = 484 FLOPS/byte
ridge_point_tf32  = TF32_TFLOPS × 10^12 / (HBM_BW × 10^9) = 354 FLOPS/byte

if intensity > ridge_point  → compute-bound  (fix: higher precision ops, more FLOPS/cycle)
if intensity < ridge_point  → memory-bound   (fix: better tiling, reduce traffic, more stages)
```

For GEMM2 (BM=32, BN=128, BK=128, FP8×FP8):
```
FLOPS_per_tile = 2 × BM × BN × BK = 2 × 32 × 128 × 128 = 1,048,576
mem_per_tile   = BM×BK×1 + BN×BK×1 = (32+128)×128 = 20,480 bytes
intensity      ≈ 51 FLOPS/byte  →  MEMORY-BOUND (well below 484 ridge)
```

### 3.2 Working Set Verification

Before committing any tile config change:
```python
def smem_per_stage(BM, BN, BK, dtype_A_bytes, dtype_B_bytes):
    return BM * BK * dtype_A_bytes + BN * BK * dtype_B_bytes

def max_stages_for_n_ctas(BM, BN, BK, dA, dB, n_ctas=2):
    per_stage = smem_per_stage(BM, BN, BK, dA, dB)
    return SMEM_LIMIT // (per_stage * n_ctas)

# Example: GEMM2 FP8×FP8, BM=32, BK=128, want 2 CTAs/SM:
per_stage = (32+128)*128*1 = 20,480 bytes
max_stages = 232,768 // (20,480 * 2) = 5  ← hard ceiling
```

If proposed stages × smem_per_stage > SMEM_LIMIT / n_ctas → reduce stages or n_ctas.

### 3.3 Expected Speedup Estimation

Estimate gain before running. Only run if estimated gain ≥ 5%.

```
Memory-bound speedup model:
  speedup ≈ (old_HBM_bytes) / (new_HBM_bytes)
  + bonus for better arithmetic_intensity (≈ 5-15% from pipeline utilization)

Compute-bound speedup model:
  speedup ≈ new_TFLOPS / old_TFLOPS  (e.g. FP8/TF32 = 3070/2250 = 1.36×)

Register pressure model:
  occupancy ≈ floor(256 / regs_per_thread)   # B200: 256 regs per warp slot
  CTAs/SM ≈ min(SMEM_LIMIT/smem_per_CTA, occupancy_limit)
```

### 3.4 Correctness Pre-check

Before running modal, reason about numerical error:
```
FP8 E4M3 epsilon ≈ 2^-3 = 0.125 per mantissa bit
Per-block quantization error ≈ eps_fp8 / sqrt(block_size) ≈ 0.002
Accumulated over K: error ≈ eps_fp8 × sqrt(K) ≈ 0.002 × sqrt(2048) ≈ 0.09 (9%)
Contest rtol=0.30 → passes if error < 30%
Contest atol=1.0  → passes if |error| < 1.0 in absolute BF16 units
```

If estimated error > 25% relative, do NOT run — revert to safer quantization.

### 3.5 Decision Gate

```
Run modal IF:
  estimated_speedup  >= 1.05   (≥5% gain)
  AND correctness_ok == True   (error << rtol)
  AND smem_fits      == True   (working set verified)
  AND NOT similar_to_a_previous_discard

Do NOT run if:
  estimated_speedup  < 1.05   (marginal gain, not worth modal cost)
  OR you changed only comments/docstrings
  OR the last 3 attempts on this tier all discarded
```

---

## 4. Strategy Catalog (apply in this order per tier)

Drawn from MLSys scheduling principles (subgraph granularity, working set, memory hierarchy) and KernelAgent optimization patterns (roofline, pipeline depth, precision).

### Strategy A — FP8×FP8 GEMM2 (already applied to medium+large)
**What**: Wire `_gemm1_swiglu_fp8out_kernel` → `_gemm2_fp8_kernel`
**When**: All tiers where GEMM2 exists and SwiGLU activations are computed
**Gain model**: A-matrix traffic reduces 4×; FP8 WGMMA is 1.36× faster if compute-bound
**Smem check**: `(BM+BN) × BK × 1 byte × stages × n_ctas ≤ SMEM_LIMIT`
**Risk**: ~9% quantization error; safe at rtol=0.30
**Status**: ✅ Applied to medium (BK=128, s=4) and large (BK=128, s=5)

---

### Strategy B — Pipeline Depth Tuning (stages)
**What**: Increase `num_stages` for GEMM kernels to hide HBM latency
**Why**: At L1 hit rate <5%, HBM round-trip is ~400-600 cycles. More stages = more in-flight requests.
**Formula**: ideal_stages ≈ HBM_latency_cycles / MMA_cycles_per_tile
**Example for GEMM2 large (BM=32, BK=128, FP8)**:
```
HBM latency       ≈ 500 cycles
MMA throughput    ≈ BM×BN×BK / FP8_ops_per_cycle ≈ 32×128×128 / (3070T/148SMs) ≈ 5 cycles per tile
ideal_stages      ≈ 500 / 5 = 100 → capped by smem: max 5 stages at 2 CTAs/SM
```
**Action**: Push stages to the smem ceiling. For each tier:
- tiny: GEMM1 s=2→4 (smem: (16+128)×128×1×4=73.7KB < 232KB ✓)
- small: GEMM2 s=8 already at smem ceiling for BK=64; switch to FP8+s=8 (160KB, 1 CTA/SM)
- medium: GEMM2 FP8 s=4 → try s=5 (120KB, 2 CTAs need 240KB > 232KB ✗ → stay at s=4)
- large: GEMM2 FP8 s=5 is ceiling; try GEMM1 s=3→4

---

### Strategy C — Block Size Sweep (BM / BK)
**What**: Sweep BLOCK_M and BLOCK_K to find better arithmetic intensity
**When**: NCU shows SM% < 60% and warp_active% < 30%
**Sweep space** (do in reasoning, not by running each):
```
For GEMM2 medium (FP8, BN=128):
  BM ∈ {16, 32, 64, 128}
  BK ∈ {64, 128}  (must equal FP8_BLK=128 for correct scale indexing)
  → BK is fixed at 128 for FP8 path; only BM varies

For GEMM1 (outputs FP8, HALF_N=128):
  BM ∈ {16, 32, 64, 128}
  BK = 128 (fixed: one FP8 quantization block per K-iter)
```
**Decision rule**: pick BM that maximises `(2×BM×BN×BK) / ((BM+BN)×BK×dtype_bytes)` while fitting smem

---

### Strategy D — Grid Multiplier Tuning
**What**: Adjust `grid_mult` (0=non-persistent, 1=1×SMS, 2=2×SMS)
**When**: total_tiles >> NUM_SMS × grid_mult → some SMs sit idle
**Rule**:
```
if total_tiles < NUM_SMS * 0.5  → use grid_mult=0 (non-persistent)
if total_tiles < NUM_SMS * 1.5  → use grid_mult=1 (1× persistent)
if total_tiles >= NUM_SMS * 1.5 → use grid_mult=2 (2× persistent)
```
**Verify** against NCU Grid Size column: grid should ≈ min(NUM_SMS × mult, total_tiles)

---

### Strategy E — Fused GEMM2 + Reduce
**What**: Use `_gemm2_fused_reduce_kernel` instead of separate GEMM2 + `_reduce_kernel`
**Eliminates**: `g2o` tensor (T×H×4 bytes FP32), `_build_assign_idx`, `_reduce_kernel` launches
**When**: small+medium tiers (scatter_add used; atomic contention is low for few experts/token)
**Risk**: Atomic contention on output tokens shared by multiple experts. For 8 experts/token:
```
# Each token written by up to 8 tiles (one per expert)
# But each tile covers a different N-range (BN=128 of H=7168)
# → No N-range collision between concurrent tiles of the same token
# → Contention only if 2 tiles for same token AND same pn → impossible (1 tile per (expert,pm,pn))
```
**Gain estimate**: Saves T×H×4 bytes of DRAM traffic (g2o buffer) + 3 kernel launches
- medium (T≈7K): saves 7K×7168×4 = 200MB of g2o traffic → significant!
- small (T≈640): saves 18MB

---

### Strategy F — Routing Kernel Fusion (tiny tier)
**What**: Collapse routing overhead from 28 launches to ≤5
**Problem**: At seq=1, routing overhead = 183μs = 44% of total time
**Current bottleneck chain**:
```
_fused_routing_kernel       (1 launch)
expert_hist.cumsum(0)        → 2 CUB launches (Init+Scan)
torch.zeros × 3              → 3 allocation + 3 fill kernels
eo[-1].item()                → GPU→CPU sync stall
_counting_sort_kernel        (1 launch)
build_two_tile_maps_diff_bm  → 2 _build_tile_map_kernel + 2 more syncs
= 12+ launches + 2 syncs
```
**Fix direction**: Use `routing_and_tilemaps_onesync()` (already implemented, line 473) which does:
```
routing kernel → cumsum → counting_sort → BOTH tile maps → ONE .tolist() sync
= 5 launches + 1 sync
```
**Gain estimate**: Replace 2 syncs with 1 = ~10-15μs saved; fewer kernel launches = ~30-50μs → total ~25-35% routing speedup

---

### Strategy G — Small-Tier FP8 GEMM2
**What**: Apply FP8×FP8 to small tier GEMM2 (seq=80)
**Config**: `_launch_gemm1_fp8out_with_tm(BM=32, BK=128, BN_G1=256, warps=8, stages=3)` + `_launch_gemm2_fp8_with_tm(BM=32, BK=128, warps=8, stages=8, grid_mult=1)`
**Smem check**: (32+128)×128×1×8 = 163,840 bytes = 160KB < 232KB at 1 CTA/SM ✓
**Note**: `_launch_gemm1_fp8out_with_tm` hardcodes `grid1 = min(_NUM_SMS * 2, tm1.shape[0])` → overshoot for small tier (should be 1×). Either fix the helper or inline the launch.

---

### Strategy H — GEMM1 Warp Count Tuning
**What**: Try `num_warps=4` vs `num_warps=8` for GEMM1 when register pressure is high
**Why**: GEMM1 with 2 accumulators (x1, x2) uses ~233 registers → limits to 1 warp group/SM
  Reducing to warps=4 may free registers → higher occupancy → better HBM latency hiding
**Check**: NCU → Launch Statistics → "Registers Per Thread" must drop below threshold

---

## 5. The Optimization Loop

```
LOOP FOREVER (per tier, cycling Priority 1→2→3→4→1→...):

  FOR each tier in [large, medium, small, tiny]:

    1. READ current kernel.py config for this tier
    2. READ latest NCU data (or use cached values from results.tsv)
    3. APPLY Reasoning Protocol (Section 3):
         - Classify bottleneck (memory/compute/launch-overhead)
         - Check smem working set
         - Estimate speedup ≥ 5%?
         - Check correctness safety
         - Is this similar to a previous discard? → skip
    4. IF gate passes:
         a. SELECT next strategy from catalog (Section 4)
         b. MODIFY solution/triton/kernel.py
         c. GIT COMMIT with descriptive message
         d. RUN benchmark:
              modal run scripts/run_modal_limited.py --note "<what changed>"
              (4 workloads: wl01/seq1, wl03/seq80, wl04/seq901, wl08/seq14107)
         e. RUN NCU profile (for next reasoning cycle):
              modal run scripts/run_profiling.py
              (runs in background — can reason while it profiles)
         f. READ reflections.md — parse the new table row
         g. LOG to results.tsv
         h. KEEP or REVERT:
              keep    if ALL tiers pass correctness AND speedup improves for target tier
                      AND no other tier regresses by > 5%
              discard if any tier fails correctness (max_rel_err > RTOL or match_ratio < 0.9)
                      OR if speedup < 1.0 on the target tier
              → on discard: git revert, update results.tsv + reflections note
         i. FILL in reflections.md: "What worked" + "Hypothesis for next run"
    5. ELSE (gate failed):
         - Log reasoning to results.tsv with status = "skip"
         - Move to next strategy or next tier

  STOP ONLY IF interrupted manually.
```

---

## 6. Workload-to-Modal-Run Mapping

The contest runs all workloads. For this optimization loop, use the 4 representative ones:

| Tier | wl index | seq_len | Primary kernel | Bottleneck |
|---|---|---|---|---|
| tiny   | wl01 | 1     | routing + GEMM1/2 | launch overhead (44%) |
| small  | wl03 | 80    | GEMM1 + GEMM2    | small tiles, partial fill |
| medium | wl04 | 901   | GEMM2 (629 μs)   | memory-bound, 40% SM |
| large  | wl08 | 14107 | GEMM2 (6195 μs)  | memory-bound, 49% SM |

**Running only target tier**: `run_modal.py` runs all 4 by default. You cannot skip tiers — treat every run as a 4-tier evaluation. This is why the gate (≥5% gain) is critical: each run validates all 4.

---

## 7. NCU Metric Extraction (for reasoning without a new profiling run)

Use the local NCU binary to extract metrics from saved `.ncu-rep` files:

```bash
NCU="/c/Program Files/NVIDIA Corporation/Nsight Compute 2025.4.1/target/windows-desktop-win7-x64/ncu.exe"

# Get key metrics for all kernels in a report
"$NCU" --import NCU/mar27/kernel_wl04_seq901.ncu-rep \
  --csv --page raw \
  --metrics "gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct" \
  | grep -E "_gemm|_reduce|_fused_routing|_counting"
```

**Key metrics to extract per kernel:**

| Metric | What it means | Target |
|---|---|---|
| `gpu__time_duration.sum` (μs) | Wall-clock kernel time | minimize |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM% (compute util) | maximize, ≥85% = excellent |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth % | ≥85% = memory-bound |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Warp occupancy % | ≥50% = good |
| `l1tex__t_sector_hit_rate.pct` | L1 cache hit rate | ≥50% = good |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | Tensor core util | ≥80% = ideal |

**Current baseline values** (from NCU/mar27, before FP8 change):

| Tier | Kernel | Duration | SM% | L1% | Warp% |
|---|---|---|---|---|---|
| tiny   | `_gemm1_swiglu_kernel` | 121 μs  | 6.9%  | ~4% | ~13% |
| tiny   | `_gemm2_kernel`        | 56 μs   | 30.0% | ~3% | ~24% |
| medium | `_gemm1_swiglu_kernel` | 403 μs  | 31.2% | 3.8%| 12.5%|
| medium | `_gemm2_kernel`        | 629 μs  | 40.4% | 2.9%| 24.2%|
| large  | `_gemm1_swiglu_kernel` | 2652 μs | 34.8% | 5.0%| 12.6%|
| large  | `_gemm2_kernel`        | 6195 μs | 49.0% | 1.7%| 22.2%|
| large  | `_reduce_kernel`       | 596 μs  | 71.8% | 19.8%|90.3%|

---

## 8. Kernel Architecture Quick Reference

The kernel dispatch tree for B200 path (`USE_FP8_TRITON=True`):

```
kernel()
└── kernel_triton(seq_len)
    ├── seq ≤ 8    → kernel_triton_tiny()
    │   routing:  routing_fused() [2 Triton + 1 sync]
    │   tile maps: build_two_tile_maps_diff_bm_gpu(BM1=16, BM2=32) [1 sync]
    │   GEMM1:    _gemm1_swiglu_kernel (BM=16, BK=128, s=2, warps=4, grid=tiles)
    │   GEMM2:    _gemm2_kernel (BM=32, BK=128, s=2, warps=4, FP32×FP8→FP32)
    │   reduce:   scatter_add inline
    │
    ├── seq ≤ 128  → kernel_triton_small()
    │   routing:  routing_fused()
    │   tile maps: build_two_tile_maps_gpu(BM=32) [1 sync]
    │   GEMM1:    _gemm1_swiglu_kernel (BM=32, BK=128, s=6, warps=8, grid=1×SMS)
    │   GEMM2:    _gemm2_kernel (BM=32, BK=64, s=8, warps=8, FP32×FP8→FP32)
    │   reduce:   scatter_add inline
    │
    ├── seq ≤ 2048 → kernel_triton_medium()
    │   routing:  routing_fused()
    │   tile maps: build_two_tile_maps_gpu(BM=64) [1 sync]
    │   GEMM1:    _gemm1_swiglu_fp8out_kernel (BM=64, BK=128, s=3, warps=8, grid=2×SMS)  ← NEW
    │   GEMM2:    _gemm2_fp8_kernel (BM=64, BK=128, s=4, warps=8, FP8×FP8→FP32)          ← NEW
    │   reduce:   scatter_add inline
    │
    └── seq > 2048 → kernel_triton_large()
        routing:  routing_fused_large() [counting sort, warps=4]
        tile maps: build_two_tile_maps_diff_bm_gpu(BM1=64, BM2=32) [1 sync]
        GEMM1:    _gemm1_swiglu_fp8out_kernel (BM=64, BK=128, s=3, warps=8, grid=2×SMS)  ← NEW
        GEMM2:    _gemm2_fp8_kernel (BM=32, BK=128, s=5, warps=8, FP8×FP8→FP32)          ← NEW
        reduce:   _launch_reduce(block_h=256)
```

Available but not currently used in dispatch:
- `_gemm2_bf16_kernel` — BF16×BF16, useful if FP8 accuracy fails
- `_gemm2_fused_reduce_kernel` — GEMM2 + atomic_add reduce fused (Strategy E)
- `_gemm2_fp8_perrow_kernel` — FP8 with per-row A scale (more accurate than per-block)
- `routing_and_tilemaps_onesync()` — routing+both tilemaps with 1 sync (Strategy F)

---

## 9. Logging Format

Two logs are maintained in parallel:

### reflections.md (auto-written by run_modal_limited.py)

After each `modal run scripts/run_modal_limited.py` the script appends a block:
```markdown
## 2026-03-28 14:35 UTC  `a3f9c12`
**Note**: FP8 GEMM2 medium+large

| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |
|------|------|----:|--------|--------:|--------:|--------:|--------:|
| e05c6c03 | tiny   |     1 | PASSED |   0.562 |  20.92x | 4.10e+03 | 1.88e+01 |
| 8f1ff9f1 | small  |    80 | PASSED |   0.862 |  18.42x | 4.10e+03 | 8.20e+02 |
| 1a4c6ba1 | medium |   901 | PASSED |   1.096 |  19.26x | 4.10e+05 | 4.10e+13 |
| 5e8dc11c | large  | 14107 | PASSED |   6.656 |   6.77x | 5.82e+05 | 5.37e+13 |

**Summary**: 4/4 PASSED | avg speedup **16.34x**

**What worked / what didn't**: _[fill in]_

**Hypothesis for next run**: _[fill in]_
```

Fill in the last two fields manually before starting the next round.

### results.tsv (manual, one row per decision)

```
round  tier    seq_len  speedup  sm_pct_gemm2  max_rel_err  status              description
1      large   14107    1.00     49.0          0.001        baseline            FP32 GEMM2
2      medium  901      1.00     40.4          0.001        baseline            FP32 GEMM2
3      large   14107    ?        ?             ?            keep/discard/crash  FP8 GEMM2 BM=32 BK=128 s=5
4      small   80       -        -             -            skip                estimated gain 3% < 5% threshold
```

---

## 10. Stopping Criteria and Success Definition

**Success per tier**: speedup ≥ 1.3× vs baseline AND sm_pct ≥ 60% AND max_rel_err ≤ RTOL
**Overall success**: all 4 tiers at success threshold
**Convergence**: 3 consecutive discards on the same tier → deprioritize, rotate to next tier
**Manual stop**: interrupt at any time; always in a clean git state (either committed or reverted)

---

## 11. First Run Sequence

> **Round 0 is done.** Baseline = V6 (commit `c150589`), recorded in `results.tsv` and `reflections.md`.
> NCU profiles are at `NCU/mar27/`. Start at Round 1.

```bash
# Round 1 — First optimization (recommended: large GEMM2 BM=32→64, stages=4)
#   Reasoning: large tier 6.77x << 19x+ for other tiers; GEMM2 at only 49% SM
#   smem check: (64+128)×128×1×4×2 = 192KB < 232KB → 2 CTAs/SM ✓
#   Expected gain: ≥10% from better arithmetic intensity (51→89 FLOPS/byte)
git commit -m "perf: large GEMM2 BM=32->64 stages=5->4 for higher arithmetic intensity"
modal run scripts/run_modal_limited.py --note "large GEMM2 BM=64 stages=4"
modal run scripts/run_profiling.py

# Round 2+ — Continue optimization loop (see Section 5)
# → Always: modify → commit → run_modal_limited → run_profiling → read reflections → next

# Final — Full validation before submission
modal run scripts/run_modal.py
# → All 18+ workloads; every row must show PASSED
```

---

## 12. Example Reasoning Trace (template to follow)

```
=== Round 3: Target = LARGE tier (seq=14107) ===

BOTTLENECK ANALYSIS:
  - _gemm2_fp8_kernel: 49% SM, 1.7% L1 hit → HBM-latency bound
  - Current smem: 20KB/stage × 5 stages × 2 CTAs = 200KB < 232KB (headroom: 32KB)
  - Current warp active: 22% → most warps stall on HBM

STRATEGY CONSIDERED: Increase stages s=5→6
  smem check: 20KB × 6 × 2 CTAs = 240KB > 232KB → FAILS 2-CTA constraint
  → cannot increase stages further at BM=32, BK=128

STRATEGY CONSIDERED: Increase BM=32→64
  smem per stage: (64+128)×128×1 = 24KB
  stages=5, 2 CTAs: 24KB × 5 × 2 = 240KB > 232KB → too tight
  stages=4, 2 CTAs: 24KB × 4 × 2 = 192KB < 232KB ✓
  arithmetic intensity: 2×64×128×128 / (24KB) = 89 FLOPS/byte (better than 51)
  tiles for T=14107: ceil(14107/64) × ceil(7168/128) = 221 × 56 = 12376 → well above 296 SMS
  expected gain: BI 51→89 FLOPS/byte = 1.75× arithmetic intensity → ~10-15% speedup
  estimate ≥ 5% threshold → PROCEED

CORRECTNESS CHECK:
  FP8 error unchanged (same quantization path), only BM changed → OK

DECISION: RUN modal with BM=64, BK=128, stages=4, grid_mult=2 for large GEMM2

CHANGE: kernel_triton_large(): _launch_gemm2_fp8_with_tm(..., BM=64, BK=128, warps=8, stages=4, ...)
NOTE: Also need to update build_two_tile_maps_diff_bm_gpu BM2=32→64

COMMIT MESSAGE: "perf: large tier GEMM2 BM=32→64 for better arithmetic intensity"
```
