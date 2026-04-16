"""
FlashInfer MLSys 2026 Track A — Fused MoE Kernel V5
=====================================================
Workload-specialized kernel dispatch based on NCU profiling:

  Tier       seq_len     GEMM1 BM  GEMM2 BM  warps  stages  grid strategy
  ─────────  ──────────  ────────  ────────  ─────  ──────  ──────────────
  tiny       ≤ 8         16        32        4      2       non-persistent
  small      9–128       32        64        4      2       1× NUM_SMS
  medium     129–2048    64        64        8      3       2× NUM_SMS
  large      > 2048      64        32        8      3       2× NUM_SMS

Rationale from NCU:
  - seq=1:  grid=48 (gemm1), 168 (gemm2) — tiles < SMs, persistent wastes CTAs
  - seq=7:  grid=112 — partial SM fill, smaller BM improves tile granularity
  - seq=62: grid=288 — near-saturated, standard config works well
  - seq=901+: grid=296 (=NUM_SMS×2) — fully saturated, deep pipeline helps
  - Register pressure: gemm1=233 regs (2 accumulators), gemm2=176 regs (1 acc)
    Smaller BLOCK_M → fewer regs → potential 2 CTAs/SM for tiny workloads

Dual-path: PyTorch FP32 on A100 | Triton FP32-dot persistent on B200
=====================================================
"""

import math
import os

import torch

if torch.cuda.is_available():
    _SM_MAJOR, _SM_MINOR = torch.cuda.get_device_capability()
else:
    _SM_MAJOR, _SM_MINOR = 0, 0
FORCE_TRITON_SM90 = os.getenv("MOE_FORCE_TRITON_SM90", "0") == "1"
USE_FP8_TRITON = (_SM_MAJOR >= 10) or (_SM_MAJOR == 9 and FORCE_TRITON_SM90)

# Constants from definition (const axes)
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
GEMM1_OUT_SIZE = 4096
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4

# Workload tier thresholds (seq_len boundaries)
_SEQ_TINY = 8
_SEQ_SMALL = 128
_SEQ_MEDIUM = 2048


# ═══════════════════════════════════════════════════════════════════
# ROUTING: Workload-specialized variants
# ═══════════════════════════════════════════════════════════════════

def _routing_core(routing_logits, routing_bias, local_expert_offset,
                  routed_scaling_factor):
    """Shared routing logic: sigmoid → group scoring → top-K selection.
    Returns (flat_tids, flat_eids, flat_wts, flat_global, seq_len, device)
    before the expert sort step.

    Optimized: avoids full [S, 256] M-matrix allocation by using gather on topk_idx.
    """
    seq_len = routing_logits.shape[0]
    device = routing_logits.device
    group_size = NUM_EXPERTS // N_GROUP

    s = torch.sigmoid(routing_logits.float())
    s_biased = s + routing_bias.float().reshape(-1) if routing_bias is not None else s

    s_grouped = s_biased.view(seq_len, N_GROUP, group_size)
    top2_vals, _ = s_grouped.topk(2, dim=-1, sorted=False)
    group_scores = top2_vals.sum(dim=-1)
    _, top_groups = group_scores.topk(TOPK_GROUP, dim=-1, sorted=False)

    group_mask = torch.zeros(seq_len, N_GROUP, device=device)
    group_mask.scatter_(1, top_groups, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(-1, -1, group_size).reshape(seq_len, NUM_EXPERTS)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_biased.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = scores_pruned.topk(TOP_K, dim=-1, sorted=False)

    # Compute weights without allocating full [S, 256] M matrix.
    # Only gather the TOP_K sigmoid scores, normalize, then use those directly.
    topk_scores = torch.gather(s, 1, topk_idx)           # [S, TOP_K]
    wt_sum = topk_scores.sum(dim=1, keepdim=True) + 1e-20
    topk_weights = (topk_scores / wt_sum) * routed_scaling_factor  # [S, TOP_K]

    local_start = int(local_expert_offset)
    local_mask = (topk_idx >= local_start) & (topk_idx < local_start + NUM_LOCAL_EXPERTS)
    local_experts = (topk_idx - local_start).clamp(0, NUM_LOCAL_EXPERTS - 1)

    flat_tokens = torch.arange(seq_len, device=device).unsqueeze(1).expand_as(local_mask)
    valid = local_mask.reshape(-1)
    flat_tids = flat_tokens.reshape(-1)[valid]
    flat_eids = local_experts.reshape(-1)[valid]
    flat_wts = topk_weights.reshape(-1)[valid]

    return flat_tids, flat_eids, flat_wts, seq_len, device


def routing_tiny(routing_logits, routing_bias, local_expert_offset,
                 routed_scaling_factor):
    """Routing for seq ≤ 8. At most 64 assignments — uses .sort() which is
    faster than argsort for very small tensors (single-warp radix sort)."""
    flat_tids, flat_eids, flat_wts, seq_len, device = _routing_core(
        routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)

    if flat_tids.numel() == 0:
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        return flat_tids.int(), flat_eids.int(), flat_wts, eo

    # .sort() is faster than argsort for tiny tensors (avoids index allocation overhead)
    sorted_eids, sort_idx = flat_eids.sort(stable=True)
    sorted_tids = flat_tids[sort_idx].int()
    sorted_eids = sorted_eids.int()
    sorted_wts = flat_wts[sort_idx]

    ec = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)
    ec.scatter_add_(0, sorted_eids.long(), torch.ones_like(sorted_eids))
    eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    eo[1:] = ec.cumsum(0)

    return sorted_tids, sorted_eids, sorted_wts, eo


def routing_standard(routing_logits, routing_bias, local_expert_offset,
                     routed_scaling_factor):
    """Routing for seq 9–2048. Standard argsort + scatter_add path."""
    flat_tids, flat_eids, flat_wts, seq_len, device = _routing_core(
        routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)

    if flat_tids.numel() == 0:
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        return flat_tids.int(), flat_eids.int(), flat_wts, eo

    sort_idx = torch.argsort(flat_eids, stable=True)
    sorted_tids = flat_tids[sort_idx].int()
    sorted_eids = flat_eids[sort_idx].int()
    sorted_wts = flat_wts[sort_idx]

    ec = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)
    ec.scatter_add_(0, sorted_eids.long(), torch.ones_like(sorted_eids))
    eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    eo[1:] = ec.cumsum(0)

    return sorted_tids, sorted_eids, sorted_wts, eo


def routing_large(routing_logits, routing_bias, local_expert_offset,
                  routed_scaling_factor):
    """Routing for seq > 2048. Uses bincount for expert offsets (avoids
    allocating ones_like tensor for 100K+ assignments)."""
    flat_tids, flat_eids, flat_wts, seq_len, device = _routing_core(
        routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)

    if flat_tids.numel() == 0:
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        return flat_tids.int(), flat_eids.int(), flat_wts, eo

    sort_idx = torch.argsort(flat_eids, stable=True)
    sorted_tids = flat_tids[sort_idx].int()
    sorted_eids = flat_eids[sort_idx].int()
    sorted_wts = flat_wts[sort_idx]

    # bincount avoids allocating a ones tensor for large T
    ec = torch.bincount(sorted_eids.long(), minlength=NUM_LOCAL_EXPERTS).int()
    eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    eo[1:] = ec.cumsum(0)

    return sorted_tids, sorted_eids, sorted_wts, eo


# ═══════════════════════════════════════════════════════════════════
# A100 PATH: PyTorch FP32 (reference-matched, proven correct on B200)
# ═══════════════════════════════════════════════════════════════════
def kernel_pytorch(routing_logits, routing_bias, hidden_states,
                   hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                   gemm2_weights, gemm2_weights_scale,
                   local_expert_offset, routed_scaling_factor, output):
    seq_len = routing_logits.shape[0]
    device = routing_logits.device

    sorted_tids, sorted_eids, sorted_wts, expert_offsets = \
        routing_standard(routing_logits, routing_bias,
                         local_expert_offset, routed_scaling_factor)

    output.zero_()
    if sorted_tids.numel() == 0:
        return output

    # Bulk dequant
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()
    A_scale_exp = A_scale.unsqueeze(-1).expand(-1, -1, BLOCK).reshape(seq_len, HIDDEN_SIZE)
    A = A_fp32 * A_scale_exp

    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13 = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13 = torch.repeat_interleave(S13, BLOCK, dim=2)
    W13 = W13_fp32 * S13

    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2 = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2 = torch.repeat_interleave(S2, BLOCK, dim=2)
    W2 = W2_fp32 * S2

    output_fp32 = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.float32, device=device)
    eo = expert_offsets.cpu().tolist()

    for e in range(NUM_LOCAL_EXPERTS):
        e_start, e_end = eo[e], eo[e + 1]
        if e_start >= e_end:
            continue

        tok_ids = sorted_tids[e_start:e_end].long()
        wts = sorted_wts[e_start:e_end]

        A_e = A[tok_ids]
        G1 = A_e.matmul(W13[e].t())

        # SwiGLU: silu(X2) * X1
        X1 = G1[:, :INTERMEDIATE_SIZE]
        X2 = G1[:, INTERMEDIATE_SIZE:]
        C = (X2 / (1.0 + torch.exp(-X2))) * X1

        O = C.matmul(W2[e].t())
        output_fp32.index_add_(0, tok_ids, O * wts.unsqueeze(1))

    output.copy_(output_fp32.to(torch.bfloat16))
    return output


# ═══════════════════════════════════════════════════════════════════
# B200 PATH: Triton kernels + workload-specialized pipelines
# ═══════════════════════════════════════════════════════════════════
if USE_FP8_TRITON:
    import triton
    import triton.language as tl

    # Number of SMs on B200 — used to size the persistent grid.
    _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count \
        if torch.cuda.is_available() else 148

    # ───────────────────────────────────────────────────────────
    # Fused routing Triton kernel — single kernel for all routing ops
    # ───────────────────────────────────────────────────────────

    @triton.jit
    def _fused_routing_kernel(
        logits_ptr, bias_ptr,
        out_tids_ptr, out_eids_ptr, out_wts_ptr,
        expert_hist_ptr,
        local_start, local_end, routed_scaling_factor,
        stride_logit_row,
        SEQ_LEN: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,   # 256
        N_GROUP: tl.constexpr,       # 8
        GROUP_SIZE: tl.constexpr,    # 32
        TOPK_GROUP: tl.constexpr,    # 4
        TOP_K: tl.constexpr,         # 8
        MAX_OUT: tl.constexpr,       # = TOP_K
        NUM_LOCAL: tl.constexpr,     # = NUM_LOCAL_EXPERTS
    ):
        """Fused routing: one CTA per token, 256 experts per program.

        All routing operations (sigmoid, group scoring, top-k, weight norm)
        executed in a single Triton kernel launch.
        Output is written unsorted; expert histogram is updated atomically.
        """
        tok = tl.program_id(0)

        experts = tl.arange(0, NUM_EXPERTS)  # [256]

        # ── Load logits + bias, apply sigmoid ─────────────────
        logit = tl.load(logits_ptr + tok * stride_logit_row + experts)
        bias  = tl.load(bias_ptr + experts)
        sig   = tl.sigmoid(logit.to(tl.float32))
        score = sig + bias.to(tl.float32)

        # ── Group scoring: top-2 sum per group ────────────────
        group_scores = tl.zeros((N_GROUP,), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (experts // GROUP_SIZE == g)
            sg   = tl.where(in_g, score, float("-inf"))
            t1   = tl.max(sg, axis=0)
            sg2  = tl.where(in_g & (score == t1), float("-inf"), sg)
            t2   = tl.max(sg2, axis=0)
            gs   = t1 + tl.maximum(t2, 0.0)
            group_scores = tl.where(tl.arange(0, N_GROUP) == g, gs, group_scores)

        # ── Top-4 groups via iterative argmax ─────────────────
        gs_work    = group_scores
        sel_groups = tl.zeros((N_GROUP,), dtype=tl.float32)
        for _ in tl.static_range(TOPK_GROUP):
            best_g  = tl.argmax(gs_work, axis=0)
            sel_groups = tl.where(tl.arange(0, N_GROUP) == best_g, 1.0, sel_groups)
            gs_work    = tl.where(tl.arange(0, N_GROUP) == best_g, float("-inf"), gs_work)

        # ── Compute per-expert group-selection mask ───────────
        group_id = experts // GROUP_SIZE
        expert_in_sel = tl.zeros((NUM_EXPERTS,), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            g_sel = tl.sum(tl.where(tl.arange(0, N_GROUP) == g, sel_groups, 0.0))
            expert_in_sel = expert_in_sel + tl.where(group_id == g, g_sel, 0.0)
        score_pruned = tl.where(expert_in_sel > 0.5, score, float("-inf"))

        # ── Top-8 experts + weight normalization ──────────────
        sp_work  = score_pruned
        wt_accum = tl.zeros((TOP_K,), dtype=tl.float32)
        eid_arr  = tl.zeros((TOP_K,), dtype=tl.int32)
        for k in tl.static_range(TOP_K):
            best_e   = tl.argmax(sp_work, axis=0)
            best_sig = tl.sum(tl.where(experts == best_e, sig, 0.0))
            wt_accum = tl.where(tl.arange(0, TOP_K) == k, best_sig, wt_accum)
            eid_arr  = tl.where(tl.arange(0, TOP_K) == k, best_e.to(tl.int32), eid_arr)
            sp_work  = tl.where(experts == best_e, float("-inf"), sp_work)

        wt_sum = tl.sum(wt_accum) + 1e-20
        wt_arr = wt_accum * (routed_scaling_factor / wt_sum)

        # ── Write local expert assignments (unsorted) + update histogram ──
        out_base = tok * MAX_OUT
        count    = 0
        for k in tl.static_range(TOP_K):
            eid = tl.sum(tl.where(tl.arange(0, TOP_K) == k, eid_arr, 0))
            wt  = tl.sum(tl.where(tl.arange(0, TOP_K) == k, wt_arr, 0.0))
            is_local = (eid >= local_start) & (eid < local_end)
            if is_local:
                local_eid = eid - local_start
                tl.store(out_tids_ptr + out_base + count, tok.to(tl.int32))
                tl.store(out_eids_ptr + out_base + count, local_eid.to(tl.int32))
                tl.store(out_wts_ptr  + out_base + count, wt.to(tl.float32))
                # Atomically increment per-expert histogram
                tl.atomic_add(expert_hist_ptr + local_eid, 1)
                count += 1
        # Mark remaining slots as invalid with eid sentinel = NUM_LOCAL
        for k in tl.static_range(TOP_K):
            if k >= count:
                tl.store(out_eids_ptr + out_base + k, NUM_LOCAL)

    @triton.jit
    def _counting_sort_kernel(
        in_tids_ptr, in_eids_ptr, in_wts_ptr,
        out_tids_ptr, out_eids_ptr, out_wts_ptr,
        expert_offsets_ptr,   # cumsum from expert_hist, [NUM_LOCAL+1] int32
        expert_ctr_ptr,       # per-expert atomic counter (starts at 0), [NUM_LOCAL] int32
        SEQ_LEN: tl.constexpr,
        TOP_K: tl.constexpr,
        NUM_LOCAL: tl.constexpr,
    ):
        """Scatter unsorted routing output to expert-sorted positions.
        One CTA per token; each token writes its local experts to the correct bucket.
        Uses atomic_add on per-expert counter to get unique write positions.
        Result: sorted by expert_id without any general sort."""
        tok = tl.program_id(0)
        in_base = tok * TOP_K
        for k in tl.static_range(TOP_K):
            eid = tl.load(in_eids_ptr + in_base + k)
            if eid < NUM_LOCAL:
                # Get write position within this expert's bucket
                pos = tl.atomic_add(expert_ctr_ptr + eid, 1)
                # Absolute write position = expert_offset[eid] + pos
                off = tl.load(expert_offsets_ptr + eid) + pos
                tl.store(out_tids_ptr + off, tl.load(in_tids_ptr + in_base + k))
                tl.store(out_eids_ptr + off, eid.to(tl.int32))
                tl.store(out_wts_ptr  + off, tl.load(in_wts_ptr + in_base + k))

    def routing_fused(routing_logits, routing_bias, local_expert_offset,
                      routed_scaling_factor):
        """Fused routing: single Triton kernel + argsort.
        Phase 1: _fused_routing_kernel computes expert assignments + histogram atomically.
        Phase 2: argsort on seq_len*TOP_K elements (no compaction sync needed).
        Returns same format as routing_standard: (sorted_tids, sorted_eids, sorted_wts, eoffs)
        """
        seq_len  = routing_logits.shape[0]
        device   = routing_logits.device
        local_start = int(local_expert_offset)
        local_end   = local_start + NUM_LOCAL_EXPERTS

        # Pre-allocate worst-case output (at most TOP_K local assignments per token)
        total_slots = seq_len * TOP_K
        out_tids  = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_eids  = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_wts   = torch.empty(total_slots, dtype=torch.float32, device=device)
        # Expert histogram: count per local expert, initialized to 0
        expert_hist = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _fused_routing_kernel[(seq_len,)](
            routing_logits, routing_bias,
            out_tids, out_eids, out_wts, expert_hist,
            local_start=local_start, local_end=local_end,
            routed_scaling_factor=float(routed_scaling_factor),
            stride_logit_row=routing_logits.stride(0),
            SEQ_LEN=seq_len,
            NUM_EXPERTS=NUM_EXPERTS, N_GROUP=N_GROUP,
            GROUP_SIZE=NUM_EXPERTS // N_GROUP,
            TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K, MAX_OUT=TOP_K,
            NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=1,  # 32 threads for 256 experts (8 elements per thread)
        )

        # Expert offsets from histogram (no bincount needed — we have expert_hist)
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)

        T = int(eo[-1].item())  # one unavoidable sync per call

        if T == 0:
            empty = torch.zeros(0, dtype=torch.int32, device=device)
            return empty, empty, torch.zeros(0, dtype=torch.float32, device=device), eo

        # Counting sort: scatter unsorted slots into expert-sorted positions
        # Faster than argsort for all T since it replaces a sort with atomic scatter
        sorted_tids = torch.empty(T, dtype=torch.int32, device=device)
        sorted_eids = torch.empty(T, dtype=torch.int32, device=device)
        sorted_wts  = torch.empty(T, dtype=torch.float32, device=device)
        expert_ctr  = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _counting_sort_kernel[(seq_len,)](
            out_tids, out_eids, out_wts,
            sorted_tids, sorted_eids, sorted_wts,
            eo, expert_ctr,
            SEQ_LEN=seq_len, TOP_K=TOP_K, NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=1,
        )

        return sorted_tids, sorted_eids, sorted_wts, eo

    def routing_fused_deferred(routing_logits, routing_bias, local_expert_offset,
                               routed_scaling_factor):
        """Routing without T sync — allocates worst-case arrays, defers T to caller.

        Avoids the eo[-1].item() sync so caller can combine it with tile-map sync.
        Returns (stids_full, seids_full, swts_full, eo, total_slots) where
        stids_full/etc. are worst-case sized; caller must slice [:T] after sync.
        """
        seq_len     = routing_logits.shape[0]
        device      = routing_logits.device
        local_start = int(local_expert_offset)
        local_end   = local_start + NUM_LOCAL_EXPERTS
        total_slots = seq_len * TOP_K

        out_tids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_eids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_wts     = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_hist = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _fused_routing_kernel[(seq_len,)](
            routing_logits, routing_bias,
            out_tids, out_eids, out_wts, expert_hist,
            local_start=local_start, local_end=local_end,
            routed_scaling_factor=float(routed_scaling_factor),
            stride_logit_row=routing_logits.stride(0),
            SEQ_LEN=seq_len,
            NUM_EXPERTS=NUM_EXPERTS, N_GROUP=N_GROUP,
            GROUP_SIZE=NUM_EXPERTS // N_GROUP,
            TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K, MAX_OUT=TOP_K,
            NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=1,
        )

        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)
        # DO NOT call eo[-1].item() here — defer to caller for batched sync
        return out_tids, out_eids, out_wts, eo, total_slots

    def routing_and_tilemaps_onesync(routing_logits, routing_bias, local_expert_offset,
                                     routed_scaling_factor, BM1, BN1, N1, BM2, BN2, N2, device):
        """Fused routing + dual tile-map: ONE GPU→CPU sync total.

        Strategy:
          1. Launch routing kernel (GPU-only, no sync)
          2. Compute eo = expert offsets (GPU cumsum, no sync)
          3. Compute m_tiles1, m_tiles2 from eo on GPU
          4. Launch counting_sort on GPU (uses eo, no T needed before launch)
             with worst-case preallocated sorted arrays
          5. ONE .tolist() sync: [T, sum_m1, sum_m2]
          6. Slice sorted arrays to [:T], build tile maps (know exact sizes)
        Returns (stids[:T], seids[:T], swts[:T], eo, tm1, tm2)
        """
        seq_len     = routing_logits.shape[0]
        local_start = int(local_expert_offset)
        local_end   = local_start + NUM_LOCAL_EXPERTS
        total_slots = seq_len * TOP_K

        # Step 1-2: routing kernel + cumsum (all GPU)
        out_tids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_eids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_wts     = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_hist = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _fused_routing_kernel[(seq_len,)](
            routing_logits, routing_bias,
            out_tids, out_eids, out_wts, expert_hist,
            local_start=local_start, local_end=local_end,
            routed_scaling_factor=float(routed_scaling_factor),
            stride_logit_row=routing_logits.stride(0),
            SEQ_LEN=seq_len,
            NUM_EXPERTS=NUM_EXPERTS, N_GROUP=N_GROUP,
            GROUP_SIZE=NUM_EXPERTS // N_GROUP,
            TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K, MAX_OUT=TOP_K,
            NUM_LOCAL=NUM_LOCAL_EXPERTS, num_warps=1,
        )

        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)

        # Step 3: compute m_tiles entirely on GPU (no sync yet)
        num_experts = NUM_LOCAL_EXPERTS
        e_starts = eo[:-1].to(torch.int64)
        M_es = (eo[1:].to(torch.int64) - e_starts).clamp(min=0)
        n_tiles_N1 = (N1 + BN1 - 1) // BN1
        n_tiles_N2 = (N2 + BN2 - 1) // BN2
        m_tiles1 = (M_es + BM1 - 1) // BM1
        m_tiles2 = (M_es + BM2 - 1) // BM2

        # Step 4: launch counting_sort on GPU with worst-case preallocated output
        # sorted arrays sized at total_slots (worst case), will slice after T is known
        sorted_tids_full = torch.empty(total_slots, dtype=torch.int32, device=device)
        sorted_eids_full = torch.empty(total_slots, dtype=torch.int32, device=device)
        sorted_wts_full  = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_ctr       = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)
        _counting_sort_kernel[(seq_len,)](
            out_tids, out_eids, out_wts,
            sorted_tids_full, sorted_eids_full, sorted_wts_full,
            eo, expert_ctr,
            SEQ_LEN=seq_len, TOP_K=TOP_K, NUM_LOCAL=NUM_LOCAL_EXPERTS, num_warps=1,
        )

        # Compute cumsum for tile maps on GPU (before sync — overlaps with counting_sort)
        tiles_per_expert1 = m_tiles1 * n_tiles_N1
        tiles_per_expert2 = m_tiles2 * n_tiles_N2
        tiles_cumsum1 = tiles_per_expert1.cumsum(0)
        tiles_cumsum2 = tiles_per_expert2.cumsum(0)

        # Step 5: ONE sync — T (routing) + sum_m1 + sum_m2 (tile maps)
        three_vals = torch.stack([eo[-1].to(torch.int64),
                                   m_tiles1.sum(),
                                   m_tiles2.sum()]).tolist()
        T, sum_m1, sum_m2 = int(three_vals[0]), int(three_vals[1]), int(three_vals[2])

        # Step 6: slice sorted arrays and build tile maps (sizes now known)
        stids = sorted_tids_full[:T]
        seids = sorted_eids_full[:T]
        swts  = sorted_wts_full[:T]

        if T == 0:
            empty_t = torch.zeros(0, 5, dtype=torch.int64, device=device)
            return stids, seids, swts, eo, empty_t, empty_t

        total_tiles1 = sum_m1 * n_tiles_N1
        total_tiles2 = sum_m2 * n_tiles_N2

        SEARCH_DEPTH = 6
        out1 = torch.empty(max(total_tiles1, 1) * 5, dtype=torch.int64, device=device)
        out2 = torch.empty(max(total_tiles2, 1) * 5, dtype=torch.int64, device=device)
        if total_tiles1 > 0:
            _build_tile_map_kernel[(total_tiles1,)](
                e_starts, M_es, tiles_cumsum1, m_tiles1, out1,
                n_tiles_N=n_tiles_N1, BM=BM1,
                num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        if total_tiles2 > 0:
            _build_tile_map_kernel[(total_tiles2,)](
                e_starts, M_es, tiles_cumsum2, m_tiles2, out2,
                n_tiles_N=n_tiles_N2, BM=BM2,
                num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        tm1 = out1.view(max(total_tiles1, 1), 5)[:total_tiles1]
        tm2 = out2.view(max(total_tiles2, 1), 5)[:total_tiles2]
        return stids, seids, swts, eo, tm1, tm2

    def routing_and_tilemaps_onesync_large(routing_logits, routing_bias, local_expert_offset,
                                           routed_scaling_factor, BM1, BN1, N1, BM2, BN2, N2, device):
        """Large-tier variant of routing_and_tilemaps_onesync.

        Differences from the standard version:
          - num_warps=4 for _fused_routing_kernel (better HBM bandwidth for large seq)
          - Syncs full eo (33 ints) alongside T, sum_m1, sum_m2 in ONE .tolist()
          - Returns eo_cpu for downstream tile-map splitting (multi-stream)
        """
        seq_len     = routing_logits.shape[0]
        local_start = int(local_expert_offset)
        local_end   = local_start + NUM_LOCAL_EXPERTS
        total_slots = seq_len * TOP_K

        out_tids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_eids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_wts     = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_hist = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _fused_routing_kernel[(seq_len,)](
            routing_logits, routing_bias,
            out_tids, out_eids, out_wts, expert_hist,
            local_start=local_start, local_end=local_end,
            routed_scaling_factor=float(routed_scaling_factor),
            stride_logit_row=routing_logits.stride(0),
            SEQ_LEN=seq_len,
            NUM_EXPERTS=NUM_EXPERTS, N_GROUP=N_GROUP,
            GROUP_SIZE=NUM_EXPERTS // N_GROUP,
            TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K, MAX_OUT=TOP_K,
            NUM_LOCAL=NUM_LOCAL_EXPERTS, num_warps=4,
        )

        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)

        num_experts = NUM_LOCAL_EXPERTS
        e_starts = eo[:-1].to(torch.int64)
        M_es = (eo[1:].to(torch.int64) - e_starts).clamp(min=0)
        n_tiles_N1 = (N1 + BN1 - 1) // BN1
        n_tiles_N2 = (N2 + BN2 - 1) // BN2
        m_tiles1 = (M_es + BM1 - 1) // BM1
        m_tiles2 = (M_es + BM2 - 1) // BM2

        sorted_tids_full = torch.empty(total_slots, dtype=torch.int32, device=device)
        sorted_eids_full = torch.empty(total_slots, dtype=torch.int32, device=device)
        sorted_wts_full  = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_ctr       = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)
        _counting_sort_kernel[(seq_len,)](
            out_tids, out_eids, out_wts,
            sorted_tids_full, sorted_eids_full, sorted_wts_full,
            eo, expert_ctr,
            SEQ_LEN=seq_len, TOP_K=TOP_K, NUM_LOCAL=NUM_LOCAL_EXPERTS, num_warps=1,
        )

        tiles_per_expert1 = m_tiles1 * n_tiles_N1
        tiles_per_expert2 = m_tiles2 * n_tiles_N2
        tiles_cumsum1 = tiles_per_expert1.cumsum(0)
        tiles_cumsum2 = tiles_per_expert2.cumsum(0)

        # ONE sync: eo (33 ints) + sum_m1 + sum_m2
        sync_tensor = torch.cat([eo.to(torch.int64), m_tiles1.sum().unsqueeze(0), m_tiles2.sum().unsqueeze(0)])
        sync_vals = sync_tensor.tolist()
        eo_cpu = [int(v) for v in sync_vals[:NUM_LOCAL_EXPERTS + 1]]
        T = eo_cpu[-1]
        sum_m1 = int(sync_vals[NUM_LOCAL_EXPERTS + 1])
        sum_m2 = int(sync_vals[NUM_LOCAL_EXPERTS + 2])

        stids = sorted_tids_full[:T]
        seids = sorted_eids_full[:T]
        swts  = sorted_wts_full[:T]

        if T == 0:
            empty_t = torch.zeros(0, 5, dtype=torch.int64, device=device)
            return stids, seids, swts, eo, eo_cpu, empty_t, empty_t

        total_tiles1 = sum_m1 * n_tiles_N1
        total_tiles2 = sum_m2 * n_tiles_N2

        SEARCH_DEPTH = 6
        out1 = torch.empty(max(total_tiles1, 1) * 5, dtype=torch.int64, device=device)
        out2 = torch.empty(max(total_tiles2, 1) * 5, dtype=torch.int64, device=device)
        if total_tiles1 > 0:
            _build_tile_map_kernel[(total_tiles1,)](
                e_starts, M_es, tiles_cumsum1, m_tiles1, out1,
                n_tiles_N=n_tiles_N1, BM=BM1,
                num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        if total_tiles2 > 0:
            _build_tile_map_kernel[(total_tiles2,)](
                e_starts, M_es, tiles_cumsum2, m_tiles2, out2,
                n_tiles_N=n_tiles_N2, BM=BM2,
                num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        tm1 = out1.view(max(total_tiles1, 1), 5)[:total_tiles1]
        tm2 = out2.view(max(total_tiles2, 1), 5)[:total_tiles2]
        return stids, seids, swts, eo, eo_cpu, tm1, tm2

    def routing_fused_argsort(routing_logits, routing_bias, local_expert_offset,
                              routed_scaling_factor):
        """Fused routing for large tier: uses argsort instead of counting sort.

        Avoids counting sort's atomic contention on 32 expert counters
        (severe at T≈seq_len≈14K, each counter hit ~437 times simultaneously).

        Uses _fused_routing_kernel (1 launch) + expert_hist for T (1 sync)
        + argsort on out_eids for sorting (no contention).

        Since sentinel=NUM_LOCAL_EXPERTS > all valid eids (0..31), argsort
        puts valid entries first. Take [:T] to get the sorted valid subset.
        """
        seq_len  = routing_logits.shape[0]
        device   = routing_logits.device
        local_start = int(local_expert_offset)
        local_end   = local_start + NUM_LOCAL_EXPERTS

        total_slots = seq_len * TOP_K
        out_tids  = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_eids  = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_wts   = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_hist = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _fused_routing_kernel[(seq_len,)](
            routing_logits, routing_bias,
            out_tids, out_eids, out_wts, expert_hist,
            local_start=local_start, local_end=local_end,
            routed_scaling_factor=float(routed_scaling_factor),
            stride_logit_row=routing_logits.stride(0),
            SEQ_LEN=seq_len,
            NUM_EXPERTS=NUM_EXPERTS, N_GROUP=N_GROUP,
            GROUP_SIZE=NUM_EXPERTS // N_GROUP,
            TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K, MAX_OUT=TOP_K,
            NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=1,
        )

        # Expert offsets from histogram (GPU cumsum)
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)

        T = int(eo[-1].item())  # one unavoidable sync

        if T == 0:
            empty = torch.zeros(0, dtype=torch.int32, device=device)
            return empty, empty, torch.zeros(0, dtype=torch.float32, device=device), eo

        # argsort on out_eids: sentinel=NUM_LOCAL_EXPERTS > valid eids 0..31
        # → valid entries appear first in sorted order, already grouped by expert
        sort_idx = torch.argsort(out_eids, stable=True)[:T]
        sorted_tids = out_tids[sort_idx]
        sorted_eids = out_eids[sort_idx]
        sorted_wts  = out_wts[sort_idx]

        return sorted_tids, sorted_eids, sorted_wts, eo

    def routing_fused_large(routing_logits, routing_bias, local_expert_offset,
                            routed_scaling_factor):
        """Fused routing for large tier: uses counting sort instead of argsort.
        Avoids sorting 95K+ elements. Two Triton kernel launches instead of argsort.
        Phase 1: _fused_routing_kernel — assignments + histogram (atomic).
        Phase 2: cumsum for expert offsets (GPU).
        Phase 3: _counting_sort_kernel — scatter to sorted positions (atomic per-expert).
        Uses num_warps=4 for large seq to improve routing throughput (more HBM bandwidth).
        """
        seq_len  = routing_logits.shape[0]
        device   = routing_logits.device
        local_start = int(local_expert_offset)
        local_end   = local_start + NUM_LOCAL_EXPERTS

        # Pre-allocate unsorted output (at most TOP_K local assignments per token)
        total_slots = seq_len * TOP_K
        out_tids  = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_eids  = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_wts   = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_hist = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _fused_routing_kernel[(seq_len,)](
            routing_logits, routing_bias,
            out_tids, out_eids, out_wts, expert_hist,
            local_start=local_start, local_end=local_end,
            routed_scaling_factor=float(routed_scaling_factor),
            stride_logit_row=routing_logits.stride(0),
            SEQ_LEN=seq_len,
            NUM_EXPERTS=NUM_EXPERTS, N_GROUP=N_GROUP,
            GROUP_SIZE=NUM_EXPERTS // N_GROUP,
            TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K, MAX_OUT=TOP_K,
            NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=4,  # 4 warps: better memory bandwidth for large seq routing
        )

        # Build expert offsets from histogram (GPU cumsum, no CPU sync yet)
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)

        T = int(eo[-1].item())  # one unavoidable sync

        if T == 0:
            empty = torch.zeros(0, dtype=torch.int32, device=device)
            return empty, empty, torch.zeros(0, dtype=torch.float32, device=device), eo

        # Counting sort: scatter unsorted slots into expert-sorted positions
        # Much faster than argsort on 95K+ elements for large T
        sorted_tids = torch.empty(T, dtype=torch.int32, device=device)
        sorted_eids = torch.empty(T, dtype=torch.int32, device=device)
        sorted_wts  = torch.empty(T, dtype=torch.float32, device=device)
        expert_ctr  = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _counting_sort_kernel[(seq_len,)](
            out_tids, out_eids, out_wts,
            sorted_tids, sorted_eids, sorted_wts,
            eo, expert_ctr,
            SEQ_LEN=seq_len, TOP_K=TOP_K, NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=1,
        )

        return sorted_tids, sorted_eids, sorted_wts, eo

    # ───────────────────────────────────────────────────────────
    # Shared GPU helpers
    # ───────────────────────────────────────────────────────────

    @triton.jit
    def _build_tile_map_kernel(
        e_starts_ptr, M_es_ptr, tiles_cumsum_ptr, m_tiles_ptr,
        out_ptr,
        n_tiles_N: tl.constexpr,
        BM: tl.constexpr,
        num_experts: tl.constexpr,
        SEARCH_DEPTH: tl.constexpr,
    ):
        """GPU kernel: build tile map without CPU sync.
        Each thread handles one tile, binary-searches for its expert."""
        tid = tl.program_id(0)
        # Binary search: find expert s.t. tiles_cumsum[eid-1] <= tid < tiles_cumsum[eid]
        lo = 0
        hi = num_experts - 1
        for _ in tl.static_range(SEARCH_DEPTH):
            mid = (lo + hi) // 2
            cs = tl.load(tiles_cumsum_ptr + mid)
            lo = tl.where(cs <= tid, mid + 1, lo)
            hi = tl.where(cs <= tid, hi, mid)
        eid = tl.minimum(lo, num_experts - 1)
        prev_cs = tl.where(eid > 0, tl.load(tiles_cumsum_ptr + eid - 1), 0)
        local_idx = tid - prev_cs

        e_start = tl.load(e_starts_ptr + eid)
        M_e     = tl.load(M_es_ptr + eid)
        m_t     = tl.load(m_tiles_ptr + eid)

        # Column-major: pm varies fast, pn varies slow
        pm = local_idx % m_t
        pn = local_idx // m_t

        base = tid.to(tl.int64) * 5
        tl.store(out_ptr + base + 0, eid.to(tl.int64))
        tl.store(out_ptr + base + 1, e_start.to(tl.int64))
        tl.store(out_ptr + base + 2, M_e.to(tl.int64))
        tl.store(out_ptr + base + 3, pm.to(tl.int64))
        tl.store(out_ptr + base + 4, pn.to(tl.int64))

    def build_tile_map_gpu(expert_offsets, N, BM, BN, device, T_hint=None):
        """Build GEMM tile map entirely on GPU.

        When T_hint is provided (total tokens, Python int), avoids GPU→CPU sync
        by using a Triton kernel to fill the tile map.

        Output: [total_tiles, 5] int64 with columns [expert_id, e_start, M_e, pm, pn].
        int64 is required for e_start so that e_start * stride doesn't overflow int32."""
        num_experts = expert_offsets.shape[0] - 1
        e_starts = expert_offsets[:-1].to(torch.int64)
        M_es = (expert_offsets[1:].to(torch.int64) - e_starts).clamp(min=0)

        n_tiles_N = (N + BN - 1) // BN
        m_tiles = (M_es + BM - 1) // BM
        tiles_per_expert = m_tiles * n_tiles_N

        if T_hint is not None and T_hint == 0:
            return torch.zeros(0, 5, dtype=torch.int64, device=device)

        if T_hint is not None:
            # Upper bound on total tiles from Python (no GPU sync)
            max_tiles = (math.ceil(T_hint / BM)) * n_tiles_N
            if max_tiles == 0:
                return torch.zeros(0, 5, dtype=torch.int64, device=device)
            # Cumsum on GPU — needed for binary search in kernel
            tiles_cumsum = tiles_per_expert.cumsum(0)  # [E] int64, stays on GPU
            # Actual total_tiles = tiles_cumsum[-1], but we avoid .item()
            # by using max_tiles as grid size and letting kernel handle bounds
            # HOWEVER: for correctness we need actual total_tiles for the GEMM loop bound
            # Use the Triton kernel path when max_tiles is the same as total_tiles
            # For now fall back: use max_tiles as allocator but still need .item() for GEMM
            # TODO: pass tiles_cumsum[-1] pointer to GEMM kernel directly
            total_tiles = int(tiles_cumsum[-1].item())
        else:
            total_tiles = int(tiles_per_expert.sum().item())
            tiles_cumsum = tiles_per_expert.cumsum(0)

        if total_tiles == 0:
            return torch.zeros(0, 5, dtype=torch.int64, device=device)

        # Use Triton kernel to build tile map (avoids repeat_interleave + multiple arange)
        if total_tiles <= 65536 * 1024:  # always true for our workloads
            out = torch.empty(total_tiles * 5, dtype=torch.int64, device=device)
            SEARCH_DEPTH = 6  # log2(32) = 5, use 6 for safety with 32 experts
            _build_tile_map_kernel[(total_tiles,)](
                e_starts, M_es, tiles_cumsum, m_tiles, out,
                n_tiles_N=n_tiles_N, BM=BM,
                num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH,
                num_warps=1)
            return out.view(total_tiles, 5)

        # Fallback for very large cases
        eids_exp = torch.repeat_interleave(
            torch.arange(num_experts, dtype=torch.int64, device=device),
            tiles_per_expert)
        exp_start_out = torch.zeros(num_experts, dtype=torch.int64, device=device)
        if num_experts > 1:
            exp_start_out[1:] = tiles_per_expert[:-1].cumsum(0)
        local_idx = (torch.arange(total_tiles, dtype=torch.int64, device=device)
                     - exp_start_out[eids_exp])
        m_t_per_tile = m_tiles[eids_exp]
        pm_exp = local_idx % m_t_per_tile
        pn_exp = local_idx // m_t_per_tile
        return torch.stack([
            eids_exp, e_starts[eids_exp], M_es[eids_exp], pm_exp, pn_exp,
        ], dim=1)

    def routing_fused_with_tilemaps(routing_logits, routing_bias, local_expert_offset,
                                    routed_scaling_factor, BM, BN1, N1, BN2, N2):
        """Fused routing + dual tile-map build with ONE GPU→CPU sync.

        Normal routing_fused + build_two_tile_maps_gpu require 2 syncs:
          1. T = eo[-1].item()  (routing)
          2. sum_m = m_tiles.sum().item()  (tile-map)
        This function combines them into ONE sync via torch.stack + tolist().

        Returns: (sorted_tids, sorted_eids, sorted_wts, eoffs, tm1, tm2)
        where BM is shared for both GEMMs.
        """
        seq_len     = routing_logits.shape[0]
        device      = routing_logits.device
        local_start = int(local_expert_offset)
        local_end   = local_start + NUM_LOCAL_EXPERTS

        total_slots = seq_len * TOP_K
        out_tids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_eids    = torch.empty(total_slots, dtype=torch.int32, device=device)
        out_wts     = torch.empty(total_slots, dtype=torch.float32, device=device)
        expert_hist = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)

        _fused_routing_kernel[(seq_len,)](
            routing_logits, routing_bias,
            out_tids, out_eids, out_wts, expert_hist,
            local_start=local_start, local_end=local_end,
            routed_scaling_factor=float(routed_scaling_factor),
            stride_logit_row=routing_logits.stride(0),
            SEQ_LEN=seq_len,
            NUM_EXPERTS=NUM_EXPERTS, N_GROUP=N_GROUP,
            GROUP_SIZE=NUM_EXPERTS // N_GROUP,
            TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K, MAX_OUT=TOP_K,
            NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=1,
        )

        # GPU: expert offsets + tile-row counts for both GEMMs (same BM)
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)
        M_es    = (eo[1:].to(torch.int64) - eo[:-1].to(torch.int64)).clamp(min=0)
        m_tiles = (M_es + BM - 1) // BM  # same for both GEMMs (same BM)

        # ONE sync: get T and sum(m_tiles) together
        both = torch.stack([eo[-1].to(torch.int64), m_tiles.sum()]).tolist()
        T, sum_m_tiles = int(both[0]), int(both[1])

        n_tiles_N1 = (N1 + BN1 - 1) // BN1
        n_tiles_N2 = (N2 + BN2 - 1) // BN2

        if T == 0:
            empty_i = torch.zeros(0, dtype=torch.int32, device=device)
            empty_f = torch.zeros(0, dtype=torch.float32, device=device)
            empty_tm = torch.zeros(0, 5, dtype=torch.int64, device=device)
            return empty_i, empty_i, empty_f, eo, empty_tm, empty_tm

        # Counting sort (now we have T)
        sorted_tids = torch.empty(T, dtype=torch.int32, device=device)
        sorted_eids = torch.empty(T, dtype=torch.int32, device=device)
        sorted_wts  = torch.empty(T, dtype=torch.float32, device=device)
        expert_ctr  = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)
        _counting_sort_kernel[(seq_len,)](
            out_tids, out_eids, out_wts,
            sorted_tids, sorted_eids, sorted_wts,
            eo, expert_ctr,
            SEQ_LEN=seq_len, TOP_K=TOP_K, NUM_LOCAL=NUM_LOCAL_EXPERTS,
            num_warps=1,
        )

        # Build both tile maps (no more syncs needed — sum_m_tiles already known)
        if sum_m_tiles == 0:
            empty_tm = torch.zeros(0, 5, dtype=torch.int64, device=device)
            return sorted_tids, sorted_eids, sorted_wts, eo, empty_tm, empty_tm

        e_starts = eo[:-1].to(torch.int64)
        total_tiles1 = sum_m_tiles * n_tiles_N1
        total_tiles2 = sum_m_tiles * n_tiles_N2
        tiles_per_expert1 = m_tiles * n_tiles_N1
        tiles_per_expert2 = m_tiles * n_tiles_N2
        tiles_cumsum1 = tiles_per_expert1.cumsum(0)
        tiles_cumsum2 = tiles_per_expert2.cumsum(0)

        SEARCH_DEPTH = 6
        out1 = torch.empty(total_tiles1 * 5, dtype=torch.int64, device=device)
        out2 = torch.empty(total_tiles2 * 5, dtype=torch.int64, device=device)
        _build_tile_map_kernel[(total_tiles1,)](
            e_starts, M_es, tiles_cumsum1, m_tiles, out1,
            n_tiles_N=n_tiles_N1, BM=BM,
            num_experts=NUM_LOCAL_EXPERTS, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        _build_tile_map_kernel[(total_tiles2,)](
            e_starts, M_es, tiles_cumsum2, m_tiles, out2,
            n_tiles_N=n_tiles_N2, BM=BM,
            num_experts=NUM_LOCAL_EXPERTS, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)

        tm1 = out1.view(total_tiles1, 5)
        tm2 = out2.view(total_tiles2, 5)
        return sorted_tids, sorted_eids, sorted_wts, eo, tm1, tm2

    def build_two_tile_maps_gpu(expert_offsets, BM, BN1, N1, BN2, N2, device):
        """Build GEMM1 and GEMM2 tile maps with a single GPU→CPU sync.

        When BM is the same for both GEMMs, m_tiles_e = ceil(count_e/BM) is shared.
        total_tiles1 = sum(m_tiles) * n_tiles_N1
        total_tiles2 = sum(m_tiles) * n_tiles_N2
        Both can be derived from a single .item() call: sum(m_tiles).
        Returns (tm1, tm2) tile maps.
        """
        num_experts = expert_offsets.shape[0] - 1
        e_starts = expert_offsets[:-1].to(torch.int64)
        M_es = (expert_offsets[1:].to(torch.int64) - e_starts).clamp(min=0)

        n_tiles_N1 = (N1 + BN1 - 1) // BN1
        n_tiles_N2 = (N2 + BN2 - 1) // BN2
        m_tiles = (M_es + BM - 1) // BM  # shared across both GEMMs

        sum_m_tiles = int(m_tiles.sum().item())  # ONE sync for both
        total_tiles1 = sum_m_tiles * n_tiles_N1
        total_tiles2 = sum_m_tiles * n_tiles_N2

        if sum_m_tiles == 0:
            empty = torch.zeros(0, 5, dtype=torch.int64, device=device)
            return empty, empty

        tiles_per_expert1 = m_tiles * n_tiles_N1
        tiles_per_expert2 = m_tiles * n_tiles_N2
        tiles_cumsum1 = tiles_per_expert1.cumsum(0)
        tiles_cumsum2 = tiles_per_expert2.cumsum(0)

        SEARCH_DEPTH = 6
        out1 = torch.empty(total_tiles1 * 5, dtype=torch.int64, device=device)
        out2 = torch.empty(total_tiles2 * 5, dtype=torch.int64, device=device)
        _build_tile_map_kernel[(total_tiles1,)](
            e_starts, M_es, tiles_cumsum1, m_tiles, out1,
            n_tiles_N=n_tiles_N1, BM=BM,
            num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        _build_tile_map_kernel[(total_tiles2,)](
            e_starts, M_es, tiles_cumsum2, m_tiles, out2,
            n_tiles_N=n_tiles_N2, BM=BM,
            num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        return out1.view(total_tiles1, 5), out2.view(total_tiles2, 5)

    def build_two_tile_maps_diff_bm_gpu(expert_offsets, BM1, BN1, N1, BM2, BN2, N2, device):
        """Build GEMM1 and GEMM2 tile maps with different BMs using a single GPU→CPU sync.

        Computes m_tiles1 and m_tiles2 separately, but sums both in one .item() call
        via a combined tensor: sum([sum(m_tiles1), sum(m_tiles2)]).
        Returns (tm1, tm2) tile maps.
        """
        num_experts = expert_offsets.shape[0] - 1
        e_starts = expert_offsets[:-1].to(torch.int64)
        M_es = (expert_offsets[1:].to(torch.int64) - e_starts).clamp(min=0)

        n_tiles_N1 = (N1 + BN1 - 1) // BN1
        n_tiles_N2 = (N2 + BN2 - 1) // BN2
        m_tiles1 = (M_es + BM1 - 1) // BM1
        m_tiles2 = (M_es + BM2 - 1) // BM2

        # One .item() call for both sums via stacked tensor
        both_sums = torch.stack([m_tiles1.sum(), m_tiles2.sum()]).tolist()
        sum_m1, sum_m2 = int(both_sums[0]), int(both_sums[1])
        total_tiles1 = sum_m1 * n_tiles_N1
        total_tiles2 = sum_m2 * n_tiles_N2

        if sum_m1 == 0 and sum_m2 == 0:
            empty = torch.zeros(0, 5, dtype=torch.int64, device=device)
            return empty, empty

        tiles_per_expert1 = m_tiles1 * n_tiles_N1
        tiles_per_expert2 = m_tiles2 * n_tiles_N2
        tiles_cumsum1 = tiles_per_expert1.cumsum(0)
        tiles_cumsum2 = tiles_per_expert2.cumsum(0)

        SEARCH_DEPTH = 6
        out1 = torch.empty(max(total_tiles1, 1) * 5, dtype=torch.int64, device=device)
        out2 = torch.empty(max(total_tiles2, 1) * 5, dtype=torch.int64, device=device)
        if total_tiles1 > 0:
            _build_tile_map_kernel[(total_tiles1,)](
                e_starts, M_es, tiles_cumsum1, m_tiles1, out1,
                n_tiles_N=n_tiles_N1, BM=BM1,
                num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        if total_tiles2 > 0:
            _build_tile_map_kernel[(total_tiles2,)](
                e_starts, M_es, tiles_cumsum2, m_tiles2, out2,
                n_tiles_N=n_tiles_N2, BM=BM2,
                num_experts=num_experts, SEARCH_DEPTH=SEARCH_DEPTH, num_warps=1)
        tm1 = out1.view(max(total_tiles1, 1), 5)[:total_tiles1]
        tm2 = out2.view(max(total_tiles2, 1), 5)[:total_tiles2]
        return tm1, tm2

    @triton.jit
    def _build_assign_idx_kernel(
        sorted_tids_ptr, reorder_ptr, tc_ptr, ts_ptr,
        T, SEQ_LEN,
        BLOCK: tl.constexpr,
    ):
        """Counting sort for _build_assign_idx: replaces argsort.
        Each CTA handles BLOCK slots. Uses atomic_add per token for write positions.
        tc_ptr has size SEQ_LEN+1 — slot SEQ_LEN is a scratch slot for masked lanes,
        so spurious atomics on masked lanes don't corrupt valid token counters."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T
        tok = tl.load(sorted_tids_ptr + offs, mask=mask, other=0).to(tl.int32)
        atomic_slot = tl.where(mask, tok, SEQ_LEN)
        pos = tl.atomic_add(tc_ptr + atomic_slot, 1)
        start = tl.load(ts_ptr + tok, mask=mask, other=0).to(tl.int32)
        dst = start + pos
        tl.store(reorder_ptr + dst, offs.to(tl.int32), mask=mask)

    def _build_assign_idx(sorted_token_ids, seq_len, device):
        T = sorted_token_ids.shape[0]
        if T == 0:
            empty_int = torch.zeros(0, dtype=torch.int32, device=device)
            tc = torch.zeros(seq_len, dtype=torch.int32, device=device)
            ts = torch.zeros(seq_len, dtype=torch.int32, device=device)
            return empty_int, ts, tc
        # Build tc (count per token) via scatter_add
        tc = torch.zeros(seq_len, dtype=torch.int32, device=device)
        tc.scatter_add_(0, sorted_token_ids.long(),
                        torch.ones(T, dtype=torch.int32, device=device))
        # Build ts (start per token) via cumsum
        ts = torch.zeros(seq_len, dtype=torch.int32, device=device)
        if seq_len > 1:
            ts[1:] = tc[:-1].cumsum(0)
        # Counting sort kernel: tc_ctr[SEQ_LEN] is scratch for masked lanes
        # After kernel, tc_ctr[:seq_len] == tc[:seq_len] (same counts)
        reorder = torch.empty(T, dtype=torch.int32, device=device)
        tc_ctr = torch.zeros(seq_len + 1, dtype=torch.int32, device=device)
        BSORT = 256
        grid_sort = (T + BSORT - 1) // BSORT
        _build_assign_idx_kernel[(grid_sort,)](
            sorted_token_ids, reorder, tc_ctr, ts,
            T=T, SEQ_LEN=seq_len, BLOCK=BSORT,
        )
        return reorder, ts, tc

    # ───────────────────────────────────────────────────────────
    # Triton JIT kernels (shared across all workload tiers)
    # Different constexpr params → different compiled variants
    # ───────────────────────────────────────────────────────────

    @triton.jit
    def _gemm1_swiglu_kernel(
        hidden_ptr, hidden_scale_ptr, sorted_ids_ptr,
        B_ptr, B_scale_ptr, tile_map_ptr, total_tiles, C_ptr,
        NUM_SMS: tl.constexpr,
        N: tl.constexpr, INTER: tl.constexpr, K: tl.constexpr,
        stride_h_row, stride_h_col,
        stride_b_expert, stride_b_row, stride_b_col,
        stride_c_row, stride_c_col,
        stride_hs_block, stride_hs_token,
        stride_bs_expert, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        HALF_N: tl.constexpr, FP8_BLK: tl.constexpr,
    ):
        start_pid = tl.program_id(0)
        for pid in tl.range(start_pid, total_tiles, NUM_SMS):
            base = tile_map_ptr + pid * 5
            eid = tl.load(base + 0).to(tl.int32)
            e_start = tl.load(base + 1)
            M_e = tl.load(base + 2)
            pm = tl.load(base + 3).to(tl.int32)
            pn = tl.load(base + 4).to(tl.int32)

            offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n_x1 = pn * HALF_N + tl.arange(0, HALF_N)
            offs_n_x2 = INTER + pn * HALF_N + tl.arange(0, HALF_N)

            mm = offs_m < M_e
            nm1 = offs_n_x1 < INTER
            nm2 = offs_n_x2 < N

            spos = e_start + offs_m
            otoks = tl.load(sorted_ids_ptr + spos, mask=mm, other=0)

            acc_x1 = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
            acc_x2 = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

            for ks in range(0, K, BLOCK_K):
                offs_k = ks + tl.arange(0, BLOCK_K)
                km = offs_k < K
                k_blk = ks // FP8_BLK

                a_fp8 = tl.load(
                    hidden_ptr + otoks[:, None] * stride_h_row + offs_k[None, :] * stride_h_col,
                    mask=mm[:, None] & km[None, :], other=0.0)

                b_x1_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert
                          + offs_n_x1[:, None] * stride_b_row
                          + offs_k[None, :] * stride_b_col,
                    mask=nm1[:, None] & km[None, :], other=0.0)

                b_x2_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert
                          + offs_n_x2[:, None] * stride_b_row
                          + offs_k[None, :] * stride_b_col,
                    mask=nm2[:, None] & km[None, :], other=0.0)

                partial_x1 = tl.dot(a_fp8, tl.trans(b_x1_fp8), out_dtype=tl.float32)
                partial_x2 = tl.dot(a_fp8, tl.trans(b_x2_fp8), out_dtype=tl.float32)

                a_s = tl.load(
                    hidden_scale_ptr + k_blk * stride_hs_block + otoks * stride_hs_token,
                    mask=mm, other=1.0)

                n_blk_x1 = pn
                b_s_x1 = tl.load(
                    B_scale_ptr + eid * stride_bs_expert
                                + n_blk_x1 * stride_bs_n + k_blk * stride_bs_k)

                n_blk_x2 = INTER // FP8_BLK + pn
                b_s_x2 = tl.load(
                    B_scale_ptr + eid * stride_bs_expert
                                + n_blk_x2 * stride_bs_n + k_blk * stride_bs_k)

                acc_x1 += partial_x1 * (a_s[:, None] * b_s_x1)
                acc_x2 += partial_x2 * (a_s[:, None] * b_s_x2)

            act = (acc_x2 * tl.sigmoid(acc_x2)) * acc_x1

            offs_out = pn * HALF_N + tl.arange(0, HALF_N)
            out_mask = offs_out < INTER

            tl.store(
                C_ptr + (e_start + offs_m[:, None]) * stride_c_row
                      + offs_out[None, :] * stride_c_col,
                act,
                mask=mm[:, None] & out_mask[None, :])

    @triton.jit
    def _gemm1_swiglu_fp8out_kernel(
        hidden_ptr, hidden_scale_ptr, sorted_ids_ptr,
        B_ptr, B_scale_ptr, tile_map_ptr, total_tiles,
        C_fp8_ptr, C_scale_ptr,
        NUM_SMS: tl.constexpr,
        N: tl.constexpr, INTER: tl.constexpr, K: tl.constexpr,
        stride_h_row, stride_h_col,
        stride_b_expert, stride_b_row, stride_b_col,
        stride_c_row, stride_c_col,
        stride_cs_row,
        stride_hs_block, stride_hs_token,
        stride_bs_expert, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        HALF_N: tl.constexpr, FP8_BLK: tl.constexpr,
    ):
        """GEMM1+SwiGLU with fused FP8 quantization of output.
        Outputs float8_e4m3fn act + per-block (HALF_N=128 wide) row scale.
        Enables FP8 tensor cores in GEMM2."""
        start_pid = tl.program_id(0)
        for pid in tl.range(start_pid, total_tiles, NUM_SMS):
            base = tile_map_ptr + pid * 5
            eid = tl.load(base + 0).to(tl.int32)
            e_start = tl.load(base + 1)
            M_e = tl.load(base + 2)
            pm = tl.load(base + 3).to(tl.int32)
            pn = tl.load(base + 4).to(tl.int32)

            offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n_x1 = pn * HALF_N + tl.arange(0, HALF_N)
            offs_n_x2 = INTER + pn * HALF_N + tl.arange(0, HALF_N)

            mm = offs_m < M_e
            nm1 = offs_n_x1 < INTER
            nm2 = offs_n_x2 < N

            spos = e_start + offs_m
            otoks = tl.load(sorted_ids_ptr + spos, mask=mm, other=0)

            acc_x1 = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
            acc_x2 = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

            for ks in range(0, K, BLOCK_K):
                offs_k = ks + tl.arange(0, BLOCK_K)
                km = offs_k < K
                k_blk = ks // FP8_BLK

                a_fp8 = tl.load(
                    hidden_ptr + otoks[:, None] * stride_h_row + offs_k[None, :] * stride_h_col,
                    mask=mm[:, None] & km[None, :], other=0.0)

                b_x1_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert
                          + offs_n_x1[:, None] * stride_b_row
                          + offs_k[None, :] * stride_b_col,
                    mask=nm1[:, None] & km[None, :], other=0.0)

                b_x2_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert
                          + offs_n_x2[:, None] * stride_b_row
                          + offs_k[None, :] * stride_b_col,
                    mask=nm2[:, None] & km[None, :], other=0.0)

                partial_x1 = tl.dot(a_fp8, tl.trans(b_x1_fp8), out_dtype=tl.float32)
                partial_x2 = tl.dot(a_fp8, tl.trans(b_x2_fp8), out_dtype=tl.float32)

                a_s = tl.load(
                    hidden_scale_ptr + k_blk * stride_hs_block + otoks * stride_hs_token,
                    mask=mm, other=1.0)

                n_blk_x1 = pn
                b_s_x1 = tl.load(
                    B_scale_ptr + eid * stride_bs_expert
                                + n_blk_x1 * stride_bs_n + k_blk * stride_bs_k)

                n_blk_x2 = INTER // FP8_BLK + pn
                b_s_x2 = tl.load(
                    B_scale_ptr + eid * stride_bs_expert
                                + n_blk_x2 * stride_bs_n + k_blk * stride_bs_k)

                acc_x1 += partial_x1 * (a_s[:, None] * b_s_x1)
                acc_x2 += partial_x2 * (a_s[:, None] * b_s_x2)

            act = (acc_x2 * tl.sigmoid(acc_x2)) * acc_x1  # [BLOCK_M, HALF_N] fp32

            # Quantize act → FP8: per-block (128-wide) scale = amax / 448
            # Using FP8 E4M3 (float8e4nv) for native FP8 tensor core support on B200
            FP8_MAX: tl.constexpr = 448.0
            amax = tl.max(tl.abs(tl.where(mm[:, None], act, 0.0)), axis=1)  # [BLOCK_M], valid rows only
            scale_a = amax / FP8_MAX + 1e-12
            act_scaled = act / scale_a[:, None]
            act_fp8 = act_scaled.to(tl.float8e4nv)

            offs_out = pn * HALF_N + tl.arange(0, HALF_N)
            out_mask = offs_out < INTER

            tl.store(
                C_fp8_ptr + (e_start + offs_m[:, None]) * stride_c_row
                          + offs_out[None, :] * stride_c_col,
                act_fp8,
                mask=mm[:, None] & out_mask[None, :])

            # C_scale layout: [T, INTER/HALF_N] — pn indexes the K-block for GEMM2
            tl.store(
                C_scale_ptr + (e_start + offs_m) * stride_cs_row + pn,
                scale_a,
                mask=mm)

    @triton.jit
    def _gemm2_fp8_kernel(
        A_fp8_ptr, A_scale_ptr, B_ptr, B_scale_ptr,
        tile_map_ptr, total_tiles, out_ptr,
        NUM_SMS: tl.constexpr,
        N: tl.constexpr, K: tl.constexpr,
        stride_a_row, stride_a_col,
        stride_as_row,
        stride_b_expert, stride_b_row, stride_b_col,
        stride_o_row, stride_o_col,
        stride_bs_expert, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        FP8_BLK: tl.constexpr,
    ):
        """GEMM2 using native FP8×FP8 tensor cores on B200 (SM 10.0).
        A is quantized FP8 activation from GEMM1 FP8-output path.
        B is FP8 weight. Both have per-(128-elem) block scales.
        Uses BLOCK_K=FP8_BLK=128 so each K-iteration is exactly one FP8 block."""
        start_pid = tl.program_id(0)
        for pid in tl.range(start_pid, total_tiles, NUM_SMS):
            base = tile_map_ptr + pid * 5
            eid = tl.load(base + 0).to(tl.int32)
            e_start = tl.load(base + 1)
            M_e = tl.load(base + 2)
            pm = tl.load(base + 3).to(tl.int32)
            pn = tl.load(base + 4).to(tl.int32)

            om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
            on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
            mmask = om < M_e
            nmask = on < N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for ks in range(0, K, BLOCK_K):
                ok = ks + tl.arange(0, BLOCK_K)
                km = ok < K
                k_blk = ks // FP8_BLK

                a_fp8 = tl.load(
                    A_fp8_ptr + (e_start + om[:, None]) * stride_a_row + ok[None, :] * stride_a_col,
                    mask=mmask[:, None] & km[None, :], other=0.0)

                b_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert + on[:, None] * stride_b_row + ok[None, :] * stride_b_col,
                    mask=nmask[:, None] & km[None, :], other=0.0)

                # Native FP8×FP8 tensor core dot product
                partial = tl.dot(a_fp8, tl.trans(b_fp8), out_dtype=tl.float32)

                a_s = tl.load(
                    A_scale_ptr + (e_start + om) * stride_as_row + k_blk,
                    mask=mmask, other=1.0)
                n_blk = (pn * BLOCK_N) // FP8_BLK
                b_s = tl.load(
                    B_scale_ptr + eid * stride_bs_expert + n_blk * stride_bs_n + k_blk * stride_bs_k)

                acc += partial * (a_s[:, None] * b_s)

            tl.store(
                out_ptr + (e_start + om[:, None]) * stride_o_row + on[None, :] * stride_o_col,
                acc,
                mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _quantize_to_fp8_rowwise_kernel(
        A_fp32_ptr, A_fp8_ptr, A_scale_ptr,
        T, stride_a_row,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Quantize FP32 activation to FP8 E4M3 with per-row (per-token) scale.
        One CTA per row. Scale = amax(|row|) / 448 → much less error than per-128-block.
        K and BLOCK_K are constexpr so static_range works."""
        row = tl.program_id(0)
        if row >= T:
            return
        FP8_MAX: tl.constexpr = 448.0
        N_BLOCKS: tl.constexpr = K // BLOCK_K
        # Compute row amax across all K elements
        row_amax = 0.0
        for kb in tl.static_range(0, N_BLOCKS):
            offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            vals = tl.load(A_fp32_ptr + row * stride_a_row + offs)
            row_amax = tl.maximum(row_amax, tl.max(tl.abs(vals)))
        scale = row_amax / FP8_MAX + 1e-12
        inv_scale = 1.0 / scale
        # Store per-row scale (single value per token)
        tl.store(A_scale_ptr + row, scale)
        # Quantize row
        for kb in tl.static_range(0, N_BLOCKS):
            offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            vals = tl.load(A_fp32_ptr + row * stride_a_row + offs)
            fp8_vals = (vals * inv_scale).to(tl.float8e4nv)
            tl.store(A_fp8_ptr + row * K + offs, fp8_vals)

    @triton.jit
    def _gemm2_fp8_perrow_kernel(
        A_fp8_ptr, A_scale_ptr, B_ptr, B_scale_ptr,
        tile_map_ptr, total_tiles, out_ptr,
        NUM_SMS: tl.constexpr,
        N: tl.constexpr, K: tl.constexpr,
        stride_a_row, stride_a_col,
        stride_b_expert, stride_b_row, stride_b_col,
        stride_o_row, stride_o_col,
        stride_bs_expert, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        FP8_BLK: tl.constexpr,
    ):
        """GEMM2 with FP8×FP8 tensor cores using per-row A scales (accurate).
        A_scale is shape [T] (one scale per token, vs per-128-block).
        scale application: acc * A_scale[row] * B_scale[n_blk, k_blk]."""
        start_pid = tl.program_id(0)
        for pid in tl.range(start_pid, total_tiles, NUM_SMS):
            base = tile_map_ptr + pid * 5
            eid = tl.load(base + 0).to(tl.int32)
            e_start = tl.load(base + 1)
            M_e = tl.load(base + 2)
            pm = tl.load(base + 3).to(tl.int32)
            pn = tl.load(base + 4).to(tl.int32)

            om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
            on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
            mmask = om < M_e
            nmask = on < N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            n_blk = (pn * BLOCK_N) // FP8_BLK

            # Load per-row A scales once (constant across all K iterations)
            a_s = tl.load(A_scale_ptr + (e_start + om), mask=mmask, other=1.0)

            for ks in range(0, K, BLOCK_K):
                ok = ks + tl.arange(0, BLOCK_K)
                km = ok < K
                k_blk = ks // FP8_BLK

                a_fp8 = tl.load(
                    A_fp8_ptr + (e_start + om[:, None]) * stride_a_row + ok[None, :] * stride_a_col,
                    mask=mmask[:, None] & km[None, :], other=0.0)

                b_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert + on[:, None] * stride_b_row + ok[None, :] * stride_b_col,
                    mask=nmask[:, None] & km[None, :], other=0.0)

                partial = tl.dot(a_fp8, tl.trans(b_fp8), out_dtype=tl.float32)

                b_s = tl.load(
                    B_scale_ptr + eid * stride_bs_expert + n_blk * stride_bs_n + k_blk * stride_bs_k)

                acc += partial * b_s

            # Apply per-row A scale once after accumulation (not per K-block)
            acc = acc * a_s[:, None]

            tl.store(
                out_ptr + (e_start + om[:, None]) * stride_o_row + on[None, :] * stride_o_col,
                acc,
                mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _gemm2_kernel(
        A_ptr, B_ptr, B_scale_ptr,
        tile_map_ptr, total_tiles, out_ptr,
        NUM_SMS: tl.constexpr,
        N: tl.constexpr, K: tl.constexpr,
        stride_a_row, stride_a_col,
        stride_b_expert, stride_b_row, stride_b_col,
        stride_o_row, stride_o_col,
        stride_bs_expert, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        FP8_BLK: tl.constexpr,
    ):
        start_pid = tl.program_id(0)
        for pid in tl.range(start_pid, total_tiles, NUM_SMS):
            base = tile_map_ptr + pid * 5
            eid = tl.load(base + 0).to(tl.int32)
            e_start = tl.load(base + 1)
            M_e = tl.load(base + 2)
            pm = tl.load(base + 3).to(tl.int32)
            pn = tl.load(base + 4).to(tl.int32)

            om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
            on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
            mmask = om < M_e
            nmask = on < N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            n_blk = (pn * BLOCK_N) // FP8_BLK

            for ks in range(0, K, BLOCK_K):
                ok = ks + tl.arange(0, BLOCK_K)
                km = ok < K
                k_blk = ks // FP8_BLK

                a = tl.load(
                    A_ptr + (e_start + om[:, None]) * stride_a_row + ok[None, :] * stride_a_col,
                    mask=mmask[:, None] & km[None, :], other=0.0)

                b_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert + on[:, None] * stride_b_row + ok[None, :] * stride_b_col,
                    mask=nmask[:, None] & km[None, :], other=0.0)
                b_fp32 = b_fp8.to(tl.float32)

                partial = tl.dot(a, tl.trans(b_fp32), out_dtype=tl.float32)

                b_s = tl.load(
                    B_scale_ptr + eid * stride_bs_expert + n_blk * stride_bs_n + k_blk * stride_bs_k)

                acc += partial * b_s

            tl.store(
                out_ptr + (e_start + om[:, None]) * stride_o_row + on[None, :] * stride_o_col,
                acc,
                mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _gemm2_bf16_kernel(
        A_ptr, B_ptr, B_scale_ptr,
        tile_map_ptr, total_tiles, out_ptr,
        NUM_SMS: tl.constexpr,
        N: tl.constexpr, K: tl.constexpr,
        stride_a_row, stride_a_col,
        stride_b_expert, stride_b_row, stride_b_col,
        stride_o_row, stride_o_col,
        stride_bs_expert, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        FP8_BLK: tl.constexpr,
    ):
        """GEMM2 using BF16 tensor cores: A(fp32→bf16) × B(fp8→bf16).
        BF16 halves register pressure for B vs FP32, enabling more pipelining.
        B200 BF16 TFLOPS: ~9000T vs TF32 2250T → potential 4× speedup if latency-limited."""
        start_pid = tl.program_id(0)
        for pid in tl.range(start_pid, total_tiles, NUM_SMS):
            base = tile_map_ptr + pid * 5
            eid = tl.load(base + 0).to(tl.int32)
            e_start = tl.load(base + 1)
            M_e = tl.load(base + 2)
            pm = tl.load(base + 3).to(tl.int32)
            pn = tl.load(base + 4).to(tl.int32)

            om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
            on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
            mmask = om < M_e
            nmask = on < N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            n_blk = (pn * BLOCK_N) // FP8_BLK

            for ks in range(0, K, BLOCK_K):
                ok = ks + tl.arange(0, BLOCK_K)
                km = ok < K
                k_blk = ks // FP8_BLK

                a_fp32 = tl.load(
                    A_ptr + (e_start + om[:, None]) * stride_a_row + ok[None, :] * stride_a_col,
                    mask=mmask[:, None] & km[None, :], other=0.0)
                a = a_fp32.to(tl.bfloat16)

                b_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert + on[:, None] * stride_b_row + ok[None, :] * stride_b_col,
                    mask=nmask[:, None] & km[None, :], other=0.0)
                b = b_fp8.to(tl.bfloat16)

                partial = tl.dot(a, tl.trans(b), out_dtype=tl.float32)

                b_s = tl.load(
                    B_scale_ptr + eid * stride_bs_expert + n_blk * stride_bs_n + k_blk * stride_bs_k)

                acc += partial * b_s

            tl.store(
                out_ptr + (e_start + om[:, None]) * stride_o_row + on[None, :] * stride_o_col,
                acc,
                mask=mmask[:, None] & nmask[None, :])

    @triton.jit
    def _reduce_kernel(
        g2_ptr, sid_ptr, sw_ptr, ts_ptr, tc_ptr, ao_ptr, out_ptr,
        seq_len, hidden_size: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        tid = tl.program_id(0)
        cb = tl.program_id(1)
        if tid >= seq_len:
            return
        offs = cb * BLOCK_H + tl.arange(0, BLOCK_H)
        m = offs < hidden_size
        start = tl.load(ts_ptr + tid)
        cnt = tl.load(tc_ptr + tid)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i in tl.static_range(8):
            valid = i < cnt
            a = tl.load(ao_ptr + start + i, mask=valid, other=0)
            w = tl.load(sw_ptr + a, mask=valid, other=0.0)
            v = tl.load(g2_ptr + a.to(tl.int64) * hidden_size + offs,
                        mask=valid & m, other=0.0).to(tl.float32)
            acc += w * v * valid.to(tl.float32)
        tl.store(out_ptr + tid.to(tl.int64) * hidden_size + offs, acc.to(tl.bfloat16), mask=m)

    @triton.jit
    def _gemm2_fused_reduce_kernel(
        A_ptr, B_ptr, B_scale_ptr,
        tile_map_ptr, total_tiles,
        sorted_tids_ptr, sorted_wts_ptr,
        out_ptr,                         # fp32 [seq_len, N], pre-zeroed
        NUM_SMS: tl.constexpr,
        N: tl.constexpr, K: tl.constexpr,
        stride_a_row, stride_a_col,
        stride_b_expert, stride_b_row, stride_b_col,
        stride_o_row,
        stride_bs_expert, stride_bs_n, stride_bs_k,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        FP8_BLK: tl.constexpr,
    ):
        """Fused GEMM2 + weighted reduce.

        Each tile computes the GEMM2 output for (expert, pm, pn) and immediately
        atomic_adds weighted rows into out_ptr[token_id, :] — no g2o buffer needed.

        Eliminates: g2o tensor (T×N fp32), _build_assign_idx kernel, _reduce_kernel.
        Atomic contention: ≤8 assignments per token write same output row, but
        different N-tiles write different column ranges (no inter-tile contention).
        """
        start_pid = tl.program_id(0)
        for pid in tl.range(start_pid, total_tiles, NUM_SMS):
            base = tile_map_ptr + pid * 5
            eid    = tl.load(base + 0).to(tl.int32)
            e_start= tl.load(base + 1)
            M_e    = tl.load(base + 2)
            pm     = tl.load(base + 3).to(tl.int32)
            pn     = tl.load(base + 4).to(tl.int32)

            om    = pm * BLOCK_M + tl.arange(0, BLOCK_M)
            on    = pn * BLOCK_N + tl.arange(0, BLOCK_N)
            mmask = om < M_e
            nmask = on < N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            n_blk = (pn * BLOCK_N) // FP8_BLK

            for ks in range(0, K, BLOCK_K):
                ok   = ks + tl.arange(0, BLOCK_K)
                km   = ok < K
                k_blk = ks // FP8_BLK

                a = tl.load(
                    A_ptr + (e_start + om[:, None]) * stride_a_row + ok[None, :] * stride_a_col,
                    mask=mmask[:, None] & km[None, :], other=0.0)

                b_fp8 = tl.load(
                    B_ptr + eid * stride_b_expert + on[:, None] * stride_b_row + ok[None, :] * stride_b_col,
                    mask=nmask[:, None] & km[None, :], other=0.0)
                b_fp32 = b_fp8.to(tl.float32)

                partial = tl.dot(a, tl.trans(b_fp32), out_dtype=tl.float32)

                b_s = tl.load(
                    B_scale_ptr + eid * stride_bs_expert + n_blk * stride_bs_n + k_blk * stride_bs_k)

                acc += partial * b_s

            # Load token IDs and weights for this tile's rows
            abs_rows = (e_start + om).to(tl.int64)  # absolute position in sorted arrays
            tids = tl.load(sorted_tids_ptr + abs_rows, mask=mmask, other=0).to(tl.int64)
            wts  = tl.load(sorted_wts_ptr  + abs_rows, mask=mmask, other=0.0)

            # Weighted atomic_add: 2D pointer matrix [BLOCK_M, BLOCK_N]
            # Each row maps to a different output token; atomic handles contention.
            row_ptrs = out_ptr + tids[:, None] * stride_o_row + on[None, :]
            weighted  = acc * wts[:, None]        # [BLOCK_M, BLOCK_N] fp32
            valid_2d  = mmask[:, None] & nmask[None, :]
            tl.atomic_add(row_ptrs, weighted, mask=valid_2d)

    # ───────────────────────────────────────────────────────────
    # Launch helpers (avoid code duplication across tiers)
    # ───────────────────────────────────────────────────────────

    # Secondary CUDA stream for overlapping tile-map2 build with GEMM1 execution
    _stream2 = torch.cuda.Stream() if torch.cuda.is_available() else None

    # Multi-stream pool for expert-group parallelism (large tier)
    _N_GEMM_STREAMS = 4
    _gemm_streams = [torch.cuda.Stream() for _ in range(_N_GEMM_STREAMS)] if torch.cuda.is_available() else []

    def _launch_gemm1(hidden_states, hidden_states_scale, stids,
                      gemm1_weights, gemm1_weights_scale,
                      eoffs, T, BM, BK, BN_G1, warps, stages, grid_mult, device):
        """Launch GEMM1+SwiGLU with tier-specific tile config."""
        tm1 = build_tile_map_gpu(eoffs, GEMM1_OUT_SIZE, BM, BN_G1, device)
        act_fp32 = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float32, device=device)
        if tm1.shape[0] > 0:
            if grid_mult == 0:
                # Non-persistent: one CTA per tile
                grid1 = tm1.shape[0]
            else:
                grid1 = min(_NUM_SMS * grid_mult, tm1.shape[0])
            _gemm1_swiglu_kernel[(grid1,)](
                hidden_states, hidden_states_scale, stids,
                gemm1_weights, gemm1_weights_scale,
                tm1, tm1.shape[0], act_fp32,
                N=GEMM1_OUT_SIZE, INTER=INTERMEDIATE_SIZE, K=HIDDEN_SIZE,
                NUM_SMS=grid1,
                stride_h_row=hidden_states.stride(0),
                stride_h_col=hidden_states.stride(1),
                stride_b_expert=gemm1_weights.stride(0),
                stride_b_row=gemm1_weights.stride(1),
                stride_b_col=gemm1_weights.stride(2),
                stride_c_row=act_fp32.stride(0),
                stride_c_col=act_fp32.stride(1),
                stride_hs_block=hidden_states_scale.stride(0),
                stride_hs_token=hidden_states_scale.stride(1),
                stride_bs_expert=gemm1_weights_scale.stride(0),
                stride_bs_n=gemm1_weights_scale.stride(1),
                stride_bs_k=gemm1_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_K=BK, HALF_N=BN_G1 // 2, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        return act_fp32

    def _launch_gemm1_prefetch_tm2(hidden_states, hidden_states_scale, stids,
                                   gemm1_weights, gemm1_weights_scale,
                                   eoffs, T, BM1, BK1, BN_G1, warps1, stages1, grid_mult1,
                                   BM2, BN2, BK2, warps2, stages2, grid_mult2, device):
        """Launch GEMM1 on default stream while building GEMM2 tile-map on stream2.
        Returns (act_fp32, tm2) — tm2 is ready when stream2 is synced."""
        # Build GEMM1 tile map on default stream
        tm1 = build_tile_map_gpu(eoffs, GEMM1_OUT_SIZE, BM1, BN_G1, device)
        act_fp32 = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float32, device=device)

        # Launch GEMM2 tile-map build on stream2 (concurrent with GEMM1)
        cur_stream = torch.cuda.current_stream()
        _stream2.wait_stream(cur_stream)  # stream2 starts after routing is done
        with torch.cuda.stream(_stream2):
            tm2 = build_tile_map_gpu(eoffs, HIDDEN_SIZE, BM2, BN2, device)

        # Launch GEMM1 on default stream (runs concurrently with tile-map build above)
        if tm1.shape[0] > 0:
            if grid_mult1 == 0:
                grid1 = tm1.shape[0]
            else:
                grid1 = min(_NUM_SMS * grid_mult1, tm1.shape[0])
            _gemm1_swiglu_kernel[(grid1,)](
                hidden_states, hidden_states_scale, stids,
                gemm1_weights, gemm1_weights_scale,
                tm1, tm1.shape[0], act_fp32,
                N=GEMM1_OUT_SIZE, INTER=INTERMEDIATE_SIZE, K=HIDDEN_SIZE,
                NUM_SMS=grid1,
                stride_h_row=hidden_states.stride(0),
                stride_h_col=hidden_states.stride(1),
                stride_b_expert=gemm1_weights.stride(0),
                stride_b_row=gemm1_weights.stride(1),
                stride_b_col=gemm1_weights.stride(2),
                stride_c_row=act_fp32.stride(0),
                stride_c_col=act_fp32.stride(1),
                stride_hs_block=hidden_states_scale.stride(0),
                stride_hs_token=hidden_states_scale.stride(1),
                stride_bs_expert=gemm1_weights_scale.stride(0),
                stride_bs_n=gemm1_weights_scale.stride(1),
                stride_bs_k=gemm1_weights_scale.stride(2),
                BLOCK_M=BM1, BLOCK_K=BK1, HALF_N=BN_G1 // 2, FP8_BLK=BLOCK,
                num_warps=warps1, num_stages=stages1)

        # Sync stream2 (tm2 must be ready before GEMM2 launches)
        cur_stream.wait_stream(_stream2)
        return act_fp32, tm2

    def _launch_gemm2(act_fp32, gemm2_weights, gemm2_weights_scale,
                      eoffs, T, BM, BN, BK, warps, stages, grid_mult, device):
        """Launch GEMM2 with tier-specific tile config."""
        tm2 = build_tile_map_gpu(eoffs, HIDDEN_SIZE, BM, BN, device)
        g2o = torch.empty(T, HIDDEN_SIZE, dtype=torch.float32, device=device)
        if tm2.shape[0] > 0:
            if grid_mult == 0:
                grid2 = tm2.shape[0]
            else:
                grid2 = min(_NUM_SMS * grid_mult, tm2.shape[0])
            _gemm2_kernel[(grid2,)](
                act_fp32, gemm2_weights, gemm2_weights_scale,
                tm2, tm2.shape[0], g2o,
                N=HIDDEN_SIZE, K=INTERMEDIATE_SIZE,
                NUM_SMS=grid2,
                stride_a_row=act_fp32.stride(0),
                stride_a_col=act_fp32.stride(1),
                stride_b_expert=gemm2_weights.stride(0),
                stride_b_row=gemm2_weights.stride(1),
                stride_b_col=gemm2_weights.stride(2),
                stride_o_row=g2o.stride(0),
                stride_o_col=g2o.stride(1),
                stride_bs_expert=gemm2_weights_scale.stride(0),
                stride_bs_n=gemm2_weights_scale.stride(1),
                stride_bs_k=gemm2_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        return g2o

    def _launch_gemm2_with_tm(act_fp32, gemm2_weights, gemm2_weights_scale,
                               tm2, T, BM, BN, BK, warps, stages, grid_mult, device):
        """Launch GEMM2 using a pre-built tile map (from _launch_gemm1_prefetch_tm2)."""
        g2o = torch.empty(T, HIDDEN_SIZE, dtype=torch.float32, device=device)
        if tm2.shape[0] > 0:
            if grid_mult == 0:
                grid2 = tm2.shape[0]
            else:
                grid2 = min(_NUM_SMS * grid_mult, tm2.shape[0])
            _gemm2_kernel[(grid2,)](
                act_fp32, gemm2_weights, gemm2_weights_scale,
                tm2, tm2.shape[0], g2o,
                N=HIDDEN_SIZE, K=INTERMEDIATE_SIZE,
                NUM_SMS=grid2,
                stride_a_row=act_fp32.stride(0),
                stride_a_col=act_fp32.stride(1),
                stride_b_expert=gemm2_weights.stride(0),
                stride_b_row=gemm2_weights.stride(1),
                stride_b_col=gemm2_weights.stride(2),
                stride_o_row=g2o.stride(0),
                stride_o_col=g2o.stride(1),
                stride_bs_expert=gemm2_weights_scale.stride(0),
                stride_bs_n=gemm2_weights_scale.stride(1),
                stride_bs_k=gemm2_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        return g2o

    def _launch_gemm2_bf16_with_tm(act_fp32, gemm2_weights, gemm2_weights_scale,
                                    tm2, T, BM, BN, BK, warps, stages, grid_mult, device):
        """Launch BF16 GEMM2: A(fp32→bf16) × B(fp8→bf16) using BF16 tensor cores.
        BF16 tensor cores: ~9000 TFLOPS vs TF32 2250 TFLOPS → potential 4× faster.
        Accuracy: BF16 ~0.78% error vs FP8's 0.44%/128-block — should pass 2% tolerance."""
        g2o = torch.empty(T, HIDDEN_SIZE, dtype=torch.float32, device=device)
        if tm2.shape[0] > 0:
            if grid_mult == 0:
                grid2 = tm2.shape[0]
            else:
                grid2 = min(_NUM_SMS * grid_mult, tm2.shape[0])
            _gemm2_bf16_kernel[(grid2,)](
                act_fp32, gemm2_weights, gemm2_weights_scale,
                tm2, tm2.shape[0], g2o,
                N=HIDDEN_SIZE, K=INTERMEDIATE_SIZE,
                NUM_SMS=grid2,
                stride_a_row=act_fp32.stride(0),
                stride_a_col=act_fp32.stride(1),
                stride_b_expert=gemm2_weights.stride(0),
                stride_b_row=gemm2_weights.stride(1),
                stride_b_col=gemm2_weights.stride(2),
                stride_o_row=g2o.stride(0),
                stride_o_col=g2o.stride(1),
                stride_bs_expert=gemm2_weights_scale.stride(0),
                stride_bs_n=gemm2_weights_scale.stride(1),
                stride_bs_k=gemm2_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        return g2o

    def _launch_gemm1_fp8out_with_tm(hidden_states, hidden_states_scale, stids,
                                      gemm1_weights, gemm1_weights_scale,
                                      tm1, T, BM, BK, BN_G1, warps, stages, device):
        """Launch GEMM1+SwiGLU with FP8 output quantization (pre-built tile map).
        Returns (act_fp8, act_scale) — FP8 activation + per-block scale for GEMM2 FP8 path.
        act_fp8: [T, INTER] float8_e4m3fn
        act_scale: [T, INTER/128] float32  (one scale per 128-wide N-tile row)"""
        n_scale_blocks = INTERMEDIATE_SIZE // (BN_G1 // 2)  # INTER/HALF_N = 2048/128 = 16
        act_fp8 = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float8_e4m3fn, device=device)
        act_scale = torch.empty(T, n_scale_blocks, dtype=torch.float32, device=device)
        if tm1.shape[0] > 0:
            grid1 = min(_NUM_SMS * 2, tm1.shape[0])
            _gemm1_swiglu_fp8out_kernel[(grid1,)](
                hidden_states, hidden_states_scale, stids,
                gemm1_weights, gemm1_weights_scale,
                tm1, tm1.shape[0],
                act_fp8, act_scale,
                N=GEMM1_OUT_SIZE, INTER=INTERMEDIATE_SIZE, K=HIDDEN_SIZE,
                NUM_SMS=grid1,
                stride_h_row=hidden_states.stride(0),
                stride_h_col=hidden_states.stride(1),
                stride_b_expert=gemm1_weights.stride(0),
                stride_b_row=gemm1_weights.stride(1),
                stride_b_col=gemm1_weights.stride(2),
                stride_c_row=act_fp8.stride(0),
                stride_c_col=act_fp8.stride(1),
                stride_cs_row=act_scale.stride(0),
                stride_hs_block=hidden_states_scale.stride(0),
                stride_hs_token=hidden_states_scale.stride(1),
                stride_bs_expert=gemm1_weights_scale.stride(0),
                stride_bs_n=gemm1_weights_scale.stride(1),
                stride_bs_k=gemm1_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_K=BK, HALF_N=BN_G1 // 2, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        return act_fp8, act_scale

    def _launch_gemm2_fp8_with_tm(act_fp8, act_scale, gemm2_weights, gemm2_weights_scale,
                                   tm2, T, BM, BN, BK, warps, stages, grid_mult, device):
        """Launch GEMM2 with FP8×FP8 tensor cores (pre-built tile map)."""
        g2o = torch.empty(T, HIDDEN_SIZE, dtype=torch.float32, device=device)
        if tm2.shape[0] > 0:
            if grid_mult == 0:
                grid2 = tm2.shape[0]
            else:
                grid2 = min(_NUM_SMS * grid_mult, tm2.shape[0])
            _gemm2_fp8_kernel[(grid2,)](
                act_fp8, act_scale, gemm2_weights, gemm2_weights_scale,
                tm2, tm2.shape[0], g2o,
                N=HIDDEN_SIZE, K=INTERMEDIATE_SIZE,
                NUM_SMS=grid2,
                stride_a_row=act_fp8.stride(0),
                stride_a_col=act_fp8.stride(1),
                stride_as_row=act_scale.stride(0),
                stride_b_expert=gemm2_weights.stride(0),
                stride_b_row=gemm2_weights.stride(1),
                stride_b_col=gemm2_weights.stride(2),
                stride_o_row=g2o.stride(0),
                stride_o_col=g2o.stride(1),
                stride_bs_expert=gemm2_weights_scale.stride(0),
                stride_bs_n=gemm2_weights_scale.stride(1),
                stride_bs_k=gemm2_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        return g2o

    def _launch_gemm2_fp8_perrow(act_fp32, gemm2_weights, gemm2_weights_scale,
                                  tm2, T, BM, BN, BK, warps, stages, grid_mult, device):
        """FP8 GEMM2 with per-row (per-token) A quantization.
        Step 1: quantize act_fp32 → fp8 with per-row scale (1 scale per token).
        Step 2: FP8×FP8 GEMM2 with per-row A scale applied after accumulation.
        Per-row scale dramatically reduces quantization error vs per-128-block:
          error ~ eps_fp8 * output_magnitude (not eps_fp8 * K * output_magnitude).
        """
        act_fp8  = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float8_e4m3fn, device=device)
        act_scale = torch.empty(T, dtype=torch.float32, device=device)  # [T] per-row
        # Quantize: one CTA per row, K=INTER=2048 must be divisible by BLOCK_K=128
        _quantize_to_fp8_rowwise_kernel[(T,)](
            act_fp32, act_fp8, act_scale,
            T=T,
            stride_a_row=act_fp32.stride(0),
            K=INTERMEDIATE_SIZE,    # constexpr: 2048
            BLOCK_K=128,            # constexpr: 2048/128=16 blocks
            num_warps=4)

        g2o = torch.empty(T, HIDDEN_SIZE, dtype=torch.float32, device=device)
        if tm2.shape[0] > 0:
            if grid_mult == 0:
                grid2 = tm2.shape[0]
            else:
                grid2 = min(_NUM_SMS * grid_mult, tm2.shape[0])
            _gemm2_fp8_perrow_kernel[(grid2,)](
                act_fp8, act_scale, gemm2_weights, gemm2_weights_scale,
                tm2, tm2.shape[0], g2o,
                N=HIDDEN_SIZE, K=INTERMEDIATE_SIZE,
                NUM_SMS=grid2,
                stride_a_row=act_fp8.stride(0),
                stride_a_col=act_fp8.stride(1),
                stride_b_expert=gemm2_weights.stride(0),
                stride_b_row=gemm2_weights.stride(1),
                stride_b_col=gemm2_weights.stride(2),
                stride_o_row=g2o.stride(0),
                stride_o_col=g2o.stride(1),
                stride_bs_expert=gemm2_weights_scale.stride(0),
                stride_bs_n=gemm2_weights_scale.stride(1),
                stride_bs_k=gemm2_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        return g2o

    def _launch_reduce(g2o, stids, swts, seq_len, device, output, block_h):
        """Launch reduce kernel with tier-specific BLOCK_H."""
        ao, ts, tc = _build_assign_idx(stids, seq_len, device)
        nhb = math.ceil(HIDDEN_SIZE / block_h)
        _reduce_kernel[(seq_len, nhb)](
            g2o, stids, swts, ts, tc, ao, output,
            seq_len=seq_len, hidden_size=HIDDEN_SIZE,
            BLOCK_H=block_h)

    def _launch_gemm2_fused_reduce(act_fp32, gemm2_weights, gemm2_weights_scale,
                                    tm2, stids, swts, seq_len,
                                    BM, BN, BK, warps, stages, grid_mult, device, output):
        """Launch fused GEMM2+reduce: no g2o buffer, direct atomic_add into out_fp32.
        Eliminates g2o tensor + _build_assign_idx + _reduce_kernel — saves 3 launches.
        out_fp32 is zeroed here; converted to bf16 after kernel completes."""
        out_fp32 = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.float32, device=device)
        if tm2.shape[0] > 0:
            if grid_mult == 0:
                grid2 = tm2.shape[0]
            else:
                grid2 = min(_NUM_SMS * grid_mult, tm2.shape[0])
            _gemm2_fused_reduce_kernel[(grid2,)](
                act_fp32, gemm2_weights, gemm2_weights_scale,
                tm2, tm2.shape[0],
                stids, swts,
                out_fp32,
                N=HIDDEN_SIZE, K=INTERMEDIATE_SIZE,
                NUM_SMS=grid2,
                stride_a_row=act_fp32.stride(0),
                stride_a_col=act_fp32.stride(1),
                stride_b_expert=gemm2_weights.stride(0),
                stride_b_row=gemm2_weights.stride(1),
                stride_b_col=gemm2_weights.stride(2),
                stride_o_row=out_fp32.stride(0),
                stride_bs_expert=gemm2_weights_scale.stride(0),
                stride_bs_n=gemm2_weights_scale.stride(1),
                stride_bs_k=gemm2_weights_scale.stride(2),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, FP8_BLK=BLOCK,
                num_warps=warps, num_stages=stages)
        output.copy_(out_fp32.to(torch.bfloat16))
        del out_fp32

    # ───────────────────────────────────────────────────────────
    # TIER 1: TINY (seq ≤ 8)
    # NCU: gemm1 grid=48, gemm2 grid=168 — tiles < SMs
    # Strategy: BM=16 (reduce wasted rows), non-persistent grid,
    #   4 warps / 2 stages → lower register pressure (~120 regs)
    #   → potential 2 CTAs/SM for latency hiding
    # ───────────────────────────────────────────────────────────

    def kernel_triton_tiny(routing_logits, routing_bias, hidden_states,
                           hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                           gemm2_weights, gemm2_weights_scale,
                           local_expert_offset, routed_scaling_factor, output):
        """Optimized for seq_len ≤ 8. ~8 active experts, 0-1 tokens each.
        Small BLOCK_M minimizes wasted tile rows. Non-persistent grid avoids
        idle CTAs when total_tiles << NUM_SMS.
        Change A: One-sync routing+tilemaps (single combined GPU→CPU sync)."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        # --- Change A: one-sync routing + tile maps ---
        stids, seids, swts, eo, tm1, tm2 = routing_and_tilemaps_onesync(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor,
            BM1=16, BN1=256, N1=GEMM1_OUT_SIZE, BM2=32, BN2=128, N2=HIDDEN_SIZE,
            device=device)

        T = stids.shape[0]
        if T == 0:
            output.zero_()
            return output

        # GEMM1+SwiGLU: BM=16, 4 warps, 2 stages, non-persistent
        act_fp32 = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float32, device=device)
        if tm1.shape[0] > 0:
            grid1 = tm1.shape[0]  # non-persistent: one CTA per tile
            _gemm1_swiglu_kernel[(grid1,)](
                hidden_states, hidden_states_scale, stids,
                gemm1_weights, gemm1_weights_scale,
                tm1, tm1.shape[0], act_fp32,
                N=GEMM1_OUT_SIZE, INTER=INTERMEDIATE_SIZE, K=HIDDEN_SIZE,
                NUM_SMS=grid1,
                stride_h_row=hidden_states.stride(0), stride_h_col=hidden_states.stride(1),
                stride_b_expert=gemm1_weights.stride(0), stride_b_row=gemm1_weights.stride(1),
                stride_b_col=gemm1_weights.stride(2),
                stride_c_row=act_fp32.stride(0), stride_c_col=act_fp32.stride(1),
                stride_hs_block=hidden_states_scale.stride(0),
                stride_hs_token=hidden_states_scale.stride(1),
                stride_bs_expert=gemm1_weights_scale.stride(0),
                stride_bs_n=gemm1_weights_scale.stride(1),
                stride_bs_k=gemm1_weights_scale.stride(2),
                BLOCK_M=16, BLOCK_K=128, HALF_N=128, FP8_BLK=BLOCK,
                num_warps=4, num_stages=2)

        # GEMM2: BM=32, BN=128, non-persistent (pre-built tm2)
        g2o = _launch_gemm2_with_tm(
            act_fp32, gemm2_weights, gemm2_weights_scale, tm2, T,
            BM=32, BN=128, BK=128, warps=4, stages=2, grid_mult=0, device=device)
        del act_fp32, tm1, tm2

        # Reduce: fp32 scatter_add + copy — replaces 6 ops with 2 (no _build_assign_idx)
        # For tiny (seq≤8), output buffer is at most 8×7168×4=230KB — trivially small
        g2o.mul_(swts.unsqueeze(1))               # [T, H] float32, in-place weighted
        out_fp32 = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.float32, device=device)
        idx = stids.long().unsqueeze(1).expand(T, HIDDEN_SIZE)
        out_fp32.scatter_add_(0, idx, g2o)
        output.copy_(out_fp32)  # auto-casts fp32→bf16
        del g2o, out_fp32, idx
        return output

    # ───────────────────────────────────────────────────────────
    # TIER 2: SMALL (seq 9–128)
    # NCU: gemm1 grid=112 (seq=7), grid=288 (seq=62)
    # Strategy: BM=32 (better granularity than 64 for ~1-4 tokens/expert),
    #   light persistent (1× NUM_SMS), 4 warps / 2 stages
    # ───────────────────────────────────────────────────────────

    def kernel_triton_small(routing_logits, routing_bias, hidden_states,
                            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                            gemm2_weights, gemm2_weights_scale,
                            local_expert_offset, routed_scaling_factor, output):
        """Optimized for seq_len 9–128. ONE combined GPU→CPU sync for routing+tilemap.
        Uses routing_fused_deferred + build_two_tile_maps_with_T_gpu for single .tolist()."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        stids, seids, swts, eoffs = routing_fused(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)

        T = stids.shape[0]
        if T == 0:
            output.zero_()
            return output

        # Build both tile maps with ONE GPU→CPU sync (shared BM=32 → shared m_tiles)
        tm1, tm2 = build_two_tile_maps_gpu(
            eoffs, BM=32, BN1=256, N1=GEMM1_OUT_SIZE, BN2=128, N2=HIDDEN_SIZE, device=device)

        # GEMM1+SwiGLU: BM=32, 8 warps, 7 stages (140KB/CTA < 232KB limit)
        act_fp32 = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float32, device=device)
        if tm1.shape[0] > 0:
            grid1 = min(_NUM_SMS, tm1.shape[0])
            _gemm1_swiglu_kernel[(grid1,)](
                hidden_states, hidden_states_scale, stids,
                gemm1_weights, gemm1_weights_scale,
                tm1, tm1.shape[0], act_fp32,
                N=GEMM1_OUT_SIZE, INTER=INTERMEDIATE_SIZE, K=HIDDEN_SIZE,
                NUM_SMS=grid1,
                stride_h_row=hidden_states.stride(0), stride_h_col=hidden_states.stride(1),
                stride_b_expert=gemm1_weights.stride(0), stride_b_row=gemm1_weights.stride(1),
                stride_b_col=gemm1_weights.stride(2),
                stride_c_row=act_fp32.stride(0), stride_c_col=act_fp32.stride(1),
                stride_hs_block=hidden_states_scale.stride(0),
                stride_hs_token=hidden_states_scale.stride(1),
                stride_bs_expert=gemm1_weights_scale.stride(0),
                stride_bs_n=gemm1_weights_scale.stride(1),
                stride_bs_k=gemm1_weights_scale.stride(2),
                BLOCK_M=32, BLOCK_K=128, HALF_N=128, FP8_BLK=BLOCK,
                num_warps=8, num_stages=7)

        # GEMM2: BM=32 BK=128 stages=6 — halves K-iters (112->56), smem 192KB/CTA
        g2o = _launch_gemm2_with_tm(
            act_fp32, gemm2_weights, gemm2_weights_scale, tm2, T,
            BM=32, BN=128, BK=128, warps=8, stages=6, grid_mult=1, device=device)
        del act_fp32, tm1, tm2

        # Reduce: fp32 scatter_add — in-place weighted
        g2o.mul_(swts.unsqueeze(1))               # [T, H] float32, in-place weighted
        out_fp32 = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.float32, device=device)
        idx = stids.long().unsqueeze(1).expand(T, HIDDEN_SIZE)
        out_fp32.scatter_add_(0, idx, g2o)
        output.copy_(out_fp32)  # auto-casts fp32→bf16
        del g2o, out_fp32, idx
        return output

    # ───────────────────────────────────────────────────────────
    # TIER 3: MEDIUM (seq 129–2048)
    # NCU: grid=288–296, near full saturation
    # Strategy: BM=64 (original), 8 warps / 3 stages, 2× NUM_SMS
    #   This is the proven config — sweet spot for most workloads
    # ───────────────────────────────────────────────────────────

    def kernel_triton_medium(routing_logits, routing_bias, hidden_states,
                             hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                             gemm2_weights, gemm2_weights_scale,
                             local_expert_offset, routed_scaling_factor, output):
        """Optimized for seq_len 129–2048.
        GEMM1: FP8×FP8 tensor cores, BM=64, BK=128, stages=3.
        GEMM2: FP32 (TF32 tensor cores on B200), BM=64, BK=32, stages=8.
        Reduce: scatter_add (faster than _reduce_kernel for medium seq)."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        stids, seids, swts, eoffs = routing_fused(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)

        T = stids.shape[0]
        if T == 0:
            output.zero_()
            return output

        # Build both tile maps with ONE GPU→CPU sync (shared BM=64 → shared m_tiles)
        tm1, tm2 = build_two_tile_maps_gpu(
            eoffs, BM=64, BN1=256, N1=GEMM1_OUT_SIZE, BN2=128, N2=HIDDEN_SIZE, device=device)

        # GEMM1+SwiGLU: BM=64, 8 warps, 4 stages → FP32 act
        # stages=4: smem=96KB/CTA, 2 CTAs/SM -> 192KB < 232KB limit
        act_fp32 = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float32, device=device)
        if tm1.shape[0] > 0:
            grid1 = min(_NUM_SMS * 2, tm1.shape[0])
            _gemm1_swiglu_kernel[(grid1,)](
                hidden_states, hidden_states_scale, stids,
                gemm1_weights, gemm1_weights_scale,
                tm1, tm1.shape[0], act_fp32,
                N=GEMM1_OUT_SIZE, INTER=INTERMEDIATE_SIZE, K=HIDDEN_SIZE,
                NUM_SMS=grid1,
                stride_h_row=hidden_states.stride(0), stride_h_col=hidden_states.stride(1),
                stride_b_expert=gemm1_weights.stride(0), stride_b_row=gemm1_weights.stride(1),
                stride_b_col=gemm1_weights.stride(2),
                stride_c_row=act_fp32.stride(0), stride_c_col=act_fp32.stride(1),
                stride_hs_block=hidden_states_scale.stride(0),
                stride_hs_token=hidden_states_scale.stride(1),
                stride_bs_expert=gemm1_weights_scale.stride(0),
                stride_bs_n=gemm1_weights_scale.stride(1),
                stride_bs_k=gemm1_weights_scale.stride(2),
                BLOCK_M=64, BLOCK_K=128, HALF_N=128, FP8_BLK=BLOCK,
                num_warps=8, num_stages=4)

        # GEMM2: BM=64 BK=64 stages=4 — smem 96KB/CTA, 2CTAs=192KB, half K iters
        g2o = _launch_gemm2_with_tm(
            act_fp32, gemm2_weights, gemm2_weights_scale, tm2, T,
            BM=64, BN=128, BK=64, warps=8, stages=4, grid_mult=2, device=device)
        del act_fp32, tm1, tm2

        # Reduce: scatter_add (faster than _reduce_kernel for medium seq)
        g2o.mul_(swts.unsqueeze(1))
        out_fp32 = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.float32, device=device)
        idx = stids.long().unsqueeze(1).expand(T, HIDDEN_SIZE)
        out_fp32.scatter_add_(0, idx, g2o)
        output.copy_(out_fp32)
        del g2o, out_fp32, idx
        return output

    # ───────────────────────────────────────────────────────────
    # TIER 4: LARGE (seq > 2048)
    # NCU: grid=296 (fully saturated), sort switches to onesweep
    # Strategy: BM=64 for both GEMMs (register-limited),
    #   8 warps / 3 stages (4 exceeds B200 232KB smem limit),
    #   BLOCK_H=256 in reduce (28 blocks vs 56 → fewer launches)
    # ───────────────────────────────────────────────────────────

    def kernel_triton_large(routing_logits, routing_bias, hidden_states,
                            hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                            gemm2_weights, gemm2_weights_scale,
                            local_expert_offset, routed_scaling_factor, output):
        """Optimized for seq_len > 2048.
        Change A: One-sync routing+tilemaps (saves ~150us from eliminating 2nd CPU-GPU sync).
        GEMM1: FP8×FP8 tensor cores, BM=64, BK=128, stages=5.
        GEMM2: FP32 (TF32 on B200), BM=64, BK=64, stages=4.
        Reduce: _launch_reduce(block_h=1024)."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        # --- Change A: one-sync routing + tile maps ---
        stids, seids, swts, eo, eo_cpu, tm1, tm2 = routing_and_tilemaps_onesync_large(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor,
            BM1=64, BN1=256, N1=GEMM1_OUT_SIZE, BM2=64, BN2=128, N2=HIDDEN_SIZE, device=device)

        T = eo_cpu[-1]
        if T == 0:
            output.zero_()
            return output

        # GEMM1+SwiGLU
        act_fp32 = torch.empty(T, INTERMEDIATE_SIZE, dtype=torch.float32, device=device)
        if tm1.shape[0] > 0:
            grid1 = min(_NUM_SMS * 2, tm1.shape[0])
            _gemm1_swiglu_kernel[(grid1,)](
                hidden_states, hidden_states_scale, stids,
                gemm1_weights, gemm1_weights_scale,
                tm1, tm1.shape[0], act_fp32,
                N=GEMM1_OUT_SIZE, INTER=INTERMEDIATE_SIZE, K=HIDDEN_SIZE,
                NUM_SMS=grid1,
                stride_h_row=hidden_states.stride(0), stride_h_col=hidden_states.stride(1),
                stride_b_expert=gemm1_weights.stride(0), stride_b_row=gemm1_weights.stride(1),
                stride_b_col=gemm1_weights.stride(2),
                stride_c_row=act_fp32.stride(0), stride_c_col=act_fp32.stride(1),
                stride_hs_block=hidden_states_scale.stride(0),
                stride_hs_token=hidden_states_scale.stride(1),
                stride_bs_expert=gemm1_weights_scale.stride(0),
                stride_bs_n=gemm1_weights_scale.stride(1),
                stride_bs_k=gemm1_weights_scale.stride(2),
                BLOCK_M=64, BLOCK_K=128, HALF_N=128, FP8_BLK=BLOCK,
                num_warps=8, num_stages=5)

        # GEMM2: FP32×FP8 (TF32 tensor cores)
        g2o = _launch_gemm2_with_tm(
            act_fp32, gemm2_weights, gemm2_weights_scale, tm2, T,
            BM=64, BN=128, BK=64, warps=8, stages=4, grid_mult=2, device=device)
        del act_fp32, tm1, tm2

        # Reduce
        _launch_reduce(g2o, stids, swts, seq_len, device, output, block_h=1024)
        del g2o
        return output

    # ───────────────────────────────────────────────────────────
    # Dispatcher: routes to the right tier
    # ───────────────────────────────────────────────────────────

    def kernel_triton(routing_logits, routing_bias, hidden_states,
                      hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                      gemm2_weights, gemm2_weights_scale,
                      local_expert_offset, routed_scaling_factor, output):
        seq_len = routing_logits.shape[0]
        if seq_len <= _SEQ_TINY:
            return kernel_triton_tiny(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                local_expert_offset, routed_scaling_factor, output)
        elif seq_len <= _SEQ_SMALL:
            return kernel_triton_small(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                local_expert_offset, routed_scaling_factor, output)
        elif seq_len <= _SEQ_MEDIUM:
            return kernel_triton_medium(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                local_expert_offset, routed_scaling_factor, output)
        else:
            return kernel_triton_large(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                local_expert_offset, routed_scaling_factor, output)


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor = None,
    **kwargs,
):
    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = local_expert_offset.item()
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = routed_scaling_factor.item()

    seq_len = routing_logits.shape[0]
    if output is None:
        output = torch.zeros(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16,
                             device=routing_logits.device)

    if USE_FP8_TRITON:
        result = kernel_triton(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor, output)
    else:
        result = kernel_pytorch(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor, output)
    return result
