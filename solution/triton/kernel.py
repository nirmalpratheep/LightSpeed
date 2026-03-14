"""
FlashInfer MLSys 2026 Track A — Fused MoE Kernel V5
=====================================================
Workload-specialized kernel dispatch based on NCU profiling:

  Tier       seq_len     GEMM1 BM  GEMM2 BM  warps  stages  grid strategy
  ─────────  ──────────  ────────  ────────  ─────  ──────  ──────────────
  tiny       ≤ 8         16        32        4      2       non-persistent
  small      9–128       32        64        4      2       1× NUM_SMS
  medium     129–2048    64        64        8      3       2× NUM_SMS
  large      > 2048      64        64        8      4       2× NUM_SMS

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

        # Sort by expert id. Invalid slots have eid=NUM_LOCAL_EXPERTS (sort to end).
        sort_idx = torch.argsort(out_eids, stable=True)
        sorted_eids_full = out_eids[sort_idx]

        # Expert offsets from histogram (no bincount needed — we have expert_hist)
        eo = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        eo[1:] = expert_hist.cumsum(0)

        T = int(eo[-1].item())  # one unavoidable sync per call

        if T == 0:
            empty = torch.zeros(0, dtype=torch.int32, device=device)
            return empty, empty, torch.zeros(0, dtype=torch.float32, device=device), eo

        sorted_tids = out_tids[sort_idx[:T]].int()
        sorted_eids = sorted_eids_full[:T].int()
        sorted_wts  = out_wts[sort_idx[:T]]

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

    def _build_assign_idx(sorted_token_ids, seq_len, device):
        T = sorted_token_ids.shape[0]
        reorder = torch.argsort(sorted_token_ids.long(), stable=True)
        tc = torch.zeros(seq_len, dtype=torch.int32, device=device)
        if T > 0:
            tc.scatter_add_(0, sorted_token_ids.long(),
                            torch.ones(T, dtype=torch.int32, device=device))
        ts = torch.zeros(seq_len, dtype=torch.int32, device=device)
        if seq_len > 1:
            ts[1:] = tc[:-1].cumsum(0)
        return reorder.int(), ts, tc

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

                n_blk = (pn * BLOCK_N) // FP8_BLK
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

    # ───────────────────────────────────────────────────────────
    # Launch helpers (avoid code duplication across tiers)
    # ───────────────────────────────────────────────────────────

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

    def _launch_reduce(g2o, stids, swts, seq_len, device, output, block_h):
        """Launch reduce kernel with tier-specific BLOCK_H."""
        ao, ts, tc = _build_assign_idx(stids, seq_len, device)
        nhb = math.ceil(HIDDEN_SIZE / block_h)
        _reduce_kernel[(seq_len, nhb)](
            g2o, stids, swts, ts, tc, ao, output,
            seq_len=seq_len, hidden_size=HIDDEN_SIZE,
            BLOCK_H=block_h)

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
        idle CTAs when total_tiles << NUM_SMS."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        torch.cuda.nvtx.range_push("ROUTE_TINY")
        stids, seids, swts, eoffs = routing_fused(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)
        torch.cuda.nvtx.range_pop()

        T = stids.shape[0]
        output.zero_()
        if T == 0:
            return output

        # GEMM1+SwiGLU: BM=16, 4 warps, 2 stages, non-persistent (grid_mult=0)
        torch.cuda.nvtx.range_push("GEMM1_TINY")
        act_fp32 = _launch_gemm1(
            hidden_states, hidden_states_scale, stids,
            gemm1_weights, gemm1_weights_scale, eoffs, T,
            BM=16, BK=128, BN_G1=256, warps=4, stages=2, grid_mult=0, device=device)
        torch.cuda.nvtx.range_pop()

        # GEMM2: BM=32, BN=128, non-persistent
        torch.cuda.nvtx.range_push("GEMM2_TINY")
        g2o = _launch_gemm2(
            act_fp32, gemm2_weights, gemm2_weights_scale, eoffs, T,
            BM=32, BN=128, BK=128, warps=4, stages=2, grid_mult=0, device=device)
        torch.cuda.nvtx.range_pop()
        del act_fp32

        # Reduce
        torch.cuda.nvtx.range_push("REDUCE_TINY")
        _launch_reduce(g2o, stids, swts, seq_len, device, output, block_h=128)
        torch.cuda.nvtx.range_pop()
        del g2o
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
        """Optimized for seq_len 9–128. Moderate tile count, light persistent
        grid (1× NUM_SMS) with BM=32 for better load balancing."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        torch.cuda.nvtx.range_push("ROUTE_SMALL")
        stids, seids, swts, eoffs = routing_fused(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)
        torch.cuda.nvtx.range_pop()

        T = stids.shape[0]
        output.zero_()
        if T == 0:
            return output

        # GEMM1+SwiGLU: BM=32, 8 warps, 5 stages, light persistent (1× NUM_SMS)
        torch.cuda.nvtx.range_push("GEMM1_SMALL")
        act_fp32 = _launch_gemm1(
            hidden_states, hidden_states_scale, stids,
            gemm1_weights, gemm1_weights_scale, eoffs, T,
            BM=32, BK=128, BN_G1=256, warps=8, stages=5, grid_mult=1, device=device)
        torch.cuda.nvtx.range_pop()

        # GEMM2: BM=32 BK=64 stages=5 warps=8 for small tier
        torch.cuda.nvtx.range_push("GEMM2_SMALL")
        g2o = _launch_gemm2(
            act_fp32, gemm2_weights, gemm2_weights_scale, eoffs, T,
            BM=32, BN=128, BK=64, warps=8, stages=5, grid_mult=1, device=device)
        torch.cuda.nvtx.range_pop()
        del act_fp32

        # Reduce
        torch.cuda.nvtx.range_push("REDUCE_SMALL")
        _launch_reduce(g2o, stids, swts, seq_len, device, output, block_h=128)
        torch.cuda.nvtx.range_pop()
        del g2o
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
        """Optimized for seq_len 129–2048. Fully saturated SMs, standard
        persistent config with BM=64, 8 warps, 3 pipeline stages."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        torch.cuda.nvtx.range_push("ROUTE_MEDIUM")
        stids, seids, swts, eoffs = routing_fused(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)
        torch.cuda.nvtx.range_pop()

        T = stids.shape[0]
        output.zero_()
        if T == 0:
            return output

        # GEMM1+SwiGLU: BM=64, 8 warps, 3 stages, full persistent (2× NUM_SMS)
        torch.cuda.nvtx.range_push("GEMM1_MEDIUM")
        act_fp32 = _launch_gemm1(
            hidden_states, hidden_states_scale, stids,
            gemm1_weights, gemm1_weights_scale, eoffs, T,
            BM=64, BK=128, BN_G1=256, warps=8, stages=3, grid_mult=2, device=device)
        torch.cuda.nvtx.range_pop()

        # GEMM2: BK=32 stages=8 — deeper pipeline within smem budget
        torch.cuda.nvtx.range_push("GEMM2_MEDIUM")
        g2o = _launch_gemm2(
            act_fp32, gemm2_weights, gemm2_weights_scale, eoffs, T,
            BM=64, BN=128, BK=32, warps=8, stages=8, grid_mult=2, device=device)
        torch.cuda.nvtx.range_pop()
        del act_fp32

        # Reduce with BLOCK_H=256 (28 blocks vs 56 → fewer launches)
        torch.cuda.nvtx.range_push("REDUCE_MEDIUM")
        _launch_reduce(g2o, stids, swts, seq_len, device, output, block_h=256)
        torch.cuda.nvtx.range_pop()
        del g2o
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
        """Optimized for seq_len > 2048. 3 pipeline stages (smem hardware limit on B200).
        Uses bincount routing to avoid allocating ones tensor for 100K+ assignments.
        BLOCK_H=256 reduce: 28 launch blocks vs 56 for fewer kernel launches."""
        seq_len = routing_logits.shape[0]
        device = routing_logits.device

        torch.cuda.nvtx.range_push("ROUTE_LARGE")
        stids, seids, swts, eoffs = routing_large(
            routing_logits, routing_bias, local_expert_offset, routed_scaling_factor)
        torch.cuda.nvtx.range_pop()

        T = stids.shape[0]
        output.zero_()
        if T == 0:
            return output

        # GEMM1+SwiGLU: BM=64, 8 warps, 3 stages (4 stages requires 245KB > B200's 232KB smem limit)
        torch.cuda.nvtx.range_push("GEMM1_LARGE")
        act_fp32 = _launch_gemm1(
            hidden_states, hidden_states_scale, stids,
            gemm1_weights, gemm1_weights_scale, eoffs, T,
            BM=64, BK=128, BN_G1=256, warps=8, stages=3, grid_mult=2, device=device)
        torch.cuda.nvtx.range_pop()

        # GEMM2: BM=32 BK=64 stages=5 for large tier — smaller tile for more parallelism
        torch.cuda.nvtx.range_push("GEMM2_LARGE")
        g2o = _launch_gemm2(
            act_fp32, gemm2_weights, gemm2_weights_scale, eoffs, T,
            BM=32, BN=128, BK=64, warps=8, stages=5, grid_mult=2, device=device)
        torch.cuda.nvtx.range_pop()
        del act_fp32

        # Reduce with BLOCK_H=256 (28 blocks vs 56 → fewer launches)
        torch.cuda.nvtx.range_push("REDUCE_LARGE")
        _launch_reduce(g2o, stids, swts, seq_len, device, output, block_h=256)
        torch.cuda.nvtx.range_pop()
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

    torch.cuda.nvtx.range_push("MOE_KERNEL")
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
    torch.cuda.nvtx.range_pop()
    return result
