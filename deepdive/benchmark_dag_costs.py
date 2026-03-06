"""
Benchmark script for MoE Kernel DAG cost model.

Measures actual latency (wall-clock, memory-only, compute) for each op
in the DeepSeek-V3 MoE kernel DAG on a B200 GPU.

Usage:
    # Single sequence length:
    python deepdive/benchmark_dag_costs.py --T 4096

    # All workloads from contest JSONL:
    python deepdive/benchmark_dag_costs.py --workloads mlsys26-contest/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl

Output:
    - Summary table to stdout
    - dag_costs.json (single T) or dag_costs_all.json (all workloads)
"""

import argparse
import json
import os
import time
from statistics import median

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H = 7168
I = 2048
E_LOCAL = 32
E_GLOBAL = 256
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def gpu_timer(fn, warmup=10, iters=100):
    """Time a GPU function using cuda events. Returns median latency in us."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us

    return median(times)


def flush_l2():
    """Flush L2 cache by reading a large buffer."""
    buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    buf.zero_()
    del buf
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Part 1: Hardware Characterization
# ---------------------------------------------------------------------------

def measure_hbm_bandwidth(warmup=10, iters=50):
    """Measure HBM bandwidth with large tensor copies."""
    results = {}
    for size_mb in [1, 10, 100, 1000]:
        n_elements = size_mb * 1024 * 1024 // 4  # fp32
        src = torch.randn(n_elements, device="cuda", dtype=torch.float32)
        dst = torch.empty_like(src)

        def copy_fn():
            dst.copy_(src)

        us = gpu_timer(copy_fn, warmup=warmup, iters=iters)
        bytes_moved = n_elements * 4  # read src + write dst = 2x, but copy is read+write
        bw_gbps = (2 * bytes_moved) / (us * 1e-6) / 1e9
        results[f"{size_mb}MB"] = {"latency_us": us, "bandwidth_GBps": round(bw_gbps, 1)}

        del src, dst
        torch.cuda.empty_cache()

    return results


def measure_peak_flops(warmup=10, iters=20):
    """Measure peak TFLOPS for different precisions via large GEMMs."""
    results = {}
    N = 4096

    for dtype, name in [
        (torch.bfloat16, "bf16"),
        (torch.float32, "fp32"),
    ]:
        a = torch.randn(N, N, device="cuda", dtype=dtype)
        b = torch.randn(N, N, device="cuda", dtype=dtype)

        def gemm_fn():
            torch.mm(a, b)

        us = gpu_timer(gemm_fn, warmup=warmup, iters=iters)
        flops = 2 * N * N * N  # 2*N^3 for matmul
        tflops = flops / (us * 1e-6) / 1e12
        results[name] = {"latency_us": round(us, 1), "tflops": round(tflops, 1)}

        del a, b
        torch.cuda.empty_cache()

    # FP8 GEMM if supported
    try:
        a_fp8 = torch.randn(N, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b_fp8 = torch.randn(N, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

        def fp8_gemm_fn():
            torch._scaled_mm(a_fp8, b_fp8.t(),
                             scale_a=torch.tensor(1.0, device="cuda"),
                             scale_b=torch.tensor(1.0, device="cuda"),
                             out_dtype=torch.bfloat16)

        us = gpu_timer(fp8_gemm_fn, warmup=warmup, iters=iters)
        flops = 2 * N * N * N
        tflops = flops / (us * 1e-6) / 1e12
        results["fp8"] = {"latency_us": round(us, 1), "tflops": round(tflops, 1)}
        del a_fp8, b_fp8
        torch.cuda.empty_cache()
    except Exception as e:
        results["fp8"] = {"error": str(e)}

    return results


def measure_launch_overhead(warmup=100, iters=1000):
    """Measure kernel launch overhead with a tiny operation."""
    x = torch.tensor(1.0, device="cuda")

    def tiny_fn():
        x + x

    us = gpu_timer(tiny_fn, warmup=warmup, iters=iters)
    return round(us, 2)


def hardware_characterization(warmup, iters):
    """Run all hardware characterization benchmarks."""
    print("=" * 70)
    print("Part 1: Hardware Characterization")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    print("HBM Bandwidth:")
    hbm = measure_hbm_bandwidth(warmup=warmup, iters=max(iters // 2, 10))
    for size, data in hbm.items():
        print(f"  {size:>8s}: {data['bandwidth_GBps']:>8.1f} GB/s  ({data['latency_us']:.1f} us)")

    # Use largest size as representative HBM BW
    hbm_bw = hbm[list(hbm.keys())[-1]]["bandwidth_GBps"]
    print()

    print("Peak FLOPS:")
    flops_results = measure_peak_flops(warmup=warmup, iters=max(iters // 5, 5))
    for name, data in flops_results.items():
        if "error" in data:
            print(f"  {name:>4s}: ERROR - {data['error']}")
        else:
            print(f"  {name:>4s}: {data['tflops']:>8.1f} TFLOPS  ({data['latency_us']:.1f} us)")
    print()

    print("Kernel Launch Overhead:")
    overhead = measure_launch_overhead()
    print(f"  {overhead:.2f} us")
    print()

    return {
        "gpu_name": gpu_name,
        "hbm_bandwidth_GBps": hbm_bw,
        "peak_flops": flops_results,
        "kernel_launch_overhead_us": overhead,
        "hbm_details": hbm,
    }


# ---------------------------------------------------------------------------
# Tensor allocation
# ---------------------------------------------------------------------------

def allocate_tensors(T, device="cuda"):
    """Allocate all input tensors at production scale."""
    Tk = T * TOP_K // E_GLOBAL  # avg tokens per expert

    tensors = {}

    # Graph inputs
    tensors["hidden_states"] = torch.randn(T, H, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    tensors["hidden_states_scale"] = torch.randn(H // BLOCK, T, device=device, dtype=torch.float32).abs() * 0.01
    tensors["gemm1_weights"] = torch.randn(E_LOCAL, 2 * I, H, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    tensors["gemm1_weights_scale"] = torch.randn(E_LOCAL, (2 * I) // BLOCK, H // BLOCK, device=device, dtype=torch.float32).abs() * 0.01
    tensors["gemm2_weights"] = torch.randn(E_LOCAL, H, I, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    tensors["gemm2_weights_scale"] = torch.randn(E_LOCAL, H // BLOCK, I // BLOCK, device=device, dtype=torch.float32).abs() * 0.01
    tensors["routing_logits"] = torch.randn(T, E_GLOBAL, device=device, dtype=torch.float32)
    tensors["routing_bias"] = torch.randn(E_GLOBAL, device=device, dtype=torch.float32) * 0.01

    # Pre-compute dequantized tensors for expert compute benchmarks
    # (so we can benchmark GEMM in isolation without including dequant cost)
    A_fp32 = tensors["hidden_states"].to(torch.float32)
    A_scale = tensors["hidden_states_scale"].to(torch.float32)
    A_scale_TH = A_scale.permute(1, 0).contiguous()
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    tensors["A"] = A_fp32 * A_scale_expanded

    W13_fp32 = tensors["gemm1_weights"].to(torch.float32)
    S13 = tensors["gemm1_weights_scale"].to(torch.float32)
    S13_exp = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_exp = torch.repeat_interleave(S13_exp, BLOCK, dim=2)
    tensors["W13"] = W13_fp32 * S13_exp

    W2_fp32 = tensors["gemm2_weights"].to(torch.float32)
    S2 = tensors["gemm2_weights_scale"].to(torch.float32)
    S2_exp = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_exp = torch.repeat_interleave(S2_exp, BLOCK, dim=2)
    tensors["W2"] = W2_fp32 * S2_exp

    # Pre-compute routing outputs for expert benchmarks
    logits = tensors["routing_logits"].to(torch.float32)
    bias = tensors["routing_bias"].to(torch.float32)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias
    group_size = E_GLOBAL // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_GLOBAL)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    routed_scaling_factor = 2.5
    weights = (weights / weights_sum) * routed_scaling_factor

    tensors["topk_idx"] = topk_idx
    tensors["weights"] = weights

    # Pre-compute per-expert token indices
    local_expert_offset = 0
    token_indices = []
    for le in range(E_LOCAL):
        ge = local_expert_offset + le
        sel = (topk_idx == ge).any(dim=1)
        idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
        token_indices.append(idx)
    tensors["token_indices"] = token_indices
    tensors["Tk"] = Tk

    return tensors


# ---------------------------------------------------------------------------
# Part 2: Per-Op Benchmarks
# ---------------------------------------------------------------------------

def bench_dequant_a(tensors, warmup, iters):
    """Op[0]: DequantA - hidden_states.float() * scale_expanded"""
    hs = tensors["hidden_states"]
    scale = tensors["hidden_states_scale"]
    T = hs.shape[0]

    # Pre-expand scale (this is part of the op)
    def fn():
        A_fp32 = hs.to(torch.float32)
        A_scale = scale.to(torch.float32)
        A_scale_TH = A_scale.permute(1, 0).contiguous()
        A_scale_expanded = (
            A_scale_TH.unsqueeze(-1)
            .repeat(1, 1, BLOCK)
            .reshape(T, H)
            .contiguous()
        )
        return A_fp32 * A_scale_expanded

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    bytes_in = T * H * 1 + (H // BLOCK) * T * 4
    bytes_out = T * H * 4
    flops = T * H
    return us, bytes_in, bytes_out, flops


def bench_dequant_w13(tensors, warmup, iters):
    """Op[1]: DequantW13 - gemm1_weights.float() * scale_expanded"""
    w = tensors["gemm1_weights"]
    s = tensors["gemm1_weights_scale"]

    def fn():
        W_fp32 = w.to(torch.float32)
        S = s.to(torch.float32)
        S_exp = torch.repeat_interleave(S, BLOCK, dim=1)
        S_exp = torch.repeat_interleave(S_exp, BLOCK, dim=2)
        return W_fp32 * S_exp

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    bytes_in = E_LOCAL * 2 * I * H * 1 + E_LOCAL * (2 * I // BLOCK) * (H // BLOCK) * 4
    bytes_out = E_LOCAL * 2 * I * H * 4
    flops = E_LOCAL * 2 * I * H
    return us, bytes_in, bytes_out, flops


def bench_dequant_w2(tensors, warmup, iters):
    """Op[2]: DequantW2 - gemm2_weights.float() * scale_expanded"""
    w = tensors["gemm2_weights"]
    s = tensors["gemm2_weights_scale"]

    def fn():
        W_fp32 = w.to(torch.float32)
        S = s.to(torch.float32)
        S_exp = torch.repeat_interleave(S, BLOCK, dim=1)
        S_exp = torch.repeat_interleave(S_exp, BLOCK, dim=2)
        return W_fp32 * S_exp

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    bytes_in = E_LOCAL * H * I * 1 + E_LOCAL * (H // BLOCK) * (I // BLOCK) * 4
    bytes_out = E_LOCAL * H * I * 4
    flops = E_LOCAL * H * I
    return us, bytes_in, bytes_out, flops


def bench_routing(tensors, warmup, iters):
    """Op[3]: Routing - sigmoid -> add_bias -> group_topk -> normalize"""
    logits = tensors["routing_logits"]
    bias = tensors["routing_bias"]
    T = logits.shape[0]

    def fn():
        s = torch.sigmoid(logits)
        s_with_bias = s + bias
        group_size = E_GLOBAL // N_GROUP
        s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
        top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
        group_scores = top2_vals.sum(dim=2)
        _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_GLOBAL)
        neg_inf = torch.finfo(torch.float32).min
        scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
        _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
        M = torch.zeros_like(s)
        M.scatter_(1, topk_idx, 1.0)
        weights = s * M
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
        weights = (weights / weights_sum) * 2.5
        return topk_idx, weights

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    bytes_in = T * E_GLOBAL * 4 + E_GLOBAL * 4
    bytes_out = T * TOP_K * 8 + T * E_GLOBAL * 4
    flops = T * E_GLOBAL * 20  # approximate
    return us, bytes_in, bytes_out, flops


def bench_gemm1(tensors, warmup, iters):
    """Op[4]: GEMM1 - A_e @ W13_e.T for all 32 experts"""
    A = tensors["A"]
    W13 = tensors["W13"]
    topk_idx = tensors["topk_idx"]
    token_indices = tensors["token_indices"]

    def fn():
        for le in range(E_LOCAL):
            idx = token_indices[le]
            if idx.numel() == 0:
                continue
            A_e = A.index_select(0, idx)
            W13_e = W13[le]
            A_e.matmul(W13_e.t())

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    # Compute totals across all experts
    total_tokens = sum(idx.numel() for idx in token_indices)
    bytes_in = total_tokens * H * 4 + E_LOCAL * 2 * I * H * 4 + total_tokens * 8  # A + W13 + idx
    bytes_out = total_tokens * 2 * I * 4
    flops = total_tokens * 2 * H * 2 * I  # 2*Tk*H*(2I) per expert summed
    return us, bytes_in, bytes_out, flops


def bench_swiglu(tensors, warmup, iters):
    """Op[5]: SwiGLU - silu(X2) * X1 for all 32 experts"""
    token_indices = tensors["token_indices"]
    # Pre-create G1 tensors for each expert
    G1s = []
    for le in range(E_LOCAL):
        Tk = token_indices[le].numel()
        if Tk == 0:
            G1s.append(None)
        else:
            G1s.append(torch.randn(Tk, 2 * I, device="cuda", dtype=torch.float32))

    def fn():
        for le in range(E_LOCAL):
            g1 = G1s[le]
            if g1 is None:
                continue
            X1 = g1[:, :I]
            X2 = g1[:, I:]
            torch.nn.functional.silu(X2) * X1

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    total_tokens = sum(idx.numel() for idx in token_indices)
    bytes_in = total_tokens * 2 * I * 4
    bytes_out = total_tokens * I * 4
    flops = total_tokens * I * 5  # silu=3ops + mul + mul
    return us, bytes_in, bytes_out, flops


def bench_gemm2(tensors, warmup, iters):
    """Op[6]: GEMM2 - C @ W2_e.T for all 32 experts"""
    W2 = tensors["W2"]
    token_indices = tensors["token_indices"]
    # Pre-create C tensors
    Cs = []
    for le in range(E_LOCAL):
        Tk = token_indices[le].numel()
        if Tk == 0:
            Cs.append(None)
        else:
            Cs.append(torch.randn(Tk, I, device="cuda", dtype=torch.float32))

    def fn():
        for le in range(E_LOCAL):
            c = Cs[le]
            if c is None:
                continue
            W2_e = W2[le]
            c.matmul(W2_e.t())

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    total_tokens = sum(idx.numel() for idx in token_indices)
    bytes_in = total_tokens * I * 4 + E_LOCAL * H * I * 4
    bytes_out = total_tokens * H * 4
    flops = total_tokens * 2 * I * H
    return us, bytes_in, bytes_out, flops


def bench_weighted_accum(tensors, warmup, iters):
    """Op[7]: WeightedAccum - output.index_add_ for all 32 experts"""
    T = tensors["routing_logits"].shape[0]
    weights = tensors["weights"]
    token_indices = tensors["token_indices"]
    # Pre-create O tensors
    Os = []
    for le in range(E_LOCAL):
        Tk = token_indices[le].numel()
        if Tk == 0:
            Os.append(None)
        else:
            Os.append(torch.randn(Tk, H, device="cuda", dtype=torch.float32))

    def fn():
        output = torch.zeros(T, H, dtype=torch.float32, device="cuda")
        for le in range(E_LOCAL):
            o = Os[le]
            if o is None:
                continue
            ge = le  # local_expert_offset=0
            idx = token_indices[le]
            w_tok = weights.index_select(0, idx)[:, ge]
            output.index_add_(0, idx, o * w_tok.unsqueeze(1))
        return output

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    total_tokens = sum(idx.numel() for idx in token_indices)
    bytes_in = total_tokens * H * 4 + T * E_GLOBAL * 4 + T * H * 4  # O + weights + output(RMW)
    bytes_out = T * H * 4
    flops = total_tokens * H * 2  # mul + add
    return us, bytes_in, bytes_out, flops


# ---------------------------------------------------------------------------
# Part 3: Memory-Only Baseline
# ---------------------------------------------------------------------------

def bench_memory_only(bytes_in, bytes_out, warmup=10, iters=50):
    """Measure pure data movement cost for equivalent bytes."""
    src = torch.empty(max(bytes_in, 1), dtype=torch.uint8, device="cuda")
    dst = torch.empty(max(bytes_out, 1), dtype=torch.uint8, device="cuda")

    def fn():
        # Simulate read of bytes_in + write of bytes_out
        _ = src.sum()  # force read from HBM
        dst.zero_()    # force write to HBM

    flush_l2()
    us = gpu_timer(fn, warmup=warmup, iters=iters)

    del src, dst
    torch.cuda.empty_cache()
    return us


# ---------------------------------------------------------------------------
# Part 4: Main
# ---------------------------------------------------------------------------

OP_NAMES = [
    "DequantA", "DequantW13", "DequantW2", "Routing",
    "GEMM1", "SwiGLU", "GEMM2", "WeightedAccum",
]

OP_TYPES = [
    "Pointwise", "Pointwise", "Pointwise", "Pointwise",
    "MatMul", "Pointwise", "MatMul", "Pointwise",
]


def fmt_bytes(b):
    if b >= 1e9:
        return f"{b / 1e9:.1f}GB"
    elif b >= 1e6:
        return f"{b / 1e6:.1f}MB"
    else:
        return f"{b / 1e3:.1f}KB"


def fmt_flops(f):
    if f >= 1e12:
        return f"{f / 1e12:.1f}T"
    elif f >= 1e9:
        return f"{f / 1e9:.1f}G"
    elif f >= 1e6:
        return f"{f / 1e6:.1f}M"
    else:
        return f"{f:.0f}"


BENCH_FNS = [
    bench_dequant_a, bench_dequant_w13, bench_dequant_w2, bench_routing,
    bench_gemm1, bench_swiglu, bench_gemm2, bench_weighted_accum,
]


def compute_roofline(bytes_in, bytes_out, flops, op_type, hw):
    """Compute theoretical roofline times for an op given hardware measurements.

    Returns (theoretical_mem_us, theoretical_compute_us, theoretical_best_us, theoretical_bound).
    """
    # HBM bandwidth in bytes/us = GB/s * 1e9 / 1e6 = GB/s * 1e3
    hbm_bw = hw.get("hbm_bandwidth_GBps", 8000)  # GB/s
    hbm_bytes_per_us = hbm_bw * 1e3  # bytes/us

    total_bytes = bytes_in + bytes_out
    theoretical_mem_us = total_bytes / hbm_bytes_per_us if hbm_bytes_per_us > 0 else 0

    # Pick peak FLOPS based on op type
    peak_flops = hw.get("peak_flops", {})
    if op_type == "MatMul":
        # Use FP8 if available, else BF16
        if "fp8" in peak_flops and "tflops" in peak_flops["fp8"]:
            tflops = peak_flops["fp8"]["tflops"]
        elif "bf16" in peak_flops and "tflops" in peak_flops["bf16"]:
            tflops = peak_flops["bf16"]["tflops"]
        else:
            tflops = 2250  # B200 BF16 fallback
    else:
        # Pointwise ops use FP32
        if "fp32" in peak_flops and "tflops" in peak_flops["fp32"]:
            tflops = peak_flops["fp32"]["tflops"]
        else:
            tflops = 1050  # B200 FP32 fallback

    flops_per_us = tflops * 1e6  # TFLOPS * 1e12 / 1e6 = flops/us
    theoretical_compute_us = flops / flops_per_us if flops_per_us > 0 else 0

    theoretical_best_us = max(theoretical_mem_us, theoretical_compute_us)
    theoretical_bound = "memory" if theoretical_mem_us >= theoretical_compute_us else "compute"

    return (
        round(theoretical_mem_us, 2),
        round(theoretical_compute_us, 2),
        round(theoretical_best_us, 2),
        theoretical_bound,
    )


def benchmark_single_T(T, warmup, iters, hw=None, device="cuda"):
    """Run all 8 op benchmarks for a single sequence length T. Returns (ops_results, base_costs)."""
    if hw is None:
        hw = {}
    tensors = allocate_tensors(T, device=device)
    torch.cuda.synchronize()

    ops_results = []
    base_costs = []

    for i, (name, op_type, bench_fn) in enumerate(zip(OP_NAMES, OP_TYPES, BENCH_FNS)):
        torch.cuda.empty_cache()

        wall_us, bytes_in, bytes_out, flops = bench_fn(tensors, warmup, iters)
        mem_us = bench_memory_only(bytes_in, bytes_out, warmup, iters)

        bound = "memory" if mem_us >= wall_us * 0.7 else "compute"
        total_bytes = bytes_in + bytes_out
        achieved_bw = total_bytes / (wall_us * 1e-6) / 1e9 if wall_us > 0 else 0
        achieved_tflops = flops / (wall_us * 1e-6) / 1e12 if wall_us > 0 else 0

        # Theoretical roofline
        theo_mem, theo_comp, theo_best, theo_bound = compute_roofline(
            bytes_in, bytes_out, flops, op_type, hw
        )

        ops_results.append({
            "name": name,
            "op_type": op_type,
            "wall_clock_us": round(wall_us, 2),
            "memory_only_us": round(mem_us, 2),
            "theoretical_mem_us": theo_mem,
            "theoretical_compute_us": theo_comp,
            "theoretical_best_us": theo_best,
            "theoretical_bound": theo_bound,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "flops": flops,
            "bound": bound,
            "achieved_bandwidth_GBps": round(achieved_bw, 1),
            "achieved_tflops": round(achieved_tflops, 3),
        })
        base_costs.append(round(theo_best, 2))

    # Free tensors
    del tensors
    torch.cuda.empty_cache()

    return ops_results, base_costs


def print_ops_table(T, ops_results, base_costs):
    """Print a formatted table for one sequence length."""
    header = (f"{'Op':<4} {'Name':<16} {'Type':<10} "
              f"{'TheoMem':>10} {'TheoComp':>10} {'TheoBest':>10} {'TheoBound':<9} "
              f"{'Wall(us)':>10} {'bytes_in':>10} {'bytes_out':>10} "
              f"{'FLOPs':>12}")
    print(f"\n--- T = {T} ---")
    print(header)
    print("-" * len(header))
    for i, (r, name, op_type) in enumerate(zip(ops_results, OP_NAMES, OP_TYPES)):
        print(f"{i:<4} {name:<16} {op_type:<10} "
              f"{r['theoretical_mem_us']:>10.1f} {r['theoretical_compute_us']:>10.1f} "
              f"{r['theoretical_best_us']:>10.1f} {r['theoretical_bound']:<9} "
              f"{r['wall_clock_us']:>10.1f} "
              f"{fmt_bytes(r['bytes_in']):>10} {fmt_bytes(r['bytes_out']):>10} "
              f"{fmt_flops(r['flops']):>12}")
    theo_total = sum(r['theoretical_best_us'] for r in ops_results)
    print(f"Theoretical best total: {theo_total:.1f} us  |  Measured total: {sum(r['wall_clock_us'] for r in ops_results):.1f} us\n")


def load_workload_seq_lens(jsonl_path):
    """Extract sorted unique seq_len values from a workload JSONL file."""
    seq_lens = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            seq_lens.append(d["workload"]["axes"]["seq_len"])
    return sorted(set(seq_lens))


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE DAG ops on GPU")
    parser.add_argument("--T", type=int, default=None,
                        help="Single sequence length to benchmark")
    parser.add_argument("--workloads", type=str, default=None,
                        help="Path to workload JSONL file (benchmarks all seq_lens)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--skip-hw", action="store_true",
                        help="Skip hardware characterization")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.set_device(0)

    # Determine sequence lengths to benchmark
    if args.workloads:
        seq_lens = load_workload_seq_lens(args.workloads)
        print(f"Loaded {len(seq_lens)} sequence lengths from workload file:")
        print(f"  {seq_lens}")
        if args.output is None:
            args.output = os.path.join(os.path.dirname(__file__), "dag_costs_all.json")
    elif args.T is not None:
        seq_lens = [args.T]
        if args.output is None:
            args.output = os.path.join(os.path.dirname(__file__), "dag_costs.json")
    else:
        # Default
        seq_lens = [4096]
        if args.output is None:
            args.output = os.path.join(os.path.dirname(__file__), "dag_costs.json")

    # Part 1: Hardware characterization
    if not args.skip_hw:
        hw = hardware_characterization(args.warmup, args.iters)
    else:
        hw = {"gpu_name": torch.cuda.get_device_name(0), "skipped": True}

    # Part 2+3: Per-op benchmarks for each sequence length
    all_results = []

    for idx, T in enumerate(seq_lens):
        print("=" * 70)
        print(f"[{idx + 1}/{len(seq_lens)}] Benchmarking T={T}")
        print("=" * 70)

        try:
            ops_results, base_costs = benchmark_single_T(
                T, args.warmup, args.iters, hw=hw, device=device
            )
            print_ops_table(T, ops_results, base_costs)

            measured_costs = [round(op["wall_clock_us"], 2) for op in ops_results]
            all_results.append({
                "T": T,
                "config": {"T": T, "H": H, "I": I, "E_local": E_LOCAL, "E_global": E_GLOBAL},
                "ops": ops_results,
                "base_costs": base_costs,  # theoretical roofline
                "measured_costs": measured_costs,
            })
        except Exception as e:
            print(f"  ERROR for T={T}: {e}")
            all_results.append({"T": T, "error": str(e)})

    # Write JSON
    if len(seq_lens) == 1:
        output_data = {
            "hardware": hw,
            **all_results[0],
        }
    else:
        output_data = {
            "hardware": hw,
            "workloads": all_results,
        }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to: {args.output}")

    # Print compact summary across all T values
    if len(seq_lens) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY: Theoretical best (roofline) per sequence length (us)")
        print("=" * 70)
        header = f"{'T':>8}  {'TheoBest':>10}  {'Measured':>10}  " + "  ".join(f"{n:>10}" for n in OP_NAMES)
        print(header)
        print("-" * len(header))
        for r in all_results:
            if "error" in r:
                print(f"{r['T']:>8}  {'ERROR':>10}")
            else:
                theo_total = sum(r["base_costs"])
                measured_total = sum(op["wall_clock_us"] for op in r["ops"])
                costs_str = "  ".join(f"{c:>10.1f}" for c in r["base_costs"])
                print(f"{r['T']:>8}  {theo_total:>10.1f}  {measured_total:>10.1f}  {costs_str}")


if __name__ == "__main__":
    main()
