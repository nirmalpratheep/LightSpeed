"""
Hardware Cost Primitives Benchmark for B200 GPU.

Measures the two fundamental costs for modeling ANY kernel DAG:
  1. Data movement: latency = f(bytes) for HBM read/write/copy, L2-warm
  2. Computation: latency = f(FLOPs, precision) for FP8/BF16/FP32 GEMM, FP32 pointwise

Usage:
    python deepdive/benchmark_dag_costs.py
    python deepdive/benchmark_dag_costs.py --output results.json
    python deepdive/benchmark_dag_costs.py --warmup 20 --iters 200

Output JSON contains raw measurements + derived formulas so any DAG can be costed as:
    cost = max(bytes / BW, flops / peak_throughput) + num_kernels * launch_overhead
"""

import argparse
import json
import os
from statistics import median

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def gpu_timer(fn, warmup=10, iters=100, flush_l2_each=False):
    """Time a GPU function using CUDA events. Returns median latency in us.

    If flush_l2_each=True, flushes L2 cache before each measured iteration
    to ensure cold-HBM measurements.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if flush_l2_each:
            _flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us

    return median(times)


# Use 512MB to ensure we exceed B200 L2 cache size
_L2_FLUSH_BUF = None

def _flush_l2():
    """Flush L2 cache by writing a large buffer (>= L2 size)."""
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    _L2_FLUSH_BUF.zero_()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Part 1: Data Movement Latency
# ---------------------------------------------------------------------------

DATA_SIZES = [
    1 * 1024,           # 1 KB
    4 * 1024,           # 4 KB
    16 * 1024,          # 16 KB
    64 * 1024,          # 64 KB
    256 * 1024,         # 256 KB
    1 * 1024 * 1024,    # 1 MB
    4 * 1024 * 1024,    # 4 MB
    16 * 1024 * 1024,   # 16 MB
    64 * 1024 * 1024,   # 64 MB
    256 * 1024 * 1024,  # 256 MB
    1024 * 1024 * 1024, # 1 GB
]


def _fmt_size(nbytes):
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.0f}GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.0f}MB"
    elif nbytes >= 1024:
        return f"{nbytes / 1024:.0f}KB"
    return f"{nbytes}B"


def bench_hbm_read(warmup, iters):
    """Measure HBM read bandwidth (cold L2) at various sizes."""
    print("  HBM Read (cold L2):")
    measurements = []
    for nbytes in DATA_SIZES:
        n_elements = nbytes // 4  # float32
        src = torch.randn(n_elements, device="cuda", dtype=torch.float32)

        # clone() = read src from HBM + write dst to HBM, but we isolate read
        # by using .clone() and measuring. The clone reads all of src.
        def fn():
            src.clone()

        us = gpu_timer(fn, warmup=warmup, iters=iters, flush_l2_each=True)
        # clone does read + write, so total bytes moved = 2 * nbytes
        bw_gbps = (2 * nbytes) / (us * 1e-6) / 1e9
        measurements.append({
            "bytes": nbytes,
            "latency_us": round(us, 3),
            "bandwidth_GBps": round(bw_gbps, 1),
        })
        print(f"    {_fmt_size(nbytes):>8s}: {us:>10.1f} us  {bw_gbps:>8.1f} GB/s")

        del src
        torch.cuda.empty_cache()

    return measurements


def bench_hbm_write(warmup, iters):
    """Measure HBM write bandwidth at various sizes."""
    print("  HBM Write (cold L2):")
    measurements = []
    for nbytes in DATA_SIZES:
        n_elements = nbytes // 4
        dst = torch.empty(n_elements, device="cuda", dtype=torch.float32)

        def fn():
            dst.zero_()

        us = gpu_timer(fn, warmup=warmup, iters=iters, flush_l2_each=True)
        bw_gbps = nbytes / (us * 1e-6) / 1e9  # write-only
        measurements.append({
            "bytes": nbytes,
            "latency_us": round(us, 3),
            "bandwidth_GBps": round(bw_gbps, 1),
        })
        print(f"    {_fmt_size(nbytes):>8s}: {us:>10.1f} us  {bw_gbps:>8.1f} GB/s")

        del dst
        torch.cuda.empty_cache()

    return measurements


def bench_hbm_copy(warmup, iters):
    """Measure HBM copy (read+write) bandwidth at various sizes."""
    print("  HBM Copy (cold L2):")
    measurements = []
    for nbytes in DATA_SIZES:
        n_elements = nbytes // 4
        src = torch.randn(n_elements, device="cuda", dtype=torch.float32)
        dst = torch.empty_like(src)

        def fn():
            dst.copy_(src)

        us = gpu_timer(fn, warmup=warmup, iters=iters, flush_l2_each=True)
        bw_gbps = (2 * nbytes) / (us * 1e-6) / 1e9  # read + write
        measurements.append({
            "bytes": nbytes,
            "latency_us": round(us, 3),
            "bandwidth_GBps": round(bw_gbps, 1),
        })
        print(f"    {_fmt_size(nbytes):>8s}: {us:>10.1f} us  {bw_gbps:>8.1f} GB/s")

        del src, dst
        torch.cuda.empty_cache()

    return measurements


def bench_l2_warm(warmup, iters):
    """Measure L2-warm read bandwidth (no L2 flush between iterations)."""
    print("  L2 Warm Read:")
    # Only test sizes that fit in L2 (up to ~128MB on B200)
    l2_sizes = [s for s in DATA_SIZES if s <= 128 * 1024 * 1024]
    measurements = []
    for nbytes in l2_sizes:
        n_elements = nbytes // 4
        src = torch.randn(n_elements, device="cuda", dtype=torch.float32)

        def fn():
            src.clone()

        # flush_l2_each=False -> data stays warm in L2 after first iteration
        us = gpu_timer(fn, warmup=warmup, iters=iters, flush_l2_each=False)
        bw_gbps = (2 * nbytes) / (us * 1e-6) / 1e9
        measurements.append({
            "bytes": nbytes,
            "latency_us": round(us, 3),
            "bandwidth_GBps": round(bw_gbps, 1),
        })
        print(f"    {_fmt_size(nbytes):>8s}: {us:>10.1f} us  {bw_gbps:>8.1f} GB/s")

        del src
        torch.cuda.empty_cache()

    return measurements


def fit_linear_model(measurements):
    """Fit latency_us = alpha + bytes / BW to measurements.

    Returns (alpha_us, bandwidth_GBps).
    Uses only the larger sizes (>= 1MB) for a stable fit.
    """
    large = [m for m in measurements if m["bytes"] >= 1 * 1024 * 1024]
    if len(large) < 2:
        large = measurements[-2:]

    x = np.array([m["bytes"] for m in large], dtype=np.float64)
    y = np.array([m["latency_us"] for m in large], dtype=np.float64)

    # Linear fit: y = a + b*x  ->  b = 1/BW (us per byte)
    coeffs = np.polyfit(x, y, 1)
    b, a = coeffs  # slope, intercept
    alpha_us = max(a, 0)  # launch overhead can't be negative
    bw_bytes_per_us = 1.0 / b if b > 0 else float("inf")
    bw_gbps = bw_bytes_per_us * 1e-3  # bytes/us -> GB/s

    return round(alpha_us, 3), round(bw_gbps, 1)


def benchmark_data_movement(warmup, iters):
    """Run all data movement benchmarks."""
    print("\n" + "=" * 70)
    print("Part 1: Data Movement Latency")
    print("=" * 70)

    hbm_read = bench_hbm_read(warmup, iters)
    hbm_write = bench_hbm_write(warmup, iters)
    hbm_copy = bench_hbm_copy(warmup, iters)
    l2_warm = bench_l2_warm(warmup, iters)

    # Fit linear models
    read_alpha, read_bw = fit_linear_model(hbm_read)
    write_alpha, write_bw = fit_linear_model(hbm_write)
    copy_alpha, copy_bw = fit_linear_model(hbm_copy)
    l2_alpha, l2_bw = fit_linear_model(l2_warm)

    print(f"\n  Derived formulas (latency_us = alpha + bytes / BW):")
    print(f"    HBM Read:  alpha={read_alpha:.1f} us, BW={read_bw:.0f} GB/s")
    print(f"    HBM Write: alpha={write_alpha:.1f} us, BW={write_bw:.0f} GB/s")
    print(f"    HBM Copy:  alpha={copy_alpha:.1f} us, BW={copy_bw:.0f} GB/s")
    print(f"    L2 Warm:   alpha={l2_alpha:.1f} us, BW={l2_bw:.0f} GB/s")

    return {
        "hbm_read": {
            "alpha_us": read_alpha,
            "peak_bandwidth_GBps": read_bw,
            "measurements": hbm_read,
        },
        "hbm_write": {
            "alpha_us": write_alpha,
            "peak_bandwidth_GBps": write_bw,
            "measurements": hbm_write,
        },
        "hbm_copy": {
            "alpha_us": copy_alpha,
            "peak_bandwidth_GBps": copy_bw,
            "measurements": hbm_copy,
        },
        "l2_warm": {
            "alpha_us": l2_alpha,
            "peak_bandwidth_GBps": l2_bw,
            "measurements": l2_warm,
        },
    }


# ---------------------------------------------------------------------------
# Part 2: Compute Latency
# ---------------------------------------------------------------------------

# Square sizes for GEMM saturation curve
GEMM_SIZES_SQUARE = [128, 256, 512, 1024, 2048, 4096, 8192]

# MoE-relevant aspect ratios: (M, N, K)
# Typical MoE shapes: small M (tokens per expert), large N and K (hidden dims)
GEMM_SIZES_MOE = [
    (1, 4096, 7168),      # T=1 per expert, GEMM1-like
    (4, 4096, 7168),
    (16, 4096, 7168),
    (64, 4096, 7168),
    (256, 4096, 7168),
    (1, 7168, 2048),      # T=1 per expert, GEMM2-like
    (16, 7168, 2048),
    (64, 7168, 2048),
    (256, 7168, 2048),
]


def bench_fp8_gemm(warmup, iters):
    """Measure FP8 tensor core GEMM throughput at various sizes."""
    print("  FP8 GEMM:")
    measurements = []

    # Square matrices
    for N in GEMM_SIZES_SQUARE:
        try:
            a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            sa = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            sb = torch.tensor(1.0, device="cuda", dtype=torch.float32)

            def fn():
                torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

            us = gpu_timer(fn, warmup=warmup, iters=iters)
            flops = 2 * N * N * N
            tflops = flops / (us * 1e-6) / 1e12
            measurements.append({
                "shape": f"{N}x{N}x{N}",
                "M": N, "N": N, "K": N,
                "flops": flops,
                "latency_us": round(us, 3),
                "tflops": round(tflops, 1),
            })
            print(f"    {N:>5}x{N:<5}: {us:>10.1f} us  {tflops:>8.1f} TFLOPS")
            del a, b, sa, sb
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {N:>5}x{N:<5}: ERROR - {e}")

    # MoE-shaped matrices
    print("  FP8 GEMM (MoE shapes):")
    for M, NN, K in GEMM_SIZES_MOE:
        try:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(NN, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            sa = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            sb = torch.tensor(1.0, device="cuda", dtype=torch.float32)

            def fn():
                torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

            us = gpu_timer(fn, warmup=warmup, iters=iters)
            flops = 2 * M * NN * K
            tflops = flops / (us * 1e-6) / 1e12
            measurements.append({
                "shape": f"{M}x{NN}x{K}",
                "M": M, "N": NN, "K": K,
                "flops": flops,
                "latency_us": round(us, 3),
                "tflops": round(tflops, 1),
            })
            print(f"    {M:>5}x{NN:>5}x{K:<5}: {us:>10.1f} us  {tflops:>8.1f} TFLOPS")
            del a, b, sa, sb
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    {M:>5}x{NN:>5}x{K:<5}: ERROR - {e}")

    return measurements


def bench_bf16_gemm(warmup, iters):
    """Measure BF16 tensor core GEMM throughput at various sizes."""
    print("  BF16 GEMM:")
    measurements = []

    for N in GEMM_SIZES_SQUARE:
        a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)

        def fn():
            torch.mm(a, b)

        us = gpu_timer(fn, warmup=warmup, iters=iters)
        flops = 2 * N * N * N
        tflops = flops / (us * 1e-6) / 1e12
        measurements.append({
            "shape": f"{N}x{N}x{N}",
            "M": N, "N": N, "K": N,
            "flops": flops,
            "latency_us": round(us, 3),
            "tflops": round(tflops, 1),
        })
        print(f"    {N:>5}x{N:<5}: {us:>10.1f} us  {tflops:>8.1f} TFLOPS")
        del a, b
        torch.cuda.empty_cache()

    return measurements


def bench_fp32_gemm(warmup, iters):
    """Measure FP32 GEMM throughput at various sizes."""
    print("  FP32 GEMM:")
    measurements = []

    for N in GEMM_SIZES_SQUARE:
        a = torch.randn(N, N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, N, device="cuda", dtype=torch.float32)

        def fn():
            torch.mm(a, b)

        us = gpu_timer(fn, warmup=warmup, iters=iters)
        flops = 2 * N * N * N
        tflops = flops / (us * 1e-6) / 1e12
        measurements.append({
            "shape": f"{N}x{N}x{N}",
            "M": N, "N": N, "K": N,
            "flops": flops,
            "latency_us": round(us, 3),
            "tflops": round(tflops, 1),
        })
        print(f"    {N:>5}x{N:<5}: {us:>10.1f} us  {tflops:>8.1f} TFLOPS")
        del a, b
        torch.cuda.empty_cache()

    return measurements


def bench_fp32_pointwise(warmup, iters):
    """Measure FP32 pointwise throughput (SwiGLU-like: silu(x) * y)."""
    print("  FP32 Pointwise (silu(x)*y):")
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000]
    measurements = []

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        def fn():
            torch.nn.functional.silu(x) * y

        us = gpu_timer(fn, warmup=warmup, iters=iters)
        # silu = x * sigmoid(x) = 3 ops, then * y = 1 op, total ~4 FLOPs/element
        # bytes: read 2*n*4 + write n*4 = 12n bytes
        flops = 4 * n
        total_bytes = 12 * n
        tflops = flops / (us * 1e-6) / 1e12
        bw_gbps = total_bytes / (us * 1e-6) / 1e9

        measurements.append({
            "elements": n,
            "flops": flops,
            "total_bytes": total_bytes,
            "latency_us": round(us, 3),
            "tflops": round(tflops, 4),
            "achieved_bw_GBps": round(bw_gbps, 1),
        })
        print(f"    {n:>12,} elem: {us:>10.1f} us  {tflops:>8.4f} TFLOPS  {bw_gbps:>8.1f} GB/s")

        del x, y
        torch.cuda.empty_cache()

    return measurements


def benchmark_compute(warmup, iters):
    """Run all compute benchmarks."""
    print("\n" + "=" * 70)
    print("Part 2: Compute Latency")
    print("=" * 70)

    fp8_gemm = bench_fp8_gemm(warmup, iters)
    bf16_gemm = bench_bf16_gemm(warmup, iters)
    fp32_gemm = bench_fp32_gemm(warmup, iters)
    fp32_pw = bench_fp32_pointwise(warmup, iters)

    # Extract peaks
    fp8_peak = max((m["tflops"] for m in fp8_gemm), default=0)
    bf16_peak = max((m["tflops"] for m in bf16_gemm), default=0)
    fp32_gemm_peak = max((m["tflops"] for m in fp32_gemm), default=0)
    fp32_peak = max((m["tflops"] for m in fp32_pw), default=0)
    pw_peak_bw = max((m["achieved_bw_GBps"] for m in fp32_pw), default=0)

    # Find saturation threshold (first size to reach 80% of peak)
    def find_saturation(measurements, key="tflops"):
        peak = max(m[key] for m in measurements)
        for m in measurements:
            if m[key] >= 0.8 * peak:
                return m.get("flops", m.get("elements", 0))
        return measurements[-1].get("flops", 0)

    fp8_sat = find_saturation(fp8_gemm) if fp8_gemm else 0
    bf16_sat = find_saturation(bf16_gemm) if bf16_gemm else 0
    fp32_gemm_sat = find_saturation(fp32_gemm) if fp32_gemm else 0

    print(f"\n  Peak throughput:")
    print(f"    FP8 GEMM:       {fp8_peak:.1f} TFLOPS (saturates at {fp8_sat:.0e} FLOPs)")
    print(f"    BF16 GEMM:      {bf16_peak:.1f} TFLOPS (saturates at {bf16_sat:.0e} FLOPs)")
    print(f"    FP32 GEMM:      {fp32_gemm_peak:.1f} TFLOPS (saturates at {fp32_gemm_sat:.0e} FLOPs)")
    print(f"    FP32 Pointwise: {fp32_peak:.4f} TFLOPS (BW-limited at {pw_peak_bw:.0f} GB/s)")

    return {
        "fp8_gemm": {
            "peak_tflops": fp8_peak,
            "saturation_threshold_flops": fp8_sat,
            "measurements": fp8_gemm,
        },
        "bf16_gemm": {
            "peak_tflops": bf16_peak,
            "saturation_threshold_flops": bf16_sat,
            "measurements": bf16_gemm,
        },
        "fp32_gemm": {
            "peak_tflops": fp32_gemm_peak,
            "saturation_threshold_flops": fp32_gemm_sat,
            "measurements": fp32_gemm,
        },
        "fp32_pointwise": {
            "peak_tflops": fp32_peak,
            "peak_bandwidth_GBps": pw_peak_bw,
            "measurements": fp32_pw,
        },
    }


# ---------------------------------------------------------------------------
# Part 3: Kernel Launch Overhead
# ---------------------------------------------------------------------------

def benchmark_launch_overhead(warmup=200, iters=2000):
    """Measure kernel launch overhead with trivial ops."""
    print("\n" + "=" * 70)
    print("Part 3: Kernel Launch Overhead")
    print("=" * 70)

    x = torch.tensor(1.0, device="cuda")

    def fn():
        x + x

    us = gpu_timer(fn, warmup=warmup, iters=iters)
    print(f"  Trivial op (scalar add): {us:.2f} us")

    # Also measure with a small tensor to see if there's size-dependent overhead
    y = torch.ones(256, device="cuda")

    def fn2():
        y + y

    us2 = gpu_timer(fn2, warmup=warmup, iters=iters)
    print(f"  Small op (256-element add): {us2:.2f} us")

    overhead = min(us, us2)
    print(f"  Launch overhead estimate: {overhead:.2f} us")

    return {
        "scalar_add_us": round(us, 2),
        "small_vector_add_us": round(us2, 2),
        "estimated_overhead_us": round(overhead, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hardware Cost Primitives Benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Measurement iterations")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.set_device(0)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # Run all benchmarks
    data_movement = benchmark_data_movement(args.warmup, args.iters)
    compute = benchmark_compute(args.warmup, args.iters)
    launch = benchmark_launch_overhead()

    # Assemble output
    output = {
        "gpu": gpu_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "config": {"warmup": args.warmup, "iters": args.iters},
        "data_movement": data_movement,
        "compute": compute,
        "kernel_launch_overhead": launch,
    }

    # Print cost model summary
    copy_bw = data_movement["hbm_copy"]["peak_bandwidth_GBps"]
    fp8_peak = compute["fp8_gemm"]["peak_tflops"]
    overhead = launch["estimated_overhead_us"]

    print("\n" + "=" * 70)
    print("COST MODEL SUMMARY")
    print("=" * 70)
    print(f"  For any fused kernel:")
    print(f"    cost_us = max(total_bytes / {copy_bw:.0f}e3, flops / {fp8_peak:.0f}e6) + N_kernels * {overhead:.1f}")
    print(f"")
    print(f"  Where:")
    print(f"    total_bytes = bytes read from HBM + bytes written to HBM")
    print(f"    flops = total FLOPs (use FP8 rate for GEMMs, FP32 rate for pointwise)")
    print(f"    N_kernels = number of separate kernel launches")

    # Write JSON
    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "hw_cost_primitives.json")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
