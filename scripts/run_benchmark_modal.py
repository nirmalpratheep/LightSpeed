"""
Run hardware cost primitives benchmark on Modal B200 GPU.

Usage:
    modal run scripts/run_benchmark_modal.py
    modal run scripts/run_benchmark_modal.py --warmup 20 --iters 100
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("hw-cost-primitives")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .add_local_dir(str(PROJECT_ROOT / "deepdive"), remote_path="/root/deepdive")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
)
def run_benchmark(warmup: int = 10, iters: int = 50) -> dict:
    """Run benchmark_dag_costs.py on B200 and return JSON results."""
    import json
    import subprocess

    cmd = [
        "python", "/root/deepdive/benchmark_dag_costs.py",
        "--warmup", str(warmup),
        "--iters", str(iters),
        "--output", "/root/hw_costs.json",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed (exit {result.returncode}):\n{result.stderr}")

    with open("/root/hw_costs.json") as f:
        return json.load(f)


@app.local_entrypoint()
def main(warmup: int = 10, iters: int = 50):
    """Run hardware cost primitives benchmark on Modal B200."""
    import json

    print("Launching hardware cost primitives benchmark on Modal B200...")

    results = run_benchmark.remote(warmup=warmup, iters=iters)

    # Save results locally
    output_file = PROJECT_ROOT / "deepdive" / "hw_cost_primitives.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    dm = results.get("data_movement", {})
    comp = results.get("compute", {})
    launch = results.get("kernel_launch_overhead", {})

    print("\n--- Cost Model Summary ---")
    if "hbm_copy" in dm:
        print(f"  HBM Copy BW: {dm['hbm_copy']['peak_bandwidth_GBps']:.0f} GB/s")
    if "l2_warm" in dm:
        print(f"  L2 Warm BW:  {dm['l2_warm']['peak_bandwidth_GBps']:.0f} GB/s")
    if "fp8_gemm" in comp:
        print(f"  FP8 Peak:    {comp['fp8_gemm']['peak_tflops']:.0f} TFLOPS")
    if "bf16_gemm" in comp:
        print(f"  BF16 Peak:   {comp['bf16_gemm']['peak_tflops']:.0f} TFLOPS")
    if launch:
        print(f"  Launch OH:   {launch['estimated_overhead_us']:.1f} us")
