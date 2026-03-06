"""
Run benchmark_dag_costs.py on Modal B200 GPU.

Usage:
    modal run scripts/run_benchmark_modal.py
    modal run scripts/run_benchmark_modal.py --T 4096
    modal run scripts/run_benchmark_modal.py --workloads
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("moe-dag-benchmark")

# Mount the deepdive directory and workloads into the container
project_mount = modal.Mount.from_local_dir(
    str(PROJECT_ROOT / "deepdive"),
    remote_path="/root/deepdive",
)
workload_mount = modal.Mount.from_local_dir(
    str(PROJECT_ROOT / "mlsys26-contest" / "workloads" / "moe"),
    remote_path="/root/workloads",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
)

WORKLOAD_JSONL = "/root/workloads/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl"


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    mounts=[project_mount, workload_mount],
)
def run_dag_benchmark(T: int = None, use_workloads: bool = False, warmup: int = 10, iters: int = 100, skip_hw: bool = False) -> dict:
    """Run benchmark_dag_costs.py on B200 and return JSON results."""
    import json
    import subprocess

    cmd = ["python", "/root/deepdive/benchmark_dag_costs.py"]
    output_path = "/root/dag_costs_results.json"

    if use_workloads:
        cmd += ["--workloads", WORKLOAD_JSONL]
    elif T is not None:
        cmd += ["--T", str(T)]
    else:
        cmd += ["--T", "4096"]

    cmd += ["--warmup", str(warmup), "--iters", str(iters), "--output", output_path]

    if skip_hw:
        cmd += ["--skip-hw"]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed (exit {result.returncode}):\n{result.stderr}")

    with open(output_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(T: int = None, workloads: bool = False, warmup: int = 10, iters: int = 100, skip_hw: bool = False):
    """Run DAG cost benchmark on Modal B200."""
    import json

    mode = "all workloads" if workloads else f"T={T or 4096}"
    print(f"Launching DAG benchmark on Modal B200 ({mode})...")

    results = run_dag_benchmark.remote(
        T=T,
        use_workloads=workloads,
        warmup=warmup,
        iters=iters,
        skip_hw=skip_hw,
    )

    # Save results locally
    output_dir = PROJECT_ROOT / "deepdive"
    output_file = output_dir / ("dag_costs_all.json" if workloads else "dag_costs.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    if "workloads" in results:
        print(f"\n{'T':>8}  {'TheoBest(us)':>12}  {'Measured(us)':>12}")
        print("-" * 36)
        for r in results["workloads"]:
            if "error" in r:
                print(f"{r['T']:>8}  {'ERROR':>12}")
            else:
                theo = sum(r["base_costs"])
                measured = sum(r["measured_costs"])
                print(f"{r['T']:>8}  {theo:>12.1f}  {measured:>12.1f}")
    elif "ops" in results:
        print(f"\nT={results.get('T', '?')}")
        print(f"  Theoretical best: {sum(results['base_costs']):.1f} us")
        print(f"  Measured:         {sum(results['measured_costs']):.1f} us")
