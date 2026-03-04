"""
FlashInfer-Bench Modal Cloud Profiling Runner.

Automatically packs the solution from source files and runs
compute-sanitizer and NCU profiling on NVIDIA B200 GPUs via Modal.

Usage:
    modal run scripts/run_profiling.py
    modal run scripts/run_profiling.py --sanitizer-only
    modal run scripts/run_profiling.py --ncu-only

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.data import Workload

app = modal.App("flashinfer-profiling")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={TRACE_SET_PATH: trace_volume})
def run_sanitizer(solution: Solution, workload: Workload, sanitizer_types: list[str]) -> str:
    """Run compute-sanitizer on Modal B200."""
    from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

    return flashinfer_bench_run_sanitizer(
        solution=solution,
        workload=workload,
        trace_set_path=TRACE_SET_PATH,
        sanitizer_types=sanitizer_types,
        timeout=300,
    )


@app.function(image=image, gpu="B200:1", timeout=600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu(solution: Solution, workload: Workload, set: str = "detailed", page: str = "details") -> str:
    """Run NCU profiling on Modal B200."""
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    return flashinfer_bench_run_ncu(
        solution=solution,
        workload=workload,
        trace_set_path=TRACE_SET_PATH,
        set=set,
        page=page,
        timeout=120,
    )


@app.local_entrypoint()
def main(
    sanitizer_only: bool = False,
    ncu_only: bool = False,
    ncu_set: str = "detailed",
    ncu_page: str = "details",
):
    """Pack solution and run profiling on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text(encoding='utf-8'))
    print(f"Loaded: {solution.name} ({solution.definition})")

    # Load first workload
    import json
    workload_dir = PROJECT_ROOT / "mlsys26-contest" / "workloads" / "moe"
    workload_file = workload_dir / f"{solution.definition}.jsonl"
    with open(workload_file, encoding='utf-8') as f:
        first_line = f.readline()
    workload = Workload.model_validate(json.loads(first_line)["workload"])
    print(f"Loaded workload: {workload.uuid[:8]}... (seq_len={workload.axes.get('seq_len', '?')})")

    run_both = not sanitizer_only and not ncu_only

    if run_both or sanitizer_only:
        print("\n" + "=" * 60)
        print("Running compute-sanitizer on Modal B200...")
        print("=" * 60)
        sanitizer_output = run_sanitizer.remote(
            solution, workload,
            sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
        )
        print(sanitizer_output)

    if run_both or ncu_only:
        print("\n" + "=" * 60)
        print("Running NCU profiling on Modal B200...")
        print("=" * 60)
        ncu_output = run_ncu.remote(solution, workload, set=ncu_set, page=ncu_page)
        print(ncu_output)

    print("\nProfiling complete.")
