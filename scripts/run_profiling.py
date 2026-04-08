"""
FlashInfer-Bench Modal Cloud NCU + Nsys Profiling Runner.

Automatically packs the solution from source files and runs
NCU and/or Nsys profiling on NVIDIA B200 GPUs via Modal for selected workloads.
The .ncu-rep / .nsys-rep files are returned and saved to the current directory.

Profiles only the kernels inside the timed execution region (matching
how the benchmark measures latency), filtering out warmup and setup.

Usage:
    modal run scripts/run_profiling.py --tier large
    modal run scripts/run_profiling.py --tier large --tool ncu
    modal run scripts/run_profiling.py --tier large --tool nsys
    modal run scripts/run_profiling.py --tier large --tool both
    modal run scripts/run_profiling.py --ncu-set detailed

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution
from flashinfer_bench.data import Workload

app = modal.App("flashinfer-profiling")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu24.04", add_python="3.12")
    .run_commands(
        "apt-get update -qq && apt-get install -y --no-install-recommends nsight-systems-cli 2>/dev/null"
        " || apt-get install -y --no-install-recommends cuda-nsight-systems-12-9 2>/dev/null"
        " || true"
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu(solution: Solution, workload: Workload, workload_idx: int, seq_len: int,
            ncu_set: str = "detailed") -> dict:
    """Run NCU profiling on Modal B200. Returns log + .ncu-rep bytes.

    Uses the _solution_runner which wraps the actual kernel call in an NVTX
    range 'flashinfer_bench_ncu_profile'. NCU filters on this range so only
    the kernels from the actual execution are profiled (not warmup/setup).
    This matches how the benchmark measures latency.
    """
    import subprocess
    import tempfile

    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solution.definition]

    rep_name = f"kernel_wl{workload_idx:02d}_seq{seq_len}"
    rep_path = f"/tmp/{rep_name}"

    with tempfile.TemporaryDirectory(prefix="fib_ncu_") as data_dir:
        data_path = Path(data_dir)
        (data_path / "definition.json").write_text(definition.model_dump_json())
        (data_path / "solution.json").write_text(solution.model_dump_json())
        (data_path / "workload.json").write_text(workload.model_dump_json())

        cmd = [
            "ncu",
            "-f",
            "--set", ncu_set,
            "--replay-mode", "kernel",
            "--target-processes", "all",
            "--import-source", "yes",
            # Only profile kernels inside the actual execution NVTX range
            # (skips warmup, JIT compilation, input allocation, etc.)
            "--nvtx",
            "--nvtx-include", "flashinfer_bench_ncu_profile/",
            "-o", rep_path,
            sys.executable, "-u", "-m",
            "flashinfer_bench.agents._solution_runner",
            "--data-dir", data_dir,
            "--device", "cuda:0",
            "--trace-set-path", TRACE_SET_PATH,
        ]

        print(f"  [NCU] Workload {workload_idx} (seq_len={seq_len})...")
        print(f"  [NCU] Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    output = result.stdout + result.stderr
    rep_file = Path(f"{rep_path}.ncu-rep")
    rep_bytes = rep_file.read_bytes() if rep_file.exists() else b""

    return {
        "name": rep_name,
        "log": output[-5000:],
        "returncode": result.returncode,
        "rep_bytes": rep_bytes,
    }


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_nsys(solution: Solution, workload: Workload, workload_idx: int, seq_len: int) -> dict:
    """Run Nsight Systems profiling on Modal B200. Returns log + .nsys-rep bytes.

    Captures CUDA API calls, kernel launches, NVTX ranges, and GPU utilization
    timeline. The NVTX range 'flashinfer_bench_ncu_profile' marks the timed
    execution window — open the .nsys-rep in Nsight Systems GUI to inspect it.
    """
    import subprocess
    import tempfile

    from flashinfer_bench import TraceSet

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solution.definition]

    rep_name = f"nsys_wl{workload_idx:02d}_seq{seq_len}"
    rep_path = f"/tmp/{rep_name}"

    with tempfile.TemporaryDirectory(prefix="fib_nsys_") as data_dir:
        data_path = Path(data_dir)
        (data_path / "definition.json").write_text(definition.model_dump_json())
        (data_path / "solution.json").write_text(solution.model_dump_json())
        (data_path / "workload.json").write_text(workload.model_dump_json())

        cmd = [
            "nsys", "profile",
            "-f", "true",
            "--output", rep_path,
            "--trace", "cuda,nvtx,osrt",
            "--stats", "true",
            sys.executable, "-u", "-m",
            "flashinfer_bench.agents._solution_runner",
            "--data-dir", data_dir,
            "--device", "cuda:0",
            "--trace-set-path", TRACE_SET_PATH,
        ]

        print(f"  [NSYS] Workload {workload_idx} (seq_len={seq_len})...")
        print(f"  [NSYS] Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    output = result.stdout + result.stderr
    rep_file = Path(f"{rep_path}.nsys-rep")
    rep_bytes = rep_file.read_bytes() if rep_file.exists() else b""

    return {
        "name": rep_name,
        "log": output[-5000:],
        "returncode": result.returncode,
        "rep_bytes": rep_bytes,
    }


# Representative workload indices — one per kernel tier to minimize profiling cost.
#   wl01 (seq=1)     → tiny   (≤8)
#   wl03 (seq=80)    → small  (9–128)
#   wl04 (seq=901)   → medium (129–2048)
#   wl08 (seq=14107) → large  (>2048)
REPRESENTATIVE_WORKLOADS = {1, 3, 4, 8}

TIER_WORKLOAD_IDX = {
    "tiny":   {1},
    "small":  {3},
    "medium": {4},
    "large":  {8},
    "all":    {1, 3, 4, 8},
}


def _load_workloads(definition: str, representative_only: bool = True) -> list[dict]:
    """Load workloads from the JSONL file.

    If representative_only is True (default), loads only one workload per
    kernel tier to cover the full complexity with minimal profiling cost.
    """
    workload_file = (
        PROJECT_ROOT / "mlsys26-contest" / "workloads" / "moe" / f"{definition}.jsonl"
    )
    workloads = []
    with open(workload_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if representative_only and i not in REPRESENTATIVE_WORKLOADS:
                continue
            data = json.loads(line)
            wl = Workload.model_validate(data["workload"])
            seq_len = data["workload"]["axes"].get("seq_len", 0)
            workloads.append({"workload": wl, "idx": i, "seq_len": seq_len})
    return workloads


def _save_result(result: dict, ext: str, tool: str) -> None:
    rep_name = result["name"]
    if result["returncode"] != 0:
        print(f"  [{tool}] {rep_name}: FAILED (rc={result['returncode']})")
        print(result["log"])
    else:
        rep_bytes = result["rep_bytes"]
        if rep_bytes:
            local_path = Path(f"{rep_name}.{ext}")
            local_path.write_bytes(rep_bytes)
            print(f"  [{tool}] {rep_name}: SUCCESS → {local_path} ({len(rep_bytes):,} bytes)")
        else:
            print(f"  [{tool}] {rep_name}: SUCCESS but no .{ext} file generated")
        print(result["log"])


@app.local_entrypoint()
def main(
    max_workloads: int = 0,
    ncu_set: str = "detailed",
    all_workloads: bool = False,
    tier: str = "all",
    tool: str = "both",
):
    """Pack solution and run NCU and/or Nsys profiling on Modal for workloads.

    Args:
        tier: Which tier to profile — tiny | small | medium | large | all (default: all)
        tool: Which profiler to run — ncu | nsys | both (default: both)
        max_workloads: Cap number of workloads (0 = no cap).
        ncu_set: NCU metric set — detailed | full | source (default: detailed).
        all_workloads: Profile every workload in the definition (not just representatives).

    Examples:
        modal run scripts/run_profiling.py --tier large
        modal run scripts/run_profiling.py --tier large --tool ncu
        modal run scripts/run_profiling.py --tier large --tool nsys
        modal run scripts/run_profiling.py --tier large --tool both
    """
    from scripts.pack_solution import pack_solution

    if tier not in TIER_WORKLOAD_IDX:
        raise ValueError(f"Unknown tier '{tier}'. Choose from: {list(TIER_WORKLOAD_IDX)}")
    if tool not in {"ncu", "nsys", "both"}:
        raise ValueError(f"Unknown tool '{tool}'. Choose from: ncu | nsys | both")

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text(encoding="utf-8"))
    print(f"Loaded: {solution.name} ({solution.definition})")

    # Load workloads filtered by tier
    selected_indices = TIER_WORKLOAD_IDX[tier]
    workloads = _load_workloads(solution.definition, representative_only=not all_workloads)
    if tier != "all":
        workloads = [wl for wl in workloads if wl["idx"] in selected_indices]
    if max_workloads > 0:
        workloads = workloads[:max_workloads]
    print(f"\nFound {len(workloads)} workloads to profile (tool={tool})")
    for wl in workloads:
        print(f"  wl{wl['idx']:02d}: seq_len={wl['seq_len']}")

    run_ncu_ = tool in {"ncu", "both"}
    run_nsys_ = tool in {"nsys", "both"}

    for wl in workloads:
        print("\n" + "=" * 60)
        print(f"Workload wl{wl['idx']:02d} (seq_len={wl['seq_len']})")
        print("=" * 60)

        if run_ncu_:
            result = run_ncu.remote(
                solution, wl["workload"], wl["idx"], wl["seq_len"],
                ncu_set=ncu_set,
            )
            _save_result(result, "ncu-rep", "NCU")

        if run_nsys_:
            result = run_nsys.remote(
                solution, wl["workload"], wl["idx"], wl["seq_len"],
            )
            _save_result(result, "nsys-rep", "NSYS")

    print("\nProfiling complete.")
