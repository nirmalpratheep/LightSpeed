"""
Local NCU & NSYS Profiling Runner for kernel.py.

Runs ncu and nsys directly as subprocesses (no Modal).
Designed for machines with direct GPU access (e.g., DGX, cloud VMs).

Usage:
    python scripts/run_local_profiling.py --max-workloads 1
    python scripts/run_local_profiling.py --ncu-only --max-workloads 3
    python scripts/run_local_profiling.py --nsys-only --max-workloads 3
    python scripts/run_local_profiling.py --output-dir ./reports

Requirements:
    - ncu (Nsight Compute) and/or nsys (Nsight Systems) in PATH
    - GPU with appropriate driver
    - FIB_DATASET_PATH env var or --trace-set-path pointing to the trace set
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.data import Workload


def _find_tool(name: str) -> str:
    """Find ncu or nsys binary, searching PATH and common install dirs."""
    import shutil
    import glob

    found = shutil.which(name)
    if found:
        return found

    search_patterns = [
        f"/opt/nvidia/nsight-compute/**/{name}",
        f"/opt/nvidia/nsight-systems/**/bin/{name}",
        f"/usr/local/cuda/bin/{name}",
        f"/usr/local/bin/{name}",
    ]
    for pattern in search_patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[-1]

    raise FileNotFoundError(
        f"Could not find '{name}'. Install Nsight tools or add to PATH."
    )


def _write_data_files(
    data_dir: Path,
    solution: Solution,
    workload: Workload,
    trace_set_path: str,
):
    """Write definition, solution, and workload JSON files for _solution_runner."""
    data_dir.mkdir(parents=True, exist_ok=True)

    trace_set = TraceSet.from_path(trace_set_path)
    definition = trace_set.definitions[solution.definition]

    (data_dir / "definition.json").write_text(definition.model_dump_json())
    (data_dir / "solution.json").write_text(solution.model_dump_json())
    (data_dir / "workload.json").write_text(workload.model_dump_json())


def _load_workloads(definition: str) -> list[dict]:
    """Load all workloads from the JSONL file."""
    workload_file = (
        PROJECT_ROOT / "mlsys26-contest" / "workloads" / "moe" / f"{definition}.jsonl"
    )
    workloads = []
    with open(workload_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            wl = Workload.model_validate(data["workload"])
            seq_len = data["workload"]["axes"].get("seq_len", 0)
            workloads.append({"workload": wl, "idx": i, "seq_len": seq_len})
    return workloads


# ═════════════════════════════════════════════════════════════
# NCU Profiling
# ═════════════════════════════════════════════════════════════
def run_ncu(
    solution: Solution,
    workload: Workload,
    workload_idx: int,
    seq_len: int,
    output_dir: Path,
    trace_set_path: str,
    device: str = "cuda:0",
    timeout: int = 600,
) -> str:
    """Run NCU profiling, generating a .ncu-rep file."""
    ncu_bin = _find_tool("ncu")
    rep_name = f"kernel_wl{workload_idx:02d}_seq{seq_len}"
    rep_path = str(output_dir / rep_name)

    with tempfile.TemporaryDirectory(prefix="fib_ncu_") as data_dir:
        _write_data_files(Path(data_dir), solution, workload, trace_set_path)

        cmd = [
            ncu_bin,
            "-f",
            "--set", "full",
            "--target-processes", "all",
            "--import-source", "yes",
            "--section", "SpeedOfLight",
            "--section", "MemoryWorkloadAnalysis",
            "--section", "ComputeWorkloadAnalysis",
            "--section", "Occupancy",
            "-o", rep_path,
            sys.executable, "-u", "-m",
            "flashinfer_bench.agents._solution_runner",
            "--data-dir", data_dir,
            "--device", device,
            "--trace-set-path", trace_set_path,
        ]

        print(f"  [NCU] Workload {workload_idx} (seq_len={seq_len})...")
        print(f"  [NCU] Using: {ncu_bin}")
        print(f"  [NCU] Output: {rep_path}.ncu-rep")
        print(f"  [NCU] Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode == 0:
        summary = f"  [NCU] Workload {workload_idx}: SUCCESS → {rep_name}.ncu-rep"
    else:
        summary = f"  [NCU] Workload {workload_idx}: FAILED (rc={result.returncode})"
        if result.stderr:
            summary += f"\n  STDERR:\n{result.stderr[-3000:]}"
        if result.stdout:
            summary += f"\n  STDOUT:\n{result.stdout[-3000:]}"

    return summary


# ═════════════════════════════════════════════════════════════
# NSYS Profiling
# ═════════════════════════════════════════════════════════════
def run_nsys(
    solution: Solution,
    workload: Workload,
    workload_idx: int,
    seq_len: int,
    output_dir: Path,
    trace_set_path: str,
    device: str = "cuda:0",
    timeout: int = 600,
) -> str:
    """Run NSYS profiling, generating a .nsys-rep file."""
    nsys_bin = _find_tool("nsys")
    rep_name = f"kernel_wl{workload_idx:02d}_seq{seq_len}"
    rep_path = str(output_dir / rep_name)

    with tempfile.TemporaryDirectory(prefix="fib_nsys_") as data_dir:
        _write_data_files(Path(data_dir), solution, workload, trace_set_path)

        cmd = [
            nsys_bin, "profile",
            "--output", rep_path,
            "--force-overwrite", "true",
            "--trace", "cuda,nvtx,osrt",
            "--cuda-memory-usage", "true",
            "--stats", "true",
            sys.executable, "-u", "-m",
            "flashinfer_bench.agents._solution_runner",
            "--data-dir", data_dir,
            "--device", device,
            "--trace-set-path", trace_set_path,
        ]

        print(f"  [NSYS] Workload {workload_idx} (seq_len={seq_len})...")
        print(f"  [NSYS] Using: {nsys_bin}")
        print(f"  [NSYS] Output: {rep_path}.nsys-rep")
        print(f"  [NSYS] Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode == 0:
        summary = f"  [NSYS] Workload {workload_idx}: SUCCESS → {rep_name}.nsys-rep"
        # Print stats if available
        if result.stdout:
            summary += f"\n{result.stdout[-5000:]}"
    else:
        summary = f"  [NSYS] Workload {workload_idx}: FAILED (rc={result.returncode})"
        if result.stderr:
            summary += f"\n  STDERR:\n{result.stderr[-3000:]}"
        if result.stdout:
            summary += f"\n  STDOUT:\n{result.stdout[-3000:]}"

    return summary


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Run NCU & NSYS profiling locally for each workload."
    )
    parser.add_argument("--ncu-only", action="store_true", help="Run NCU profiling only")
    parser.add_argument("--nsys-only", action="store_true", help="Run NSYS profiling only")
    parser.add_argument("--max-workloads", type=int, default=0, help="Max workloads to profile (0 = all)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports", help="Output directory for reports")
    parser.add_argument("--trace-set-path", type=str, default=None, help="Path to trace set (default: FIB_DATASET_PATH env)")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per workload in seconds")
    args = parser.parse_args()

    # Resolve trace set path
    trace_set_path = args.trace_set_path or os.environ.get("FIB_DATASET_PATH")
    if not trace_set_path:
        print("ERROR: Provide --trace-set-path or set FIB_DATASET_PATH env var.", file=sys.stderr)
        sys.exit(1)

    # Pack solution
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(
        solution_path.read_text(encoding="utf-8")
    )
    print(f"Loaded: {solution.name} ({solution.definition})")

    # Load workloads
    workloads = _load_workloads(solution.definition)
    if args.max_workloads > 0:
        workloads = workloads[: args.max_workloads]
    print(f"\nFound {len(workloads)} workloads to profile")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    run_both = not args.ncu_only and not args.nsys_only

    # ── NCU ───────────────────────────────────────────────────
    if run_both or args.ncu_only:
        print("\n" + "=" * 60)
        print("Running NCU profiling...")
        print("=" * 60)
        for wl in workloads:
            result = run_ncu(
                solution, wl["workload"], wl["idx"], wl["seq_len"],
                args.output_dir, trace_set_path, args.device, args.timeout,
            )
            print(result)

    # ── NSYS ──────────────────────────────────────────────────
    if run_both or args.nsys_only:
        print("\n" + "=" * 60)
        print("Running NSYS profiling...")
        print("=" * 60)
        for wl in workloads:
            result = run_nsys(
                solution, wl["workload"], wl["idx"], wl["seq_len"],
                args.output_dir, trace_set_path, args.device, args.timeout,
            )
            print(result)

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print(f"Reports saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
