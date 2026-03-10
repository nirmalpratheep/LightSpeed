"""
NCU & NSYS Profiling Runner for kernel.py on Modal B200.

Generates separate .ncu-rep and .nsys-rep files per workload.
Uses flashinfer_bench.agents._solution_runner as the profiled subprocess,
matching the same pattern as the built-in flashinfer_bench NCU agent.

Usage:
    modal run scripts/run_ncu_nsys_profiling.py
    modal run scripts/run_ncu_nsys_profiling.py --ncu-only
    modal run scripts/run_ncu_nsys_profiling.py --nsys-only
    modal run scripts/run_ncu_nsys_profiling.py --max-workloads 5

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.data import Workload

app = modal.App("flashinfer-ncu-nsys-profiling")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
output_volume = modal.Volume.from_name("profiling-reports", create_if_missing=True)
TRACE_SET_PATH = "/data"
OUTPUT_PATH = "/reports"

# Use debian_slim base (better Modal driver compat) and install profiling tools
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "wget", "gnupg2", "software-properties-common",
    )
    .run_commands(
        # Add NVIDIA apt repo for nsight tools
        "wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | apt-key add -",
        'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda.list',
        "apt-get update",
        # Install pinned versions of Nsight Compute and Nsight Systems
        "apt-get install -y --no-install-recommends nsight-compute-2025.4.1 nsight-systems-2025.5.2",
        "apt-get clean && rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def _find_tool(name: str) -> str:
    """Find ncu or nsys binary on the container, searching common paths."""
    import shutil
    import glob

    # Check PATH first
    found = shutil.which(name)
    if found:
        return found

    # Common install locations for nsight tools
    search_dirs = [
        f"/opt/nvidia/nsight-compute/*/{name}",
        f"/opt/nvidia/nsight-systems/**/bin/{name}",
        f"/usr/local/cuda/nsight-compute/*/{name}",
        f"/usr/local/cuda/bin/{name}",
        f"/usr/local/nv/nsight-compute/*/{name}",
        f"/usr/local/nv/nsight-systems/**/bin/{name}",
        f"/usr/lib/nsight-compute/*/{name}",
        f"/usr/lib/nsight-systems/**/bin/{name}",
    ]
    for pattern in search_dirs:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[-1]  # newest version

    raise FileNotFoundError(
        f"Could not find '{name}'. Searched PATH and common install dirs."
    )


def _write_data_files(data_dir: str, solution: Solution, workload: Workload, trace_set_path: str):
    """Write definition, solution, and workload JSON files for _solution_runner."""
    from pathlib import Path
    from flashinfer_bench import TraceSet

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    trace_set = TraceSet.from_path(trace_set_path)
    definition = trace_set.definitions[solution.definition]

    (data_path / "definition.json").write_text(definition.model_dump_json())
    (data_path / "solution.json").write_text(solution.model_dump_json())
    (data_path / "workload.json").write_text(workload.model_dump_json())


@app.function(
    image=image,
    gpu="B200:1",
    timeout=900,
    volumes={TRACE_SET_PATH: trace_volume, OUTPUT_PATH: output_volume},
)
def discover_tools() -> str:
    """Discover available NCU/NSYS binaries and GPU info on the container."""
    import subprocess
    import shutil
    import glob

    lines = []

    # GPU info
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        lines.append(f"GPU: {r.stdout.strip()}")
    except Exception as e:
        lines.append(f"nvidia-smi: {e}")

    # Find NCU
    for name in ["ncu", "nsys"]:
        found = shutil.which(name)
        lines.append(f"{name} in PATH: {found or 'NOT FOUND'}")

        # Search common dirs
        for pattern in [
            f"/opt/nvidia/**/{name}",
            f"/usr/local/**/{name}",
            f"/usr/lib/**/{name}",
            f"/usr/bin/{name}",
        ]:
            matches = glob.glob(pattern, recursive=True)
            for m in matches[:5]:
                lines.append(f"  found: {m}")

    # Version checks
    for name in ["ncu", "nsys"]:
        binary = shutil.which(name)
        if binary:
            try:
                r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
                lines.append(f"{name} --version: {r.stdout.strip()[:200]}")
            except Exception as e:
                lines.append(f"{name} --version failed: {e}")

    return "\n".join(lines)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=900,
    volumes={TRACE_SET_PATH: trace_volume, OUTPUT_PATH: output_volume},
)
def run_ncu_profile(
    solution: Solution,
    workload: Workload,
    workload_idx: int,
    seq_len: int,
) -> str:
    """Run NCU profiling, generating a .ncu-rep file."""
    import subprocess
    import sys
    import tempfile

    ncu_bin = _find_tool("ncu")
    rep_name = f"kernel_wl{workload_idx:02d}_seq{seq_len}"
    rep_path = f"{OUTPUT_PATH}/{rep_name}"

    with tempfile.TemporaryDirectory(prefix="fib_ncu_") as data_dir:
        _write_data_files(data_dir, solution, workload, TRACE_SET_PATH)

        cmd = [
            ncu_bin,
            "--set", "full",
            "--nvtx",
            "--nvtx-include", "flashinfer_bench_ncu_profile",
            "--export", rep_path,
            "-f",
            "--target-processes", "all",
            sys.executable, "-u", "-m",
            "flashinfer_bench.agents._solution_runner",
            "--data-dir", data_dir,
            "--device", "cuda:0",
            "--trace-set-path", TRACE_SET_PATH,
        ]

        print(f"[NCU] Profiling workload {workload_idx} (seq_len={seq_len})...")
        print(f"[NCU] Using: {ncu_bin}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    output_volume.commit()

    summary = f"[NCU] Workload {workload_idx} (seq_len={seq_len}): "
    if result.returncode == 0:
        summary += f"SUCCESS -> {rep_name}.ncu-rep"
    else:
        summary += f"FAILED (rc={result.returncode})"
        if result.stderr:
            summary += f"\nSTDERR:\n{result.stderr[-2000:]}"
        if result.stdout:
            summary += f"\nSTDOUT:\n{result.stdout[-2000:]}"

    return summary


@app.function(
    image=image,
    gpu="B200:1",
    timeout=900,
    volumes={TRACE_SET_PATH: trace_volume, OUTPUT_PATH: output_volume},
)
def run_nsys_profile(
    solution: Solution,
    workload: Workload,
    workload_idx: int,
    seq_len: int,
) -> str:
    """Run NSYS profiling, generating a .nsys-rep file."""
    import subprocess
    import sys
    import tempfile

    nsys_bin = _find_tool("nsys")
    rep_name = f"kernel_wl{workload_idx:02d}_seq{seq_len}"
    rep_path = f"{OUTPUT_PATH}/{rep_name}"

    with tempfile.TemporaryDirectory(prefix="fib_nsys_") as data_dir:
        _write_data_files(data_dir, solution, workload, TRACE_SET_PATH)

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
            "--device", "cuda:0",
            "--trace-set-path", TRACE_SET_PATH,
        ]

        print(f"[NSYS] Profiling workload {workload_idx} (seq_len={seq_len})...")
        print(f"[NSYS] Using: {nsys_bin}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    output_volume.commit()

    summary = f"[NSYS] Workload {workload_idx} (seq_len={seq_len}): "
    if result.returncode == 0:
        summary += f"SUCCESS -> {rep_name}.nsys-rep"
    else:
        summary += f"FAILED (rc={result.returncode})"
        if result.stderr:
            summary += f"\nSTDERR:\n{result.stderr[-2000:]}"
        if result.stdout:
            summary += f"\nSTDOUT:\n{result.stdout[-2000:]}"

    return summary


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


@app.local_entrypoint()
def main(
    ncu_only: bool = False,
    nsys_only: bool = False,
    max_workloads: int = 0,
    discover: bool = False,
):
    """Pack solution and run NCU/NSYS profiling for each workload."""
    from scripts.pack_solution import pack_solution

    if discover:
        print("Discovering available tools on Modal B200...")
        print(discover_tools.remote())
        return

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(
        solution_path.read_text(encoding="utf-8")
    )
    print(f"Loaded: {solution.name} ({solution.definition})")

    workloads = _load_workloads(solution.definition)
    if max_workloads > 0:
        workloads = workloads[:max_workloads]
    print(f"\nFound {len(workloads)} workloads to profile")

    run_both = not ncu_only and not nsys_only

    # --- NCU profiling ---
    if run_both or ncu_only:
        print("\n" + "=" * 60)
        print("Running NCU profiling (generating .ncu-rep per workload)...")
        print("=" * 60)

        ncu_futures = []
        for wl in workloads:
            future = run_ncu_profile.spawn(
                solution, wl["workload"], wl["idx"], wl["seq_len"]
            )
            ncu_futures.append(future)

        for future in ncu_futures:
            print(future.get())

    # --- NSYS profiling ---
    if run_both or nsys_only:
        print("\n" + "=" * 60)
        print("Running NSYS profiling (generating .nsys-rep per workload)...")
        print("=" * 60)

        nsys_futures = []
        for wl in workloads:
            future = run_nsys_profile.spawn(
                solution, wl["workload"], wl["idx"], wl["seq_len"]
            )
            nsys_futures.append(future)

        for future in nsys_futures:
            print(future.get())

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print(f"Reports saved to Modal volume 'profiling-reports'")
    print("Download with: modal volume get profiling-reports /local/path/")
    print("=" * 60)
