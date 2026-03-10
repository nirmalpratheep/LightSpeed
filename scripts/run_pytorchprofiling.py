"""
PyTorch Profiling Runner for kernel.py on Modal B200.

Uses torch.profiler in-process → Chrome trace JSON + summary table.
Works reliably on Modal B200 (no nsys/QuadD clock calibration needed).

Usage:
    modal run scripts/run_pytorchprofiling.py --max-workloads 1
    modal run scripts/run_pytorchprofiling.py --max-workloads 5 --profile-iters 10

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

app = modal.App("flashinfer-pytorch-profiling")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
output_volume = modal.Volume.from_name("profiling-reports", create_if_missing=True)
TRACE_SET_PATH = "/data"
OUTPUT_PATH = "/reports"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=900,
    volumes={TRACE_SET_PATH: trace_volume, OUTPUT_PATH: output_volume},
)
def run_pytorch_profile(
    solution: Solution,
    workload: Workload,
    workload_idx: int,
    seq_len: int,
    num_warmup: int = 3,
    num_profile_iters: int = 5,
) -> str:
    """Run PyTorch Profiler in-process.  Exports Chrome trace JSON + summary."""
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

    print(f"[PROFILE] Workload {workload_idx} (seq_len={seq_len})...")

    # ── Build solution (mirrors _solution_runner) ────────────────────
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solution.definition]

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, Path(TRACE_SET_PATH))

    device = "cuda:0"
    inputs = gen_inputs(definition, workload, device, safe_tensors)
    outputs = allocate_outputs(definition, inputs, device)

    # ── Warmup (JIT compile Triton kernels) ──────────────────────────
    for _ in range(num_warmup):
        with torch.no_grad():
            runnable.call_destination_passing(*inputs, *outputs)
        torch.cuda.synchronize()
    print(f"[PROFILE] Warmup done ({num_warmup} iters)")

    # ── Profile ──────────────────────────────────────────────────────
    trace_name = f"trace_wl{workload_idx:02d}_seq{seq_len}"
    trace_path = f"{OUTPUT_PATH}/{trace_name}.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(num_profile_iters):
            with record_function(f"moe_kernel_iter{i}"):
                with torch.no_grad():
                    runnable.call_destination_passing(*inputs, *outputs)
                torch.cuda.synchronize()

    # ── Export chrome trace to volume ────────────────────────────────
    prof.export_chrome_trace(trace_path)
    output_volume.commit()
    print(f"[PROFILE] Chrome trace saved → {trace_name}.json")

    # ── Build summary ────────────────────────────────────────────────
    summary_lines = [
        f"[PROFILE] Workload {workload_idx} (seq_len={seq_len}): SUCCESS",
        f"   Chrome trace: {trace_name}.json",
        f"   Iterations profiled: {num_profile_iters}",
        "",
        "── CUDA Kernel Summary (sorted by CUDA time) ──",
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20
        ),
        "",
        "── Top CPU Operations ──",
        prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=10
        ),
    ]

    # Memory summary if available
    try:
        mem_table = prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=10
        )
        summary_lines.extend(["", "── CUDA Memory Usage ──", mem_table])
    except Exception:
        pass

    runnable.cleanup()
    return "\n".join(summary_lines)


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
    max_workloads: int = 0,
    profile_iters: int = 5,
):
    """Pack solution and run PyTorch profiling for each workload."""
    from scripts.pack_solution import pack_solution

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

    print("\n" + "=" * 60)
    print("Running PyTorch Profiler (generating Chrome trace per workload)...")
    print("=" * 60)

    prof_futures = []
    for wl in workloads:
        future = run_pytorch_profile.spawn(
            solution, wl["workload"], wl["idx"], wl["seq_len"],
            num_profile_iters=profile_iters,
        )
        prof_futures.append(future)

    for future in prof_futures:
        print(future.get())

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("Reports saved to Modal volume 'profiling-reports'")
    print("Download with: modal volume get profiling-reports / reports/")
    print("View traces at: chrome://tracing  or  https://ui.perfetto.dev")
    print("=" * 60)
