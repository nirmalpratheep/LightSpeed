"""
Reference Kernel Modal Cloud Benchmark Runner.

Runs kernel_ref.py as a solution on NVIDIA B200 GPUs via Modal to establish
baseline performance numbers.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, Solution, TraceSet
from flashinfer_bench.agents import pack_solution_from_files

app = modal.App("flashinfer-bench-ref")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            # Capture log for error diagnosis
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

            # Print log for errors
            if result.get("log") and status != "PASS":
                print(f"    log: {result['log']}")


# Build the wrapper by reading kernel_ref.py and embedding it directly.
# This avoids inter-module import issues inside the framework's sandbox.
def _build_wrapper_code() -> str:
    kernel_ref_path = PROJECT_ROOT / "solution" / "triton" / "kernel_ref.py"
    ref_source = kernel_ref_path.read_text(encoding="utf-8")

    return f'''\
import torch

# Embedded kernel_ref source
{ref_source}

# Rename to avoid collision with the entry-point name
_ref_run = run

@torch.no_grad()
def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    output=None,
    **kwargs,
):
    # Framework may pass scalars as tensors
    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = local_expert_offset.item()
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = routed_scaling_factor.item()

    result = _ref_run(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
    )
    if output is not None:
        output.copy_(result)
        return output
    return result
'''


def pack_ref_solution() -> Solution:
    """Pack kernel_ref.py with a wrapper as a single-file solution."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    wrapper_code = _build_wrapper_code()

    # Create a temp directory with the single combined kernel.py
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write combined wrapper (includes embedded kernel_ref code) as entry point
        (tmpdir / "kernel.py").write_text(wrapper_code, encoding="utf-8")

        spec = BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point="kernel.py::kernel",
        )

        solution = pack_solution_from_files(
            path=str(tmpdir),
            spec=spec,
            name="reference-baseline",
            definition=config["solution"]["definition"],
            author=config["solution"]["author"],
        )

    return solution


@app.local_entrypoint()
def main():
    """Pack kernel_ref.py and run benchmark on Modal."""
    print("Packing reference kernel (kernel_ref.py) with wrapper...")
    solution = pack_ref_solution()
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning reference benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
