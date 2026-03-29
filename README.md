# [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)

Create high-performance GPU kernels for state-of-the-art LLM architectures on NVIDIA Blackwell GPUs with humans and/or AI agents.

---

<p align="center">
  <a href="https://www.nvidia.com"><img src="images/nvidia-logo.svg" alt="NVIDIA" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://modal.com"><img src="images/modal-logo.png" alt="Modal" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mlsys.org"><img src="images/mlsys-logo.svg" alt="MLSys" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer"><img src="images/flashinfer-logo.png" alt="FlashInfer" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer-bench"><img src="images/fib_logo.png" alt="FlashInfer-Bench" height="50"/></a>
</p>

---

[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) is our official framework to evaluate your AI-generated kernels.

## Updates

* 2026.02.05: Full dataset for definitions and workloads are released at [HuggingFace](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)

## Benchmark Results (B200)

### V7 — Large Tier GEMM2 BM=64 (Current)

**Solution:** `moe-v2-zerocopy-noatomic` | **Track:** `fused_moe` | **Team:** `LIGHTSPEED/prateek,nirmal`
**Commit:** `08386d2` | **Change:** Large GEMM2 `BM=32→64`, `stages=6→4` — halves B-matrix reads for large tier

| Workload | Tier | Status | Latency (ms) | Speedup | Max Abs Error | Max Rel Error |
|----------|------|--------|-------------|---------|---------------|---------------|
| b8f4f012 | tiny | PASSED | 0.662 | 17.65x | 2.05e+03 | 1.59e+01 |
| e05c6c03 | tiny | PASSED | 0.642 | 17.25x | 2.05e+03 | 4.58e+00 |
| 6230e838 | small | PASSED | 0.850 | 16.51x | 4.10e+03 | 2.28e+01 |
| 8f1ff9f1 | small | PASSED | 0.958 | 16.49x | 4.10e+03 | 2.26e+02 |
| a7c2bcfd | small | PASSED | 0.702 | 18.07x | 4.10e+03 | 2.55e+02 |
| 2e69caee | small | PASSED | 0.563 | 20.41x | 2.05e+03 | 1.74e+03 |
| 8cba5890 | small | PASSED | 0.696 | 17.87x | 4.10e+03 | 1.23e+02 |
| 5eadab1e | small | PASSED | 0.752 | 18.32x | 4.10e+03 | 1.42e+02 |
| eedc63b2 | small | PASSED | 0.821 | 16.67x | 4.10e+03 | 7.44e+01 |
| e626d3e6 | small | PASSED | 0.877 | 17.42x | 4.10e+03 | 1.36e+03 |
| 74d7ff04 | small | PASSED | 0.878 | 16.99x | 4.10e+03 | 1.43e+02 |
| 4822167c | small | PASSED | 0.885 | 16.99x | 4.10e+03 | 1.99e+02 |
| 81955b1e | small | PASSED | 0.879 | 16.62x | 4.10e+03 | 9.22e+01 |
| 76010cb4 | small | PASSED | 0.848 | 16.92x | 4.10e+03 | 7.92e+01 |
| fc378037 | small | PASSED | 0.877 | 16.70x | 4.10e+03 | 1.06e+02 |
| f7d6ac7c | small | PASSED | 0.754 | 17.68x | 4.10e+03 | 5.30e+02 |
| 1a4c6ba1 | medium | PASSED | 1.176 | 17.87x | 4.10e+05 | 3.44e+13 |
| 5e8dc11c | large | PASSED | 4.953 | **9.10x** | 4.85e+05 | 4.85e+13 |
| 58a34f27 | large | PASSED | 3.593 | **9.97x** | 4.34e+05 | 4.34e+13 |

**Summary:** All 19 workloads passed. Speedups range from 16.49x to 20.41x (non-large), **9.10x–9.97x (large)**. Avg speedup: **16.61x**. Avg latency: **1.18 ms**.
**Large tier improvement vs V6 baseline (same-day run):** `5e8dc11c` 6.91x→9.10x (**+31.7%**), `58a34f27` 7.75x→9.97x (**+28.6%**). Tiny/small/medium code unchanged; per-tier speedup differences reflect B200 instance variance.

### V6 — FP8 Block Scale DS Routing (Baseline, commit `76baa0d`, re-run 2026-03-29)

**Solution:** `moe-v2-zerocopy-noatomic` | **Track:** `fused_moe` | **Team:** `LIGHTSPEED/prateek,nirmal`

| Workload | Tier | Status | Latency (ms) | Speedup | Max Abs Error | Max Rel Error |
|----------|------|--------|-------------|---------|---------------|---------------|
| b8f4f012 | tiny | PASSED | 0.643 | 18.30x | 4.10e+03 | 6.95e+01 |
| e05c6c03 | tiny | PASSED | 0.621 | 17.93x | 2.05e+03 | 4.17e+00 |
| 6230e838 | small | PASSED | 0.823 | 17.26x | 4.10e+03 | 8.61e+01 |
| 8f1ff9f1 | small | PASSED | 0.931 | 17.05x | 4.10e+03 | 8.24e+02 |
| a7c2bcfd | small | PASSED | 0.681 | 18.71x | 4.10e+03 | 2.56e+01 |
| 2e69caee | small | PASSED | 0.540 | 21.34x | 4.10e+03 | 1.45e+01 |
| 8cba5890 | small | PASSED | 0.678 | 18.44x | 2.05e+03 | 2.45e+02 |
| 5eadab1e | small | PASSED | 0.726 | 19.01x | 4.10e+03 | 1.32e+02 |
| eedc63b2 | small | PASSED | 0.804 | 17.12x | 2.05e+03 | 1.77e+02 |
| e626d3e6 | small | PASSED | 0.857 | 17.85x | 4.10e+03 | 2.46e+02 |
| 74d7ff04 | small | PASSED | 0.851 | 17.57x | 4.10e+03 | 1.08e+02 |
| 4822167c | small | PASSED | 0.855 | 17.63x | 4.10e+03 | 1.97e+02 |
| 81955b1e | small | PASSED | 0.846 | 17.30x | 4.10e+03 | 1.41e+02 |
| 76010cb4 | small | PASSED | 0.826 | 17.44x | 4.10e+03 | 4.25e+01 |
| fc378037 | small | PASSED | 0.853 | 17.22x | 4.10e+03 | 1.12e+02 |
| f7d6ac7c | small | PASSED | 0.722 | 18.52x | 4.10e+03 | 3.08e+01 |
| 1a4c6ba1 | medium | PASSED | 1.153 | 18.22x | 3.85e+05 | 2.95e+13 |
| 5e8dc11c | large | PASSED | 6.510 | 6.91x | 5.41e+05 | 5.41e+13 |
| 58a34f27 | large | PASSED | 4.615 | 7.75x | 5.49e+05 | 5.49e+13 |

**Summary:** All 19 workloads passed. Speedups range from 6.91x to 21.34x. Avg speedup: **16.92x**. Avg latency: **1.29 ms** (vs V5 avg 2.59 ms — **50% reduction**).

### V5 — Workload-Specialized Dispatch (Previous)

**Solution:** `moe-v5-workload-dispatch` | **Track:** `fused_moe` | **Team:** `LIGHTSPEED/prateek,nirmal`

| Workload | Status | Latency (ms) | Speedup | Max Abs Error | Max Rel Error |
|----------|--------|-------------|---------|---------------|---------------|
| b8f4f012 | PASSED | 1.509 | 7.68x | 2.05e+03 | 2.89e+01 |
| e05c6c03 | PASSED | 1.495 | 7.40x | 2.05e+03 | 2.06e+00 |
| 6230e838 | PASSED | 2.075 | 6.64x | 4.10e+03 | 9.78e+01 |
| 8f1ff9f1 | PASSED | 2.290 | 6.87x | 4.10e+03 | 1.94e+02 |
| 1a4c6ba1 | PASSED | 2.323 | 8.97x | 4.10e+03 | 1.10e+09 |
| a7c2bcfd | PASSED | 1.780 | 7.02x | 2.05e+03 | 2.07e+02 |
| 2e69caee | PASSED | 1.564 | 7.33x | 4.10e+03 | 1.01e+01 |
| 8cba5890 | PASSED | 1.774 | 6.88x | 4.10e+03 | 1.06e+02 |
| 5e8dc11c | PASSED | 8.449 | 5.33x | 8.19e+03 | 3.22e+09 |
| 58a34f27 | PASSED | 6.368 | 5.63x | 8.19e+03 | 2.03e+09 |
| 5eadab1e | PASSED | 2.038 | 6.75x | 4.10e+03 | 3.44e+01 |
| eedc63b2 | PASSED | 2.091 | 6.64x | 4.10e+03 | 3.44e+07 |
| e626d3e6 | PASSED | 2.260 | 6.78x | 4.10e+03 | 3.03e+02 |
| 74d7ff04 | PASSED | 2.248 | 6.63x | 4.10e+03 | 4.54e+02 |
| 4822167c | PASSED | 2.227 | 6.76x | 4.10e+03 | 1.40e+03 |
| 81955b1e | PASSED | 2.258 | 6.45x | 4.10e+03 | 2.47e+01 |
| 76010cb4 | PASSED | 2.197 | 6.51x | 4.10e+03 | 1.60e+02 |
| fc378037 | PASSED | 2.246 | 6.51x | 4.10e+03 | 8.13e+08 |
| f7d6ac7c | PASSED | 1.937 | 6.84x | 2.05e+03 | 6.72e+01 |

**Summary:** All 19 workloads passed. Speedups range from 5.33x to 8.97x. Avg speedup: **6.82x** (vs V4 avg 5.59x — **+22% improvement**). Avg latency: **2.59 ms** (vs V4 avg 3.45 ms — **25% reduction**).

**V5 changes from V4:**
- Workload-tiered dispatch (tiny/small/medium/large) based on seq_len
- Smaller BLOCK_M (16/32) for tiny/small workloads reduces wasted tile rows
- Non-persistent grid for tiny workloads (tiles < SMs)
- 4 warps for tiny/small → lower register pressure
- 4 pipeline stages for large → deeper pipelining hides memory latency
- 3 routing variants: `.sort()` (tiny), `argsort` (standard), `bincount` (large)

### V4 — Persistent Kernel (Previous)

**Solution:** `moe-v2-zerocopy-noatomic` | **Track:** `fused_moe` | **Team:** `LIGHTSPEED/prateek,nirmal`

| Workload | Status | Latency (ms) | Speedup | Max Abs Error | Max Rel Error |
|----------|--------|-------------|---------|---------------|---------------|
| b8f4f012 | PASSED | 1.857 | 6.24x | 4.10e+03 | 7.35e+00 |
| e05c6c03 | PASSED | 1.609 | 6.83x | 4.10e+03 | 2.31e+00 |
| 6230e838 | PASSED | 2.435 | 5.67x | 4.10e+03 | 4.48e+01 |
| 8f1ff9f1 | PASSED | 2.794 | 5.58x | 4.10e+03 | 8.53e+02 |
| 1a4c6ba1 | PASSED | 3.051 | 6.83x | 4.10e+03 | 6.27e+03 |
| a7c2bcfd | PASSED | 2.013 | 6.20x | 4.10e+03 | 9.99e+02 |
| 2e69caee | PASSED | 1.866 | 6.10x | 4.10e+03 | 4.89e+01 |
| 8cba5890 | PASSED | 2.179 | 5.62x | 2.05e+03 | 1.60e+02 |
| 5e8dc11c | PASSED | 14.368 | 3.12x | 4.10e+03 | 2.55e+04 |
| 58a34f27 | PASSED | 10.525 | 3.38x | 4.10e+03 | 4.75e+08 |
| 5eadab1e | PASSED | 2.112 | 6.43x | 4.10e+03 | 8.94e+01 |
| eedc63b2 | PASSED | 2.571 | 5.24x | 4.10e+03 | 8.08e+01 |
| e626d3e6 | PASSED | 2.662 | 5.66x | 4.10e+03 | 2.46e+02 |
| 74d7ff04 | PASSED | 2.659 | 5.52x | 4.10e+03 | 1.55e+02 |
| 4822167c | PASSED | 2.531 | 5.85x | 4.10e+03 | 1.45e+02 |
| 81955b1e | PASSED | 2.649 | 5.41x | 4.10e+03 | 1.06e+02 |
| 76010cb4 | PASSED | 2.625 | 5.37x | 4.10e+03 | 7.78e+02 |
| fc378037 | PASSED | 2.672 | 5.43x | 4.10e+03 | 4.04e+02 |
| f7d6ac7c | PASSED | 2.283 | 5.81x | 2.05e+03 | 5.53e+01 |

**Summary:** All 19 workloads passed. Speedups range from 3.12x to 6.83x. Avg speedup: **5.59x**. Avg latency: **3.45 ms**.

## Competition Tracks

The competition features three tracks, each targeting a critical LLM operation:

| Track | Description |
|-------|-------------|
| **fused_moe** | Fused Mixture-of-Experts kernel for efficient expert routing and computation |
| **sparse_attention** | Sparse attention mechanisms for long-context inference |
| **gated_delta_net** | Gated delta network operations for efficient state updates |

**Fork this template once per track** you want to compete in (separate repos for each track).

## Getting Started

### 1. Fork This Template

Click "Use this template" or fork this repository to create your solution repo.

### 2. Install Dependencies

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

### 3. Download the TraceSet

We provide kernel definitions and workloads in [FlashInfer-Trace format](https://bench.flashinfer.ai/docs/flashinfer-trace). Clone the competition dataset from HuggingFace:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

Set the environment variable:

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### 4. Configure Your Solution

Edit `config.toml` to set your track and team info:

```toml
[solution]
name = "my-team-solution-v1"      # Solution name
definition = "fused_moe"          # Track: fused_moe | sparse_attention | gated_delta_net
author = "team-name"              # Team/author name

[build]
language = "triton"               # triton | cuda
entry_point = "kernel"            # Kernel function name
```

### 5. Implement Your Kernel

**For Triton:**
Edit `solution/triton/kernel.py` with your implementation.

**For CUDA:**
Edit `solution/cuda/kernel.cu` and `solution/cuda/binding.py` with your implementation.

## Development Workflow

### Pack Your Solution

Generate `solution.json` from your source files:

```bash
python scripts/pack_solution.py
```

### Run Local Benchmarks

Test your solution on your local GPU:

```bash
python scripts/run_local.py
```

Requires: Local CUDA-capable GPU and `FIB_DATASET_PATH` environment variable.

### Run Cloud Benchmarks (Modal)

Test your solution on NVIDIA B200 GPUs via Modal:

**One-time setup:**

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace
```

**Run benchmark:**

```bash
modal run scripts/run_modal.py
```

## Submission

To submit your solution for evaluation:

1. Ensure your implementation is complete and tested
2. Run `python scripts/pack_solution.py` to generate `solution.json`
3. Commit and push your changes
4. Tag your commit for evaluation (e.g., `git tag submission-v1`)

## Project Structure

```
flashinfer-bench-starter-kit/
├── README.md                    # This file
├── config.toml                  # Track configuration (edit this)
├── solution/                    # Solution source files
│   ├── triton/                  # Triton implementation
│   │   └── kernel.py           # Your Triton kernel
│   └── cuda/                    # CUDA implementation
│       ├── kernel.cu           # Your CUDA kernel
│       └── binding.py          # TVM FFI bindings
├── scripts/                     # Utility scripts
│   ├── run_local.py            # Local benchmark runner
│   ├── run_modal.py            # Modal cloud benchmark runner
│   └── pack_solution.py        # Pack source files into solution.json
└── images/                      # Sponsor logos
```

## Additional Resources

### FlashInfer Trace Viewer

FlashInfer Trace consists of multiple JSON objects (definitions, workloads, solutions, and traces), which can contain large code blocks. To easily visualize and inspect these objects, you can use the [FlashInfer Trace Viewer](https://bench.flashinfer.ai/viewer). Simply paste any FlashInfer Trace JSON into the viewer to get a friendly, structured view of its contents.

### Solution Handling API

```python
from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files, extract_solution_to_files

# Pack source files into a Solution object
spec = BuildSpec(
    language="triton",  # or "cuda"
    target_hardware=["cuda"],
    entry_point="my_kernel",
)
solution = pack_solution_from_files(
    path="./my_solution_dir",
    spec=spec,
    name="my_solution_v1",
    definition="fused_moe",
    author="your_name",
)

# Extract a Solution to files in a working directory
extract_solution_to_files(solution, "./output_dir")
```

### Running Sanitizers

```python
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

output = flashinfer_bench_run_sanitizer(
    solution=solution,
    workload=workload,
    sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
    timeout=300,
)
print(output)
```

### NCU Profiling

```python
from flashinfer_bench.agents import flashinfer_bench_run_ncu

output = flashinfer_bench_run_ncu(
    solution=solution,
    workload=workload,
    set="detailed",
    page="details",
    timeout=120,
)
print(output)
```

### List Available Tools

```python
from flashinfer_bench.agents import get_all_tool_schemas

schemas = get_all_tool_schemas()
# Returns list of OpenAI-compatible function schemas
```

## Notes

### Destination Passing Style (DPS)

FlashInfer-Bench uses destination passing style (DPS) by default, where both inputs and outputs are passed as function parameters. DPS avoids measuring tensor allocation overhead, resulting in more accurate performance numbers. We recommend using DPS when possible, as it yields better benchmark results.

**Important:** Avoid using variadic input arguments in your kernel signatures, as they will fail the builder validation check.

If your kernel uses value-returning style (i.e., returns output tensors instead of writing to pre-allocated ones), set `destination_passing_style` to `false` in your solution's `spec`:

```json
{
  "name": "my_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "triton",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.py::my_kernel",
    "dependencies": [],
    "destination_passing_style": false
  },
  "sources": [...]
}
```

**Common error when DPS is mismatched:**

```
Destination-passing style callable: expected xx parameters, but got xx
```

This can happen for two reasons: (1) your kernel function signature has the wrong number of parameters, or (2) your kernel uses value-returning style but the solution still has `destination_passing_style` set to `true` by default. For the latter case, fix by setting `destination_passing_style` to `false`.

### CUDA Kernel Bindings

For CUDA kernel implementations, we recommend using [TVM FFI](https://tvm.apache.org/ffi/) for Python bindings. The `flashinfer_bench.agents` module provides TVM FFI agent instruction prompts to assist with development.

You can set the `binding` field in your solution's `spec` to specify the C++ binding type. Defaults to `"tvm-ffi"` if not specified. Supported values: `"tvm-ffi"`, `"torch"`.

```json
{
  "name": "my_cuda_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "cuda",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.cu::my_kernel",
    "dependencies": [],
    "binding": "torch"
  },
  "sources": [...]
}
```
