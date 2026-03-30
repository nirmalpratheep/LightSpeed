"""
FlashInfer-Bench Modal Benchmark Runner — 4 Representative Workloads.

Runs only one workload per kernel tier to measure all four tiers
with minimum modal cost. Prints a compact result table and appends
the run to reflections.md for the optimization loop.

  wl01  idx=1  seq=1      → tiny   (≤8)
  wl03  idx=3  seq=80     → small  (9–128)
  wl04  idx=4  seq=901    → medium (129–2048)
  wl08  idx=8  seq=14107  → large  (>2048)

Usage:
    modal run scripts/run_modal_limited.py
    modal run scripts/run_modal_limited.py --tier large
    modal run scripts/run_modal_limited.py --tier large --note "GEMM1 BK=64"
    modal run scripts/run_modal_limited.py --iterations 50

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import json
import re
import subprocess
import sys
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench-limited")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

# Mirrors REPRESENTATIVE_WORKLOADS in run_profiling.py — one per tier.
REPRESENTATIVE_INDICES = {1, 3, 4, 8}

TIER_WORKLOAD_IDX = {
    "tiny":   {1},
    "small":  {3},
    "medium": {4},
    "large":  {8},
    "all":    {1, 3, 4, 8},
}

TIER_LABELS = {
    range(0, 9):    "tiny",
    range(9, 129):  "small",
    range(129, 2049): "medium",
    range(2049, 10 ** 9): "large",
}


def _tier(seq: int) -> str:
    for rng, label in TIER_LABELS.items():
        if seq in rng:
            return label
    return "?"


def _load_representative_uuids(definition: str, indices: set[int] | None = None) -> list[str]:
    """Return UUIDs for representative workloads (loaded from local JSONL).

    Args:
        definition: The definition name to load workloads for.
        indices: Which workload indices to include. Defaults to all 4 representatives.
    """
    if indices is None:
        indices = REPRESENTATIVE_INDICES
    workload_file = (
        PROJECT_ROOT / "mlsys26-contest" / "workloads" / "moe"
        / f"{definition}.jsonl"
    )
    uuids: list[str] = []
    with open(workload_file, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in indices:
                data = json.loads(line)
                uuids.append(data["workload"]["uuid"])
    return uuids


# ── Remote function ────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_benchmark_limited(
    solution: Solution,
    representative_uuids: list[str],
    config: BenchmarkConfig | None = None,
) -> dict:
    """Run benchmark on Modal B200 for the 4 representative workloads only."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    all_workloads = trace_set.workloads.get(solution.definition, [])

    uuid_set = set(representative_uuids)
    workloads = [wl for wl in all_workloads if wl.workload.uuid in uuid_set]

    if not workloads:
        raise ValueError(
            f"None of the representative UUIDs found in trace set. "
            f"Looking for: {uuid_set}"
        )

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
    results: dict[str, dict] = {}

    for trace in traces:
        if not trace.evaluation:
            continue
        entry: dict = {
            "uuid": trace.workload.uuid,
            "seq_len": trace.workload.axes.get("seq_len", 0),
            "status": trace.evaluation.status.value,
            "latency_ms": None,
            "reference_latency_ms": None,
            "speedup_factor": None,
            "max_abs_error": None,
            "max_rel_error": None,
        }
        if trace.evaluation.performance:
            p = trace.evaluation.performance
            entry["latency_ms"] = p.latency_ms
            entry["reference_latency_ms"] = p.reference_latency_ms
            entry["speedup_factor"] = p.speedup_factor
        if trace.evaluation.correctness:
            c = trace.evaluation.correctness
            entry["max_abs_error"] = c.max_absolute_error
            entry["max_rel_error"] = c.max_relative_error
        results[trace.workload.uuid] = entry

    return results


# ── Local helpers ──────────────────────────────────────────────────────────────

def _print_table(results: dict) -> None:
    """Print compact result table matching contest output style."""
    rows = sorted(results.values(), key=lambda r: r.get("seq_len", 0))

    sep = "|" + "-" * 79 + "|"
    hdr = (
        f"| {'UUID':>8} | {'Tier':>6} | {'seq':>6} | "
        f"{'Status':>6} | {'lat(ms)':>8} | {'speedup':>8} | "
        f"{'abs_err':>8} | {'rel_err':>8} |"
    )
    print(sep)
    print(hdr)
    print(sep)

    for r in rows:
        seq = r.get("seq_len", 0)
        lat = r.get("latency_ms")
        spd = r.get("speedup_factor")
        abs_e = r.get("max_abs_error")
        rel_e = r.get("max_rel_error")

        print(
            f"| {r['uuid'][:8]:>8} | {_tier(seq):>6} | {seq:>6} | "
            f"{r.get('status','?'):>6} | "
            f"{f'{lat:.3f}' if lat is not None else 'N/A':>8} | "
            f"{f'{spd:.2f}x' if spd is not None else 'N/A':>8} | "
            f"{f'{abs_e:.2e}' if abs_e is not None else 'N/A':>8} | "
            f"{f'{rel_e:.2e}' if rel_e is not None else 'N/A':>8} |"
        )

    print(sep)

    passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
    total = len(results)
    speedups = [r["speedup_factor"] for r in results.values() if r.get("speedup_factor") is not None]
    avg_spd = sum(speedups) / len(speedups) if speedups else 0.0
    min_spd = min(speedups) if speedups else 0.0
    max_spd = max(speedups) if speedups else 0.0

    print(
        f"\nResult : {passed}/{total} PASSED | "
        f"avg={avg_spd:.2f}x  min={min_spd:.2f}x  max={max_spd:.2f}x"
    )


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _next_round_number() -> int:
    """Return the next round number by scanning experiments/ folder."""
    exp_dir = PROJECT_ROOT / "experiments"
    if not exp_dir.exists():
        return 1
    rounds = []
    for d in exp_dir.iterdir():
        if d.is_dir():
            import re
            m = re.match(r"round_(\d+)", d.name)
            if m:
                rounds.append(int(m.group(1)))
    return max(rounds, default=-1) + 1


def _save_experiment_folder(results: dict, note: str, round_num: int) -> Path:
    """Create experiments/round_NNN_<commit>_<slug>/ with all run artifacts."""
    commit = _git_commit_short()
    slug = note.lower().replace(" ", "_").replace("/", "-")[:40] if note else "run"
    slug = "".join(c for c in slug if c.isalnum() or c in "-_")
    folder_name = f"round_{round_num:03d}_{commit}_{slug}"
    folder = PROJECT_ROOT / "experiments" / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = sorted(results.values(), key=lambda r: r.get("seq_len", 0))
    speedups = [r["speedup_factor"] for r in results.values() if r.get("speedup_factor") is not None]
    avg_spd = sum(speedups) / len(speedups) if speedups else 0.0
    passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
    all_passed = passed == len(results)

    # config.json
    config = {
        "round": round_num, "commit": commit, "ts": ts,
        "note": note, "tier": "all", "strategy": "optimization",
        "changes": note,
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # results.json
    workloads = []
    for r in rows:
        seq = r.get("seq_len", 0)
        workloads.append({
            "uuid":        r["uuid"][:8],
            "tier":        _tier(seq),
            "seq":         seq,
            "status":      r.get("status", "?"),
            "lat_ms":      r.get("latency_ms"),
            "speedup":     r.get("speedup_factor"),
            "max_abs_err": r.get("max_abs_error"),
            "max_rel_err": r.get("max_rel_error"),
        })
    results_doc = {
        "round": round_num, "commit": commit, "ts": ts,
        "all_passed": all_passed, "avg_speedup": round(avg_spd, 2),
        "workloads": workloads,
    }
    (folder / "results.json").write_text(json.dumps(results_doc, indent=2), encoding="utf-8")

    # notes.md template
    notes_lines = [
        f"# Round {round_num} — {note or 'Run'}",
        f"",
        f"**Commit**: {commit}",
        f"**Date**: {ts}",
        f"",
        "## Results",
        "",
        "| Tier | seq | lat(ms) | speedup | Status |",
        "|------|----:|--------:|--------:|--------|",
    ]
    for r in rows:
        seq = r.get("seq_len", 0)
        lat = r.get("latency_ms")
        spd = r.get("speedup_factor")
        notes_lines.append(
            f"| {_tier(seq)} | {seq} | "
            f"{f'{lat:.3f}' if lat is not None else 'N/A'} | "
            f"{f'{spd:.2f}x' if spd is not None else 'N/A'} | "
            f"{r.get('status', '?')} |"
        )
    notes_lines += [
        f"",
        f"**Avg speedup**: {avg_spd:.2f}x",
        "",
        "## Observations",
        "",
        "_[fill in after reviewing results]_",
        "",
        "## Hypothesis for Next Run",
        "",
        "_[fill in before starting next round]_",
    ]
    (folder / "notes.md").write_text("\n".join(notes_lines), encoding="utf-8")

    # Save kernel diff
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD~1", "--", "solution/triton/kernel.py"],
            cwd=str(PROJECT_ROOT), text=True
        )
        if diff.strip():
            (folder / "kernel.patch").write_text(diff, encoding="utf-8")
    except Exception:
        pass

    print(f"\nExperiment saved: {folder}")
    return folder


def _append_to_reflections(results: dict, note: str) -> None:
    """Append this run's table to reflections.md."""
    reflections_path = PROJECT_ROOT / "reflections.md"

    commit = _git_commit_short()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = sorted(results.values(), key=lambda r: r.get("seq_len", 0))

    passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
    total = len(results)
    speedups = [r["speedup_factor"] for r in results.values() if r.get("speedup_factor") is not None]
    avg_spd = sum(speedups) / len(speedups) if speedups else 0.0

    lines = [
        f"\n## {ts}  `{commit}`",
        f"**Note**: {note}" if note else "",
        "",
        "| UUID | Tier | seq | Status | lat(ms) | speedup | abs_err | rel_err |",
        "|------|------|----:|--------|--------:|--------:|--------:|--------:|",
    ]

    for r in rows:
        seq = r.get("seq_len", 0)
        lat = r.get("latency_ms")
        spd = r.get("speedup_factor")
        abs_e = r.get("max_abs_error")
        rel_e = r.get("max_rel_error")
        lines.append(
            f"| {r['uuid'][:8]} | {_tier(seq)} | {seq} | "
            f"{r.get('status', '?')} | "
            f"{f'{lat:.3f}' if lat is not None else 'N/A'} | "
            f"{f'{spd:.2f}x' if spd is not None else 'N/A'} | "
            f"{f'{abs_e:.2e}' if abs_e is not None else 'N/A'} | "
            f"{f'{rel_e:.2e}' if rel_e is not None else 'N/A'} |"
        )

    lines += [
        "",
        f"**Summary**: {passed}/{total} PASSED | avg speedup **{avg_spd:.2f}x**",
        "",
        "**What worked / what didn't**: _[fill in]_",
        "",
        "**Hypothesis for next run**: _[fill in]_",
        "",
        "---",
    ]

    with open(reflections_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(line for line in lines if line is not None))

    print(f"\nAppended results to {reflections_path}")


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    iterations: int = 100,
    warmup: int = 3,
    trials: int = 5,
    note: str = "",
    round_num: int = -1,
    tier: str = "all",
) -> None:
    """Pack solution and run benchmark on Modal for representative workloads.

    Args:
        iterations: Number of timed iterations per workload (default 100).
        warmup:     Number of warmup runs (default 3).
        trials:     Number of independent trials (default 5).
        note:       Short description of what changed, appended to reflections.md.
        round_num:  Override round number (default: auto-detect from experiments/).
        tier:       Which tier to benchmark — tiny | small | medium | large | all (default: all).

    Examples:
        modal run scripts/run_modal_limited.py --tier large
        modal run scripts/run_modal_limited.py --tier large --note "GEMM1 BK=64"
        modal run scripts/run_modal_limited.py
    """
    from scripts.pack_solution import pack_solution

    if tier not in TIER_WORKLOAD_IDX:
        raise ValueError(f"Unknown tier '{tier}'. Choose from: {list(TIER_WORKLOAD_IDX)}")

    selected_indices = TIER_WORKLOAD_IDX[tier]

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text(encoding="utf-8"))
    print(f"Loaded : {solution.name}  ({solution.definition})")

    print(f"\nResolving workload UUIDs for tier={tier}...")
    rep_uuids = _load_representative_uuids(solution.definition, selected_indices)
    for idx, uid in zip(sorted(selected_indices), rep_uuids):
        print(f"  idx={idx}  uuid={uid[:8]}...")

    config = BenchmarkConfig(warmup_runs=warmup, iterations=iterations, num_trials=trials)

    print(
        f"\nRunning benchmark on Modal B200  "
        f"({len(rep_uuids)} workload(s) [{tier}] | warmup={warmup} | iter={iterations} | trials={trials})..."
    )
    results = run_benchmark_limited.remote(solution, rep_uuids, config)

    if not results:
        print("No results returned!")
        return

    print()
    _print_table(results)

    rnum = round_num if round_num >= 0 else _next_round_number()
    _save_experiment_folder(results, note, rnum)
    _append_to_reflections(results, note)

    # Regenerate dashboard
    try:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "update_dashboard_data.py")],
            check=True,
        )
        print("dashboard_data.js regenerated — refresh dashboard.html")
    except Exception as e:
        print(f"[WARN] update_dashboard_data.py failed: {e}")
