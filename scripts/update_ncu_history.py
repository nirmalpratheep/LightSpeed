"""
Extract NCU hardware metrics from the 4 representative .ncu-rep files and
append a new entry to ncu_history.jsonl.  Then regenerates dashboard_data.js.

Run after every `modal run scripts/run_profiling.py`:

    python scripts/update_ncu_history.py --round 1 --note "after FP8 GEMM2 BM=64"

The script searches for .ncu-rep files in:
  1. The directory given by --ncu-dir (default: current working directory)
  2. NCU/<latest dated subfolder>/
"""

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ── NCU binary search ──────────────────────────────────────────────────────────

NCU_CANDIDATES = [
    # Windows Nsight Compute installations
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\target\windows-desktop-win7-x64\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\target\windows-desktop-win7-x64\ncu.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\target\windows-desktop-win7-x64\ncu.exe",
    # Linux / Modal
    "/usr/local/cuda/bin/ncu",
    "/usr/bin/ncu",
    "ncu",  # in PATH
]

METRICS = ",".join([
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
])

# The kernel name substrings we track (matched against CSV "Kernel Name" column)
TRACKED_KERNELS = [
    "_gemm1_swiglu_kernel",
    "_gemm1_swiglu_fp8out_kernel",
    "_gemm2_kernel",
    "_gemm2_fp8_kernel",
    "_gemm2_bf16_kernel",
    "_reduce_kernel",
    "_fused_routing_kernel",
]

# Representative workload .ncu-rep file patterns
REP_FILES = {
    "tiny":   ("kernel_wl01_seq1.ncu-rep",   1),
    "small":  ("kernel_wl03_seq80.ncu-rep",  80),
    "medium": ("kernel_wl04_seq901.ncu-rep", 901),
    "large":  ("kernel_wl08_seq14107.ncu-rep", 14107),
}


def find_ncu() -> str | None:
    for candidate in NCU_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path)
    # Try which/where
    try:
        result = subprocess.run(["where", "ncu"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0]
    except FileNotFoundError:
        pass
    try:
        result = subprocess.run(["which", "ncu"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def find_rep_file(name: str, search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        p = d / name
        if p.exists():
            return p
    return None


def extract_metrics(ncu: str, rep_file: Path) -> list[dict]:
    """Run ncu --import and parse CSV output. Returns list of per-kernel dicts."""
    cmd = [
        ncu,
        "--import", str(rep_file),
        "--csv", "--page", "raw",
        "--metrics", METRICS,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] ncu timed out on {rep_file.name}")
        return []

    if result.returncode != 0:
        print(f"  [WARN] ncu failed on {rep_file.name}:\n{result.stderr[-500:]}")
        return []

    # Filter out non-CSV lines (ncu sometimes prints warnings before CSV)
    lines = result.stdout.splitlines()
    csv_start = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")), None)
    if csv_start is None:
        print(f"  [WARN] No CSV header found in ncu output for {rep_file.name}")
        return []

    reader = csv.DictReader(io.StringIO("\n".join(lines[csv_start:])))

    # --page raw produces wide-format CSV: one row per kernel invocation,
    # metric names as column headers. Skip the units row (Kernel Name is empty).
    agg: dict[str, dict[str, list[float]]] = {}
    metric_cols = METRICS.split(",")
    for row in reader:
        kname = row.get("Kernel Name", "").strip()
        if not kname:
            continue  # units row or blank
        matched = next((t for t in TRACKED_KERNELS if t in kname), None)
        if matched is None:
            continue
        for mname in metric_cols:
            mval = row.get(mname, "").replace(",", "").strip()
            if not mval:
                continue
            try:
                val = float(mval)
            except ValueError:
                continue
            agg.setdefault(kname, {}).setdefault(mname, []).append(val)

    results = []
    for kname, metrics in agg.items():
        def avg(key):
            vals = metrics.get(key, [])
            return round(sum(vals) / len(vals), 2) if vals else None

        def sumv(key):
            vals = metrics.get(key, [])
            return round(sum(vals), 2) if vals else None

        dur_ns = sumv("gpu__time_duration.sum")
        results.append({
            "kernel_raw": kname,
            "kernel": next((t for t in TRACKED_KERNELS if t in kname), kname),
            "dur_us":   round(dur_ns / 1000, 1) if dur_ns is not None else None,
            "sm_pct":   avg("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "hbm_pct":  avg("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            "l1_pct":   avg("l1tex__t_sector_hit_rate.pct"),
            "warp_pct": avg("sm__warps_active.avg.pct_of_peak_sustained_active"),
        })

    return results


def main():
    ap = argparse.ArgumentParser(description="Append NCU metrics to ncu_history.jsonl")
    ap.add_argument("--round", type=int, required=True, help="Experiment round number")
    ap.add_argument("--commit", type=str, default="", help="Git commit short hash")
    ap.add_argument("--note",   type=str, default="", help="Short description of changes")
    ap.add_argument("--ncu-dir", type=str, default="", help="Directory containing .ncu-rep files")
    args = ap.parse_args()

    # Resolve search directories for .ncu-rep files
    search_dirs: list[Path] = []
    if args.ncu_dir:
        search_dirs.append(Path(args.ncu_dir))
    search_dirs.append(PROJECT_ROOT)  # repo root
    # Find latest NCU/<date>/ subfolder
    ncu_base = PROJECT_ROOT / "NCU"
    if ncu_base.exists():
        dated = sorted([d for d in ncu_base.iterdir() if d.is_dir()], reverse=True)
        search_dirs.extend(dated)

    # Resolve commit if not provided
    commit = args.commit
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(PROJECT_ROOT), text=True
            ).strip()
        except Exception:
            commit = "unknown"

    ncu = find_ncu()
    if ncu is None:
        print("[ERROR] ncu binary not found. Install Nsight Compute or add ncu to PATH.")
        sys.exit(1)
    print(f"Using ncu: {ncu}")

    entry: dict = {
        "round":   args.round,
        "commit":  commit,
        "ts":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "note":    args.note,
        "kernels": [],
    }

    for tier, (fname, seq) in REP_FILES.items():
        rep_path = find_rep_file(fname, search_dirs)
        if rep_path is None:
            print(f"  [SKIP] {fname} not found in search dirs")
            continue
        print(f"  Extracting {tier} ({fname})...")
        kernels = extract_metrics(ncu, rep_path)
        for k in kernels:
            k["tier"] = tier
            k["seq"]  = seq
        entry["kernels"].extend(kernels)
        print(f"    -> {len(kernels)} kernels extracted")

    # Save ncu_metrics.json into the matching experiments/round_NNN_*/ folder
    import re as _re
    exp_dir = PROJECT_ROOT / "experiments"
    round_folder = None
    if exp_dir.exists():
        candidates = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir()
             and _re.match(rf"round_{args.round:03d}_", d.name)],
        )
        if candidates:
            round_folder = candidates[-1]

    ncu_doc = {
        "round":   args.round,
        "commit":  commit,
        "ts":      entry["ts"],
        "note":    args.note,
        "source":  args.ncu_dir or "auto-detected",
        "kernels": entry["kernels"],
    }

    if round_folder:
        ncu_path = round_folder / "ncu_metrics.json"
        ncu_path.write_text(json.dumps(ncu_doc, indent=2), encoding="utf-8")
        print(f"Saved ncu_metrics.json -> {ncu_path}")

        # Also copy .ncu-rep files into round_folder/ncu/ (best-effort, skip if disk full)
        ncu_rep_dir = round_folder / "ncu"
        try:
            ncu_rep_dir.mkdir(exist_ok=True)
            for tier, (fname, _seq) in REP_FILES.items():
                rep_path = find_rep_file(fname, search_dirs)
                if rep_path:
                    import shutil
                    shutil.copy2(rep_path, ncu_rep_dir / fname)
                    print(f"  Copied {fname} -> {ncu_rep_dir.name}/")
        except OSError as e:
            print(f"  [WARN] Could not copy .ncu-rep files: {e}")
    else:
        print(f"[WARN] No experiments/round_{args.round:03d}_* folder found; ncu_metrics.json not saved to experiment.")

    # Also append to ncu_history.jsonl (legacy flat log)
    out_path = PROJECT_ROOT / "ncu_history.jsonl"
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"\nAppended round {args.round} to {out_path}")

    # Regenerate dashboard data
    try:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "update_dashboard_data.py")],
            check=True,
        )
    except Exception as e:
        print(f"[WARN] update_dashboard_data.py failed: {e}")


if __name__ == "__main__":
    main()
