#!/usr/bin/env python3
"""
Generate a graphviz Critical Path DAG for kernel.py (Triton MoE).

Produces the same style as refkernel.pdf / newkernel.pdf:
  - Color-coded nodes: green=Triton COMPUTE, purple=Routing, blue=bookkeeping, red=zero
  - Ellipse nodes for data tensors (green=weights, cream=intermediates)
  - Cost model latency predictions in every compute node
  - Critical path highlighted in red

Usage:
    python deepdive/generate_kernel_dag.py
    python deepdive/generate_kernel_dag.py --T 4096
    python deepdive/generate_kernel_dag.py --T 4096 --output deepdive/newkernel_dag
    python deepdive/generate_kernel_dag.py --all-workloads
"""

import argparse
import os
import sys

try:
    import graphviz
except ImportError:
    print("graphviz not installed. Run: pip install graphviz", file=sys.stderr)
    sys.exit(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from kernel_cost_model import load_hw, lookup_gemm_us, lookup_hbm_us

# ── MoE constants ──────────────────────────────────────────────────────────
H        = 7168
I        = 2048
E_LOCAL  = 32
E_GLOBAL = 256
TOP_K    = 8

# ── Workload tiers (matching kernel.py dispatch) ───────────────────────────
WORKLOADS = [
    ("tiny",   1),
    ("small",  80),
    ("medium", 901),
    ("large",  14107),
]

# ── Colour palette ─────────────────────────────────────────────────────────
C_ROUTE   = "#7d3c98"   # purple   — routing / sort (PyTorch MEM)
C_COMPUTE = "#1a5276"   # dark teal/blue — Triton GEMM kernels
C_BOOK    = "#1f618d"   # blue     — bookkeeping (tile_map, assign_idx)
C_ZERO    = "#cb4335"   # red      — output.zero_()
C_W_FILL  = "#1e8449"   # green    — weight/input tensor ellipses
C_W_FONT  = "white"
C_I_FILL  = "#fdfde0"   # cream    — intermediate tensor ellipses
C_I_FONT  = "#333333"
C_CRIT    = "red"       # critical path edges


# ── Formatting helpers ─────────────────────────────────────────────────────
def _fmt(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.3f} ms"
    return f"{us:.1f} us"


def _compute_costs(T: int, hw: dict) -> dict:
    """
    Predict per-stage latency for the kernel.py (Triton) pipeline.

    Token accounting:
      Tk        = avg tokens per local expert  = T * TOP_K // E_GLOBAL
      T_sorted  = total locally-dispatched (token, expert) pairs
                = T * TOP_K * E_LOCAL // E_GLOBAL  ≈ T (since TOP_K * E_LOCAL == E_GLOBAL)
    """
    Tk       = max(1, T * TOP_K // E_GLOBAL)
    T_sorted = max(1, T * TOP_K * E_LOCAL // E_GLOBAL)

    # ── Routing  (~20 MEM kernels) ─────────────────────────────────────
    # sigmoid [T, E_GLOBAL] BF16 + topk + scatter/normalize
    r_bytes = T * E_GLOBAL * 2 * 2   # read + write, BF16
    r_n     = 20
    r_mem   = lookup_hbm_us(r_bytes, hw) * r_n
    r_us    = r_mem + hw["launch_us"] * r_n

    # ── Sort by expert  (~18 MEM kernels) ─────────────────────────────
    # argsort + gather + expert_offsets scatter
    s_bytes = T_sorted * 4 * 3
    s_n     = 18
    s_mem   = lookup_hbm_us(s_bytes, hw) * s_n
    s_us    = s_mem + hw["launch_us"] * s_n

    # ── build_tile_map GEMM1  (~9 MEM kernels) ────────────────────────
    # tile_map [n_tiles, 5] int64,  GEMM1: E×ceil(Tk/64)×16 tiles
    n_tiles_g1 = E_LOCAL * max(1, (Tk + 63) // 64) * 16
    tm1_bytes  = n_tiles_g1 * 5 * 8
    tm1_n      = 9
    tm1_mem    = lookup_hbm_us(tm1_bytes, hw) * tm1_n
    tm1_us     = tm1_mem + hw["launch_us"] * tm1_n

    # ── build_tile_map GEMM2  (~9 MEM kernels) ────────────────────────
    # GEMM2: E×ceil(Tk/64)×56 tiles
    n_tiles_g2 = E_LOCAL * max(1, (Tk + 63) // 64) * 56
    tm2_bytes  = n_tiles_g2 * 5 * 8
    tm2_n      = 9
    tm2_mem    = lookup_hbm_us(tm2_bytes, hw) * tm2_n
    tm2_us     = tm2_mem + hw["launch_us"] * tm2_n

    # ── build_assign_idx  (~7 MEM kernels) ────────────────────────────
    # argsort + scatter_add + cumsum on [T_sorted]
    ai_bytes = T_sorted * 4 * 3
    ai_n     = 7
    ai_mem   = lookup_hbm_us(ai_bytes, hw) * ai_n
    ai_us    = ai_mem + hw["launch_us"] * ai_n

    # ── output.zero_()  (1 MEM kernel) ────────────────────────────────
    # zero [T, H] BF16
    z_bytes = T * H * 2
    z_mem   = lookup_hbm_us(z_bytes, hw)
    z_us    = z_mem + hw["launch_us"]

    # ── GEMM1 + SwiGLU  (1 persistent Triton kernel, FP8) ─────────────
    # [T_sorted, H] @ [E_LOCAL, 2I, H]^T  → [T_sorted, 2I]  FP8→FP32
    # then silu(x2) * x1                  → [T_sorted, I]   FP32
    g1_comp   = lookup_gemm_us(T_sorted, 2 * I, H, "fp8", hw)
    g1_rbytes = T_sorted * H * 1 + E_LOCAL * 2 * I * H * 1   # FP8 inputs
    g1_wbytes = T_sorted * I * 4                               # FP32 output
    g1_mem    = lookup_hbm_us(g1_rbytes + g1_wbytes, hw)
    g1_us     = max(g1_comp, g1_mem) + hw["launch_us"]
    g1_bnd    = "COMPUTE" if g1_comp >= g1_mem else "MEM"

    # ── GEMM2  (1 persistent Triton kernel, FP32 accum / FP8 weights) ─
    # [T_sorted, I] @ [E_LOCAL, H, I]^T  → [T_sorted, H]  FP32
    g2_comp   = lookup_gemm_us(T_sorted, H, I, "fp32", hw)
    g2_rbytes = T_sorted * I * 4 + E_LOCAL * H * I * 1         # FP32 acts + FP8 W2
    g2_wbytes = T_sorted * H * 4                                # FP32 output
    g2_mem    = lookup_hbm_us(g2_rbytes + g2_wbytes, hw)
    g2_us     = max(g2_comp, g2_mem) + hw["launch_us"]
    g2_bnd    = "COMPUTE" if g2_comp >= g2_mem else "MEM"

    # ── Reduce  (1 Triton kernel) ──────────────────────────────────────
    # weighted scatter-add [T_sorted, H] FP32 → [T, H] BF16
    rd_bytes = T_sorted * H * 4 + T * H * 2
    rd_mem   = lookup_hbm_us(rd_bytes, hw)
    rd_us    = rd_mem + hw["launch_us"]

    total = r_us + s_us + tm1_us + tm2_us + ai_us + z_us + g1_us + g2_us + rd_us
    # Critical path: routing → sort → tilemap1 → gemm1 → gemm2 → reduce
    # (tilemap2, assign, zero run in parallel with tilemap1→gemm path)
    crit  = r_us + s_us + tm1_us + g1_us + g2_us + rd_us

    return {
        "routing":   dict(mem=r_mem,   comp=0.,        n=r_n,   total=r_us,   bound="MEM"),
        "sort":      dict(mem=s_mem,   comp=0.,        n=s_n,   total=s_us,   bound="MEM"),
        "tilemap1":  dict(mem=tm1_mem, comp=0.,        n=tm1_n, total=tm1_us, bound="MEM"),
        "tilemap2":  dict(mem=tm2_mem, comp=0.,        n=tm2_n, total=tm2_us, bound="MEM"),
        "assign":    dict(mem=ai_mem,  comp=0.,        n=ai_n,  total=ai_us,  bound="MEM"),
        "zero":      dict(mem=z_mem,   comp=0.,        n=1,     total=z_us,   bound="MEM"),
        "gemm1":     dict(mem=g1_mem,  comp=g1_comp,   n=1,     total=g1_us,  bound=g1_bnd),
        "gemm2":     dict(mem=g2_mem,  comp=g2_comp,   n=1,     total=g2_us,  bound=g2_bnd),
        "reduce":    dict(mem=rd_mem,  comp=0.,        n=1,     total=rd_us,  bound="MEM"),
        "_total":    total,
        "_crit":     crit,
        "_Tk":       Tk,
        "_T_sorted": T_sorted,
        "_n_tiles_g1": n_tiles_g1,
        "_n_tiles_g2": n_tiles_g2,
    }


def _node_label(name: str, d: dict, total_us: float, detail: str = "") -> str:
    """Build the multi-line label for a compute / bookkeeping node."""
    pct  = 100.0 * d["total"] / total_us if total_us else 0.0
    n    = d["n"]
    kern = f"{n} kernel{'s' if n != 1 else ''}"
    lines = [name]
    if detail:
        lines.append(detail)
    lines.append(kern)
    lines.append(f"mem: {_fmt(d['mem'])} | comp: {_fmt(d['comp'])}")
    lines.append(f"total: {_fmt(d['total'])} ({pct:.1f}%) [{d['bound']}]")
    return "\n".join(lines)


def _timeline_text(costs: dict, T: int) -> str:
    """Build the sequential execution timeline (top-right text box)."""
    total_us = costs["_total"]
    stages = [
        ("routing",  "Routing"),
        ("sort",     "Sort by expert"),
        ("tilemap1", "build_tile_map GEMM1"),
        ("tilemap2", "build_tile_map GEMM2"),
        ("assign",   "build_assign_idx"),
        ("zero",     "output.zero_()"),
        ("gemm1",    "GEMM1+SwiGLU"),
        ("gemm2",    "GEMM2"),
        ("reduce",   "Reduce"),
    ]
    lines = [f"Sequential Execution Timeline (T={T})"]
    cum = 0.0
    for key, label in stages:
        d = costs[key]
        cum += d["total"]
        lines.append(
            f"{label}: mem={_fmt(d['mem'])} comp={_fmt(d['comp'])}"
            f" -> {_fmt(d['total'])} [{d['bound'].lower()}] (cum={_fmt(cum)})"
        )
    lines.append("=" * 48)
    lines.append(f"Total: {_fmt(total_us)}")
    return "\n".join(lines)


def generate_dag(T: int, hw: dict, output: str = "deepdive/newkernel_dag") -> None:
    """Generate and render the graphviz DAG for kernel.py at a given T."""
    costs    = _compute_costs(T, hw)
    total_us = costs["_total"]
    Tk       = costs["_Tk"]
    T_sorted = costs["_T_sorted"]
    ntg1     = costs["_n_tiles_g1"]
    ntg2     = costs["_n_tiles_g2"]

    # In DOT format, \n inside a quoted string is a literal newline — use \\n for
    # a newline in the rendered label.  Also avoid -> which is an edge operator.
    title = (
        f"kernel.py (Triton) — Critical Path DAG  —  T={T}, "
        f"Predicted Total: {_fmt(total_us)}\\n"
        f"Green = COMPUTE (3 Triton launches) | Purple = Routing (PyTorch MEM) "
        f"| Blue = bookkeeping\\n"
        f"Red = on critical path: Route  Sort  TileMap  GEMM1+SwiGLU  GEMM2  Reduce  "
        f"| Thickness proportional to latency"
    )

    dot = graphviz.Digraph(name="kernel_dag", comment="kernel.py Triton DAG")
    dot.attr(
        rankdir="TB",
        fontname="Helvetica",
        fontsize="11",
        label=title,
        labelloc="b",
        labeljust="c",
        nodesep="0.5",
        ranksep="0.6",
        splines="spline",
        pad="0.4",
    )
    dot.attr("node", fontname="Helvetica", fontsize="10", margin="0.15,0.10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # ── Tensor ellipse helpers ─────────────────────────────────────────
    def _wt(**kw):  # weight/input tensor (green)
        return dict(shape="ellipse", style="filled",
                    fillcolor=C_W_FILL, fontcolor=C_W_FONT, fontsize="9", **kw)

    def _it(**kw):  # intermediate tensor (cream)
        return dict(shape="ellipse", style="filled",
                    fillcolor=C_I_FILL, fontcolor=C_I_FONT, fontsize="9", **kw)

    def _box(color):  # compute node (rectangle)
        return dict(shape="box", style="filled,rounded",
                    fillcolor=color, fontcolor="white")

    # ── Input weight / bias tensors (green ellipses) ───────────────────
    dot.node("routing_logits", f"routing_logits\n[T={T}, {E_GLOBAL}] BF16",  **_wt())
    dot.node("routing_bias",   f"routing_bias\n[{E_GLOBAL}] BF16",            **_wt())
    dot.node("hidden_states",  f"hidden_states\n[T={T}, {H}] FP8",            **_wt())
    dot.node("hs_scale",       f"hidden_states_scale\n[56, T] FP32",           **_wt())
    dot.node("w13",            f"gemm1_weights\n[{E_LOCAL}, {2*I}, {H}] FP8", **_wt())
    dot.node("w13_scale",      f"gemm1_weights_scale\n[{E_LOCAL}, 32, 56]",   **_wt())
    dot.node("w2",             f"gemm2_weights\n[{E_LOCAL}, {H}, {I}] FP8",   **_wt())
    dot.node("w2_scale",       f"gemm2_weights_scale\n[{E_LOCAL}, 56, 16]",   **_wt())

    # ── Intermediate / output tensors (cream ellipses) ─────────────────
    dot.node("topk_out",    f"topk_idx,\nweights",                          **_it())
    dot.node("sorted_tids", f"sorted_tids\n[{T_sorted}] int32",             **_it())
    dot.node("sorted_wts",  f"sorted_wts\n[{T_sorted}] fp32",               **_it())
    dot.node("expert_off",  f"expert_offsets\n[{E_LOCAL+1}] int32",         **_it())
    dot.node("tile_g1",     f"tile_map\n[{ntg1}, 5] int64",                 **_it())
    dot.node("tile_g2",     f"tile_map\n[{ntg2}, 5] int64",                 **_it())
    dot.node("act_fp32",    f"act_fp32\n[{T_sorted}, {I}] FP32",            **_it())
    dot.node("g2o",         f"g2o\n[{T_sorted}, {H}] FP32",                 **_it())
    dot.node("output_buf",  f"output buf\n[T={T}, {H}] BF16",               **_it())
    dot.node("reorder",     f"reorder,\nts, tc",                             **_it())
    dot.node("output",      f"output\n[T={T}, {H}] BF16",                   **_it())

    # ── Compute / bookkeeping nodes (coloured rectangles) ──────────────
    c = costs

    # Routing (purple)
    dot.node("routing",
             _node_label("Routing (L51-70)", c["routing"], total_us,
                         "sigmoid topk scatter\nnormalize weights"),
             **_box(C_ROUTE))

    dot.node("sort",
             _node_label("Sort by expert (L73-95)", c["sort"], total_us,
                         "local_mask argsort\nexpert_offsets"),
             **_box(C_ROUTE))

    # Bookkeeping (blue)
    dot.node("tilemap1",
             _node_label("build_tile_map (L479)", c["tilemap1"], total_us,
                         f"GEMM1: E*ceil(Tk/{64})*16 = {ntg1} tiles"),
             **_box(C_BOOK))

    dot.node("tilemap2",
             _node_label("build_tile_map (L505)", c["tilemap2"], total_us,
                         f"GEMM2: E*ceil(Tk/{64})*56 = {ntg2} tiles"),
             **_box(C_BOOK))

    dot.node("assign",
             _node_label("build_assign_idx (L529)", c["assign"], total_us,
                         "argsort scatter_add cumsum"),
             **_box(C_BOOK))

    # Zero (red)
    dot.node("zero",
             _node_label("output.zero_() (L463)", c["zero"], total_us),
             **_box(C_ZERO))

    # Triton GEMM compute (dark teal)
    dot.node("gemm1",
             _node_label("GEMM1+SwiGLU (L483)", c["gemm1"], total_us,
                         f"1 persistent Triton kernel\n"
                         f"FP8 dot  FP32 acc  silu(X2)*X1\n"
                         f"[{T_sorted},{H}] @ [E,{2*I},{H}] -> [{T_sorted},{I}]"),
             **_box(C_COMPUTE))

    dot.node("gemm2",
             _node_label("GEMM2 (L509)", c["gemm2"], total_us,
                         f"1 persistent Triton kernel\n"
                         f"FP32 dot  (A=FP32, B=FP8->FP32)\n"
                         f"[{T_sorted},{I}] @ [E,{H},{I}] -> [{T_sorted},{H}]"),
             **_box(C_COMPUTE))

    dot.node("reduce",
             _node_label("Reduce (L532)", c["reduce"], total_us,
                         f"1 Triton kernel\n"
                         f"weighted scatter-add BF16\n"
                         f"grid [T={T}, ceil({H}/128)]"),
             **_box(C_COMPUTE))

    # ── Timeline annotation (top-right text box, like refkernel.pdf) ───
    dot.node("timeline",
             _timeline_text(costs, T),
             shape="note", style="filled", fillcolor="#fffef0",
             fontcolor="#333", fontsize="8", fontname="Courier",
             margin="0.2,0.1")

    # ── Rank groups  ───────────────────────────────────────────────────
    with dot.subgraph() as s:
        s.attr(rank="source")
        s.node("timeline")
        s.node("routing_logits"); s.node("routing_bias")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("sorted_tids"); s.node("expert_off"); s.node("sorted_wts")

    # Weight inputs and tile maps share rank with GEMM1
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("w13"); s.node("w13_scale")
        s.node("tilemap1")
        s.node("hidden_states"); s.node("hs_scale")
        s.node("tilemap2")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("gemm1")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("w2"); s.node("w2_scale")
        s.node("gemm2")
        s.node("assign")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("zero"); s.node("reduce")

    # ── Edges ──────────────────────────────────────────────────────────
    # Edge thickness proportional to latency (like refkernel.pdf)
    max_us = max(c[k]["total"] for k in
                 ("routing","sort","tilemap1","gemm1","gemm2","reduce"))

    def _pw(key: str, scale: float = 4.0) -> str:
        return f"{max(0.5, scale * c[key]['total'] / max_us):.2f}"

    CRIT = dict(color=C_CRIT,    weight="10")
    NORM = dict(color="#555555", penwidth="1.0")

    # --- Input tensors → Routing
    dot.edge("routing_logits", "routing", **NORM)
    dot.edge("routing_bias",   "routing", **NORM)

    # --- Routing → Sort  (critical)
    dot.edge("routing", "topk_out",  penwidth=_pw("routing"), **CRIT)
    dot.edge("topk_out", "sort",     penwidth=_pw("routing"), **CRIT)

    # --- Sort → outputs
    dot.edge("sort", "sorted_tids",  label="token ids", penwidth=_pw("sort"), **CRIT)
    dot.edge("sort", "expert_off",   label="",          penwidth=_pw("sort"), **CRIT)
    dot.edge("sort", "sorted_wts",   **NORM)

    # --- expert_off → TileMaps  (GEMM1 branch is critical)
    dot.edge("expert_off", "tilemap1", penwidth=_pw("sort"),    **CRIT)
    dot.edge("expert_off", "tilemap2",                           **NORM)
    dot.edge("tilemap1",   "tile_g1",  penwidth=_pw("tilemap1"), **CRIT)
    dot.edge("tilemap2",   "tile_g2",                            **NORM)

    # --- GEMM1 inputs
    dot.edge("sorted_tids",  "gemm1", label="token ids",  penwidth=_pw("tilemap1"), **CRIT)
    dot.edge("tile_g1",      "gemm1", label="tile_map",   penwidth=_pw("tilemap1"), **CRIT)
    dot.edge("hidden_states","gemm1", label="FP8 A",      **NORM)
    dot.edge("hs_scale",     "gemm1", label="A scale",    **NORM)
    dot.edge("w13",          "gemm1", label="FP8 W13",    **NORM)
    dot.edge("w13_scale",    "gemm1", label="W13 scale",  **NORM)

    # --- GEMM1 → GEMM2  (critical)
    dot.edge("gemm1", "act_fp32",
             label=f"act_fp32\n[{T_sorted}, {I}]",
             penwidth=_pw("gemm1"), **CRIT)
    dot.edge("act_fp32", "gemm2", penwidth=_pw("gemm1"), **CRIT)

    # --- GEMM2 inputs
    dot.edge("tile_g2",  "gemm2", label="tile_map",  **NORM)
    dot.edge("w2",       "gemm2", label="FP8 W2",    **NORM)
    dot.edge("w2_scale", "gemm2", label="W2 scale",  **NORM)

    # --- GEMM2 → Reduce  (critical)
    dot.edge("gemm2", "g2o",
             label=f"g2o\n[{T_sorted}, {H}]",
             penwidth=_pw("gemm2"), **CRIT)
    dot.edge("g2o", "reduce", penwidth=_pw("gemm2"), **CRIT)

    # --- Side paths → Reduce
    dot.edge("sorted_tids", "assign",     label="token ids",    **NORM)
    dot.edge("assign",      "reorder",                           **NORM)
    dot.edge("reorder",     "reduce",     label="reorder,ts,tc", **NORM)
    dot.edge("sorted_wts",  "reduce",     label="weights",       **NORM)
    dot.edge("zero",        "output_buf",                         **NORM)
    dot.edge("output_buf",  "reduce",     label="output buf",    **NORM)

    # --- Reduce → output  (critical)
    dot.edge("reduce", "output", penwidth=_pw("reduce"), **CRIT)

    # ── Render ─────────────────────────────────────────────────────────
    dot.render(output, format="pdf", cleanup=True)
    out_path = f"{output}.pdf"
    print(f"Saved -> {out_path}")

    # Also print timeline to stdout
    print()
    print(_timeline_text(costs, T))
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate Critical Path DAG PDF for kernel.py (Triton MoE)"
    )
    p.add_argument("--T", type=int, default=4096,
                   help="Input token count (default: 4096)")
    p.add_argument("--output", default="deepdive/newkernel_dag",
                   help="Output path without extension "
                        "(default: deepdive/newkernel_dag)")
    p.add_argument("--all-workloads", action="store_true",
                   help="Generate one PDF per canonical workload tier "
                        "(tiny/small/medium/large)")
    args = p.parse_args()

    hw = load_hw()

    if args.all_workloads:
        for name, T in WORKLOADS:
            out = f"deepdive/newkernel_dag_{name}_T{T}"
            generate_dag(T, hw, out)
    else:
        generate_dag(args.T, hw, args.output)


if __name__ == "__main__":
    main()
