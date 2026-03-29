"""
AST-Based DAG Cost Model for MoE Kernels -- B200 GPU
=====================================================

Parses kernel Python source code via AST to automatically build a compute DAG,
then predicts latency using measured B200 hardware primitives.

Usage:
    python deepdive/kernel_cost_model.py solution/triton/kernel_ref.py --T 52
    python deepdive/kernel_cost_model.py solution/triton/kernel.py     --T 52
    python deepdive/kernel_cost_model.py --compare --T 52
    python deepdive/kernel_cost_model.py --all-workloads
    python deepdive/kernel_cost_model.py solution/triton/kernel.py --T 1 --dag-graph dag_seq1.png
    python deepdive/kernel_cost_model.py solution/triton/kernel.py --T 1 \\
        --ncu NCU/kernel_wl01_seq1.ncu-rep --dag-graph dag_seq1.png

How it works:
  1) Parse the .py file with Python's ast module
  2) Walk the AST to find: function calls, loops, tensor ops, shapes
  3) Build DagNode list from detected operations
  4) Predict latency using measured GEMM lookups + HBM bandwidth interpolation
  5) (Optional) Load NCU profile report, map profiled kernels to DAG stages,
     and render a compute/memory latency DAG graph (requires matplotlib).

DAG Graph:
  Each node = one logical pipeline stage (Routing, TileMap, GEMM1, GEMM2, Reduce).
  Node colour  = bottleneck:  RED = compute-bound  |  BLUE = memory-bound
                               ORANGE = launch-overhead-dominated
  Two rows per node: predicted latency (cost model) vs actual (NCU profiling).
  Edges carry the data tensor size flowing between stages.
  Critical path is highlighted in red.
"""

import argparse
import ast
import csv
import json
import math
import os
import subprocess
import sys
import textwrap


# =====================================================================
# Hardware Cost Primitives (B200 measured)
# =====================================================================
def load_hw(path=None):
    if path is None:
        for p in ["hw_cost_primitives.json", "deepdive/hw_cost_primitives.json"]:
            if os.path.exists(p):
                path = p
                break
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        hw = {
            "hbm_bw_GBps": raw["data_movement"]["hbm_copy"]["measurements"][-1]["bandwidth_GBps"],
            "fp8_tflops":  raw["compute"]["fp8_gemm"]["peak_tflops"],
            "bf16_tflops": raw["compute"]["bf16_gemm"]["peak_tflops"],
            "fp32_tflops": raw["compute"]["fp32_gemm"]["peak_tflops"],
            "launch_us":   raw["kernel_launch_overhead"]["estimated_overhead_us"],
        }
        # Store measured GEMM lookup tables keyed by (N, K, precision)
        # Each entry: list of (M, latency_us) sorted by M
        hw["gemm_tables"] = {}
        for prec in ("fp8", "fp32"):
            key = f"{prec}_gemm"
            for m in raw["compute"][key]["measurements"]:
                N, K = m["N"], m["K"]
                tkey = (N, K, prec)
                if tkey not in hw["gemm_tables"]:
                    hw["gemm_tables"][tkey] = []
                hw["gemm_tables"][tkey].append((m["M"], m["latency_us"]))
        # Sort each table by M
        for tkey in hw["gemm_tables"]:
            hw["gemm_tables"][tkey].sort()

        # Store HBM copy bandwidth table: list of (bytes, latency_us)
        hw["hbm_table"] = [(m["bytes"], m["latency_us"])
                           for m in raw["data_movement"]["hbm_copy"]["measurements"]]
        hw["hbm_table"].sort()
        return hw
    return {
        "hbm_bw_GBps": 6347., "fp8_tflops": 3070., "bf16_tflops": 1628.,
        "fp32_tflops": 64., "launch_us": 12.29,
        "gemm_tables": {}, "hbm_table": [],
    }


def _log_interp(x, x0, y0, x1, y1):
    """Log-linear interpolation between two points."""
    if x1 == x0:
        return y0
    lx = math.log(max(x, 1))
    lx0 = math.log(max(x0, 1))
    lx1 = math.log(max(x1, 1))
    if lx1 == lx0:
        return y0
    t = (lx - lx0) / (lx1 - lx0)
    return y0 + t * (y1 - y0)


def lookup_gemm_us(M, N, K, prec, hw):
    """Look up measured GEMM latency by interpolating on M dimension.
    Returns latency in microseconds for one GEMM call."""
    tkey = (N, K, prec)
    table = hw.get("gemm_tables", {}).get(tkey)
    if not table:
        # Fallback: roofline with peak TFLOPS
        flops = 2 * M * N * K
        pk = hw.get(f"{prec}_tflops", hw["fp32_tflops"])
        return max(hw["launch_us"], flops / (pk * 1e6))

    # Binary search for bracket
    Ms = [t[0] for t in table]
    Ls = [t[1] for t in table]
    if M <= Ms[0]:
        return Ls[0]
    if M >= Ms[-1]:
        # Extrapolate linearly from last two points
        return _log_interp(M, Ms[-2], Ls[-2], Ms[-1], Ls[-1])
    for i in range(len(Ms) - 1):
        if Ms[i] <= M <= Ms[i + 1]:
            return _log_interp(M, Ms[i], Ls[i], Ms[i + 1], Ls[i + 1])
    return Ls[-1]


def lookup_hbm_us(nbytes, hw):
    """Look up HBM transfer latency by interpolating measured bandwidth curve."""
    table = hw.get("hbm_table")
    if not table or nbytes <= 0:
        return max(0, nbytes / (hw["hbm_bw_GBps"] * 1e3))
    Bs = [t[0] for t in table]
    Ls = [t[1] for t in table]
    if nbytes <= Bs[0]:
        return Ls[0]
    if nbytes >= Bs[-1]:
        # Extrapolate: use peak bandwidth beyond largest measurement
        bw = hw["hbm_bw_GBps"] * 1e3  # GB/s -> bytes/us
        return nbytes / bw
    for i in range(len(Bs) - 1):
        if Bs[i] <= nbytes <= Bs[i + 1]:
            return _log_interp(nbytes, Bs[i], Ls[i], Bs[i + 1], Ls[i + 1])
    return nbytes / (hw["hbm_bw_GBps"] * 1e3)


# =====================================================================
# DAG Node
# =====================================================================
class DagNode:
    """One node in the kernel compute DAG."""
    __slots__ = ("name", "bytes_rw", "flops", "precision", "n_launches",
                 "hw_util", "src_line", "note", "gemm_shape")

    def __init__(self, name, bytes_rw, flops=0, precision="mem",
                 n_launches=1, hw_util=1.0, src_line=0, note="",
                 gemm_shape=None):
        self.name = name
        self.bytes_rw = int(bytes_rw)
        self.flops = int(flops)
        self.precision = precision
        self.n_launches = n_launches
        self.hw_util = hw_util
        self.src_line = src_line
        self.note = note
        self.gemm_shape = gemm_shape  # (M, N, K, prec) for GEMM lookup

    def cost_us(self, hw):
        """Compute cost using measured lookups when available."""
        # GEMM nodes: use measured lookup table
        if self.gemm_shape:
            M, N, K, prec = self.gemm_shape
            per_launch = lookup_gemm_us(M, N, K, prec, hw)
            return per_launch * self.n_launches, "COMPUTE"

        # Memory-only nodes: use HBM bandwidth interpolation
        if self.precision == "mem" or self.flops == 0:
            mem_us = lookup_hbm_us(self.bytes_rw, hw) if self.bytes_rw > 0 else 0
            per_launch = mem_us + hw["launch_us"]
            return per_launch * self.n_launches, "MEM"

        # Compute nodes without GEMM lookup: roofline
        mem_us = lookup_hbm_us(self.bytes_rw, hw) if self.bytes_rw > 0 else 0
        pk = f"{self.precision}_tflops"
        peak = hw.get(pk, hw["fp32_tflops"])
        comp_us = self.flops / (peak * self.hw_util * 1e6)
        bound = "MEM" if mem_us >= comp_us else "COMPUTE"
        per_launch = max(mem_us, comp_us) + hw["launch_us"]
        return per_launch * self.n_launches, bound


def predict_dag(nodes, hw, verbose=False):
    total = 0.0
    if verbose:
        print(f"  {'Node':<28} {'Line':>4} {'Bytes':>12} {'FLOPs':>14} "
              f"{'Prec':<5} {'#L':>3} {'Cost(us)':>10} {'Bound':<7} Note")
        print("  " + "-" * 105)
    for n in nodes:
        c, b = n.cost_us(hw)
        if verbose:
            print(f"  {n.name:<28} {n.src_line:>4} {n.bytes_rw:>12,} "
                  f"{n.flops:>14,} {n.precision:<5} {n.n_launches:>3} "
                  f"{c:>10.1f} {b:<7} {n.note}")
        total += c
    if verbose:
        print(f"  {'-'*105}")
        print(f"  TOTAL {total:>80.1f} us = {total/1e3:.3f} ms")
    return total / 1000.0


# =====================================================================
# MoE geometry constants
# =====================================================================
H = 7168
I = 2048
E_LOCAL = 32
E_GLOBAL = 256
BLOCK = 128
TOP_K = 8


# =====================================================================
# AST-based DAG extraction
# =====================================================================

class TensorShape:
    def __init__(self, dims, dtype="fp32", nbytes_elem=4):
        self.dims = dims
        self.dtype = dtype
        self.nbytes_elem = nbytes_elem

    def numel(self, ctx):
        result = 1
        for d in self.dims:
            result *= _eval_dim(d, ctx)
        return result

    def nbytes(self, ctx):
        return self.numel(ctx) * self.nbytes_elem


def _eval_dim(d, ctx):
    if isinstance(d, (int, float)):
        return int(d)
    if isinstance(d, str):
        return ctx.get(d, 1)
    return 1


def _dtype_bytes(dtype_str):
    m = {"fp8": 1, "float8_e4m3fn": 1, "fp16": 2, "bf16": 2, "bfloat16": 2,
         "fp32": 4, "float32": 4, "float": 4, "int32": 4, "int64": 8, "bool": 1}
    return m.get(dtype_str, 4)


def _detect_dtype(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return "float32"


class KernelAnalyzer(ast.NodeVisitor):
    """Walk AST of a kernel source file to extract compute DAG nodes."""

    def __init__(self, T, kernel_type="auto"):
        self.T = T
        self.Tk = max(1, T * TOP_K // E_GLOBAL)
        self.T_sorted = max(1, T * TOP_K * E_LOCAL // E_GLOBAL)
        self.kernel_type = kernel_type
        self.nodes = []
        self.loop_mult = 1
        self.in_expert_loop = False
        self.ctx = {"T": T, "H": H, "I": I, "Tk": self.Tk,
                    "E_local": E_LOCAL, "E_global": E_GLOBAL,
                    "BLOCK": BLOCK, "TOP_K": TOP_K,
                    "T_sorted": self.T_sorted,
                    "seq_len": T, "HIDDEN_SIZE": H,
                    "INTERMEDIATE_SIZE": I, "GEMM1_OUT_SIZE": 2*I,
                    "NUM_LOCAL_EXPERTS": E_LOCAL, "NUM_EXPERTS": E_GLOBAL}
        self.has_triton = False
        self._detected_fns = set()

    def analyze(self, source_code):
        tree = ast.parse(source_code)
        func_defs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "triton":
                self.has_triton = True
            if isinstance(node, ast.FunctionDef):
                func_defs[node.name] = node

        fn = None
        if self.has_triton and "kernel_triton" in func_defs:
            fn = func_defs["kernel_triton"]
        elif "kernel_pytorch" in func_defs:
            fn = func_defs["kernel_pytorch"]
        elif "run" in func_defs:
            fn = func_defs["run"]
        elif "kernel" in func_defs:
            fn = func_defs["kernel"]

        if fn:
            self._analyze_function(fn)
        return self.nodes

    def _analyze_function(self, func_node):
        if func_node.name in self._detected_fns:
            return
        self._detected_fns.add(func_node.name)
        for stmt in ast.iter_child_nodes(func_node):
            self._visit_stmt(stmt)

    def _visit_stmt(self, stmt):
        line = getattr(stmt, "lineno", 0)
        if isinstance(stmt, ast.Assign):
            self._visit_assign(stmt, line)
        elif isinstance(stmt, ast.Expr):
            self._visit_expr(stmt.value, line)
        elif isinstance(stmt, ast.For):
            self._visit_for(stmt, line)
        elif isinstance(stmt, ast.If):
            for s in stmt.body:
                self._visit_stmt(s)
            for s in stmt.orelse:
                self._visit_stmt(s)
        elif isinstance(stmt, ast.Return):
            self._visit_return(stmt, line)

    def _visit_for(self, for_node, line):
        loop_count = self._get_loop_range(for_node)
        if loop_count and loop_count >= E_LOCAL:
            old_mult = self.loop_mult
            old_in_loop = self.in_expert_loop
            self.loop_mult = loop_count
            self.in_expert_loop = True
            for s in for_node.body:
                self._visit_stmt(s)
            self.loop_mult = old_mult
            self.in_expert_loop = old_in_loop
        else:
            for s in for_node.body:
                self._visit_stmt(s)

    def _get_loop_range(self, for_node):
        if isinstance(for_node.iter, ast.Call):
            func = for_node.iter
            if isinstance(func.func, ast.Name) and func.func.id == "range":
                if func.args:
                    return self._eval_const(func.args[-1])
        return None

    def _visit_assign(self, assign, line):
        target_name = self._get_target_name(assign.targets[0]) if assign.targets else ""
        value = assign.value
        self._detect_op(value, target_name, line)

    def _visit_expr(self, expr, line):
        self._detect_op(expr, "", line)

    def _visit_return(self, ret, line):
        if ret.value:
            self._detect_op(ret.value, "_return", line)

    def _detect_op(self, node, target, line):
        if node is None:
            return

        if isinstance(node, ast.Call):
            func = node.func

            if isinstance(func, ast.Attribute) and func.attr == "to":
                self._emit_cast(target, node, line)
                return

            if isinstance(func, ast.Attribute) and func.attr in ("matmul", "mm"):
                self._emit_matmul(target, func.value, node.args[0], line)
                return

            if self._is_torch_call(func, "repeat_interleave"):
                self._emit_repeat_interleave(target, node, line)
                return

            if self._is_torch_call(func, "sigmoid"):
                self._emit_pointwise(target, "sigmoid", line,
                                     read_bytes=self.T * E_GLOBAL * 4,
                                     write_bytes=self.T * E_GLOBAL * 4)
                return

            if self._is_torch_call(func, "topk"):
                self._emit_pointwise(target, "topk", line,
                                     read_bytes=self.T * E_GLOBAL * 4,
                                     write_bytes=self.T * 8 * 8)
                return

            if self._is_torch_call(func, "exp"):
                return

            if self._is_torch_call(func, "nonzero"):
                self._emit_pointwise(target, "nonzero", line,
                                     read_bytes=self.T * 8 * 8,
                                     write_bytes=self.Tk * 8)
                return

            if self._is_torch_call(func, "argsort"):
                self._emit_pointwise(target, "argsort", line,
                                     read_bytes=self.T_sorted * 4,
                                     write_bytes=self.T_sorted * 4,
                                     n_launches=4)
                return

            if self._is_torch_call(func, "zeros"):
                self._emit_zeros(target, node, line)
                return

            if isinstance(func, ast.Attribute) and func.attr == "index_select":
                self._emit_gather(target, line)
                return

            if isinstance(func, ast.Attribute) and func.attr == "index_add_":
                self._emit_scatter(target, line)
                return

            if isinstance(func, ast.Attribute) and func.attr in ("scatter_", "scatter_add_"):
                self._emit_pointwise(target, func.attr, line,
                                     read_bytes=self.T * E_GLOBAL * 4,
                                     write_bytes=self.T * E_GLOBAL * 4)
                return

            if isinstance(func, ast.Attribute) and func.attr == "permute":
                self._emit_pointwise(target, "permute", line,
                                     read_bytes=56 * self.T * 4,
                                     write_bytes=56 * self.T * 4)
                return

            if isinstance(func, ast.Attribute) and func.attr == "zero_":
                self._emit_pointwise(target, "zero_", line,
                                     read_bytes=0,
                                     write_bytes=self.T * H * 2,
                                     n_launches=1)
                return

            if isinstance(func, ast.Attribute) and func.attr == "copy_":
                self._emit_pointwise(target, "copy_", line,
                                     read_bytes=self.T * H * 4,
                                     write_bytes=self.T * H * 2)
                return

            if isinstance(func, ast.Attribute) and func.attr in (
                    "view", "reshape", "unsqueeze", "squeeze", "contiguous",
                    "expand", "expand_as", "clamp", "numel", "item",
                    "tolist", "cpu"):
                return

            if isinstance(func, ast.Name) and func.id == "deepseek_v3_routing":
                self._emit_routing(line)
                return

            if isinstance(func, ast.Name) and func.id == "build_tile_map_gpu":
                self._emit_tile_map(target, line)
                return

            if isinstance(func, ast.Name) and func.id == "_build_assign_idx":
                self._emit_assign_idx(target, line)
                return

            if isinstance(node.func, ast.Subscript):
                kernel_name = self._get_subscript_kernel_name(node.func)
                if kernel_name:
                    self._emit_triton_kernel(kernel_name, node, line)
                    return

            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Subscript):
                kernel_name = self._get_subscript_kernel_name(func.value)
                if kernel_name:
                    self._emit_triton_kernel(kernel_name, node, line)
                    return

        if isinstance(node, ast.BinOp):
            self._detect_binop(node, target, line)
            return

        if isinstance(node, ast.Subscript):
            return

    def _detect_binop(self, node, target, line):
        if isinstance(node.op, ast.Mult):
            if self.in_expert_loop:
                rw = self.Tk * H * 4 * 2
                self.nodes.append(DagNode(
                    f"{'x'+str(self.loop_mult)}_{target}_mul",
                    rw, precision="mem", n_launches=self.loop_mult,
                    src_line=line, note="per-expert elementwise"))
            elif target and any(kw in target.lower() for kw in
                                ("w13", "w2", "a_scale", "a_fp32", "a ")):
                if "w13" in target.lower():
                    rw = E_LOCAL * 2*I*H * 4 * 2
                elif "w2" in target.lower():
                    rw = E_LOCAL * H*I * 4 * 2
                else:
                    rw = self.T * H * 4 * 2
                self.nodes.append(DagNode(
                    f"{target}_mul", rw, precision="mem",
                    n_launches=1, src_line=line, note="dequant multiply"))

    # ---- Emit helpers ----

    def _emit_cast(self, target, call_node, line):
        src_dtype = "fp8"
        dst_dtype = "fp32"
        if call_node.args:
            dst_dtype = _detect_dtype(call_node.args[0])

        if "hidden_states" in (target or "").lower() or target == "A_fp32":
            nbytes_read = self.T * H * _dtype_bytes(src_dtype)
            nbytes_write = self.T * H * _dtype_bytes(dst_dtype)
        elif "gemm1_weights" in (target or "").lower() or "W13" in (target or ""):
            nbytes_read = E_LOCAL * 2*I*H * _dtype_bytes(src_dtype)
            nbytes_write = E_LOCAL * 2*I*H * _dtype_bytes(dst_dtype)
        elif "gemm2_weights" in (target or "").lower() or "W2" in (target or ""):
            nbytes_read = E_LOCAL * H*I * _dtype_bytes(src_dtype)
            nbytes_write = E_LOCAL * H*I * _dtype_bytes(dst_dtype)
        elif "routing" in (target or "").lower() or "logits" in (target or "").lower():
            nbytes_read = self.T * E_GLOBAL * _dtype_bytes(src_dtype)
            nbytes_write = self.T * E_GLOBAL * _dtype_bytes(dst_dtype)
        elif "S13" in (target or "") or "gemm1_weights_scale" in (target or ""):
            nbytes_read = E_LOCAL * 32 * 56 * 4
            nbytes_write = E_LOCAL * 32 * 56 * 4
            self.nodes.append(DagNode(f"{target}_cast", nbytes_read + nbytes_write,
                                      precision="mem", n_launches=1, src_line=line,
                                      note="scale cast"))
            return
        elif "S2" in (target or "") or "gemm2_weights_scale" in (target or ""):
            nbytes_read = E_LOCAL * 56 * 16 * 4
            nbytes_write = E_LOCAL * 56 * 16 * 4
            self.nodes.append(DagNode(f"{target}_cast", nbytes_read + nbytes_write,
                                      precision="mem", n_launches=1, src_line=line,
                                      note="scale cast"))
            return
        else:
            return

        self.nodes.append(DagNode(
            f"{target}_cast", nbytes_read + nbytes_write,
            precision="mem", n_launches=1, src_line=line,
            note=f"{src_dtype}->{dst_dtype}"))

    def _emit_matmul(self, target, lhs, rhs, line):
        """Emit a GEMM node using measured lookup table."""
        if self.in_expert_loop:
            rhs_src = ast.dump(rhs)
            if "W13" in rhs_src or "gemm1" in rhs_src:
                M, N, K = self.Tk, 2*I, H
                label = f"x{self.loop_mult}_GEMM1"
                # kernel_ref uses FP32 matmul: [Tk,H] @ [H,4096]
                # Measured shape: M x 4096 x 7168 FP32
                gemm_shape = (M, 4096, 7168, "fp32")
            elif "W2" in rhs_src or "gemm2" in rhs_src:
                M, N, K = self.Tk, H, I
                label = f"x{self.loop_mult}_GEMM2"
                # Measured shape: M x 7168 x 2048 FP32
                gemm_shape = (M, 7168, 2048, "fp32")
            else:
                M, N, K = self.Tk, H, H
                label = f"x{self.loop_mult}_GEMM"
                gemm_shape = None

            read_bytes = (M*K + N*K) * 4
            write_bytes = M*N * 4
            flops = 2 * M * N * K
            self.nodes.append(DagNode(
                label, read_bytes + write_bytes, flops=flops,
                precision="fp32", n_launches=self.loop_mult,
                src_line=line,
                note=f"[{M},{K}]@[{K},{N}] FP32",
                gemm_shape=gemm_shape))
        else:
            self.nodes.append(DagNode(
                f"{target}_mm", self.T * H * 4 * 3, flops=2*self.T*H*H,
                precision="fp32", n_launches=1, src_line=line))

    def _emit_repeat_interleave(self, target, call_node, line):
        src = ast.dump(call_node)
        if "S13" in src or "gemm1" in src:
            rw = E_LOCAL * 2*I*H * 4 * 2
            label = f"{target}_expand"
        elif "S2" in src or "gemm2" in src:
            rw = E_LOCAL * H*I * 4 * 2
            label = f"{target}_expand"
        else:
            rw = self.T * H * 4 * 2
            label = f"{target}_expand"
        self.nodes.append(DagNode(label, rw, precision="mem",
                                  n_launches=1, src_line=line,
                                  note="repeat_interleave"))

    def _emit_pointwise(self, target, op_name, line, read_bytes=0,
                        write_bytes=0, n_launches=1):
        mult = self.loop_mult if self.in_expert_loop else 1
        self.nodes.append(DagNode(
            f"{target}_{op_name}" if target else op_name,
            read_bytes + write_bytes, precision="mem",
            n_launches=n_launches * mult, src_line=line,
            note=op_name))

    def _emit_zeros(self, target, call_node, line):
        if "output" in (target or "").lower():
            nbytes = self.T * H * 4
        else:
            nbytes = self.T * H * 4
        self.nodes.append(DagNode(
            f"{target}_zeros", nbytes, precision="mem",
            n_launches=1, src_line=line, note="allocation+zero"))

    def _emit_gather(self, target, line):
        if self.in_expert_loop:
            rw = self.Tk * H * 4 + self.Tk * 8
            self.nodes.append(DagNode(
                f"x{self.loop_mult}_gather", rw, precision="mem",
                n_launches=self.loop_mult, src_line=line,
                note="index_select per expert"))
        else:
            rw = self.T * H * 4
            self.nodes.append(DagNode(
                f"{target}_gather", rw, precision="mem",
                n_launches=1, src_line=line))

    def _emit_scatter(self, target, line):
        if self.in_expert_loop:
            rw = self.Tk * H * 4 * 2 + self.Tk * 8
            self.nodes.append(DagNode(
                f"x{self.loop_mult}_scatter", rw, precision="mem",
                n_launches=self.loop_mult, src_line=line,
                note="index_add_ per expert"))

    def _emit_routing(self, line):
        routing_bytes = self.T * E_GLOBAL * 4 * 8
        self.nodes.append(DagNode(
            "routing", routing_bytes, precision="mem",
            n_launches=13, src_line=line,
            note="deepseek_v3_routing"))

    def _emit_tile_map(self, target, line):
        Tk_avg = max(1, self.T_sorted // E_LOCAL)
        n_tiles = E_LOCAL * max(1, math.ceil(Tk_avg / 64)) * 16
        rw = n_tiles * 5 * 8 + E_LOCAL * 8 * 3
        self.nodes.append(DagNode(
            f"{target}_tile_map", rw, precision="mem",
            n_launches=2, src_line=line,
            note=f"build_tile_map_gpu (~{n_tiles} tiles)"))

    def _emit_assign_idx(self, target, line):
        rw = self.T_sorted * 4 * 6 + self.T * 4 * 3
        self.nodes.append(DagNode(
            "build_assign_idx", rw, precision="mem",
            n_launches=3, src_line=line,
            note="argsort+scatter_add+cumsum"))

    def _emit_triton_kernel(self, kernel_name, call_node, line):
        """Emit Triton kernel nodes using measured GEMM lookups."""
        if "gemm1" in kernel_name.lower() and "swiglu" in kernel_name.lower():
            # GEMM1+SwiGLU persistent kernel: FP8 tensor cores
            # Total GEMM across all experts: T_sorted x 4096 x 7168
            # But it's a persistent kernel processing E_LOCAL experts sequentially
            # Model as: sum of per-expert FP8 GEMMs + SwiGLU fused
            g1_read = (E_LOCAL * 2*I*H * 1        # W13 FP8
                       + self.T_sorted * H * 1     # hidden FP8
                       + E_LOCAL * 32*56 * 4        # W13 scales
                       + 56 * self.T * 4            # hidden scales
                       + self.T_sorted * 4)         # sorted_ids
            g1_write = self.T_sorted * I * 4       # act_fp32
            g1_flops = 2 * self.T_sorted * H * 2*I + self.T_sorted * I * 5

            # Use measured FP8 GEMM lookup: per-expert Tk x 4096 x 7168
            self.nodes.append(DagNode(
                "GEMM1+SwiGLU", g1_read + g1_write,
                flops=g1_flops, precision="fp8", n_launches=E_LOCAL,
                src_line=line,
                note=f"persistent FP8, Tk={self.Tk}, T_sorted={self.T_sorted}",
                gemm_shape=(self.Tk, 4096, 7168, "fp8")))

        elif "gemm2" in kernel_name.lower():
            # GEMM2 persistent kernel: FP32 CUDA cores (A=FP32 SwiGLU output, B=FP8->FP32)
            g2_read = (self.T_sorted * I * 4       # act_fp32
                       + E_LOCAL * H*I * 1          # W2 FP8
                       + E_LOCAL * 56*16 * 4)       # W2 scales
            g2_write = self.T_sorted * H * 4       # g2o FP32
            g2_flops = 2 * self.T_sorted * I * H

            # Use measured FP32 GEMM lookup: per-expert Tk x 7168 x 2048
            self.nodes.append(DagNode(
                "GEMM2", g2_read + g2_write,
                flops=g2_flops, precision="fp32", n_launches=E_LOCAL,
                src_line=line,
                note=f"persistent FP32 dot, Tk={self.Tk}, T_sorted={self.T_sorted}",
                gemm_shape=(self.Tk, 7168, 2048, "fp32")))

        elif "reduce" in kernel_name.lower():
            r_read = self.T_sorted * H * 4 + self.T_sorted * 4 * 3
            r_write = self.T * H * 2
            self.nodes.append(DagNode(
                "reduce", r_read + r_write, precision="mem",
                n_launches=1, src_line=line,
                note=f"scatter-reduce [{self.T},{H}] BF16"))
        else:
            self.nodes.append(DagNode(
                kernel_name, self.T * H * 4, precision="mem",
                n_launches=1, src_line=line,
                note="unknown triton kernel"))

    # ---- AST helpers ----

    def _get_target_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self._get_target_name(node.value)
        if isinstance(node, ast.Tuple):
            names = [self._get_target_name(e) for e in node.elts]
            return "_".join(n for n in names if n)
        return ""

    def _is_torch_call(self, func, method_name):
        if isinstance(func, ast.Attribute) and func.attr == method_name:
            if isinstance(func.value, ast.Name):
                return func.value.id in ("torch", "F", "tl")
            return True
        if isinstance(func, ast.Name) and func.id == method_name:
            return True
        return False

    def _get_subscript_kernel_name(self, node):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                name = node.value.id
                if name.startswith("_") and "kernel" in name.lower():
                    return name
            if isinstance(node.value, ast.Attribute):
                return node.value.attr
        return None

    def _eval_const(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.ctx.get(node.id, None)
        if isinstance(node, ast.Attribute):
            return None
        return None


def analyze_kernel_file(filepath, T):
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    analyzer = KernelAnalyzer(T)
    nodes = analyzer.analyze(source)
    return nodes, os.path.basename(filepath)


# =====================================================================
# NCU Report Parsing
# =====================================================================

# Maps NCU kernel name substrings → logical DAG stage label.
# Order matters: first match wins.
_NCU_STAGE_MAP = [
    ("_fused_routing_kernel",            "routing"),
    ("_counting_sort_kernel",            "routing"),
    ("DeviceScan",                       "routing"),
    ("CatArrayBatchedCopy",              "routing"),
    ("scatter_gather",                   "reduce"),
    ("_build_tile_map_kernel",           "tile_map"),
    ("_gemm1_swiglu_kernel",             "GEMM1+SwiGLU"),
    ("_gemm2_kernel",                    "GEMM2"),
    # Catch-all torch elementwise / unrolled kernels are split by position.
    # We tag them generically then reassign by sequence in assign_ncu_stages().
    ("FillFunctor<int>",                 "zero_int"),
    ("FillFunctor<float>",               "zero_float"),
    ("direct_copy_kernel",               "copy"),
    ("elementwise_kernel",               "elementwise"),
    ("reduce_kernel",                    "reduce_aux"),
    ("bfloat16_copy",                    "output_copy"),
    ("BUnaryFunctor",                    "routing"),
    ("AUnaryFunctor",                    "routing"),
    ("CUDAFunctor",                      "routing"),
    ("CUDAFunctorOnSelf",                "routing"),
    ("launch_clamp",                     "routing"),
    ("gpu_kernel_impl_nocast",           "elementwise"),
]

# Per-stage: which NCU kernels are compute-bound vs memory-bound?
# A stage is "launch-overhead dominated" when actual SM + DRAM util are both < 1 %.
_COMPUTE_STAGES = {"GEMM1+SwiGLU", "GEMM2"}


def _ncu_stage(kernel_name: str) -> str:
    """Return the logical DAG stage for a raw NCU kernel name."""
    for fragment, stage in _NCU_STAGE_MAP:
        if fragment in kernel_name:
            return stage
    return "other"


def parse_ncu_report(ncu_path: str) -> list[dict]:
    """
    Call ncu CLI to export the .ncu-rep file as CSV (--page details) and
    parse it into a list of per-kernel dicts with keys:
        id, name, stage, duration_us,
        sm_throughput_pct, dram_throughput_pct, mem_throughput_GBps
    Returns [] on failure.
    """
    # Locate ncu executable
    ncu_exe = None
    for candidate in [
        "ncu",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\ncu.bat",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat",
        r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\ncu.bat",
    ]:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                ncu_exe = candidate
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if ncu_exe is None:
        print("[NCU] WARNING: ncu executable not found; skipping NCU data.", file=sys.stderr)
        return []

    try:
        result = subprocess.run(
            [ncu_exe, "--import", ncu_path, "--csv", "--page", "details"],
            capture_output=True, text=True, timeout=120
        )
        raw = result.stdout
    except subprocess.TimeoutExpired:
        print("[NCU] WARNING: ncu timed out.", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[NCU] WARNING: {e}", file=sys.stderr)
        return []

    # Parse CSV — skip non-CSV preamble lines (like .bashrc errors)
    lines = [l for l in raw.splitlines() if l.startswith('"')]
    if not lines:
        print("[NCU] WARNING: no CSV data returned from ncu.", file=sys.stderr)
        return []

    reader = csv.DictReader(iter(lines))
    kernels: dict[tuple, dict] = {}
    order: list[tuple] = []

    for row in reader:
        kname = row.get("Kernel Name", "")
        metric = row.get("Metric Name", "")
        raw_val = row.get("Metric Value", "").replace(",", "")
        try:
            val = float(raw_val)
        except ValueError:
            val = 0.0
        kid = int(row.get("ID", "0") or "0")
        key = (kid, kname)
        if key not in kernels:
            kernels[key] = {}
            order.append(key)
        kernels[key][metric] = val

    result_list = []
    for (kid, kname) in order:
        m = kernels[(kid, kname)]
        dur = m.get("Duration", 0.0)
        sm = m.get("Compute (SM) Throughput", 0.0)
        dram = m.get("DRAM Throughput", 0.0)
        mem_bw = m.get("Memory Throughput", 0.0)   # GB/s raw value
        result_list.append({
            "id": kid,
            "name": kname,
            "stage": _ncu_stage(kname),
            "duration_us": dur,
            "sm_throughput_pct": sm,
            "dram_throughput_pct": dram,
            "mem_throughput_GBps": mem_bw,
        })
    return result_list


def aggregate_ncu_by_stage(ncu_kernels: list[dict]) -> dict[str, dict]:
    """
    Aggregate per-kernel NCU data into per-stage totals.
    Returns dict keyed by stage name with:
        total_us, n_kernels, avg_sm_pct, avg_dram_pct,
        max_sm_pct, max_dram_pct, kernels (list)
    """
    stages: dict[str, dict] = {}
    for k in ncu_kernels:
        s = k["stage"]
        if s not in stages:
            stages[s] = {
                "total_us": 0.0,
                "n_kernels": 0,
                "sm_sum": 0.0,
                "dram_sum": 0.0,
                "max_sm_pct": 0.0,
                "max_dram_pct": 0.0,
                "kernels": [],
            }
        stages[s]["total_us"] += k["duration_us"]
        stages[s]["n_kernels"] += 1
        stages[s]["sm_sum"] += k["sm_throughput_pct"]
        stages[s]["dram_sum"] += k["dram_throughput_pct"]
        stages[s]["max_sm_pct"] = max(stages[s]["max_sm_pct"], k["sm_throughput_pct"])
        stages[s]["max_dram_pct"] = max(stages[s]["max_dram_pct"], k["dram_throughput_pct"])
        stages[s]["kernels"].append(k)

    for s in stages:
        n = stages[s]["n_kernels"]
        stages[s]["avg_sm_pct"] = stages[s]["sm_sum"] / n if n else 0
        stages[s]["avg_dram_pct"] = stages[s]["dram_sum"] / n if n else 0
    return stages


# =====================================================================
# DAG Stage Definitions (logical pipeline)
# =====================================================================

# Ordered pipeline stages for the MoE forward pass.
# Each entry: (stage_id, label, dag_node_name_substrings, inputs, outputs)
_PIPELINE_STAGES = [
    {
        "id": "routing",
        "label": "Routing\n(sigmoid→topK→sort)",
        "dag_names": ["routing", "argsort", "sort", "scatter_add", "cumsum",
                      "zero_int", "copy"],
        "inputs": ["routing_logits\n[T×256 FP32]", "routing_bias\n[256 FP32]"],
        "outputs": ["sorted_tids\n[Tk int32]", "sorted_eids\n[Tk int32]",
                    "expert_offsets\n[33 int32]"],
        "ncu_stages": ["routing", "zero_int", "copy", "reduce_aux"],
    },
    {
        "id": "tile_map",
        "label": "Tile Map\n(build_tile_map)",
        "dag_names": ["tile_map", "build_assign"],
        "inputs": ["expert_offsets\n[33 int32]"],
        "outputs": ["tile_ids\n[n_tiles × 5]"],
        "ncu_stages": ["tile_map"],
    },
    {
        "id": "gemm1",
        "label": "GEMM1 + SwiGLU\n(FP8 → FP32)",
        "dag_names": ["GEMM1", "SwiGLU"],
        "inputs": ["hidden_states\n[T×7168 FP8]",
                   "W1/W3\n[E×4096×7168 FP8]",
                   "tile_ids"],
        "outputs": ["act_fp32\n[Tk×2048 FP32]"],
        "ncu_stages": ["GEMM1+SwiGLU"],
    },
    {
        "id": "gemm2",
        "label": "GEMM2\n(FP32 dot)",
        "dag_names": ["GEMM2"],
        "inputs": ["act_fp32\n[Tk×2048 FP32]",
                   "W2\n[E×7168×2048 FP8]"],
        "outputs": ["g2o_fp32\n[Tk×7168 FP32]"],
        "ncu_stages": ["GEMM2"],
    },
    {
        "id": "reduce",
        "label": "Scatter-Reduce\n(weighted sum → BF16)",
        "dag_names": ["reduce", "scatter", "output_copy", "zero_float",
                      "elementwise"],
        "inputs": ["g2o_fp32\n[Tk×7168 FP32]",
                   "sorted_wts\n[Tk FP32]",
                   "sorted_tids"],
        "outputs": ["output\n[T×7168 BF16]"],
        "ncu_stages": ["reduce", "scatter", "output_copy", "zero_float",
                       "elementwise", "reduce_aux", "other"],
    },
]


def _classify_stage_bound(ncu_stage_data: dict | None, stage_id: str) -> str:
    """
    Returns 'COMPUTE', 'MEMORY', or 'LAUNCH' based on NCU data or stage_id.
    LAUNCH = both SM and DRAM utilisation are very low (launch-overhead dominated).
    """
    if ncu_stage_data is None:
        return "COMPUTE" if stage_id in ("gemm1", "gemm2") else "MEMORY"
    sm = ncu_stage_data.get("avg_sm_pct", 0)
    dram = ncu_stage_data.get("avg_dram_pct", 0)
    if sm < 1.0 and dram < 1.0:
        return "LAUNCH"
    if sm >= dram:
        return "COMPUTE"
    return "MEMORY"


# =====================================================================
# DAG Graph Visualisation
# =====================================================================

def plot_dag_graph(
    dag_nodes: list,
    hw: dict,
    ncu_kernels: list[dict] | None = None,
    output_path: str = "dag_latency.png",
    title: str = "MoE Kernel DAG — Compute / Memory Latency",
    T: int = 1,
):
    """
    Render a compute/memory latency DAG graph for the MoE kernel pipeline.

    Left panel  — vertical DAG flow (stage nodes + arrows)
    Right panel — horizontal bar chart (predicted vs NCU latency per stage)

    Colour scheme:
      RED    = compute-bound   (SM util > DRAM util)
      BLUE   = memory-bound    (DRAM util > SM util)
      ORANGE = launch-overhead (both SM and DRAM < 1%)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        import numpy as np
    except ImportError:
        print("[DAG] matplotlib not installed — cannot generate graph.", file=sys.stderr)
        print("[DAG] Install with:  pip install matplotlib", file=sys.stderr)
        return

    # ── Aggregate NCU data ────────────────────────────────────────────
    ncu_by_stage: dict[str, dict] = {}
    if ncu_kernels:
        raw_agg = aggregate_ncu_by_stage(ncu_kernels)
        for ps in _PIPELINE_STAGES:
            total_us = sum(raw_agg.get(s, {}).get("total_us", 0)
                           for s in ps["ncu_stages"])
            n_kernels = sum(raw_agg.get(s, {}).get("n_kernels", 0)
                            for s in ps["ncu_stages"])
            sm_sum, dram_sum, count = 0.0, 0.0, 0
            for s in ps["ncu_stages"]:
                if s in raw_agg:
                    sm_sum += raw_agg[s]["sm_sum"]
                    dram_sum += raw_agg[s]["dram_sum"]
                    count += raw_agg[s]["n_kernels"]
            ncu_by_stage[ps["id"]] = {
                "total_us": total_us,
                "n_kernels": n_kernels,
                "avg_sm_pct": sm_sum / count if count else 0,
                "avg_dram_pct": dram_sum / count if count else 0,
            }

    # ── Predict latency per stage ─────────────────────────────────────
    def _pred_stage(stage) -> float:
        total = 0.0
        for n in dag_nodes:
            if any(sub.lower() in n.name.lower() for sub in stage["dag_names"]):
                c, _ = n.cost_us(hw)
                total += c
        return total

    # ── Palette ───────────────────────────────────────────────────────
    C = {"COMPUTE": "#c0392b", "MEMORY": "#2980b9", "LAUNCH": "#e67e22"}
    BADGE = {"COMPUTE": "COMPUTE BOUND", "MEMORY": "MEMORY BOUND",
             "LAUNCH": "LAUNCH OVERHEAD"}

    # ── Per-stage data ────────────────────────────────────────────────
    stage_data = []
    for ps in _PIPELINE_STAGES:
        ncu = ncu_by_stage.get(ps["id"])
        bound = _classify_stage_bound(ncu, ps["id"])
        stage_data.append({
            "ps": ps,
            "bound": bound,
            "colour": C[bound],
            "pred_us": _pred_stage(ps),
            "ncu_us": ncu["total_us"] if ncu else None,
            "n_kernels": ncu["n_kernels"] if ncu else None,
            "avg_sm": ncu["avg_sm_pct"] if ncu else None,
            "avg_dram": ncu["avg_dram_pct"] if ncu else None,
        })

    n_stages = len(stage_data)

    # ── Figure: two columns ───────────────────────────────────────────
    fig, (ax, ax_bar) = plt.subplots(
        1, 2,
        figsize=(18, 14),
        gridspec_kw={"width_ratios": [1, 1]},
        facecolor="#f0f2f5",
    )
    for a in (ax, ax_bar):
        a.set_facecolor("#f0f2f5")

    # ══════════════════════════════════════════════════════════════════
    # LEFT PANEL — DAG flow (custom drawing, y=0 at bottom)
    # ══════════════════════════════════════════════════════════════════
    W = 10.0
    cx = W / 2
    node_w = 8.5
    node_h = 1.70
    gap = 0.70           # vertical gap between nodes
    y_step = node_h + gap

    # Assign y positions bottom-up: stage 0 at top → highest y value
    # node i occupies: y_bot[i] = total_height - (i+1)*y_step - header_space
    header_space = 2.2
    total_dag_h = header_space + n_stages * y_step + 0.4

    ax.set_xlim(0, W)
    ax.set_ylim(0, total_dag_h)
    ax.axis("off")

    x_left = cx - node_w / 2

    # y_top of stage i (y increases upward, stage 0 is at top = large y)
    def stage_y_top(i):
        return total_dag_h - header_space - i * y_step

    def stage_y_bot(i):
        return stage_y_top(i) - node_h

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(cx, total_dag_h - 0.5, title,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="#1a252f")
    ax.text(cx, total_dag_h - 1.05,
            f"T={T} · H={H} · I={I} · E_local={E_LOCAL} · TOP_K={TOP_K}",
            ha="center", va="center", fontsize=8, color="#666")

    for i, sd in enumerate(stage_data):
        yt = stage_y_top(i)    # top edge y
        yb = stage_y_bot(i)    # bottom edge y
        col = sd["colour"]
        pred_us = sd["pred_us"]
        ncu_us = sd["ncu_us"]

        # Shadow
        ax.add_patch(FancyBboxPatch(
            (x_left + 0.08, yb - 0.08), node_w, node_h,
            boxstyle="round,pad=0.06", linewidth=0,
            facecolor="#c8ccd4", zorder=1))

        # Main box (outline)
        ax.add_patch(FancyBboxPatch(
            (x_left, yb), node_w, node_h,
            boxstyle="round,pad=0.06", linewidth=2,
            edgecolor=col, facecolor="white", zorder=2))

        # Header strip (top of node)
        hdr_h = 0.42
        ax.add_patch(FancyBboxPatch(
            (x_left, yt - hdr_h), node_w, hdr_h,
            boxstyle="round,pad=0.03", linewidth=0,
            facecolor=col, zorder=3))

        # Stage label + badge
        ax.text(x_left + 0.16, yt - hdr_h / 2,
                sd["ps"]["label"].replace("\n", "  "),
                ha="left", va="center", fontsize=9, fontweight="bold",
                color="white", zorder=4)
        ax.text(x_left + node_w - 0.12, yt - hdr_h / 2,
                f"[{BADGE[sd['bound']]}]",
                ha="right", va="center", fontsize=6.5,
                color="white", fontstyle="italic", zorder=4, alpha=0.9)

        # Body rows
        body_top = yt - hdr_h
        body_bot = yb
        body_h = body_top - body_bot
        rh = body_h / 3.8   # row height

        # Row 1: Predicted
        r1y = body_top - rh * 0.6
        pred_str = (f"{pred_us/1000:.3f} ms" if pred_us >= 1000
                    else f"{pred_us:.1f} μs") if pred_us > 0 else "N/A (Triton)"
        ax.text(x_left + 0.18, r1y, "Predicted:",
                ha="left", va="center", fontsize=8, color="#666")
        ax.text(x_left + node_w - 0.16, r1y, pred_str,
                ha="right", va="center", fontsize=8.5, fontweight="bold",
                color="#333")

        # Row 2: NCU actual
        r2y = body_top - rh * 1.6
        if ncu_us is not None:
            ncu_str = (f"{ncu_us/1000:.3f} ms" if ncu_us >= 1000
                       else f"{ncu_us:.1f} μs")
            ratio = ncu_us / pred_us if pred_us > 0 else None
            ratio_str = f"  (×{ratio:.1f})" if ratio else ""
            ratio_col = ("#27ae60" if (ratio and ratio < 1.5) else
                         "#e67e22" if (ratio and ratio < 3.0) else col)
            ax.text(x_left + 0.18, r2y, "NCU actual:",
                    ha="left", va="center", fontsize=8, color="#666")
            ax.text(x_left + node_w - 0.16, r2y, f"{ncu_str}{ratio_str}",
                    ha="right", va="center", fontsize=8.5, fontweight="bold",
                    color=ratio_col)
        else:
            ax.text(x_left + 0.18, r2y, "NCU actual:",
                    ha="left", va="center", fontsize=8, color="#aaa")
            ax.text(x_left + node_w - 0.16, r2y, "no data",
                    ha="right", va="center", fontsize=8, color="#aaa")

        # Row 3: Utilisation
        r3y = body_top - rh * 2.6
        if sd["avg_sm"] is not None:
            util_str = (f"SM: {sd['avg_sm']:.1f}%    "
                        f"DRAM: {sd['avg_dram']:.1f}%    "
                        f"({sd['n_kernels']} kernels)")
        else:
            util_str = f"bottleneck: {BADGE[sd['bound']]}"
        ax.text(x_left + 0.18, r3y, util_str,
                ha="left", va="center", fontsize=7.5, color="#444")

        # Mini comparison bar at bottom of node
        bar_bot = body_bot + 0.10
        bh = 0.13
        bmax = node_w - 0.50
        ref = max(pred_us, ncu_us if ncu_us else 0, 1.0)

        if pred_us > 0:
            pw = bmax * min(pred_us / ref, 1.0)
            ax.add_patch(FancyBboxPatch(
                (x_left + 0.22, bar_bot + bh + 0.03), pw, bh,
                boxstyle="round,pad=0.008", linewidth=0,
                facecolor="#95a5a6", zorder=3))
            ax.text(x_left + 0.22 + pw + 0.06, bar_bot + bh * 1.5 + 0.03,
                    "pred", fontsize=5.5, color="#888", va="center")

        if ncu_us:
            aw = bmax * min(ncu_us / ref, 1.0)
            ax.add_patch(FancyBboxPatch(
                (x_left + 0.22, bar_bot), aw, bh,
                boxstyle="round,pad=0.008", linewidth=0,
                facecolor=col, alpha=0.82, zorder=3))
            ax.text(x_left + 0.22 + aw + 0.06, bar_bot + bh * 0.5,
                    "NCU", fontsize=5.5, color=col, va="center")

    # ── Arrows between nodes ──────────────────────────────────────────
    for i in range(n_stages - 1):
        y_start = stage_y_bot(i)         # bottom of node i
        y_end = stage_y_top(i + 1)       # top of node i+1
        ax.annotate(
            "",
            xy=(cx, y_end), xytext=(cx, y_start),
            arrowprops=dict(
                arrowstyle="-|>", color="#7f8c8d",
                lw=2.0, mutation_scale=16),
            zorder=5)

        outputs = _PIPELINE_STAGES[i]["outputs"]
        if outputs:
            mid_y = (y_start + y_end) / 2
            ax.text(cx + 0.25, mid_y, outputs[0],
                    ha="left", va="center", fontsize=6.5, color="#7f8c8d",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="#bdc3c7", lw=0.5),
                    zorder=6)

    # ── Side inputs for GEMM stages ───────────────────────────────────
    _side_inputs_map = {
        "gemm1": [("hidden_states\n[T×7168 FP8]", 0.28),
                  ("W1/W3 weights\n[E×4096×7168 FP8]", 0.72)],
        "gemm2": [("W2 weights\n[E×7168×2048 FP8]", 0.5)],
    }
    for sid, items in _side_inputs_map.items():
        idx = next((ii for ii, s in enumerate(stage_data)
                    if s["ps"]["id"] == sid), None)
        if idx is None:
            continue
        for lbl, frac in items:
            yt, yb = stage_y_top(idx), stage_y_bot(idx)
            y_att = yb + (yt - yb) * (1 - frac)
            ax.annotate(
                lbl,
                xy=(x_left, y_att),
                xytext=(x_left - 2.2, y_att),
                fontsize=6.5, color="#555",
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#aaa",
                                lw=1.0, mutation_scale=10),
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ecf0f1",
                          edgecolor="#bdc3c7", lw=0.6),
                zorder=6, clip_on=False)

    # ── Legend ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=C["COMPUTE"], label="Compute bound  (SM% > DRAM%)"),
        mpatches.Patch(color=C["MEMORY"],  label="Memory bound   (DRAM% > SM%)"),
        mpatches.Patch(color=C["LAUNCH"],  label="Launch-overhead (<1% SM & DRAM)"),
        mpatches.Patch(color="#95a5a6",    label="Predicted (cost model)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left",
              fontsize=7.5, framealpha=0.92, edgecolor="#bdc3c7")

    # ══════════════════════════════════════════════════════════════════
    # RIGHT PANEL — horizontal bar chart
    # ══════════════════════════════════════════════════════════════════
    import numpy as np

    labels = [sd["ps"]["label"].replace("\n", "\n") for sd in stage_data]
    pred_vals = np.array([sd["pred_us"] for sd in stage_data])
    ncu_vals  = np.array([sd["ncu_us"] if sd["ncu_us"] else 0.0
                          for sd in stage_data])
    colours   = [sd["colour"] for sd in stage_data]

    y_pos = np.arange(n_stages)
    bar_h = 0.35

    ax_bar.set_facecolor("#f0f2f5")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#cccccc")
    ax_bar.spines["bottom"].set_color("#cccccc")

    # Predicted bars (grey)
    has_pred = pred_vals.sum() > 0
    if has_pred:
        bars_pred = ax_bar.barh(y_pos + bar_h / 2, pred_vals,
                                height=bar_h, color="#95a5a6",
                                label="Predicted (cost model)")

    # NCU bars (coloured per stage)
    for i, (y, v, col) in enumerate(zip(y_pos, ncu_vals, colours)):
        if v > 0:
            ax_bar.barh(y - bar_h / 2, v, height=bar_h,
                        color=col, alpha=0.85,
                        label="NCU actual" if i == 0 else "")

    # Value labels on bars
    x_max = max(pred_vals.max() if has_pred else 0,
                ncu_vals.max(), 1.0)
    for i, (y, pv, av, col) in enumerate(
            zip(y_pos, pred_vals, ncu_vals, colours)):
        if has_pred and pv > 0:
            pstr = (f"{pv/1000:.2f}ms" if pv >= 1000 else f"{pv:.0f}μs")
            ax_bar.text(pv + x_max * 0.01, y + bar_h / 2, pstr,
                        va="center", fontsize=7.5, color="#666")
        if av > 0:
            astr = (f"{av/1000:.2f}ms" if av >= 1000 else f"{av:.0f}μs")
            ax_bar.text(av + x_max * 0.01, y - bar_h / 2, astr,
                        va="center", fontsize=7.5, color=col,
                        fontweight="bold")

    # y-axis labels
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=8.5)
    ax_bar.invert_yaxis()

    # x-axis
    ax_bar.set_xlabel("Latency (μs)", fontsize=9)
    ax_bar.tick_params(axis="x", labelsize=8)

    # Totals annotation
    total_pred = pred_vals.sum()
    total_ncu  = ncu_vals.sum()
    totals_str = (
        f"Total  pred: "
        f"{'N/A' if not has_pred else (f'{total_pred/1000:.3f}ms' if total_pred>=1000 else f'{total_pred:.0f}μs')}"
        f"   NCU: {total_ncu/1000:.3f}ms"
    )
    ax_bar.set_title(totals_str, fontsize=9, color="#1a252f", pad=6)
    ax_bar.set_xlabel("Latency (μs)", fontsize=9)

    # Bottleneck insight box
    dominant = max(stage_data, key=lambda s: s["ncu_us"] or s["pred_us"])
    dom_time = dominant["ncu_us"] if dominant["ncu_us"] else dominant["pred_us"]
    insight = (
        f"Dominant stage: {dominant['ps']['label'].replace(chr(10),' ')}  "
        f"({dom_time:.0f}μs)\n"
        f"Bottleneck: {BADGE[dominant['bound']]}"
    )
    ax_bar.text(0.98, 0.02, insight,
                transform=ax_bar.transAxes,
                ha="right", va="bottom", fontsize=8,
                color="#1a252f",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bdc3c7", lw=1))

    legend_handles2 = [
        mpatches.Patch(color="#95a5a6", label="Predicted (cost model)"),
        mpatches.Patch(color=C["COMPUTE"], label="NCU: Compute bound"),
        mpatches.Patch(color=C["MEMORY"],  label="NCU: Memory bound"),
        mpatches.Patch(color=C["LAUNCH"],  label="NCU: Launch overhead"),
    ]
    ax_bar.legend(handles=legend_handles2, fontsize=7.5,
                  framealpha=0.9, loc="lower right")

    ax_bar.set_title(
        f"Stage Latency Breakdown\n{totals_str}",
        fontsize=9, color="#1a252f", pad=8)

    fig.suptitle(title, fontsize=12, fontweight="bold",
                 color="#1a252f", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[DAG] Graph saved to: {output_path}")


# =====================================================================
# Workload Data
# =====================================================================
WORKLOADS = [
    ("b8f4f012",     7,   192,  11.705,  1.857),
    ("e05c6c03",     1,    32,  10.939,  1.609),
    ("6230e838",    32,    32,  13.704,  2.435),
    ("8f1ff9f1",    80,    96,  15.699,  2.794),
    ("1a4c6ba1",   901,    96,  20.875,  3.051),
    ("a7c2bcfd",    16,   224,  12.465,  2.013),
    ("2e69caee",    15,    32,  11.492,  1.866),
    ("8cba5890",    14,     0,  12.152,  2.179),
    ("5e8dc11c", 14107,    32,  44.830, 14.368),
    ("58a34f27", 11948,   128,  35.542, 10.525),
    ("5eadab1e",    62,    96,  13.883,  2.112),
    ("eedc63b2",    59,   160,  13.521,  2.571),
    ("e626d3e6",    58,    64,  15.095,  2.662),
    ("74d7ff04",    57,    96,  14.972,  2.659),
    ("4822167c",    56,    64,  14.975,  2.531),
    ("81955b1e",    55,   128,  14.714,  2.649),
    ("76010cb4",    54,   128,  14.379,  2.625),
    ("fc378037",    53,    32,  14.693,  2.672),
    ("f7d6ac7c",    52,   160,  13.105,  2.283),
]


# =====================================================================
# Main
# =====================================================================
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description="AST-based DAG cost model for MoE kernels (B200)")
    parser.add_argument("kernel_file", nargs="?", default=None,
                        help="Path to kernel .py file to analyze")
    parser.add_argument("--T", type=int, default=52,
                        help="Sequence length (number of tokens)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare kernel_ref.py vs kernel.py")
    parser.add_argument("--all-workloads", action="store_true",
                        help="Run all 19 workloads for both kernels")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--hw-json", type=str, default=None)
    parser.add_argument(
        "--ncu", type=str, default=None, metavar="NCU_REPORT",
        help="Path to an .ncu-rep file; overlays actual profiled latency on the DAG graph.")
    parser.add_argument(
        "--dag-graph", type=str, default=None, metavar="OUTPUT_PNG",
        help="Generate a compute/memory latency DAG graph and save to OUTPUT_PNG "
             "(requires matplotlib). Combine with --ncu to show measured vs predicted.")
    args = parser.parse_args()

    hw = load_hw(args.hw_json)
    print(f"B200 HW: {hw['hbm_bw_GBps']:.0f} GB/s | "
          f"FP8={hw['fp8_tflops']:.0f}T | BF16={hw['bf16_tflops']:.0f}T | "
          f"FP32={hw['fp32_tflops']:.1f}T | launch={hw['launch_us']:.1f}us")
    print()

    ref_path = None
    new_path = None
    for base in [".", "solution/triton", "../solution/triton"]:
        rp = os.path.join(base, "kernel_ref.py")
        np = os.path.join(base, "kernel.py")
        if os.path.exists(rp): ref_path = rp
        if os.path.exists(np): new_path = np

    if args.dag_graph:
        # Determine which kernel file to analyse
        kernel_path = args.kernel_file
        if kernel_path is None:
            for base in [".", "solution/triton", "../solution/triton"]:
                p = os.path.join(base, "kernel.py")
                if os.path.exists(p):
                    kernel_path = p
                    break
        if kernel_path is None:
            print("ERROR: specify a kernel .py file or ensure solution/triton/kernel.py exists.")
            return

        nodes, name = analyze_kernel_file(kernel_path, args.T)
        print(f"Kernel: {name}  |  T={args.T}  |  {len(nodes)} DAG nodes")
        predict_dag(nodes, hw, verbose=args.verbose)

        # Load NCU data if provided
        ncu_kernels = None
        if args.ncu:
            print(f"Loading NCU report: {args.ncu}")
            ncu_kernels = parse_ncu_report(args.ncu)
            if ncu_kernels:
                print(f"  → {len(ncu_kernels)} kernels parsed from NCU report")
                # Print per-stage summary
                agg = aggregate_ncu_by_stage(ncu_kernels)
                print(f"\n  {'Stage':<22} {'Total(μs)':>10} {'#K':>4} "
                      f"{'SM%':>7} {'DRAM%':>7}")
                print("  " + "-" * 56)
                total_ncu = 0.0
                for s, d in sorted(agg.items(), key=lambda x: -x[1]["total_us"]):
                    print(f"  {s:<22} {d['total_us']:>10.1f} {d['n_kernels']:>4} "
                          f"{d['avg_sm_pct']:>7.2f} {d['avg_dram_pct']:>7.2f}")
                    total_ncu += d["total_us"]
                print(f"  {'TOTAL':<22} {total_ncu:>10.1f}")

        plot_dag_graph(
            dag_nodes=nodes,
            hw=hw,
            ncu_kernels=ncu_kernels,
            output_path=args.dag_graph,
            title=f"MoE Kernel DAG — {os.path.basename(kernel_path)}  T={args.T}",
            T=args.T,
        )
        return

    if args.kernel_file:
        nodes, name = analyze_kernel_file(args.kernel_file, args.T)
        print(f"Kernel: {name}  |  T={args.T}  |  {len(nodes)} DAG nodes")
        pred = predict_dag(nodes, hw, verbose=True)
        print(f"\nPredicted: {pred:.3f} ms")
        return

    if args.compare:
        for path, label in [(ref_path, "kernel_ref.py"), (new_path, "kernel.py")]:
            if not path:
                print(f"  {label}: NOT FOUND")
                continue
            nodes, _ = analyze_kernel_file(path, args.T)
            print(f"\n{'='*80}")
            print(f"{label}  |  T={args.T}  |  {len(nodes)} DAG nodes")
            print(f"{'='*80}")
            pred = predict_dag(nodes, hw, verbose=args.verbose)
            print(f"Predicted: {pred:.3f} ms")
        return

    if args.all_workloads:
        if not ref_path or not new_path:
            print("ERROR: need both kernel_ref.py and kernel.py on path")
            return

        print(f"{'Workload':<10} {'T':>6} | {'Ref Pred':>9} {'Ref Act':>9} "
              f"{'R':>5} | {'New Pred':>9} {'New Act':>9} {'R':>5} | "
              f"{'PSpd':>5} {'ASpd':>5}")
        print("-" * 95)

        ref_ratios, new_ratios = [], []
        for uid, T, off, ref_ms, new_ms in WORKLOADS:
            ref_nodes, _ = analyze_kernel_file(ref_path, T)
            ref_pred = predict_dag(ref_nodes, hw)
            new_nodes, _ = analyze_kernel_file(new_path, T)
            new_pred = predict_dag(new_nodes, hw)

            rr = ref_pred / ref_ms if ref_ms > 0 else 0
            nr = new_pred / new_ms if new_ms > 0 else 0
            ref_ratios.append(rr)
            new_ratios.append(nr)
            pspd = ref_pred / new_pred if new_pred > 0 else 0
            aspd = ref_ms / new_ms if new_ms > 0 else 0

            print(f"{uid:<10} {T:>6} | {ref_pred:>8.2f}ms {ref_ms:>8.2f}ms "
                  f"{rr:>4.2f}x | {new_pred:>8.2f}ms {new_ms:>8.2f}ms "
                  f"{nr:>4.2f}x | {pspd:>4.1f}x {aspd:>4.1f}x")

        print(f"\nRef  pred/actual: avg={sum(ref_ratios)/len(ref_ratios):.2f}x  "
              f"range=[{min(ref_ratios):.2f}, {max(ref_ratios):.2f}]")
        print(f"New  pred/actual: avg={sum(new_ratios)/len(new_ratios):.2f}x  "
              f"range=[{min(new_ratios):.2f}, {max(new_ratios):.2f}]")
        return

    if ref_path:
        nodes, name = analyze_kernel_file(ref_path, args.T)
        print(f"Default: analyzing {name} at T={args.T}")
        pred = predict_dag(nodes, hw, verbose=True)
        print(f"\nPredicted: {pred:.3f} ms")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
