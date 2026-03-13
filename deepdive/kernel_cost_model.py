"""
AST-Based DAG Cost Model for MoE Kernels -- B200 GPU
=====================================================

Parses kernel Python source code via AST to automatically build a compute DAG,
then predicts latency using measured B200 hardware primitives.

Usage:
    python deepdive/new_kernel_cost_model.py solution/triton/kernel_ref.py --T 52
    python deepdive/new_kernel_cost_model.py solution/triton/kernel.py     --T 52
    python deepdive/new_kernel_cost_model.py --compare --T 52
    python deepdive/new_kernel_cost_model.py --all-workloads

How it works:
  1) Parse the .py file with Python's ast module
  2) Walk the AST to find: function calls, loops, tensor ops, shapes
  3) Build DagNode list from detected operations
  4) Predict latency using measured GEMM lookups + HBM bandwidth interpolation
"""

import argparse
import ast
import json
import math
import os
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
