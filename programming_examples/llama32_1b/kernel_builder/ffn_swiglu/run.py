#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FFN SwiGLU Multi-Launch — Self-contained test.

Builds a single AIR function with 4 sequential air.launch operations
(Gate GEMM → Up GEMM → SwiGLU → Down GEMM), compiles to ELF,
runs on NPU, and validates against CPU F32 reference.

Usage:
    make run                    # compile + run + validate
    make print                  # print combined MLIR
    python3 run.py -p           # same as make print
    python3 run.py              # compile + run + validate
    python3 run.py --profile    # compile + run + profile
"""

import argparse
import os
import re
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from air.ir import *
from air.dialects.air import *
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

# ---------------------------------------------------------------------------
# MLIR text stitching utilities
# ---------------------------------------------------------------------------


def _extract_between_func_and_return(mlir_text):
    """Extract func body (between func signature and return)."""
    lines = mlir_text.split("\n")
    body_start = body_end = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            body_start = i + 1
    for i in range(len(lines) - 1, body_start, -1):
        if lines[i].strip() == "return":
            body_end = i
            break
    return "\n".join(lines[body_start:body_end])


def _extract_affine_maps(mlir_text):
    return [l for l in mlir_text.split("\n") if l.startswith("#map")]


def _extract_private_funcs(mlir_text):
    return [l for l in mlir_text.split("\n") if "func.func private" in l]


def _rename_all(text, prefix):
    """Rename all SSA values, affine maps, and symbols with a unique prefix."""
    # Affine maps (longest first)
    for name in sorted(set(re.findall(r"#map\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)

    # SSA word values (%argN, %cN, %allocN, etc.)
    for name in sorted(set(re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"%{prefix}_{name[1:]}", text)

    # SSA numbered values (%0, %1, ...)
    for name in sorted(
        set(re.findall(r"%\d+", text)), key=lambda x: int(x[1:]), reverse=True
    ):
        text = text.replace(name, f"%{prefix}_n{name[1:]}")

    # Symbol names (@seg, @herd) but NOT external kernel functions
    extern_funcs = {"@silu_and_mul_bf16", "@zero_vectorized_bf16", "@matmul_bf16"}
    for name in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if name not in extern_funcs:
            text = text.replace(name, f"@{prefix}_{name[1:]}")

    return text


def _fix_launch_func_args(text, prefix, arg_map):
    """Fix func-arg references in launch's args() clause after _rename_all."""
    for orig_idx, combined_idx in arg_map.items():
        old_ref = f"%{prefix}_arg{orig_idx}"
        new_ref = f"%arg{combined_idx}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    return text


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_ffn_module(
    seq_len=2048,
    emb_dim=2048,
    hidden_dim=8192,
    gate_tile_m=64,
    gate_tile_k_l2=64,
    gate_tile_k_l1=32,
    gate_tile_n=128,
    gate_herd_m=8,
    gate_herd_n=4,
    down_tile_m=64,
    down_tile_k_l2=256,
    down_tile_k_l1=32,
    down_tile_n=64,
    down_herd_m=8,
    down_herd_n=4,
    swiglu_tile_n=4096,
    swiglu_herd_x=8,
    swiglu_herd_y=1,
    print_kernels=False,
):
    """Build multi-launch FFN: 4 air.launch ops in one func.

    Args:
        print_kernels: If True, print each sub-kernel's MLIR before stitching.
    """
    from llama32_1b.kernel_builder.gemm_builder import _build_gemm_module

    # Import silu_and_mul from same directory as this file
    import importlib.util

    _silu_path = os.path.join(os.path.dirname(__file__), "silu_and_mul.py")
    _spec = importlib.util.spec_from_file_location("silu_and_mul", _silu_path)
    _silu_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_silu_mod)
    build_swiglu = _silu_mod.build_module_2d

    # Build each kernel independently
    print("  [1/4] Gate GEMM...")
    gate_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    print("  [2/4] Up GEMM...")
    up_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            hidden_dim,
            gate_tile_m,
            gate_tile_k_l2,
            gate_tile_k_l1,
            gate_tile_n,
            gate_herd_m,
            gate_herd_n,
        )
    )

    print("  [3/4] SwiGLU...")
    swiglu_ir = str(
        build_swiglu(
            seq_len,
            hidden_dim,
            swiglu_tile_n,
            bfloat16,
            herd_x=swiglu_herd_x,
            herd_y=swiglu_herd_y,
        )
    )

    print("  [4/4] Down GEMM...")
    down_ir = str(
        _build_gemm_module(
            seq_len,
            hidden_dim,
            emb_dim,
            down_tile_m,
            down_tile_k_l2,
            down_tile_k_l1,
            down_tile_n,
            down_herd_m,
            down_herd_n,
        )
    )

    if print_kernels:
        for name, ir in [
            ("Gate GEMM", gate_ir),
            ("Up GEMM", up_ir),
            ("SwiGLU", swiglu_ir),
            ("Down GEMM", down_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # Extract, rename, remap
    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (gate_ir, "g", {0: 0, 1: 1, 2: 2}),
        (up_ir, "u", {0: 0, 1: 3, 2: 4}),
        (swiglu_ir, "s", {0: 2, 1: 4, 2: 5}),
        (down_ir, "d", {0: 5, 1: 6, 2: 7}),
    ]:
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        body = _rename_all(body, prefix)
        maps = [_rename_all(m, prefix) for m in maps]
        body = _fix_launch_func_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    privates = _extract_private_funcs(swiglu_ir)

    # Assemble
    n_hidden = seq_len * hidden_dim
    combined = "\n".join(maps_all) + f"""
module {{
  {"  ".join(p.strip() + chr(10) for p in privates)}  func.func @ffn_block(
    %arg0: memref<{seq_len}x{emb_dim}xbf16>,
    %arg1: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg2: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg3: memref<{emb_dim}x{hidden_dim}xbf16>,
    %arg4: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg5: memref<{seq_len}x{hidden_dim}xbf16>,
    %arg6: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg7: memref<{seq_len}x{emb_dim}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
{bodies[2]}
{bodies[3]}
    return
  }}
}}
"""

    from air.ir import Module, Context

    with Context() as ctx:
        module = Module.parse(combined, ctx)
        print(f"  Module: {len(combined.splitlines())} lines, parsed OK")
        return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def ffn_reference(x, w_gate, w_up, w_down):
    """CPU F32 FFN reference: output = SwiGLU(x @ W_gate, x @ W_up) @ W_down."""
    x_f32 = x.astype(np.float32)
    gate = x_f32 @ w_gate.astype(np.float32)
    up = x_f32 @ w_up.astype(np.float32)
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    swiglu = (gate * sigmoid) * up
    return (swiglu @ w_down.astype(np.float32)).astype(bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFN SwiGLU multi-launch test")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--print-kernels",
        action="store_true",
        help="Print each sub-kernel's MLIR before stitching",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile kernel execution"
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument(
        "--iterations", type=int, default=5, help="Profiling iterations"
    )
    args = parser.parse_args()

    seq_len, emb_dim, hidden_dim = args.seq_len, args.emb_dim, args.hidden_dim
    print(
        f"FFN Multi-Launch: seq_len={seq_len}, emb_dim={emb_dim}, hidden_dim={hidden_dim}"
    )

    module = build_ffn_module(
        seq_len, emb_dim, hidden_dim, print_kernels=args.print_kernels
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    # Test data
    np.random.seed(42)
    x = (np.random.randn(seq_len, emb_dim) * 1.0).astype(bfloat16)
    w_gate = (np.random.randn(emb_dim, hidden_dim) * 0.1).astype(bfloat16)
    w_up = (np.random.randn(emb_dim, hidden_dim) * 0.1).astype(bfloat16)
    w_down = (np.random.randn(hidden_dim, emb_dim) * 0.01).astype(bfloat16)
    gate_buf = np.zeros((seq_len, hidden_dim), dtype=bfloat16)
    up_buf = np.zeros((seq_len, hidden_dim), dtype=bfloat16)
    swiglu_buf = np.zeros((seq_len, hidden_dim), dtype=bfloat16)

    output_ref = ffn_reference(x, w_gate, w_up, w_down)

    if args.profile:
        # Profile mode: compile, load, run N iterations
        import pyxrt as xrt
        import filelock

        print("Compiling...")
        backend = XRTBackend(
            verbose=False,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="ffn_block",
        )
        artifact = backend.compile(module)

        print("Loading...")
        with filelock.FileLock("/tmp/npu.lock"):
            invoker = backend.load(artifact)

        inputs = [
            x,
            w_gate,
            gate_buf,
            w_up,
            up_buf,
            swiglu_buf,
            w_down,
            np.zeros((seq_len, emb_dim), dtype=bfloat16),
        ]
        sizes = [a.size * a.itemsize for a in inputs]
        bos = [xrt.ext.bo(backend.device, s) for s in sizes]

        # Warmup
        for i, a in enumerate(inputs):
            bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
            bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        run = xrt.run(backend.kernel)
        for i, bo in enumerate(bos):
            run.set_arg(i, bo)
        run.start()
        run.wait2()

        # Timed iterations
        times_kernel, times_total = [], []
        for it in range(args.iterations):
            t0 = time.perf_counter()
            for i, a in enumerate(inputs):
                bos[i].write(a.view(np.int16) if a.dtype == bfloat16 else a, 0)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            tk0 = time.perf_counter()
            run = xrt.run(backend.kernel)
            for i, bo in enumerate(bos):
                run.set_arg(i, bo)
            run.start()
            run.wait2()
            tk1 = time.perf_counter()
            for bo in bos:
                bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            t1 = time.perf_counter()
            times_kernel.append((tk1 - tk0) * 1000)
            times_total.append((t1 - t0) * 1000)

        # Check correctness
        output_bo = bos[-1]
        output_data = output_bo.read(sizes[-1], 0).view(np.int16).view(bfloat16)
        output_data = output_data.reshape(seq_len, emb_dim).astype(np.float32)
        ref_flat = output_ref.astype(np.float32).flatten()
        corr = np.corrcoef(output_data.flatten(), ref_flat)[0, 1]

        backend.unload()

        print(f"\n{'='*60}")
        print(f"PROFILING ({args.iterations} iterations)")
        print(f"{'='*60}")
        print(
            f"  Kernel (4 launches): avg={np.mean(times_kernel):.1f}ms  min={np.min(times_kernel):.1f}ms  max={np.max(times_kernel):.1f}ms"
        )
        print(
            f"  Total (write+run+read): avg={np.mean(times_total):.1f}ms  min={np.min(times_total):.1f}ms  max={np.max(times_total):.1f}ms"
        )
        print(f"  Host overhead: {np.mean(times_total) - np.mean(times_kernel):.1f}ms")
        print(f"  Correlation: {corr:.6f}")
        print(f"\nComparison:")
        print(f"  4 separate kernels: ~109ms kernel, ~149ms total")
        print(
            f"  Multi-launch:       {np.min(times_kernel):.1f}ms kernel, {np.min(times_total):.1f}ms total"
        )
        print(
            f"  Speedup:            {109/np.min(times_kernel):.2f}x kernel, {149/np.min(times_total):.2f}x total"
        )
        print(f"  IRON fused FFN:     57.4ms total")
        status = "PASS" if corr > 0.999 else "FAIL"
        print(f"\n  {status} (corr={corr:.6f})")

    else:
        # Correctness test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf",
            instance_name="ffn_block",
        )
        exit(
            runner.run_test(
                module,
                inputs=[x, w_gate, gate_buf, w_up, up_buf, swiglu_buf, w_down],
                expected_outputs=[output_ref.reshape(-1)],
                rtol=0.04,
                atol=4.0,
                min_correlation=0.999,
            )
        )
