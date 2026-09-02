# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + QKV GEMV + RoPE Q+K -- 6-launch multi-launch ELF for decode.

Merges the decode attention front-half into a single ELF:
  L1: RMSNorm    [1,1]  x * norm_w -> normed        (M=1, N=2048)
  L2: Q GEMV     [8,1]  wq @ normed -> q             (M=2048, K=2048)
  L3: K GEMV     [8,1]  wk @ normed -> k             (M=512, K=2048)
  L4: V GEMV     [8,1]  wv @ normed -> v             (M=512, K=2048)
  L5: RoPE Q     [1,1]  q * lut_q -> q_roped         (N=32, dim=64)
  L6: RoPE K     [1,1]  k * lut_k -> k_roped         (N=8, dim=64)

All shared buffers are 1D (matching GEMV/RoPE expectations). RMSNorm
operates at M=1 for decode, so its 2D (1, emb_dim) I/O is equivalent to
1D (emb_dim). A custom wrapper builds the RMSNorm launch with 1D func
args and expand_shape/collapse_shape conversions inside the launch body.

13 func args (6 launches):
    %arg0:  x_in        memref<2048xbf16>         RMSNorm input (1D decode)
    %arg1:  norm_w      memref<2048xbf16>         RMSNorm weight
    %arg2:  normed      memref<2048xbf16>         RMSNorm output / GEMV input
    %arg3:  wq          memref<2048x2048xbf16>    Q weight (transposed)
    %arg4:  q           memref<2048xbf16>         Q output / RoPE Q input
    %arg5:  wk          memref<512x2048xbf16>     K weight (transposed)
    %arg6:  k           memref<512xbf16>          K output / RoPE K input
    %arg7:  wv          memref<512x2048xbf16>     V weight (transposed)
    %arg8:  v           memref<512xbf16>          V output (final)
    %arg9:  lut_q       memref<2048xbf16>         RoPE Q LUT (32*64)
    %arg10: lut_k       memref<512xbf16>          RoPE K LUT (8*64)
    %arg11: q_roped     memref<2048xbf16>         RoPE Q output (final)
    %arg12: k_roped     memref<512xbf16>          RoPE K output (final)

Usage:
    python3 rms_gemv_rope_multi.py -p           # print combined MLIR
    python3 rms_gemv_rope_multi.py              # compile + run + validate
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "matrix_vector_multiplication",
        "bf16",
    ),
)

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

from air import api as air
from air.api import ops
from air.api.types import f32, i32
from shared.builders.rms_gemms_rope_multi import _api_dtype

from shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
)

range_ = for_

EPS = 1e-5


# ---------------------------------------------------------------------------
# 1D RMSNorm wrapper (accepts 1D args, converts to 2D inside launch)
# ---------------------------------------------------------------------------


def _build_rms_1d(n, np_dtype, vector_size=16, target="npu2"):
    """Build RMSNorm for M=1 with 1D func args (decode-friendly).

    The standard weighted_rms_norm builds with 2D (M, N) I/O memrefs. For
    decode (M=1) the GEMV expects 1D (N,) input, so the func args are 1D.

    Func signature: (x_1d: [N], weight: [N], out_1d: [N])

    Three lines of DSL for what the predecessor spelled as ~90: an explicit
    1D->2D expand_shape at launch scope, a vector-width bf16 accumulator filled
    by a hand-rolled loop of subview + two transfer_reads + mulf + addf, a
    store/load inserted between the mulf and the addf to break the chain, a
    horizontal vector.reduction, and an extf/rsqrt/truncf triple.

    None of that structure survives here -- ops.reduce_add is one op and there
    is nowhere to put the chain break -- and it does not need to: the emitted
    air.insts.bin is byte-identical to the predecessor's at every N the callers
    use. The accumulate stays bf16 and the division stays a division, which is
    where this differs from weighted_rms_norm (f32 accumulate, multiply by the
    reciprocal); those are numerics, not spelling, so they are kept.
    """
    assert n % vector_size == 0, (n, vector_size)

    dtype = _api_dtype(np_dtype)
    X = air.tensor([n], dtype)
    W = air.tensor([n], dtype)
    Y = air.tensor([n], dtype)

    with air.launch(name="rms_norm_1d") as launch:

        @launch.body
        def _():
            with air.segment(name="rms_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(1), range(1)], name="rms_herd", shape=(1, 1)
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            row = air.alloc(
                                [n], dtype, scope=h.private(), vector=vector_size
                            )
                            wt = air.alloc(
                                [n], dtype, scope=h.private(), vector=vector_size
                            )
                            out = air.alloc(
                                [n], dtype, scope=h.private(), vector=vector_size
                            )
                            acc = air.alloc(
                                [1], dtype, scope=h.private(), vector=vector_size
                            )
                            rstd = air.alloc(
                                [1], dtype, scope=h.private(), vector=vector_size
                            )

                            ops.load(wt, W[:])
                            ops.load(row, X[:])
                            acc[:] = ops.reduce_add(row[:] * row[:])
                            rstd[:] = ops.cast(
                                ops.rsqrt(ops.cast(acc[:] / n + EPS, f32)), dtype
                            )
                            out[:] = row[:] * rstd[:] * wt[:]
                            ops.store(out, Y[:])

    return launch.build(target=target)


# ---------------------------------------------------------------------------
# 1D RoPE launch builder (accepts 1D args, herd processes rows of head_dim)
# ---------------------------------------------------------------------------


def _build_rope_1d(n_rows, embed_dim, np_dtype, herd_x=1, target="npu2"):
    """Build a RoPE launch with 1D func args (for decode GEMV compatibility).

    Func signature:
      (in_1d: [total], lut_1d: [total], out_1d: [total])

    The herd processes n_rows rows of embed_dim elements each.

    Args:
        n_rows:    Number of RoPE rows (n_heads for Q, n_kv_heads for K)
        embed_dim: RoPE column width per row (head_dim=64)
        herd_x:    Number of tiles for row-parallel
        target:    NPU generation to build for; see _build_rope_2d.

    The 2-D sibling in rms_gemms_rope_multi.py carries the reasoning: the
    hand-built AffineMap becomes Python arithmetic on the tile coordinates,
    and the emitted air.insts.bin is byte-identical to the predecessor's.
    """
    assert embed_dim % 16 == 0
    total = n_rows * embed_dim
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert n_rows % total_tiles == 0
    rows_per_tile = n_rows // total_tiles

    dtype = _api_dtype(np_dtype)
    IN = air.tensor([total], dtype)
    LUT = air.tensor([total], dtype)
    OUT = air.tensor([total], dtype)

    rope = air.extern("rope", link_with="rope.o", scalars=[i32])

    with air.launch(name="rope_1d") as launch:

        @launch.body
        def _():
            with air.segment(name="rope_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)],
                        name="rope_herd",
                        shape=(herd_x, herd_y),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_in = air.alloc([embed_dim], dtype, scope=h.private())
                            l1_lut = air.alloc([embed_dim], dtype, scope=h.private())
                            l1_out = air.alloc([embed_dim], dtype, scope=h.private())

                            for local_row in air.sequential(0, rows_per_tile):
                                row = (
                                    local_row + (tx * herd_y + ty) * rows_per_tile
                                ) * embed_dim

                                ops.load(l1_in, IN[row : row + embed_dim])
                                ops.load(l1_lut, LUT[row : row + embed_dim])
                                rope(l1_in, l1_lut, l1_out, embed_dim)
                                ops.store(l1_out, OUT[row : row + embed_dim])

    return launch.build(target=target)


# External kernel function names shared across all sub-kernels
_EXTERN_FUNCS = {
    "@zero_vectorized_bf16",  # RMSNorm (if used)
    "@matvec_vectorized_bf16_bf16",  # GEMV
    "@linalg_fill_bf16",  # GEMV
    "@rope",  # RoPE
}


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_rms_gemv_rope_module(
    emb_dim=2048,
    kv_dim=512,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    # GEMV tile config
    tile_m=8,
    m_input=4,
    herd_m=8,
    # RoPE config
    rope_herd_x=1,
    print_kernels=False,
):
    """Build 6-launch module: RMSNorm + Q/K/V GEMVs + RoPE Q + RoPE K.

    All shared buffers are 1D memrefs (decode: M=1 tokens).

    Returns:
        Module with func @rms_gemv_rope and 13 memref args:
            %arg0:  x_in        (emb_dim,)             RMSNorm input
            %arg1:  norm_w      (emb_dim,)             RMSNorm weight
            %arg2:  normed      (emb_dim,)             RMSNorm output
            %arg3:  wq          (emb_dim, emb_dim)     Q weight
            %arg4:  q           (emb_dim,)             Q GEMV output
            %arg5:  wk          (kv_dim, emb_dim)      K weight
            %arg6:  k           (kv_dim,)              K GEMV output
            %arg7:  wv          (kv_dim, emb_dim)      V weight
            %arg8:  v           (kv_dim,)              V GEMV output
            %arg9:  lut_q       (emb_dim,)             RoPE Q LUT
            %arg10: lut_k       (kv_dim,)              RoPE K LUT
            %arg11: q_roped     (emb_dim,)             RoPE Q output
            %arg12: k_roped     (kv_dim,)              RoPE K output
    """
    from matvec import build_module as build_gemv

    q_total = n_heads * head_dim  # = emb_dim = 2048
    k_total = n_kv_heads * head_dim  # = kv_dim = 512

    assert q_total == emb_dim
    assert k_total == kv_dim

    # ---- Build sub-kernels ----

    # 1. RMSNorm at M=1 with 1D I/O (custom wrapper)
    print("  [1/6] RMSNorm (decode, 1D wrapper)...")
    rms_ir = str(_build_rms_1d(emb_dim, bfloat16, 16))

    # 2-4. Q/K/V GEMVs (already produce air.launch with 1D I/O)
    print("  [2/6] Q GEMV...")
    q_ir = str(
        build_gemv(emb_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16)
    )

    print("  [3/6] K GEMV...")
    k_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    print("  [4/6] V GEMV...")
    v_ir = str(build_gemv(kv_dim, emb_dim, tile_m, m_input, herd_m, bfloat16, bfloat16))

    # 5-6. RoPE Q/K (1D in/out, launch+segment wrapper)
    # Decode: Q has n_heads=32 rows of head_dim=64, K has n_kv_heads=8 rows
    print(
        f"  [5/6] RoPE Q (n_rows={n_heads}, embed_dim={head_dim}, "
        f"herd_x={rope_herd_x})..."
    )
    rope_q_ir = str(_build_rope_1d(n_heads, head_dim, bfloat16, rope_herd_x))

    print(
        f"  [6/6] RoPE K (n_rows={n_kv_heads}, embed_dim={head_dim}, "
        f"herd_x={rope_herd_x})..."
    )
    rope_k_ir = str(_build_rope_1d(n_kv_heads, head_dim, bfloat16, rope_herd_x))

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Q GEMV", q_ir),
            ("K GEMV", k_ir),
            ("V GEMV", v_ir),
            ("RoPE Q", rope_q_ir),
            ("RoPE K", rope_k_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch ----
    # Arg mapping (combined func arg indices):
    #   RMSNorm:  {0->0, 1->1, 2->2}       (x_in, norm_w, normed)
    #   Q GEMV:   {0->3, 1->2, 2->4}       (wq, normed, q)
    #   K GEMV:   {0->5, 1->2, 2->6}       (wk, normed, k)
    #   V GEMV:   {0->7, 1->2, 2->8}       (wv, normed, v)
    #   RoPE Q:   {0->4, 1->9, 2->11}      (q, lut_q, q_roped)
    #   RoPE K:   {0->6, 1->10, 2->12}     (k, lut_k, k_roped)

    # Privates only from q_ir (GEMV externs) and rope_q_ir (@rope); the other
    # slices carry no private decls of their own.
    base_args = [
        FuncArg("%arg0", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{kv_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{kv_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{kv_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{q_total}xbf16>"),
        FuncArg("%arg10", f"memref<{k_total}xbf16>"),
        FuncArg("%arg11", f"memref<{q_total}xbf16>"),
        FuncArg("%arg12", f"memref<{k_total}xbf16>"),
    ]
    slices = [
        KernelSlice(rms_ir, "r", {0: 0, 1: 1, 2: 2}, private_from=False),
        KernelSlice(q_ir, "q", {0: 3, 1: 2, 2: 4}),
        KernelSlice(k_ir, "k", {0: 5, 1: 2, 2: 6}, private_from=False),
        KernelSlice(v_ir, "v", {0: 7, 1: 2, 2: 8}, private_from=False),
        KernelSlice(rope_q_ir, "rq", {0: 4, 1: 9, 2: 11}),
        KernelSlice(rope_k_ir, "rk", {0: 6, 1: 10, 2: 12}, private_from=False),
    ]
    module = stitch_elf(
        "rms_gemv_rope",
        base_args,
        slices,
        extra_externs=_EXTERN_FUNCS,
    )
    print(
        f"  Module: {len(str(module).splitlines())} lines, "
        f"13 args, 6 launches, parsed OK"
    )
    return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def _rms_norm_ref(x_1d, weight, eps=1e-5):
    """CPU RMSNorm reference for 1D input (M=1 decode)."""
    x_f32 = x_1d.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2) + eps)
    return (x_f32 / rms * weight.astype(np.float32)).astype(bfloat16)


def _rope_ref(x_flat, lut_flat, head_dim):
    """CPU RoPE reference for flat 1D arrays."""
    x = x_flat.astype(np.float32).reshape(-1, head_dim)
    lut = lut_flat.astype(np.float32).reshape(-1, head_dim)
    out = np.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * lut[:, 0::2] - x[:, 1::2] * lut[:, 1::2]
    out[:, 1::2] = x[:, 0::2] * lut[:, 1::2] + x[:, 1::2] * lut[:, 0::2]
    return out.astype(bfloat16).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EMB_DIM = 2048
    KV_DIM = 512
    N_HEADS = 32
    N_KV_HEADS = 8
    HEAD_DIM = 64

    parser = argparse.ArgumentParser(
        description="RMSNorm + QKV GEMV + RoPE QK multi-launch decode test"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print combined MLIR and exit",
    )
    parser.add_argument("--print-kernels", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
    )
    args = parser.parse_args()

    print(
        f"RMS+GEMV+RoPE Multi-Launch (decode): emb={EMB_DIM}, "
        f"kv={KV_DIM}, heads={N_HEADS}/{N_KV_HEADS}, dk={HEAD_DIM}"
    )

    module = build_rms_gemv_rope_module(
        emb_dim=EMB_DIM,
        kv_dim=KV_DIM,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        print_kernels=args.print_kernels,
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rms_gemv_rope",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run: build test data, run, verify ----
    np.random.seed(42)

    # Inputs
    x_in = np.random.uniform(-1.0, 1.0, (EMB_DIM,)).astype(bfloat16)
    norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)
    wq = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    wk = np.random.uniform(-0.1, 0.1, (KV_DIM, EMB_DIM)).astype(bfloat16)
    wv = np.random.uniform(-0.1, 0.1, (KV_DIM, EMB_DIM)).astype(bfloat16)

    # RoPE LUTs (decode: single position, one row per head)
    from rope_lut.rope_lut import generate_lut

    # For decode, LUT is just one position: (1, head_dim) repeated per head
    # But the LUT shape must match the total: n_heads * head_dim = emb_dim
    # Use position 0 for simplicity in test
    base_lut_row = generate_lut(1, HEAD_DIM, bfloat16)  # (1, 64)
    lut_q = np.tile(base_lut_row, (N_HEADS, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(base_lut_row, (N_KV_HEADS, 1)).flatten().astype(bfloat16)

    # CPU reference
    print("Computing CPU reference...")
    normed_ref = _rms_norm_ref(x_in, norm_w)
    q_ref = np.dot(wq.astype(np.float32), normed_ref.astype(np.float32)).astype(
        bfloat16
    )
    k_ref = np.dot(wk.astype(np.float32), normed_ref.astype(np.float32)).astype(
        bfloat16
    )
    v_ref = np.dot(wv.astype(np.float32), normed_ref.astype(np.float32)).astype(
        bfloat16
    )

    # Apply RoPE
    q_roped_ref = _rope_ref(q_ref, lut_q, HEAD_DIM)
    k_roped_ref = _rope_ref(k_ref, lut_k, HEAD_DIM)

    # Output buffers (zeroed)
    normed_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    q_buf = np.zeros(EMB_DIM, dtype=bfloat16)
    k_buf = np.zeros(KV_DIM, dtype=bfloat16)
    v_buf = np.zeros(KV_DIM, dtype=bfloat16)

    # Func signature: 13 args
    # (x_in, norm_w, normed, wq, q, wk, k, wv, v, lut_q, lut_k, q_roped, k_roped)
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="rms_gemv_rope",
    )

    # XRTRunner: inputs = first N args, expected_outputs = last M args.
    # Last 2 (arg11=q_roped, arg12=k_roped) are outputs.
    # First 11 (arg0-arg10) are inputs (including zeroed intermediates).
    exit(
        runner.run_test(
            module,
            inputs=[
                x_in,  # arg0
                norm_w,  # arg1
                normed_buf,  # arg2 (intermediate, zeroed)
                wq,  # arg3
                q_buf,  # arg4 (intermediate, zeroed)
                wk,  # arg5
                k_buf,  # arg6 (intermediate, zeroed)
                wv,  # arg7
                v_buf,  # arg8 (V output, also an intermediate)
                lut_q,  # arg9
                lut_k,  # arg10
            ],
            expected_outputs=[
                q_roped_ref,  # arg11
                k_roped_ref,  # arg12
            ],
            rtol=0.2,
            atol=0.5,
            min_correlation=0.99,
        )
    )
