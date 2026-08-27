# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# bfp16ebs8 A x bfp16ebs8 B -> bf16/f32 C GEMM on NPU2, running the mlir-aie
# reference kernel mm_bfp.cc unmodified.
#
# AIR has no bfp16ebs8 element type (mlir-aie's !aiex.bfp<> lowers to i72 only
# inside aie.device, i.e. after every AIR pass, and AIR's BD lowering assumes an
# int-or-float element type). So A and B cross the AIR boundary as plain i8 and
# the kernel reinterprets via aie::block_vector<bfp16ebs8>, the same type-pun
# bf16_x_bfp16 uses for its weights.
#
# Layout: bfp16ebs8 packs 8 scalars into 9 bytes, so a logical [H, W] matrix is
# an [H, W*9//8] byte matrix. mlir-aie's host-side shuffleMatrixForBfp16ebs8()
# reorders 8x8 sub-tiles WITHIN each tile box only, leaving the global matrix
# row-major -- so A and B here are byte-for-byte what mlir-aie's own
# bfp_test.cpp produces. B is consumed transposed, i.e. stored [N, K*9//8].
#
# C is accumulated in bfp16ebs8 by the kernel and converted to bf16 or f32 on
# the way out (bfp_cvt.cc), so the output is an ordinary float tensor that
# XRTRunner checks against a float reference exactly like the bf16 example.

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineSymbolExpr,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ShapedType,
    StridedLayoutAttr,
    StringAttr,
    UnitAttr,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    MemorySpace,
    T,
    dma_memcpy_nd,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects import arith
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.scf import ForOp, for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

from bfp16_utils import float_to_bfp16ebs8, nbytes, shuffle_bfp16ebs8

np.random.seed(42)

KERNEL_OBJ_NAME = "mm_bfp.o"

# bfp16ebs8 MMUL is 8x8x8 (mac_8x8_8x8T).
MMUL_R = MMUL_S = MMUL_T = 8


def _scaled(iv, c):
    """affine_apply of (s0 -> s0 * c)."""
    m = AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(c))],
    )
    return affine_apply(m, [iv])


@module_builder
def build_module(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
    np_dtype_out,
    arch="aie2p",
):
    r, s, t = MMUL_R, MMUL_S, MMUL_T

    # bfp16ebs8 is an AIE2P type; there is no aie2 lowering for it.
    assert arch == "aie2p", "bfp16ebs8 requires arch aie2p"

    # mm_bfp.cc static_asserts
    assert tile_m % (2 * r) == 0, "tile_m must be a multiple of 16"
    # mm_bfp.cc's static_assert allows tile_m == 16, but at that size rowA is 2
    # while the z loop still carries AIE_LOOP_MIN_ITERATION_COUNT(4), and Peano
    # fails with "ran out of registers during register allocation". 32 is the
    # smallest tile_m the reference kernel actually compiles at on AIE2P.
    assert tile_m >= 32, (
        f"tile_m ({tile_m}) must be >= 32: mm_bfp.cc exhausts the register "
        "file at tile_m=16 on AIE2P"
    )
    assert tile_n % (2 * t) == 0, "tile_n must be a multiple of 16"
    assert tile_k_l1 % s == 0, "tile_k_l1 must be a multiple of 8"
    # shuffleMatrixForBfp16ebs8 asserts (tile width is in elements)
    assert tile_k_l1 % 64 == 0, "tile_k_l1 must be a multiple of 64 (host shuffle)"
    assert tile_n % 64 == 0, "tile_n must be a multiple of 64 (host shuffle)"
    # 4-byte shim BD granularity on the L3 row strides of the packed operands
    assert k % 32 == 0, "K must be a multiple of 32 (K*9/8 must be 4-byte aligned)"

    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0

    k_per_l2 = tile_k_l2 // tile_k_l1
    xrt_dtype_out = type_mapper(np_dtype_out)

    KB = nbytes(k)  # L3 A/B row stride, bytes
    a_row_b = nbytes(tile_k_l1)  # one L3 row of an A/B tile box
    a_l1_b = nbytes(tile_m * tile_k_l1)
    b_l1_b = nbytes(tile_n * tile_k_l1)
    c_l1_b = nbytes(tile_m * tile_n)  # bfp16 accumulator, bytes
    c_elems = tile_m * tile_n  # converted output, elements
    MB, NB = tile_m // r, tile_n // t  # 8x8 blocks per tile

    i8 = IntegerType.get_signless(8)
    l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

    # ---- L3 (caller-facing). A/B are packed bytes; C is a real float tensor.
    A_l3_ty = MemRefType.get([m, KB], i8)
    B_l3_ty = MemRefType.get([n, KB], i8)  # B stored transposed
    C_l3_ty = MemRefType.get([m, n], xrt_dtype_out)

    # ---- L1. A/B tile boxes are contiguous in sub-tile-major order, which is
    # exactly what mm_bfp.cc's block_vector_input_buffer_stream .seek(z*colA)
    # expects, so no strided views are needed.
    l1TyA = MemRefType.get([a_l1_b], i8, memory_space=l1_ms)
    l1TyB = MemRefType.get([b_l1_b], i8, memory_space=l1_ms)
    l1TyAccHerd = MemRefType.get([herd_m, herd_n, c_l1_b], i8, memory_space=l1_ms)
    l1TyOutHerd = MemRefType.get(
        [herd_m, herd_n, c_elems], xrt_dtype_out, memory_space=l1_ms
    )
    acc_sv_layout = StridedLayoutAttr.get(
        ShapedType.get_dynamic_size(), [herd_n * c_l1_b, c_l1_b, 1]
    )
    out_sv_layout = StridedLayoutAttr.get(
        ShapedType.get_dynamic_size(), [herd_n * c_elems, c_elems, 1]
    )
    l1TyAccSub = MemRefType.get(
        [1, 1, c_l1_b], i8, memory_space=l1_ms, layout=acc_sv_layout
    )
    l1TyOutSub = MemRefType.get(
        [1, 1, c_elems], xrt_dtype_out, memory_space=l1_ms, layout=out_sv_layout
    )

    # ---- External kernel decls. mm_bfp.cc + bfp_cvt.cc are linked into one .o.
    def _extern(name, arg_tys):
        f = FuncOp(name, (arg_tys, []), visibility="private")
        f.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return f

    zero_func = _extern("zero_kernel", [l1TyAccSub])
    matmul_func = _extern("matmul_vectorized_bfp16", [l1TyA, l1TyB, l1TyAccSub])
    cvt_name = "bfp16_to_bf16_mn" if np_dtype_out == bfloat16 else "bfp16_to_f32_mn"
    cvt_func = _extern(cvt_name, [l1TyAccSub, l1TyOutSub])

    @FuncOp.from_py_func(A_l3_ty, B_l3_ty, C_l3_ty)
    def matmul_bfp16(arg0, arg1, arg2):
        launch_size = [m // (tile_m * herd_m), n // (tile_n * herd_n)]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(ivx, ivy, sx, sy, l3_a, l3_b, l3_c):
            @segment(name="matmul_seg", operands=[ivx, ivy, l3_a, l3_b, l3_c])
            def segment_body(ivx_s, ivy_s, l3_a_s, l3_b_s, l3_c_s):
                l2TyA = MemRefType.get(
                    [herd_m, k_per_l2, a_l1_b], i8, memory_space=l2_ms
                )
                l2TyB = MemRefType.get(
                    [herd_n, k_per_l2, b_l1_b], i8, memory_space=l2_ms
                )
                l2TyC = MemRefType.get(
                    [herd_m, herd_n, tile_m, tile_n], xrt_dtype_out, memory_space=l2_ms
                )

                l2_a = AllocOp(l2TyA, [], [])
                l2_b = AllocOp(l2TyB, [], [])
                l2_c = AllocOp(l2TyC, [], [])
                # Segment-shared so the accumulator survives across the
                # zero / compute / drain herd invocations.
                l1_acc = AllocOp(l1TyAccHerd, [], [])
                l1_out = AllocOp(l1TyOutHerd, [], [])

                off_x_rows = _scaled(ivx_s, tile_m * herd_m)  # A/C row offset
                off_y_rows = _scaled(ivy_s, tile_n * herd_n)  # B row offset
                off_y_cols = _scaled(ivy_s, tile_n * herd_n)  # C column offset

                # ---- Herd #1: zero the bfp16 L1 accumulator.
                @herd(name="herd_0", sizes=[herd_m, herd_n], operands=[l1_acc])
                def herd_init(tx, ty, _sx, _sy, acc):
                    CallOp(
                        zero_func,
                        [
                            subview(
                                acc,
                                offsets=[tx, ty, 0],
                                sizes=[1, 1, c_l1_b],
                                strides=[1, 1, 1],
                            )
                        ],
                    )

                # ---- Segment-level K-l2 loop.
                for i in for_(0, k // tile_k_l2):
                    k_off_b = _scaled(i, nbytes(tile_k_l2))

                    # L3 -> L2: gather tile boxes. The two innermost dims
                    # (tile_m, a_row_b) flatten into the contiguous destination.
                    dma_memcpy_nd(
                        l2_a,
                        l3_a_s,
                        src_offsets=[0, 0, off_x_rows, k_off_b],
                        src_sizes=[herd_m, k_per_l2, tile_m, a_row_b],
                        src_strides=[tile_m * KB, a_row_b, KB, 1],
                    )
                    dma_memcpy_nd(
                        l2_b,
                        l3_b_s,
                        src_offsets=[0, 0, off_y_rows, k_off_b],
                        src_sizes=[herd_n, k_per_l2, tile_n, a_row_b],
                        src_strides=[tile_n * KB, a_row_b, KB, 1],
                    )

                    # ---- Herd #2: accumulate over the K-l1 chunks.
                    @herd(
                        name="herd_0",
                        sizes=[herd_m, herd_n],
                        operands=[l1_acc, l2_a, l2_b],
                    )
                    def herd_compute(tx, ty, _sx, _sy, acc, a2, b2):
                        a1 = AllocOp(l1TyA, [], [])
                        b1 = AllocOp(l1TyB, [], [])
                        loop = for_
                        for j in loop(0, k_per_l2):
                            dma_memcpy_nd(
                                a1,
                                a2,
                                src_offsets=[tx, j, 0],
                                src_sizes=[1, 1, a_l1_b],
                                src_strides=[k_per_l2 * a_l1_b, a_l1_b, 1],
                            )
                            dma_memcpy_nd(
                                b1,
                                b2,
                                src_offsets=[ty, j, 0],
                                src_sizes=[1, 1, b_l1_b],
                                src_strides=[k_per_l2 * b_l1_b, b_l1_b, 1],
                            )
                            CallOp(
                                matmul_func,
                                [
                                    a1,
                                    b1,
                                    subview(
                                        acc,
                                        offsets=[tx, ty, 0],
                                        sizes=[1, 1, c_l1_b],
                                        strides=[1, 1, 1],
                                    ),
                                ],
                            )
                            yield_([])
                        DeallocOp(a1)
                        DeallocOp(b1)

                    yield_([])

                # ---- Herd #3: convert bfp16 -> output dtype, then drain.
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_acc, l1_out, l2_c],
                )
                def herd_drain(tx, ty, _sx, _sy, acc, out, c2):
                    CallOp(
                        cvt_func,
                        [
                            subview(
                                acc,
                                offsets=[tx, ty, 0],
                                sizes=[1, 1, c_l1_b],
                                strides=[1, 1, 1],
                            ),
                            subview(
                                out,
                                offsets=[tx, ty, 0],
                                sizes=[1, 1, c_elems],
                                strides=[1, 1, 1],
                            ),
                        ],
                    )
                    # L1 (8x8-block order) -> L2 (row-major). Nesting is
                    # [m_b, n_b, m_i, n_i] so the L1 side is a single
                    # contiguous run and only the L2 side needs the permute.
                    dma_memcpy_nd(
                        c2,
                        out,
                        dst_offsets=[tx, ty, 0, 0, 0, 0],
                        dst_sizes=[1, 1, MB, NB, r, t],
                        dst_strides=[
                            herd_n * c_elems,
                            c_elems,
                            r * tile_n,
                            t,
                            tile_n,
                            1,
                        ],
                        src_offsets=[tx, ty, 0, 0, 0, 0],
                        src_sizes=[1, 1, MB, NB, r, t],
                        src_strides=[
                            herd_n * c_elems,
                            c_elems,
                            NB * r * t,
                            r * t,
                            t,
                            1,
                        ],
                    )

                # ---- L2 -> L3.
                dma_memcpy_nd(
                    l3_c_s,
                    l2_c,
                    dst_offsets=[0, off_x_rows, 0, off_y_cols],
                    dst_sizes=[herd_m, tile_m, herd_n, tile_n],
                    dst_strides=[tile_m * n, n, tile_n, 1],
                    src_offsets=[0, 0, 0, 0],
                    src_sizes=[herd_m, tile_m, herd_n, tile_n],
                    src_strides=[herd_n * c_elems, tile_n, c_elems, 1],
                )

                DeallocOp(l2_a)
                DeallocOp(l2_b)
                DeallocOp(l2_c)
                DeallocOp(l1_acc)
                DeallocOp(l1_out)


def pack_operands(A, B, tile_m, tile_k_l1, tile_n):
    """Float A[M,K], B[K,N] -> the packed+shuffled device buffers.

    Byte-for-byte what mlir-aie's bfp_test.cpp builds: quantize with
    floatToBfp16(), then shuffleMatrixForBfp16ebs8() with the per-core tile
    box. B is quantized transposed because the kernel consumes it that way.
    """
    A_dev = shuffle_bfp16ebs8(float_to_bfp16ebs8(A), tile_m, tile_k_l1)
    B_dev = shuffle_bfp16ebs8(
        float_to_bfp16ebs8(np.ascontiguousarray(B.T)), tile_n, tile_k_l1
    )
    return A_dev, B_dev


if __name__ == "__main__":
    # Default values.
    M = 512
    K = 512
    N = 512
    TILE_M = 64
    TILE_K_L2 = 128
    TILE_K_L1 = 64
    TILE_N = 64
    HERD_M = 4
    HERD_N = 4
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the bfp16ebs8 matmul example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--m", type=int, default=M, help="M dimension size in a (MxK) * (KxN) matmul"
    )
    parser.add_argument(
        "--k", type=int, default=K, help="K dimension size in a (MxK) * (KxN) matmul"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="N dimension size in a (MxK) * (KxN) matmul",
    )
    parser.add_argument(
        "--tile-m", type=int, default=TILE_M, help="M dimension size of each L1 tile"
    )
    parser.add_argument(
        "--tile-k-l2",
        type=int,
        default=TILE_K_L2,
        help="K dimension size of each L2 tile",
    )
    parser.add_argument(
        "--tile-k-l1",
        type=int,
        default=TILE_K_L1,
        help="K dimension size of each L1 tile",
    )
    parser.add_argument(
        "--tile-n", type=int, default=TILE_N, help="N dimension size of each L1 tile"
    )
    parser.add_argument(
        "--herd-m",
        type=int,
        default=HERD_M,
        help="Number of L1 tiles along the M dimension",
    )
    parser.add_argument(
        "--herd-n",
        type=int,
        default=HERD_N,
        help="Number of L1 tiles along the N dimension",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-xclbin", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure compilation mode: compile-only (no XRT, no xclbin), compile-and-xclbin (requires XRT, generates xclbin), or compile-and-run (requires XRT, generates xclbin and runs)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2p",
        help="Target AIE architecture (aie2p only; bfp16ebs8 is an AIE2P type)",
    )
    parser.add_argument(
        "--output-dtype",
        type=str,
        choices=["bf16", "f32"],
        default=None,
        dest="output_dtype",
        help="Override output data type (default: f32 for aie2p)",
    )
    parser.add_argument(
        "--runtime-loop-tiling",
        type=int,
        default=None,
        dest="runtime_loop_tiling",
        help="Tile factor for the runtime launch loop (default: auto -- 2, or 1 "
        "for small herds, see the note in the source)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after 10 warmup) and "
        "print Latency + GFLOPs in addition to the correctness check",
    )
    args = parser.parse_args()

    # bfp16ebs8 (v8bfp16ebs8 / mac_8x8_8x8T) exists only on AIE2P. There is no
    # aie2 fallback -- neither the type nor the MMUL intrinsic is available, and
    # xchesscc has no bfp16ebs8 codegen path either. Fail loudly rather than
    # emitting IR that cannot be lowered.
    if args.arch != "aie2p":
        print(
            f"Error: --arch {args.arch} is not supported by this example.",
            file=sys.stderr,
        )
        print(
            "bfp16ebs8 (v8bfp16ebs8) and its mac_8x8_8x8T MMUL are AIE2P-only; "
            "NPU1/aie2 has no block-floating-point datapath. Re-run with "
            "--arch aie2p on an NPU2 (Strix) device.",
            file=sys.stderr,
        )
        sys.exit(1)

    # aie2p defaults to f32 output for better precision.
    # Can be overridden with --output-dtype.
    if args.output_dtype == "bf16":
        pass  # keep default bfloat16
    elif args.output_dtype == "f32":
        OUTPUT_DATATYPE = np.float32
    elif args.arch == "aie2p":
        OUTPUT_DATATYPE = np.float32

    # bfp16ebs8 kernels are Peano-only; xchesscc has no codegen path for them.
    if args.compile_mode != "compile-only" and not os.environ.get("PEANO_INSTALL_DIR"):
        print(
            "Error: PEANO_INSTALL_DIR environment variable is not set.",
            file=sys.stderr,
        )
        print(
            "Peano is required for bfp16ebs8; the Chess toolchain has no "
            "bfp16ebs8 codegen path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Runtime launch-loop tiling. [2, 2] batches the per-launch DMA task
    # triplets and is worth ~26% throughput (183us vs 232us at 512^3 / 4x4).
    #
    # It is NOT safe on small herds: when the herd is small enough that AIR
    # gives the C drain a single shim channel, the shim-BD fold collapses that
    # drain across every launch iteration into one task with one await, so only
    # the first launch iteration's output is ever written while all the input
    # fills still run. Observed at herd 2x2 / 2x1 / 1x2 with more than one
    # launch iteration (herd 4x2 and larger split C over 4 channels and are
    # fine). Same class as the shim-BD stride-fold / wait-pairing fixes in
    # mlir-air #1810 and #1815. Override with --runtime-loop-tiling.
    n_launch = (args.m // (args.tile_m * args.herd_m)) * (
        args.n // (args.tile_n * args.herd_n)
    )
    n_cores = args.herd_m * args.herd_n
    if args.runtime_loop_tiling is not None:
        runtime_loop_tiling_sizes = [args.runtime_loop_tiling] * 2
    elif n_launch > 1 and n_cores < 8:
        runtime_loop_tiling_sizes = [1, 1]
        print(
            f"note: herd {args.herd_m}x{args.herd_n} ({n_cores} cores) with "
            f"{n_launch} launch iterations -- using runtime_loop_tiling_sizes="
            "[1, 1]; [2, 2] drops all but the first launch iteration's output "
            "on herds this small. Override with --runtime-loop-tiling."
        )
    else:
        runtime_loop_tiling_sizes = [2, 2]

    # mm_bfp.cc keeps C in bfp16ebs8 between K-l1 chunks, re-quantizing the
    # accumulator to an 8-bit mantissa with a shared exponent on every chunk.
    # Error therefore grows with the number of round-trips, K/tile_k_l1,
    # independently of the input distribution (measured mean_rel_L1 at
    # M=128,N=2048: 32 round-trips -> 0.27 randn / 0.09 integer; 128
    # round-trips -> 1.23 randn / 0.41 integer). Past ~64 round-trips the
    # result is not usable at any sane tolerance.
    n_roundtrips = args.k // args.tile_k_l1
    if n_roundtrips > 64:
        print(
            f"warning: K/tile_k_l1 = {n_roundtrips} accumulator round-trips. "
            "mm_bfp.cc re-quantizes C to bfp16ebs8 on every K-l1 chunk, so this "
            "shape will lose too much precision to pass the check. Raise "
            "--tile-k-l1 or use a shallower K.",
            file=sys.stderr,
        )

    mlir_module = build_module(
        args.m,
        args.k,
        args.n,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
        OUTPUT_DATATYPE,
        args.arch,
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Variance-normalized inputs following PyTorch's
    # random_matrix_with_scaled_reduction_dim: randn / sqrt(K). This keeps
    # output variance ~1 regardless of K, so relative tolerance behaves
    # consistently across matrix sizes. Same generator as the bf16 example.
    scale = 1.0 / np.sqrt(args.k)
    input_a = (np.random.randn(args.m, args.k) * scale).astype(np.float32)
    input_b = (np.random.randn(args.k, args.n) * scale).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        # Device operands are the bfp16ebs8-quantized, sub-tile-shuffled bytes.
        packed_a, packed_b = pack_operands(
            input_a, input_b, args.tile_m, args.tile_k_l1, args.tile_n
        )

        # Full reference: f32 matmul over the whole output, cast to the output
        # dtype. Computing the reference from the ORIGINAL floats (not the
        # de-quantized operands) means the check covers the bfp16 input
        # quantization too, which is what mlir-aie's bfp_test.cpp does.
        reference = (input_a @ input_b).astype(OUTPUT_DATATYPE)

        # Tolerances are sized to the kernel's measured worst-case error.
        #
        # bfp16ebs8 shares one 8-bit exponent across 8 elements, so an element
        # sitting in a block with a much larger neighbour loses mantissa bits.
        # On Gaussian inputs that is the dominant error term -- it comes from
        # quantizing A and B, not from the output dtype, which is why bf16 and
        # f32 output need the same tolerance here.
        #
        # The check is therefore absolute-error driven: with randn/sqrt(K)
        # inputs the reference has std ~1/sqrt(K), which shrinks with K while
        # the bfp16 error floor does not, so mean_rel_L1 grows with K
        # (measured: 2.1e-2 at K=64, 6.9e-2 at K=512, 2.7e-1 at K=2048).
        # Measured abs_err max across every Makefile target and both output
        # dtypes is 8.5e-3 .. 1.5e-2, so atol below leaves ~1.7x headroom.
        #
        # For reference, mlir-aie's own bfp_test.cpp sidesteps this by drawing
        # inputs from rand()%16 ("limiting to 16 to avoid precision loss
        # issues") -- integers of similar magnitude share an exponent almost
        # losslessly. This example keeps the bf16 example's randn/sqrt(K)
        # generator instead, so the numbers above are the honest bfp16 error
        # on high-dynamic-range data.
        test_rtol, test_atol = 5e-2, 2.5e-2

        ###### Compile and test
        runner_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": runtime_loop_tiling_sizes,
            "stack_size": 2048,
            "report_precision": True,
            "n_perf_iters": args.perf_iters,
            "perf_flops": (
                (2.0 * args.m * args.k * args.n) if args.perf_iters > 0 else None
            ),
        }

        runner = XRTRunner(**runner_kwargs, instance_name="matmul_bfp16")
        exit(
            runner.run_test(
                mlir_module,
                inputs=[packed_a, packed_b],
                expected_outputs=[reference],
                rtol=test_rtol,
                atol=test_atol,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        ###### Compile and generate xclbin (requires XRT, no execution)
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
            stack_size=2048,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()

    elif args.compile_mode == "compile-only":
        ###### Compile only (without XRT dependencies)
        backend = XRTBackend(
            verbose=args.verbose,
            target_device="npu2",
            output_format="none",
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
            stack_size=2048,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()

        print("Compilation completed successfully!")
        sys.exit(0)
