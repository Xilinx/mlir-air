# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RoPE (Rotary Position Embeddings) with On-Chip Sin/Cos Computation

Applies rotary position embeddings to Q and K vectors (V is passed through):
  Q_out[2i]   = Q[2i] * cos(pos * freq_i) - Q[2i+1] * sin(pos * freq_i)
  Q_out[2i+1] = Q[2i] * sin(pos * freq_i) + Q[2i+1] * cos(pos * freq_i)
  (same for K; V is unchanged)

where freq_i = 1 / (theta ^ (2i / head_size)), theta = 10000.

Unlike the LUT-based variant, sin/cos values are computed on-chip using
Chebyshev polynomial approximation. The frequency table is hardcoded in
the kernel (head_size=48). No external sin/cos input is needed.

Input format: [num_heads * 3 * head_size] — Q, K, V concatenated per head.
Uses external C++ kernels compiled from rope.cc.

Note: The kernel uses Chess-specific shuffle intrinsics and is currently
XFAIL on Peano. See rope_lut/ for a Peano-compatible alternative.
"""

import argparse
import numpy as np
from math import cos, sin
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.affine import apply as affine_apply
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(num_heads, head_size, herd_n, np_dtype_in):
    assert (num_heads * head_size) % herd_n == 0
    inout_size = [3 * num_heads * head_size]  # Q, K, V concatenated
    xrt_dtype_in = type_mapper(np_dtype_in)

    # L3 MemRefTypes
    memrefTyIn = MemRefType.get(inout_size, xrt_dtype_in)
    memrefTyOut = MemRefType.get(inout_size, xrt_dtype_in)

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyHeadSizeByTwo = MemRefType.get(
        shape=[head_size // 2],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyThreeByHeadSize = MemRefType.get(
        shape=[3, head_size],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )

    # External kernel functions
    cosf_poly_func = FuncOp(
        "cosf_bf16_24_8",
        ([l1MemrefTyHeadSizeByTwo, l1MemrefTyHeadSizeByTwo], []),
        visibility="private",
    )
    sinf_poly_func = FuncOp(
        "sinf_bf16_24_8",
        ([l1MemrefTyHeadSizeByTwo, l1MemrefTyHeadSizeByTwo], []),
        visibility="private",
    )
    freq_pos_func = FuncOp(
        "freq_pos_bf16_24_8",
        ([T.i32(), l1MemrefTyHeadSizeByTwo], []),
        visibility="private",
    )
    shuffle_apply_rope_func = FuncOp(
        "shuffle_apply_rope_bf16_48",
        (
            [
                T.i32(),
                l1MemrefTyHeadSizeByTwo,
                l1MemrefTyHeadSizeByTwo,
                l1MemrefTyThreeByHeadSize,
            ],
            [],
        ),
        visibility="private",
    )
    vector_copy_func = FuncOp(
        "vector_copy_bf16_192_16",
        ([l1MemrefTyThreeByHeadSize, l1MemrefTyThreeByHeadSize], []),
        visibility="private",
    )
    for func in [
        freq_pos_func,
        sinf_poly_func,
        cosf_poly_func,
        shuffle_apply_rope_func,
        vector_copy_func,
    ]:
        func.attributes["link_with"] = StringAttr.get("rope.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def rope_sincos(arg0, arg1):

        @herd(
            name="herd_0",
            sizes=[1, herd_n],
            operands=[arg0, arg1],
        )
        def herd_body(_tx, _ty, _sx, _sy, _l3_in, _l3_out):

            one_const = ConstantOp(IntegerAttr.get(T.i32(), 1), None)

            for t in range_(0, num_heads * 3 * head_size, herd_n * 3 * head_size):
                l1_in_data = AllocOp(l1MemrefTyThreeByHeadSize, [], [])
                l1_out_data = AllocOp(l1MemrefTyThreeByHeadSize, [], [])
                l1_freq_pos_data = AllocOp(l1MemrefTyHeadSizeByTwo, [], [])

                offset_map = AffineMap.get(
                    0,
                    2,
                    [
                        AffineExpr.get_add(
                            AffineSymbolExpr.get(0),
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(1),
                                AffineConstantExpr.get(3 * head_size),
                            ),
                        )
                    ],
                )
                offset = affine_apply(offset_map, [t, _ty])

                dma_memcpy_nd(
                    l1_in_data,
                    _l3_in,
                    src_offsets=[offset],
                    src_sizes=[3 * head_size],
                    src_strides=[1],
                )

                # Copy input to output buffer (V passes through unchanged)
                CallOp(vector_copy_func, [l1_in_data, l1_out_data])

                # Compute frequency * position on-chip
                CallOp(freq_pos_func, [one_const, l1_freq_pos_data])

                # Compute sin/cos via Chebyshev polynomial
                l1_sinf_vec = AllocOp(l1MemrefTyHeadSizeByTwo, [], [])
                l1_cosf_vec = AllocOp(l1MemrefTyHeadSizeByTwo, [], [])
                CallOp(sinf_poly_func, [l1_freq_pos_data, l1_sinf_vec])
                CallOp(cosf_poly_func, [l1_freq_pos_data, l1_cosf_vec])

                # Apply rotation to Q (offset 0) and K (offset head_size)
                rope_offset_q = ConstantOp(IntegerAttr.get(T.i32(), 0), None)
                CallOp(
                    shuffle_apply_rope_func,
                    [rope_offset_q, l1_cosf_vec, l1_sinf_vec, l1_out_data],
                )
                rope_offset_k = ConstantOp(IntegerAttr.get(T.i32(), head_size), None)
                CallOp(
                    shuffle_apply_rope_func,
                    [rope_offset_k, l1_cosf_vec, l1_sinf_vec, l1_out_data],
                )

                dma_memcpy_nd(
                    _l3_out,
                    l1_out_data,
                    dst_offsets=[offset],
                    dst_sizes=[3 * head_size],
                    dst_strides=[1],
                )
                DeallocOp(l1_sinf_vec)
                DeallocOp(l1_cosf_vec)
                DeallocOp(l1_freq_pos_data)
                DeallocOp(l1_in_data)
                DeallocOp(l1_out_data)
                yield_([])

        herd_body.attributes["link_with"] = StringAttr.get("rope.o")


if __name__ == "__main__":
    HEAD_SIZE = 48
    NUM_HEADS = 8
    HERD_N = 4
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the RoPE (on-chip sin/cos) example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--head-size", type=int, default=HEAD_SIZE, help="Head size")
    parser.add_argument(
        "--num-heads", type=int, default=NUM_HEADS, help="Number of heads"
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
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module(
        args.num_heads, args.head_size, args.herd_n, INPUT_DATATYPE
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    num_tiles = args.num_heads
    inputs = np.random.randn(num_tiles, 3 * args.head_size).astype(INPUT_DATATYPE)
    outputs = inputs.copy()

    # Reference: apply rotation to Q and K, leave V unchanged
    for i in range(num_tiles):
        for s in range(0, args.head_size, 2):
            freq = 1.0 / pow(10000.0, float(s) / float(args.head_size))
            val = 1 * freq  # position = 1

            fcr = cos(val)
            fci = sin(val)

            # Rotate Q
            v0 = outputs[i][s]
            v1 = outputs[i][s + 1]
            outputs[i][s] = v0 * fcr - v1 * fci
            outputs[i][s + 1] = v0 * fci + v1 * fcr

            # Rotate K
            v0 = outputs[i][s + args.head_size]
            v1 = outputs[i][s + args.head_size + 1]
            outputs[i][s + args.head_size] = v0 * fcr - v1 * fci
            outputs[i][s + args.head_size + 1] = v0 * fci + v1 * fcr

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[inputs],
                expected_outputs=[outputs],
                rtol=1e1,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
