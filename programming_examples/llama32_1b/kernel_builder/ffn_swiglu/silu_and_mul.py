# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SiLU + Elementwise Multiply Kernel

Element-wise SiLU and multiply: output[i] = SiLU(gate[i]) * up[i]
where SiLU(x) = x * sigmoid(x)

Uses an external C++ kernel (silu_and_mul.cc) compiled with Peano.
The kernel processes data in tiles using a 1x2 herd (2 AIE tiles).
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.memref import collapse_shape as memref_collapse_shape
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(n, tile_n, np_dtype_in, herd_x=1, herd_y=None):
    xrt_dtype = type_mapper(np_dtype_in)
    if herd_y is None:
        herd_y = 2  # default backward-compatible
    total_tiles = herd_x * herd_y
    assert (
        n % (tile_n * total_tiles) == 0
    ), f"n ({n}) must be divisible by tile_n * total_tiles ({tile_n * total_tiles})"

    # L3 types
    l3MemrefTy = MemRefType.get([n], xrt_dtype)

    # L1 types
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTy = MemRefType.get(
        shape=[tile_n], element_type=xrt_dtype, memory_space=l1_mem_space
    )

    # External kernel declaration
    silu_mul_func = FuncOp(
        "silu_and_mul_bf16",
        ([l1MemrefTy, l1MemrefTy, l1MemrefTy, T.i32()], []),
        visibility="private",
    )
    silu_mul_func.attributes["link_with"] = StringAttr.get("silu_and_mul.o")
    silu_mul_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3MemrefTy, l3MemrefTy, l3MemrefTy)
    def silu_and_mul(arg0, arg1, arg2):
        # arg0 = gate [n], arg1 = up [n], arg2 = output [n]

        @launch(operands=[arg0, arg1, arg2])
        def silu_mul_launch(l_gate, l_up, l_out):

            @segment(name="silu_mul_seg", operands=[l_gate, l_up, l_out])
            def silu_mul_seg(s_gate, s_up, s_out):

                @herd(
                    name="herd_0",
                    sizes=[herd_x, herd_y],
                    operands=[s_gate, s_up, s_out],
                )
                def herd_body(_tx, _ty, _sx, _sy, l3_gate, l3_up, l3_out):
                    l1_gate = AllocOp(l1MemrefTy, [], [])
                    l1_up = AllocOp(l1MemrefTy, [], [])
                    l1_out = AllocOp(l1MemrefTy, [], [])

                    tile_n_i32 = ConstantOp(T.i32(), tile_n)

                    for loop_iv in range_(0, n, tile_n * total_tiles):
                        # Compute linear tile index: tx * herd_y + ty
                        offset_map = AffineMap.get(
                            0,
                            3,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineExpr.get_mul(
                                        AffineExpr.get_add(
                                            AffineExpr.get_mul(
                                                AffineSymbolExpr.get(1),
                                                AffineConstantExpr.get(herd_y),
                                            ),
                                            AffineSymbolExpr.get(2),
                                        ),
                                        AffineConstantExpr.get(tile_n),
                                    ),
                                )
                            ],
                        )
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])

                        dma_memcpy_nd(
                            l1_gate,
                            l3_gate,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_up,
                            l3_up,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )

                        CallOp(silu_mul_func, [l1_gate, l1_up, l1_out, tile_n_i32])

                        dma_memcpy_nd(
                            l3_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_gate)
                    DeallocOp(l1_up)
                    DeallocOp(l1_out)

                herd_body.attributes["link_with"] = StringAttr.get(
                    "silu_and_mul.o"
                )


@module_builder
def build_module_2d(rows, cols, tile_n, np_dtype_in, herd_x=8, herd_y=1):
    """Build SwiGLU module with 2D memref inputs: memref<rows x cols x bf16>.

    Same computation as build_module but accepts 2D memrefs for compatibility
    with GEMM outputs. Collapses 2D → 1D at the segment level before
    passing to the herd.
    """
    n = rows * cols
    xrt_dtype = type_mapper(np_dtype_in)
    total_tiles = herd_x * herd_y
    assert n % (tile_n * total_tiles) == 0

    l3_2d_ty = MemRefType.get([rows, cols], xrt_dtype)
    l3_1d_ty = MemRefType.get([n], xrt_dtype)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1TileTy = MemRefType.get(
        shape=[tile_n], element_type=xrt_dtype, memory_space=l1_space
    )

    silu_mul_func = FuncOp(
        "silu_and_mul_bf16",
        ([l1TileTy, l1TileTy, l1TileTy, T.i32()], []),
        visibility="private",
    )
    silu_mul_func.attributes["link_with"] = StringAttr.get("silu_and_mul.o")
    silu_mul_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_2d_ty)
    def silu_and_mul_2d(arg0, arg1, arg2):

        @launch(operands=[arg0, arg1, arg2])
        def silu_mul_launch(l_gate, l_up, l_out):

            # Collapse 2D → 1D at segment level
            gate_1d = memref_collapse_shape(l3_1d_ty, l_gate, [[0, 1]])
            up_1d = memref_collapse_shape(l3_1d_ty, l_up, [[0, 1]])
            out_1d = memref_collapse_shape(l3_1d_ty, l_out, [[0, 1]])

            @segment(name="silu_mul_seg", operands=[gate_1d, up_1d, out_1d])
            def silu_mul_seg(s_gate, s_up, s_out):

                @herd(
                    name="herd_0",
                    sizes=[herd_x, herd_y],
                    operands=[s_gate, s_up, s_out],
                )
                def herd_body(_tx, _ty, _sx, _sy, l3_gate, l3_up, l3_out):
                    l1_gate = AllocOp(l1TileTy, [], [])
                    l1_up = AllocOp(l1TileTy, [], [])
                    l1_out = AllocOp(l1TileTy, [], [])

                    tile_n_i32 = ConstantOp(T.i32(), tile_n)

                    for loop_iv in range_(0, n, tile_n * total_tiles):
                        offset_map = AffineMap.get(
                            0,
                            3,
                            [
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineExpr.get_mul(
                                        AffineExpr.get_add(
                                            AffineExpr.get_mul(
                                                AffineSymbolExpr.get(1),
                                                AffineConstantExpr.get(herd_y),
                                            ),
                                            AffineSymbolExpr.get(2),
                                        ),
                                        AffineConstantExpr.get(tile_n),
                                    ),
                                )
                            ],
                        )
                        offset = affine_apply(offset_map, [loop_iv, _tx, _ty])

                        dma_memcpy_nd(
                            l1_gate,
                            l3_gate,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        dma_memcpy_nd(
                            l1_up,
                            l3_up,
                            src_offsets=[offset],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        CallOp(silu_mul_func, [l1_gate, l1_up, l1_out, tile_n_i32])
                        dma_memcpy_nd(
                            l3_out,
                            l1_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])

                    DeallocOp(l1_gate)
                    DeallocOp(l1_up)
                    DeallocOp(l1_out)

                herd_body.attributes["link_with"] = StringAttr.get(
                    "silu_and_mul.o"
                )


def silu_reference(x):
    """Reference SiLU implementation in F32."""
    x_f32 = x.astype(np.float32)
    return x_f32 * (1.0 / (1.0 + np.exp(-x_f32)))


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="silu_and_mul.py",
        description="Builds, runs, and tests the standalone SwiGLU activation kernel",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
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

    mlir_module = build_module(args.n, args.tile_n, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(42)
    gate = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)
    up = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)

    # Reference: SiLU(gate) * up
    silu_gate = silu_reference(gate)
    expected = (silu_gate * up.astype(np.float32)).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="silu_and_mul",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[gate, up],
                expected_outputs=[expected],
                rtol=5e-2,
                atol=5e-1,
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
