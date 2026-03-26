# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Row-wise argmax: out[row] = argmax_col(A[row, col])
# For each row, find the column index of the maximum element.
# Input: [ne1, ne0] f32 (ne0 contiguous), Output: [ne1] i32.
#
# GGML layout: [ne0, ne1] where ne0 is contiguous.
# GGML op: argmax [10, 500] -> [500, 1]
#   ne0=10 (classes, contiguous, reduced), ne1=500 (batch, rows, tiled).
# In numpy row-major: input is (ne1=500, ne0=10), output is (500,) i32.
# Argmax is over ne0 (axis=1, columns) for each row.

import argparse
import math
import numpy as np

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from air.extras import types as extrasT

np.random.seed(42)

range_ = for_


@module_builder
def build_module(num_rows, num_cols, tile_rows, herd_n, ne0_actual):
    """Build row-wise argmax module.

    num_rows = ne1 (padded, tiled across herd).
    num_cols = ne0 (padded, contiguous, reduced per row).
    ne0_actual = actual number of columns to reduce over.
    """
    assert num_rows % (tile_rows * herd_n) == 0

    xrt_dtype_f32 = type_mapper(np.float32)
    xrt_dtype_i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()
    l1_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)

    # L3 MemRefTypes
    # Input: (ne1, ne0) = (num_rows, num_cols)
    memrefTyA = MemRefType.get([num_rows, num_cols], xrt_dtype_f32)
    # Output: (ne1,) = (num_rows,) i32
    memrefTyOut = MemRefType.get([num_rows], xrt_dtype_i32)

    # L1: load tile_rows rows, each with num_cols columns
    l1TileTy = MemRefType.get(
        shape=[tile_rows, num_cols],
        element_type=xrt_dtype_f32,
        memory_space=l1_mem_space,
    )
    l1OutTy = MemRefType.get(
        shape=[tile_rows], element_type=xrt_dtype_i32, memory_space=l1_mem_space
    )

    @FuncOp.from_py_func(memrefTyA, memrefTyOut)
    def argmax(arg_a, arg_out):
        launch_size = [1, num_rows // tile_rows // herd_n]

        @launch(operands=[arg_a, arg_out], sizes=launch_size)
        def launch_body(
            launch_ivx, launch_ivy, launch_sizex, launch_sizey, l3_a, l3_out
        ):

            @segment(
                name="argmax_seg",
                operands=[launch_ivy, l3_a, l3_out],
            )
            def segment_body(launch_ivy_s, l3_a_s, l3_out_s):
                c_tile_rows_herd_n = ConstantOp(
                    IntegerAttr.get(IndexType.get(), tile_rows * herd_n), None
                )
                launch_offset_row = arith.MulIOp(launch_ivy_s, c_tile_rows_herd_n)

                @herd(
                    name="herd_0",
                    sizes=[1, herd_n],
                    operands=[
                        launch_offset_row,
                        l3_a_s,
                        l3_out_s,
                    ],
                )
                def herd_body(tx, ty, _sx, _sy, _loff_row, _l3_a, _l3_out):
                    l1_tile = AllocOp(l1TileTy, [], [])
                    l1_out = AllocOp(l1OutTy, [], [])

                    # row_offset = launch_offset_row + ty * tile_rows
                    row_offset_map = AffineMap.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(1),
                                    AffineConstantExpr.get(tile_rows),
                                ),
                            )
                        ],
                    )
                    row_offset = affine_apply(row_offset_map, [_loff_row, ty])

                    # DMA: load tile_rows rows, all num_cols columns
                    dma_memcpy_nd(
                        l1_tile,
                        _l3_a,
                        src_offsets=[row_offset, 0],
                        src_sizes=[tile_rows, num_cols],
                        src_strides=[num_cols, 1],
                    )

                    # Scalar argmax per row over ne0_actual columns
                    c0 = ConstantOp(index_type, 0)
                    c1 = ConstantOp(index_type, 1)
                    c_tile_rows_cst = ConstantOp(index_type, tile_rows)
                    c_ne0_actual = ConstantOp(index_type, ne0_actual)
                    neg_inf = arith.ConstantOp(
                        xrt_dtype_f32,
                        FloatAttr.get(xrt_dtype_f32, float("-inf")),
                    )
                    c0_i32 = arith.ConstantOp(xrt_dtype_i32, 0)

                    for row in range_(c0, c_tile_rows_cst, c1):
                        # Argmax over ne0_actual columns for this row
                        for col, (current_max_val, current_max_idx), results in range_(
                            c0,
                            c_ne0_actual,
                            c1,
                            iter_args=[neg_inf, c0_i32],
                        ):
                            val = load(l1_tile, [row, col])
                            cmp = arith.CmpFOp(
                                arith.CmpFPredicate.OGT,
                                val,
                                current_max_val,
                            )
                            new_max_val = arith.SelectOp(cmp, val, current_max_val)
                            col_i32 = arith.IndexCastOp(xrt_dtype_i32, col)
                            new_max_idx = arith.SelectOp(cmp, col_i32, current_max_idx)
                            yield_([new_max_val, new_max_idx])

                        store(results[1], l1_out, [row])
                        yield_([])

                    # DMA output back to L3
                    dma_memcpy_nd(
                        _l3_out,
                        l1_out,
                        dst_offsets=[row_offset],
                        dst_sizes=[tile_rows],
                        dst_strides=[1],
                    )

                    DeallocOp(l1_tile)
                    DeallocOp(l1_out)


if __name__ == "__main__":
    # GGML [10, 500]: ne0=10 (classes), ne1=500 (batch)
    # numpy: (500, 10)
    NE0_ACTUAL = 10  # cols (classes, reduced)
    NE1_ACTUAL = 500  # rows (batch, tiled)
    TILE_ROWS = 32
    HERD_N = 4

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Row-wise argmax: out[row] = argmax_col(A[row,col])",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--ne0", type=int, default=NE0_ACTUAL, help="ne0 (cols, classes, reduced)"
    )
    parser.add_argument(
        "--ne1", type=int, default=NE1_ACTUAL, help="ne1 (rows, batch, tiled)"
    )
    parser.add_argument("--tile-rows", type=int, default=TILE_ROWS)
    parser.add_argument("--herd-n", type=int, default=HERD_N)
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )

    args = parser.parse_args()

    ne0_actual = args.ne0
    ne1_actual = args.ne1
    TILE_ROWS = args.tile_rows
    HERD_N = args.herd_n

    # Pad ne0 (cols) to multiple of 16 for DMA alignment
    ne0_padded = math.ceil(ne0_actual / 16) * 16
    # Pad ne1 (rows) to tile-aligned
    ne1_padded = math.ceil(ne1_actual / (TILE_ROWS * HERD_N)) * (TILE_ROWS * HERD_N)

    if args.verbose:
        print(f"ne0_actual={ne0_actual} (cols), ne1_actual={ne1_actual} (rows)")
        print(f"ne0_padded={ne0_padded}, ne1_padded={ne1_padded}")
        print(f"TILE_ROWS={TILE_ROWS}, HERD_N={HERD_N}")

    mlir_module = build_module(ne1_padded, ne0_padded, TILE_ROWS, HERD_N, ne0_actual)

    # Host-side padding (no air.actual_sizes needed; scalar loop uses ne0_actual)
    needs_padding = False

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Host data: (ne1, ne0) = (rows, cols) in numpy row-major
    input_a = np.zeros((ne1_padded, ne0_padded), dtype=np.float32)
    input_a[:ne1_actual, :ne0_actual] = (
        np.random.randn(ne1_actual, ne0_actual) * 4
    ).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        # Golden: argmax over ne0 (axis=1, cols) for each row
        argmax_golden = np.argmax(input_a[:ne1_actual, :ne0_actual], axis=1).astype(
            np.int32
        )

        argmax_golden_padded = np.zeros(ne1_padded, dtype=np.int32)
        argmax_golden_padded[:ne1_actual] = argmax_golden

        # Sample indices
        num_samples = min(100, ne1_actual)
        sampled_row_indices = np.random.choice(ne1_actual, num_samples, replace=False)
        boundary_rows = [0, ne1_actual - 1]
        if ne1_actual - TILE_ROWS + 1 > 0:
            boundary_rows.append(ne1_actual - TILE_ROWS + 1)
        sampled_row_indices = np.unique(
            np.concatenate([sampled_row_indices, boundary_rows])
        )

        sampled_indices = np.vstack([sampled_row_indices])
        sampled_values = argmax_golden_padded[sampled_row_indices]

        sampled_data = {
            "shape": (ne1_padded,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="xclbin",
            instance_name="argmax",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=0,
                atol=0,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="xclbin",
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
