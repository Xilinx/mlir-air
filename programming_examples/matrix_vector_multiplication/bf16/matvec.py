# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Matrix-vector multiplication (GEMV): C[M] = A[M,K] @ B[K]
# BF16 input/output, accfloat accumulation in the kernel.
#
# Follows the IRON GEMV design style:
#   - Full K vector B loaded once into L1 (no K tiling)
#   - A rows streamed through L2→L1 in m_input-sized chunks
#   - Output C accumulated in L1, written back once
#
# The outer M tiling is handled by `launch` (each launch instance handles
# tile_m_l2 output rows). The inner m_input loop is inside the herd.
# L2 holds tile_m_l2 * K elements of A — must fit in MemTile (256KB).

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(m, k, tile_m_l2, m_input, np_dtype_in, np_dtype_out):
    assert m % tile_m_l2 == 0, f"M ({m}) must be divisible by tile_m_l2 ({tile_m_l2})"
    assert (
        tile_m_l2 % m_input == 0
    ), f"tile_m_l2 ({tile_m_l2}) must be divisible by m_input ({m_input})"
    assert k % 64 == 0, f"K ({k}) must be divisible by 64 (vector width)"

    # Check L2 capacity: A tile must fit in MemTile
    l2_a_bytes = tile_m_l2 * k * 2  # bf16 = 2 bytes
    assert l2_a_bytes <= 256 * 1024, (
        f"L2 A tile ({l2_a_bytes} bytes) exceeds MemTile capacity (256KB). "
        f"Reduce tile_m_l2 or K."
    )

    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)

    # L3 MemRefTypes
    memrefTyA = MemRefType.get([m, k], xrt_dtype_in)  # matrix A[M,K]
    memrefTyB = MemRefType.get([k], xrt_dtype_in)  # vector B[K]
    memrefTyC = MemRefType.get([m], xrt_dtype_out)  # output C[M]

    # L2 MemRefTypes
    # L2 holds full tile_m_l2 rows of A so the inner loop can be inside the herd
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyA = MemRefType.get(
        shape=[tile_m_l2, k], element_type=xrt_dtype_in, memory_space=l2_mem_space
    )
    l2MemrefTyB = MemRefType.get(
        shape=[k], element_type=xrt_dtype_in, memory_space=l2_mem_space
    )
    l2MemrefTyC = MemRefType.get(
        shape=[tile_m_l2], element_type=xrt_dtype_out, memory_space=l2_mem_space
    )

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1MemrefTyA = MemRefType.get(
        shape=[m_input, k], element_type=xrt_dtype_in, memory_space=l1_mem_space
    )
    l1MemrefTyB = MemRefType.get(
        shape=[k], element_type=xrt_dtype_in, memory_space=l1_mem_space
    )
    l1MemrefTyC = MemRefType.get(
        shape=[tile_m_l2], element_type=xrt_dtype_out, memory_space=l1_mem_space
    )

    # External kernel declarations
    # matvec_vectorized_bf16_bf16(i32 m, i32 k, i32 row_offset, A, B, C)
    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([T.i32(), T.i32(), T.i32(), l1MemrefTyA, l1MemrefTyB, l1MemrefTyC], []),
        visibility="private",
    )
    # linalg_fill_bf16(bf16 val, C) — matches VecMat pattern
    linalg_fill_func = FuncOp(
        "linalg_fill_bf16",
        ([xrt_dtype_out, l1MemrefTyC], []),
        visibility="private",
    )
    for func in [matvec_func, linalg_fill_func]:
        func.attributes["link_with"] = StringAttr.get("mv.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyC)
    def matvec_bf16(arg0, arg1, arg2):

        # Each launch instance handles tile_m_l2 output rows.
        launch_size = [m // tile_m_l2, 1]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a_data,
            l3_b_data,
            l3_c_data,
        ):

            @segment(
                name="matvec_bf16_0",
                operands=[launch_ivx, l3_a_data, l3_b_data, l3_c_data],
            )
            def segment_body(
                launch_ivx_s,
                l3_a_data_s,
                l3_b_data_s,
                l3_c_data_s,
            ):
                # Affine map for launch_ivx: row offset = launch_ivx * tile_m_l2
                launch_ivx_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m_l2),
                        )
                    ],
                )
                launch_offset_m = affine_apply(launch_ivx_map, [launch_ivx_s])

                # L2 memref allocs
                l2_a_data = AllocOp(l2MemrefTyA, [], [])
                l2_b_data = AllocOp(l2MemrefTyB, [], [])
                l2_c_data = AllocOp(l2MemrefTyC, [], [])
                # L1 memref allocs
                l1_a_data = AllocOp(l1MemrefTyA, [], [])
                l1_b_data = AllocOp(l1MemrefTyB, [], [])
                l1_c_data = AllocOp(l1MemrefTyC, [], [])

                # --- Load B: L3 → L2 ---
                dma_memcpy_nd(
                    l2_b_data,
                    l3_b_data_s,
                    src_offsets=[],
                    src_sizes=[k],
                    src_strides=[1],
                )

                # --- Load A tile: L3 → L2 (all tile_m_l2 rows at once) ---
                dma_memcpy_nd(
                    l2_a_data,
                    l3_a_data_s,
                    src_offsets=[launch_offset_m, 0],
                    src_sizes=[tile_m_l2, k],
                    src_strides=[k, 1],
                )

                # --- Zero-fill C in L1 ---
                @herd(
                    name="herd_0",
                    sizes=[1, 1],
                    operands=[l1_c_data],
                )
                def herd_body(_tx, _ty, _sx, _sy, _l1_c):
                    zero_const = ConstantOp(FloatAttr.get(xrt_dtype_out, 0), None)
                    CallOp(linalg_fill_func, [zero_const, _l1_c])

                herd_body.attributes["link_with"] = StringAttr.get("mv.o")

                # Affine map: s0 * constant
                def make_scale_map(scale):
                    return AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(scale),
                            )
                        ],
                    )

                # --- Compute: inner m_input loop inside herd ---
                @herd(
                    name="herd_0",
                    sizes=[1, 1],
                    operands=[
                        l1_a_data,
                        l1_b_data,
                        l1_c_data,
                        l2_a_data,
                        l2_b_data,
                    ],
                )
                def herd_body(_tx, _ty, _sx, _sy, _l1_a, _l1_b, _l1_c, _l2_a, _l2_b):
                    # DMA B: L2 → L1 (once, before the loop)
                    dma_memcpy_nd(
                        _l1_b,
                        _l2_b,
                        src_offsets=[],
                        src_sizes=[k],
                        src_strides=[1],
                    )

                    for j_m in range_(0, tile_m_l2 // m_input):
                        # Compute L2 row offset for A: j_m * m_input
                        l2_a_row_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(m_input),
                                )
                            ],
                        )
                        l2_a_row_offset = affine_apply(l2_a_row_map, [j_m])

                        # DMA A: L2[j_m*m_input:, K] → L1
                        dma_memcpy_nd(
                            _l1_a,
                            _l2_a,
                            src_offsets=[l2_a_row_offset, 0],
                            src_sizes=[m_input, k],
                            src_strides=[k, 1],
                        )

                        # Compute row_offset for kernel
                        row_offset_i32 = arith.index_cast(T.i32(), l2_a_row_offset)
                        m_const = ConstantOp(IntegerAttr.get(T.i32(), m_input), None)
                        k_const = ConstantOp(IntegerAttr.get(T.i32(), k), None)

                        CallOp(
                            matvec_func,
                            [
                                m_const,
                                k_const,
                                row_offset_i32,
                                _l1_a,
                                _l1_b,
                                _l1_c,
                            ],
                        )

                        yield_([])

                herd_body.attributes["link_with"] = StringAttr.get("mv.o")

                # --- Writeback C: L1 → L2 ---
                @herd(
                    name="herd_0",
                    sizes=[1, 1],
                    operands=[l1_c_data, l2_c_data],
                )
                def herd_body(_tx, _ty, _sx, _sy, _l1_c, _l2_c):
                    dma_memcpy_nd(
                        _l2_c,
                        _l1_c,
                        src_offsets=[],
                        src_sizes=[tile_m_l2],
                        src_strides=[1],
                    )

                herd_body.attributes["link_with"] = StringAttr.get("mv.o")

                # --- Writeback C: L2 → L3 ---
                dma_memcpy_nd(
                    l3_c_data_s,
                    l2_c_data,
                    dst_offsets=[launch_offset_m],
                    dst_sizes=[tile_m_l2],
                    dst_strides=[1],
                )

                DeallocOp(l2_a_data)
                DeallocOp(l2_b_data)
                DeallocOp(l2_c_data)
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_c_data)


if __name__ == "__main__":
    # Default values (M=2048, K=8192 matching IRON test case 1)
    M = 2048
    K = 8192
    TILE_M_L2 = 2048
    M_INPUT = 1
    INPUT_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="matvec.py",
        description="Builds, runs, and tests the bf16 matrix-vector multiplication (GEMV) example",
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
        "--m",
        type=int,
        default=M,
        help="M dimension (matrix rows / output size)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="K dimension (matrix columns / vector length)",
    )
    parser.add_argument(
        "--tile-m-l2",
        type=int,
        default=TILE_M_L2,
        help="Number of output rows per L2 output tile",
    )
    parser.add_argument(
        "--m-input",
        type=int,
        default=M_INPUT,
        help="Number of matrix rows per kernel call",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-and-xclbin"],
        dest="compile_mode",
        default="compile-and-run",
        help="compile-and-run (default): compile and validate; compile-and-xclbin: generate xclbin only",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.m,
        args.k,
        args.tile_m_l2,
        args.m_input,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        # Generate test data matching IRON reference.py pattern
        np.random.seed(42)
        input_a = (np.random.randn(args.m, args.k) * 4).astype(INPUT_DATATYPE)
        input_b = (np.random.randn(args.k) * 4).astype(INPUT_DATATYPE)
        # Compute reference in float32 for accuracy, cast back to output dtype
        output_c = np.dot(
            input_a.astype(np.float32), input_b.astype(np.float32)
        ).astype(OUTPUT_DATATYPE)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="matvec_bf16",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                expected_outputs=[output_c],
                rtol=0.04,
                atol=1e-3,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
