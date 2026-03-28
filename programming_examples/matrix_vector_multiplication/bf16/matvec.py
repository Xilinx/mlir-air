# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Matrix-vector multiplication (GEMV): C[M] = A[M,K] @ B[K]
# BF16 input/output, accfloat accumulation in the kernel.
#
# L2 (MemTile) staging for A and C; B goes L3→L1 directly.
# Multi-column support: herd_m AIE columns process independent row chunks
# in parallel. Each column handles tile_m output rows per launch.
# The outer M tiling is handled by `launch` (each launch instance handles
# herd_m * tile_m output rows). The inner m_input loop is inside the herd.

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
def build_module(m, k, tile_m, m_input, herd_m, np_dtype_in, np_dtype_out):
    assert (
        m % (tile_m * herd_m) == 0
    ), f"M ({m}) must be divisible by tile_m * herd_m ({tile_m * herd_m})"
    assert (
        tile_m % m_input == 0
    ), f"tile_m ({tile_m}) must be divisible by m_input ({m_input})"
    assert k % 64 == 0, f"K ({k}) must be divisible by 64 (vector width)"

    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)

    # L3 MemRefTypes
    memrefTyA = MemRefType.get([m, k], xrt_dtype_in)
    memrefTyB = MemRefType.get([k], xrt_dtype_in)
    memrefTyC = MemRefType.get([m], xrt_dtype_out)

    # L2 MemRefTypes
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyA = MemRefType.get(
        shape=[herd_m, tile_m, k],
        element_type=xrt_dtype_in,
        memory_space=l2_mem_space,
    )
    l2MemrefTyC = MemRefType.get(
        shape=[herd_m, tile_m],
        element_type=xrt_dtype_out,
        memory_space=l2_mem_space,
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
        shape=[tile_m],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )

    # External kernel declarations
    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([T.i32(), T.i32(), T.i32(), l1MemrefTyA, l1MemrefTyB, l1MemrefTyC], []),
        visibility="private",
    )
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

        # Each launch handles herd_m * tile_m output rows.
        launch_size = [m // tile_m // herd_m, 1]

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
                # Affine map: row offset = launch_ivx * tile_m * herd_m
                launch_ivx_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                launch_offset_m = affine_apply(launch_ivx_map, [launch_ivx_s])

                # Alloc L2 (A and C; B skips L2) and L1
                l2_a_data = AllocOp(l2MemrefTyA, [], [])
                l2_c_data = AllocOp(l2MemrefTyC, [], [])
                l1_a_data = AllocOp(l1MemrefTyA, [], [])
                l1_b_data = AllocOp(l1MemrefTyB, [], [])
                l1_c_data = AllocOp(l1MemrefTyC, [], [])

                # L3→L2: A (all herd_m * tile_m rows)
                dma_memcpy_nd(
                    l2_a_data,
                    l3_a_data_s,
                    src_offsets=[0, launch_offset_m, 0],
                    src_sizes=[herd_m, tile_m, k],
                    src_strides=[tile_m * k, k, 1],
                )

                # Single compute herd: zero-fill + loop(A L2→L1 + B L3→L1, kernel) + C L1→L2
                # B skips L2 — loaded directly L3→L1 inside loop for channel hoisting
                @herd(
                    name="herd_0",
                    sizes=[herd_m, 1],
                    operands=[
                        l1_a_data,
                        l1_b_data,
                        l1_c_data,
                        l2_a_data,
                        l3_b_data_s,
                        l2_c_data,
                    ],
                )
                def herd_body(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _l1_a,
                    _l1_b,
                    _l1_c,
                    _l2_a,
                    _l3_b,
                    _l2_c,
                ):
                    # Zero-fill C
                    zero = ConstantOp(FloatAttr.get(xrt_dtype_out, 0), None)
                    CallOp(linalg_fill_func, [zero, _l1_c])

                    for j_m in range_(0, tile_m // m_input):
                        j_m_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(m_input),
                                )
                            ],
                        )
                        j_m_offset = affine_apply(j_m_map, [j_m])

                        # L3→L1: B directly (inside loop for channel hoisting)
                        dma_memcpy_nd(
                            _l1_b,
                            _l3_b,
                            src_offsets=[],
                            src_sizes=[k],
                            src_strides=[1],
                        )

                        # L2→L1: A[_tx, j_m*m_input:, :]
                        dma_memcpy_nd(
                            _l1_a,
                            _l2_a,
                            src_offsets=[_tx, j_m_offset, 0],
                            src_sizes=[1, m_input, k],
                            src_strides=[tile_m * k, k, 1],
                        )

                        # Kernel
                        row_offset_i32 = arith.index_cast(T.i32(), j_m_offset)
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

                    # L1→L2: C writeback (per-column offset on L2 side)
                    dma_memcpy_nd(
                        _l2_c,
                        _l1_c,
                        dst_offsets=[_tx, 0],
                        dst_sizes=[1, tile_m],
                        dst_strides=[tile_m, 1],
                        src_offsets=[],
                        src_sizes=[tile_m],
                        src_strides=[1],
                    )

                herd_body.attributes["link_with"] = StringAttr.get("mv.o")

                # L2→L3: C
                dma_memcpy_nd(
                    l3_c_data_s,
                    l2_c_data,
                    dst_offsets=[launch_offset_m],
                    dst_sizes=[herd_m * tile_m],
                    dst_strides=[1],
                    src_offsets=[0, 0],
                    src_sizes=[herd_m, tile_m],
                    src_strides=[tile_m, 1],
                )

                DeallocOp(l2_a_data)
                DeallocOp(l2_c_data)
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_c_data)


if __name__ == "__main__":
    # Default values (M=2048, K=8192, 4 AIE columns)
    M = 2048
    K = 8192
    TILE_M = 4
    M_INPUT = 1
    HERD_M = 4
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
        default=TILE_M,
        dest="tile_m",
        help="Number of output rows per tile per column",
    )
    parser.add_argument(
        "--m-input",
        type=int,
        default=M_INPUT,
        help="Number of matrix rows per kernel call",
    )
    parser.add_argument(
        "--herd-m",
        type=int,
        default=HERD_M,
        help="Number of AIE columns (parallel compute tiles along M dimension)",
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
    parser.add_argument(
        "--debug-ir",
        action="store_true",
        dest="debug_ir",
        help="Emit IR after each pass into debug_ir/ directory",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.m,
        args.k,
        args.tile_m,
        args.m_input,
        args.herd_m,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        np.random.seed(42)
        input_a = (np.random.randn(args.m, args.k) * 4).astype(INPUT_DATATYPE)
        input_b = (np.random.randn(args.k) * 4).astype(INPUT_DATATYPE)
        output_c = np.dot(
            input_a.astype(np.float32), input_b.astype(np.float32)
        ).astype(OUTPUT_DATATYPE)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            output_format=args.output_format,
            instance_name="matvec_bf16",
            debug_ir=args.debug_ir,
            use_lock_race_condition_fix=True,
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
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            use_lock_race_condition_fix=True,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
