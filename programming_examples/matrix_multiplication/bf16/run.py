# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import os
import sys
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.linalg import fill
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store, subview
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from air.extras import types as extrasT
from air.dialects.linalg.opdsl.lang import *
import air.dialects.linalg.opdsl.lang as linalg_lang

range_ = for_


@linalg_structured_op()
def block_matmul(
    A=TensorDef(linalg_lang.TV.T1, S.a, S.c, S.f, S.d, S.g, S.i),
    B=TensorDef(linalg_lang.TV.T2, S.b, S.c, S.e, S.f, S.i, S.h),
    C=TensorDef(linalg_lang.TV.U, S.b, S.a, S.e, S.d, S.g, S.h, output=True),
):
    domain(D.a, D.b, D.c, D.d, D.e, D.f, D.g, D.h, D.i)
    C[D.b, D.a, D.e, D.d, D.g, D.h] += (
        TypeFn.cast_signed(linalg_lang.TV.U, A[D.a, D.c, D.f, D.d, D.g, D.i])
    ) * (TypeFn.cast_signed(linalg_lang.TV.U, B[D.b, D.c, D.e, D.f, D.i, D.h]))


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
    np_dtype_in,
    np_dtype_out,
    arch="aie2",
    direct_codegen=False,
):
    assert m % tile_m == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert n % tile_n == 0
    a_size = [m, k]
    b_size = [k, n]
    c_size = [m, n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)

    # Architecture-specific matrix multiplication dimensions
    # aie2p with direct codegen uses 8x8x8, otherwise uses 4x8x4
    # aie2 always uses 4x8x4
    if arch == "aie2p" and direct_codegen:
        mmul_mkn = [8, 8, 8]  # For aie2p with BFP16 emulation (direct codegen)
    else:
        mmul_mkn = [4, 8, 4]  # For aie2 or aie2p without direct codegen

    # L3 MemRefTypes
    memrefTyA = MemRefType.get(a_size, xrt_dtype_in)
    memrefTyB = MemRefType.get(b_size, xrt_dtype_in)
    memrefTyOut = MemRefType.get(c_size, xrt_dtype_out)

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)
    a_l1_size = [
        1,
        1,
        tile_k_l1 // mmul_mkn[1],
        tile_m // mmul_mkn[0],
        mmul_mkn[0],
        mmul_mkn[1],
    ]
    b_l1_size = [
        1,
        1,
        tile_n // mmul_mkn[2],
        tile_k_l1 // mmul_mkn[1],
        mmul_mkn[1],
        mmul_mkn[2],
    ]
    c_l1_size = [
        1,
        1,
        tile_n // mmul_mkn[2],
        tile_m // mmul_mkn[0],
        mmul_mkn[0],
        mmul_mkn[2],
    ]
    c_herd_l1_size = [
        herd_m,
        herd_n,
        tile_n // mmul_mkn[2],
        tile_m // mmul_mkn[0],
        mmul_mkn[0],
        mmul_mkn[2],
    ]
    l1MemrefTyA = MemRefType.get(
        shape=a_l1_size,
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyB = MemRefType.get(
        shape=b_l1_size,
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    # Each core's result buffer is a subview of the global result buffer
    layout = StridedLayoutAttr.get(
        ShapedType.get_dynamic_size(),
        [
            tile_m * tile_n * herd_n,
            tile_m * tile_n,
            tile_m * mmul_mkn[2],
            mmul_mkn[0] * mmul_mkn[2],
            mmul_mkn[2],
            1,
        ],
    )
    l1MemrefTyC = MemRefType.get(
        shape=c_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
        layout=layout,
    )
    l1MemrefTyCHerd = MemRefType.get(
        shape=c_herd_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyOut)
    def matmul_bf16(arg0, arg1, arg2):

        launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

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
                name="matmul_seg",
                operands=[launch_ivx, launch_ivy, l3_a_data, l3_b_data, l3_c_data],
            )
            def segment_body(
                launch_ivx_s,
                launch_ivy_s,
                l3_a_data_s,
                l3_b_data_s,
                l3_c_data_s,
            ):
                # L2 MemRefTypes
                a_size_l2 = [herd_m, 1, tile_m, tile_k_l2]
                b_size_l2 = [1, herd_n, tile_k_l2, tile_n]
                c_size_l2 = [herd_m, herd_n, tile_m, tile_n]
                l2_mem_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L2)
                l2MemrefTyA = MemRefType.get(
                    shape=a_size_l2,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyB = MemRefType.get(
                    shape=b_size_l2,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyC = MemRefType.get(
                    shape=c_size_l2,
                    element_type=xrt_dtype_out,
                    memory_space=l2_mem_space,
                )
                # L2 memref allocs
                l2_a_data = AllocOp(l2MemrefTyA, [], [])
                l2_b_data = AllocOp(l2MemrefTyB, [], [])
                l2_c_data = AllocOp(l2MemrefTyC, [], [])
                # L1 memref allocs
                l1_a_data = AllocOp(l1MemrefTyA, [], [])
                l1_b_data = AllocOp(l1MemrefTyB, [], [])
                l1_c_data = AllocOp(l1MemrefTyCHerd, [], [])

                # Affine map for launch iv
                launch_ix_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                launch_iy_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_n * herd_n),
                        )
                    ],
                )
                launch_offset_x = affine_apply(launch_ix_map, [launch_ivx_s])
                launch_offset_y = affine_apply(launch_iy_map, [launch_ivy_s])

                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_a_data, l1_b_data, l1_c_data, l2_a_data, l2_b_data],
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
                    _l2_b,
                ):

                    l1_c_subview = subview(
                        _l1_c,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[
                            1,
                            1,
                            tile_n // mmul_mkn[2],
                            tile_m // mmul_mkn[0],
                            mmul_mkn[0],
                            mmul_mkn[2],
                        ],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    zero_const = ConstantOp(FloatAttr.get(xrt_dtype_out, 0.0), None)
                    zero_fill = fill(zero_const, outs=[l1_c_subview])

                for i in range_(0, k // tile_k_l2):
                    # Affine map for k (l2) loop iv
                    reduction_l2_iv_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(tile_k_l2),
                            )
                        ],
                    )
                    reduction_offset = affine_apply(reduction_l2_iv_map, [i])
                    dma_memcpy_nd(
                        l2_a_data,
                        l3_a_data_s,
                        src_offsets=[0, 0, launch_offset_x, reduction_offset],
                        src_sizes=[herd_m, 1, tile_m, tile_k_l2],
                        src_strides=[k * tile_m, tile_k_l2, k, 1],
                    )
                    dma_memcpy_nd(
                        l2_b_data,
                        l3_b_data_s,
                        src_offsets=[0, 0, reduction_offset, launch_offset_y],
                        src_sizes=[1, herd_n, tile_k_l2, tile_n],
                        src_strides=[n * tile_k_l2, tile_n, n, 1],
                    )

                    @herd(
                        name="herd_0",
                        sizes=[herd_m, herd_n],
                        operands=[
                            l1_a_data,
                            l1_b_data,
                            l1_c_data,
                            l2_a_data,
                            l2_b_data,
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
                        _l2_b,
                    ):
                        for j in range_(0, tile_k_l2 // tile_k_l1):
                            # Affine map for k (l1) loop iv
                            reduction_l1_iv_map = AffineMap.get(
                                0,
                                1,
                                [
                                    AffineExpr.get_mul(
                                        AffineSymbolExpr.get(0),
                                        AffineConstantExpr.get(tile_k_l1),
                                    )
                                ],
                            )
                            reduction_l1_offset = affine_apply(reduction_l1_iv_map, [j])
                            dma_memcpy_nd(
                                _l1_a,
                                _l2_a,
                                src_offsets=[_tx, 0, 0, 0, 0, reduction_l1_offset],
                                src_sizes=[
                                    1,
                                    1,
                                    tile_k_l1 // mmul_mkn[1],
                                    tile_m // mmul_mkn[0],
                                    mmul_mkn[0],
                                    mmul_mkn[1],
                                ],
                                src_strides=[
                                    tile_m * tile_k_l2,
                                    tile_m * tile_k_l2,
                                    mmul_mkn[1],
                                    tile_k_l2 * mmul_mkn[0],
                                    tile_k_l2,
                                    1,
                                ],
                            )
                            dma_memcpy_nd(
                                _l1_b,
                                _l2_b,
                                src_offsets=[0, _ty, 0, 0, reduction_l1_offset, 0],
                                src_sizes=[
                                    1,
                                    1,
                                    tile_n // mmul_mkn[2],
                                    tile_k_l1 // mmul_mkn[1],
                                    mmul_mkn[1],
                                    mmul_mkn[2],
                                ],
                                src_strides=[
                                    herd_n * tile_n * tile_k_l2,
                                    tile_n * tile_k_l2,
                                    mmul_mkn[2],
                                    tile_n * mmul_mkn[1],
                                    tile_n,
                                    1,
                                ],
                            )
                            l1_c_subview = subview(
                                _l1_c,
                                offsets=[_tx, _ty, 0, 0, 0, 0],
                                sizes=[
                                    1,
                                    1,
                                    tile_n // mmul_mkn[2],
                                    tile_m // mmul_mkn[0],
                                    mmul_mkn[0],
                                    mmul_mkn[2],
                                ],
                                strides=[1, 1, 1, 1, 1, 1],
                            )
                            matmul = block_matmul(_l1_a, _l1_b, outs=[l1_c_subview])
                            yield_([])

                    yield_([])

                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[
                        l1_a_data,
                        l1_b_data,
                        l1_c_data,
                        l2_a_data,
                        l2_b_data,
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
                    _l2_b,
                    _l2_c,
                ):
                    l1_c_subview = subview(
                        _l1_c,
                        offsets=[_tx, _ty, 0, 0, 0, 0],
                        sizes=[
                            1,
                            1,
                            tile_n // mmul_mkn[2],
                            tile_m // mmul_mkn[0],
                            mmul_mkn[0],
                            mmul_mkn[2],
                        ],
                        strides=[1, 1, 1, 1, 1, 1],
                    )
                    dma_memcpy_nd(
                        _l2_c,
                        _l1_c,
                        dst_offsets=[_tx, _ty, 0, 0],
                        dst_sizes=[1, 1, tile_m, tile_n],
                        dst_strides=[
                            herd_n * tile_m * tile_n,
                            tile_m * tile_n,
                            tile_n,
                            1,
                        ],
                        src_offsets=[_tx, _ty, 0, 0, 0, 0],
                        src_sizes=[
                            1,
                            1,
                            tile_m // mmul_mkn[0],
                            mmul_mkn[0],
                            tile_n // mmul_mkn[2],
                            mmul_mkn[2],
                        ],
                        src_strides=[
                            herd_n * tile_m * tile_n,
                            tile_m * tile_n,
                            mmul_mkn[2] * mmul_mkn[0],
                            mmul_mkn[2],
                            tile_m * mmul_mkn[2],
                            1,
                        ],
                    )

                dma_memcpy_nd(
                    l3_c_data_s,
                    l2_c_data,
                    dst_offsets=[launch_offset_x, launch_offset_y],
                    dst_sizes=[herd_m * tile_m, herd_n * tile_n],
                    dst_strides=[n, 1],
                    src_offsets=[0, 0, 0, 0],
                    src_sizes=[herd_m, tile_m, herd_n, tile_n],
                    src_strides=[tile_m * herd_n * tile_n, tile_n, tile_m * tile_n, 1],
                )

                DeallocOp(l2_a_data)
                DeallocOp(l2_b_data)
                DeallocOp(l2_c_data)
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_c_data)


if __name__ == "__main__":
    # Default values.
    M = 512
    K = 512
    N = 512
    TILE_M = 128
    TILE_K_L2 = 128
    TILE_K_L1 = 32
    TILE_N = 64
    HERD_M = 4
    HERD_N = 4
    INPUT_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16  # also supports np.float32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
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
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
    )
    parser.add_argument(
        "--direct-codegen",
        action="store_true",
        help="Enable direct code generation mode (compiles directly without extra kernel library)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2",
        help="Target AIE architecture (aie2 or aie2p)",
    )
    args = parser.parse_args()

    # Check for PEANO_INSTALL_DIR if direct codegen is enabled
    if args.direct_codegen:
        if not os.environ.get("PEANO_INSTALL_DIR"):
            print(
                "Error: PEANO_INSTALL_DIR environment variable is not set.",
                file=sys.stderr,
            )
            print("Peano is needed for direct code generation mode.", file=sys.stderr)
            sys.exit(1)

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
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
        args.arch,
        args.direct_codegen,
    )

    # Vectorization - only run if direct codegen mode is enabled
    if args.direct_codegen:
        transform_ir_string = """
        transform.with_pdl_patterns {
        ^bb0(%arg0: !pdl.operation):
            transform.sequence %arg0 : !pdl.operation failures(propagate) {
            ^bb1(%arg1: !pdl.operation):

                %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                transform.apply_patterns to %func0 {
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
                } : !pdl.operation


                %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation

                %inner_most_matmul, %vec_loops:3 =
                  transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
                  : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)  
                %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
                  transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
                  : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)  
                transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !pdl.operation
                transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !pdl.operation

                %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                %inner_most_fills, %vec_fill_loops:2 =
                  transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
                  : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)

                %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                %vectorized_herds = transform.air.herd_vectorize %herds
                
                %herd1, %herd2, %herd3 = transform.split_handle %vectorized_herds : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
                %scf_fors = transform.structured.match ops{["scf.for"]} in %herd2 : (!pdl.operation) -> !pdl.operation
                %for1, %for2, %for3, %for4 = transform.split_handle %scf_fors : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

                // Apply LICM to the innermost loop to hoist invariant reads
                transform.apply_licm to %for4 : !pdl.operation

                %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                transform.apply_patterns to %func1 {
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
                    transform.apply_patterns.memref.fold_memref_alias_ops
                } : !pdl.operation
                
                // Eliminate redundant vector.transfer_read operations
                %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1
                
                // Hoist loop-invariant vector transfers out of innermost loop
                %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                %vectorized_herds_1 = transform.air.herd_vectorize %herds_1
                %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %vectorized_herds_1 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
                %all_reads_in_herd2 = transform.structured.match ops{["vector.transfer_read"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
                %all_writes_in_herd2 = transform.structured.match ops{["vector.transfer_write"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
                
                // Split handles to get individual read/write operations
                %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
                %for1_1, %for2_1, %for3_1, %for4_1 = transform.split_handle %scf_fors_1 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
                // The innermost loop has 4 read-write pairs accessing arg22
                %read0, %read1, %read2, %read3, %read4, %read5, %read6, %read7 = transform.split_handle %all_reads_in_herd2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
                %write0, %write1, %write2, %write3 = transform.split_handle %all_writes_in_herd2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
                
                %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]}
                
                // Hoist each read/write pair from the innermost loop (%for1_1)
                // Pair 1: reads[2] (%8) and writes[0] (%13) - accessing [arg27, arg26]
                %for1_1_updated = transform.air.hoist_loop_invariant_transfers %read2, %write0, %for1_1
                // // Pair 2: reads[4] (%17) and writes[1] (%22) - accessing [arg27+1, arg26]
                %for1_1_updated_1 = transform.air.hoist_loop_invariant_transfers %read4, %write1, %for1_1_updated
                // Pair 3: reads[6] (%27) and writes[2] (%32) - accessing [arg27, arg26+1]
                %for1_1_updated_2 = transform.air.hoist_loop_invariant_transfers %read6, %write2, %for1_1_updated_1
                // Pair 4: reads[7] (%38) and writes[3] (%43) - accessing [arg27+1, arg26+1]
                %for1_1_updated_3 = transform.air.hoist_loop_invariant_transfers %read7, %write3, %for1_1_updated_2

                %for1_1_updated_4 = transform.air.flatten_for_iter_args %for1_1_updated_3
                %for1_1_updated_5 = transform.air.hoist_vector_transfer_pointers %for1_1_updated_4
 
                %fors_to_hoist_ptrs = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!pdl.operation) -> !pdl.operation
                %for_ptr1, %for_ptr2, %for_ptr3, %for_ptr4 = transform.split_handle %fors_to_hoist_ptrs: (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
 
                // Hoist the 4 extf/truncf pairs from the innermost loop
                %all_extf_loop = transform.structured.match ops{["arith.extf"]} in %for_ptr1 : (!pdl.operation) -> !pdl.operation
                %all_truncf_loop = transform.structured.match ops{["arith.truncf"]} in %for_ptr1 : (!pdl.operation) -> !pdl.operation
                
                // Split to get individual operations (4 extf total)
                %extf_bf16_1, %extf_bf16_2, %extf_bf16_3, %extf_bf16_4 = transform.split_handle %all_extf_loop : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
                
                // The 4 truncf ops correspond to the 4 vector.contract results
                %truncf_1, %truncf_2, %truncf_3, %truncf_4 = transform.split_handle %all_truncf_loop : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
                
                // Hoist first pair
                %for1_1_hoisted_1 = transform.air.hoist_cast_pair %extf_bf16_1, %truncf_1, %for_ptr1
                
                // Re-match and hoist second pair
                %all_extf_loop_2 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_1 : (!pdl.operation) -> !pdl.operation
                %all_truncf_loop_2 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_1 : (!pdl.operation) -> !pdl.operation
                %extf_bf16_2_new, %e2_5, %e2_6 = transform.split_handle %all_extf_loop_2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
                %truncf_2_1, %truncf_2_2, %truncf_2_3 = transform.split_handle %all_truncf_loop_2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
                %for1_1_hoisted_2 = transform.air.hoist_cast_pair %extf_bf16_2_new, %truncf_2_1, %for1_1_hoisted_1
                
                // Re-match and hoist third pair
                %all_extf_loop_3 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_2 : (!pdl.operation) -> !pdl.operation
                %all_truncf_loop_3 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_2 : (!pdl.operation) -> !pdl.operation
                %extf_bf16_3_new, %e3_7 = transform.split_handle %all_extf_loop_3 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
                %truncf_3_1, %truncf_3_2 = transform.split_handle %all_truncf_loop_3 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
                %for1_1_hoisted_3 = transform.air.hoist_cast_pair %extf_bf16_3_new, %truncf_3_1, %for1_1_hoisted_2
                
                // Re-match and hoist fourth pair
                %all_extf_loop_4 = transform.structured.match ops{["arith.extf"]} in %for1_1_hoisted_3 : (!pdl.operation) -> !pdl.operation
                %all_truncf_loop_4 = transform.structured.match ops{["arith.truncf"]} in %for1_1_hoisted_3 : (!pdl.operation) -> !pdl.operation
                %for1_1_hoisted_final = transform.air.hoist_cast_pair %all_extf_loop_4, %all_truncf_loop_4, %for1_1_hoisted_3

                %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
                transform.apply_patterns to %func2 {
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
                    transform.apply_patterns.memref.fold_memref_alias_ops
                } : !pdl.operation
            }
        }
                
        """
        transform_ir = Module.parse(transform_ir_string, context=mlir_module.context)
        run_transform(transform_ir, mlir_module)

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, args.m * args.k, dtype=INPUT_DATATYPE).reshape(
        args.m, args.k
    )
    input_b = np.arange(0, args.k * args.n, dtype=INPUT_DATATYPE).reshape(
        args.k, args.n
    )

    if args.compile_mode == "compile-and-run":
        # Stochastically sample num_sample results, and pass to XRTRunner backend for verification.
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, args.m, num_samples),  # i indices
                np.random.randint(0, args.n, num_samples),  # j indices
            ]
        )

        # Compute reference results for sampled indices
        sampled_values = np.array(
            [
                np.sum(
                    (
                        input_a[i, :].astype(OUTPUT_DATATYPE)
                        * input_b[:, j].astype(OUTPUT_DATATYPE)
                    ),
                    dtype=OUTPUT_DATATYPE,
                )
                for i, j in zip(*sampled_indices)
            ],
            dtype=OUTPUT_DATATYPE,
        )

        # Store as a dictionary
        sampled_data = {
            "shape": (args.m, args.n),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        ###### Compile and test
        runner_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        }
        # Only use external kernel library if NOT in direct codegen mode
        if not args.direct_codegen:
            runner_kwargs["lower_linalg_to_func"] = "mm.o"

        runner = XRTRunner(**runner_kwargs)
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e0,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
        }
        # Only use external kernel library if NOT in direct codegen mode
        if not args.direct_codegen:
            backend_kwargs["lower_linalg_to_func"] = "mm.o"

        backend = XRTBackend(**backend_kwargs)
        module_function = backend.compile(mlir_module)

        backend.unload()
