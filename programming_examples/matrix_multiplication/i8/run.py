# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse

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

range_ = for_


@linalg_structured_op()
def block_matmul(
    A=TensorDef(T, S.a, S.c, S.f, S.d, S.g, S.i),
    B=TensorDef(T, S.b, S.c, S.e, S.f, S.i, S.h),
    C=TensorDef(U, S.b, S.a, S.e, S.d, S.g, S.h, output=True),
):
    domain(D.a, D.b, D.c, D.d, D.e, D.f, D.g, D.h, D.i)
    C[D.b, D.a, D.e, D.d, D.g, D.h] += (
        TypeFn.cast_signed(U, A[D.a, D.c, D.f, D.d, D.g, D.i])
    ) * (TypeFn.cast_signed(U, B[D.b, D.c, D.e, D.f, D.i, D.h]))


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

    mmul_mkn = [4, 8, 8]

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
                    zero_const = ConstantOp(IntegerAttr.get(xrt_dtype_out, 0), None)
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
    TILE_M = 64
    TILE_K_L2 = 256
    TILE_K_L1 = 64
    TILE_N = 128
    HERD_M = 4
    HERD_N = 4
    INPUT_DATATYPE = np.int8
    OUTPUT_DATATYPE = np.int16

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
    args = parser.parse_args()

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
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, args.m * args.k, dtype=np.int64).reshape(args.m, args.k) % 7
    input_a = input_a.astype(INPUT_DATATYPE)
    input_b = np.arange(0, args.k * args.n, dtype=np.int64).reshape(args.k, args.n) % 7
    input_b = input_b.astype(INPUT_DATATYPE)

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
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[2, 2],
            lower_linalg_to_func="mm.o",
        )
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
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[2, 2],
            lower_linalg_to_func="mm.o",
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
