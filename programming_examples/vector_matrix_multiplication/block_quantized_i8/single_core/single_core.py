# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_


@module_builder
def build_module(k, n, bs, tile_k, tile_n, np_dtype_in, np_dtype_acc, np_dtype_out):
    assert k % tile_k == 0
    assert n % tile_n == 0
    a_size = [k]
    a_s_size = [k // bs]
    b_size = [k, n]
    b_s_size = [k // bs, n]
    c_size = [n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_acc = type_mapper(np_dtype_acc)
    xrt_dtype_out = type_mapper(np_dtype_out)

    mmul_mkn = [1, 16, 8]

    # L3 MemRefTypes
    memrefTyA = MemRefType.get(a_size, xrt_dtype_in)
    memrefTyAS = MemRefType.get(a_s_size, xrt_dtype_out)
    memrefTyB = MemRefType.get(b_size, xrt_dtype_in)
    memrefTyBS = MemRefType.get(b_s_size, xrt_dtype_out)
    memrefTyOut = MemRefType.get(c_size, xrt_dtype_out)

    Channel("aL3ToL2")
    Channel("bL3ToL2")
    Channel("aL2ToL1")
    Channel("bL2ToL1")
    Channel("cL1ToL2")
    Channel("cL2ToL3")

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

    a_l1_size = [tile_k // mmul_mkn[1], mmul_mkn[1]]
    a_s_l1_size = [tile_k // bs]
    b_l1_size = [tile_n // mmul_mkn[2], tile_k // mmul_mkn[1], mmul_mkn[1], mmul_mkn[2]]
    b_s_l1_size = [tile_n // mmul_mkn[2], tile_k // bs, mmul_mkn[2]]
    l1MemrefTyA = MemRefType.get(
        shape=a_l1_size,
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyAS = MemRefType.get(
        shape=a_s_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyB = MemRefType.get(
        shape=b_l1_size,
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    l1MemrefTyBS = MemRefType.get(
        shape=b_s_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    c_l1_size = [tile_n // mmul_mkn[2], mmul_mkn[2]]
    l1MemrefTyC = MemRefType.get(
        shape=c_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    linalg_fill_func = FuncOp(
        "linalg_fill_i32_view16x8xi32as2",
        ([xrt_dtype_out, l1MemrefTyC], []),
        visibility="private",
    )
    vecmat_func = FuncOp(
        "vecmat_i8_f32_i32_32",
        ([l1MemrefTyA, l1MemrefTyAS, l1MemrefTyB, l1MemrefTyBS, l1MemrefTyC], []),
        visibility="private",
    )
    for func in [linalg_fill_func, vecmat_func]:
        func.attributes["link_with"] = StringAttr.get("vm.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyA, memrefTyAS, memrefTyB, memrefTyBS, memrefTyOut)
    def vecmat_i8(arg0, arg1, arg2, arg3, arg4):

        launch_size = [1, n // tile_n]

        @launch(operands=[arg0, arg1, arg2, arg3, arg4], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a_data,
            l3_a_scale,
            l3_b_data,
            l3_b_scale,
            l3_c_data,
        ):
            ChannelPut(
                "aL3ToL2",
                l3_a_data,
                offsets=[],
                sizes=[],
                strides=[],
            )
            ChannelPut(
                "aL3ToL2",
                l3_a_scale,
                offsets=[],
                sizes=[],
                strides=[],
            )

            # Affine map for launch iv
            launch_ivy_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(tile_n),
                    )
                ],
            )
            launch_offset_y = affine_apply(launch_ivy_map, [launch_ivy])
            ChannelPut(
                "bL3ToL2",
                l3_b_data,
                offsets=[0, launch_offset_y],
                sizes=[k, tile_n],
                strides=[n, 1],
            )
            ChannelPut(
                "bL3ToL2",
                l3_b_scale,
                offsets=[0, launch_offset_y],
                sizes=[k // bs, tile_n],
                strides=[n, 1],
            )
            ChannelGet(
                "cL2ToL3",
                l3_c_data,
                offsets=[launch_offset_y],
                sizes=[tile_n],
                strides=[1],
            )

            @segment(name="vecmat_i8_0")
            def segment_body():
                # L2 MemRefTypes
                a_size_l2 = [k]
                a_s_size_l2 = [k // bs]
                b_size_l2 = [k, tile_n]
                b_s_size_l2 = [k // bs, tile_n]
                c_size_l2 = [tile_n]
                l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
                l2MemrefTyA = MemRefType.get(
                    shape=a_size_l2,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyAS = MemRefType.get(
                    shape=a_s_size_l2,
                    element_type=xrt_dtype_out,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyB = MemRefType.get(
                    shape=b_size_l2,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyBS = MemRefType.get(
                    shape=b_s_size_l2,
                    element_type=xrt_dtype_out,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyC = MemRefType.get(
                    shape=c_size_l2,
                    element_type=xrt_dtype_out,
                    memory_space=l2_mem_space,
                )
                l2_a_data = AllocOp(l2MemrefTyA, [], [])
                l2_a_scale = AllocOp(l2MemrefTyAS, [], [])

                ChannelGet(
                    "aL3ToL2",
                    l2_a_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )
                ChannelGet(
                    "aL3ToL2",
                    l2_a_scale,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )

                l2_b_data = AllocOp(l2MemrefTyB, [], [])
                l2_b_scale = AllocOp(l2MemrefTyBS, [], [])

                ChannelGet(
                    "bL3ToL2",
                    l2_b_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )
                ChannelGet(
                    "bL3ToL2",
                    l2_b_scale,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )

                # L2 to L1 data packing and unpacking, using offsets, sizes and strides
                a_l2l1_pack_size = [tile_k]
                a_l2l1_pack_stride = [1]
                a_s_l2l1_pack_size = [tile_k // bs]
                a_s_l2l1_pack_stride = [1]
                b_l2l1_pack_size = [tile_n // mmul_mkn[2], tile_k, mmul_mkn[2]]
                b_l2l1_pack_stride = [mmul_mkn[2], tile_n, 1]
                b_s_l2l1_pack_size = [tile_n // mmul_mkn[2], tile_k // bs, mmul_mkn[2]]
                b_s_l2l1_pack_stride = [mmul_mkn[2], tile_n, 1]

                for_iv_map_data = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_k),
                        )
                    ],
                )
                for_iv_map_scale = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_k // bs),
                        )
                    ],
                )

                for i in range_(0, k // tile_k):
                    for_iv_data_offset = affine_apply(for_iv_map_data, [i])
                    a_l2l1_pack_offset = [for_iv_data_offset]
                    ChannelPut(
                        "aL2ToL1",
                        l2_a_data,
                        offsets=a_l2l1_pack_offset,
                        sizes=a_l2l1_pack_size,
                        strides=a_l2l1_pack_stride,
                    )
                    for_iv_scale_offset = affine_apply(for_iv_map_scale, [i])
                    a_s_l2l1_pack_offset = [for_iv_scale_offset]
                    ChannelPut(
                        "aL2ToL1",
                        l2_a_scale,
                        offsets=a_s_l2l1_pack_offset,
                        sizes=a_s_l2l1_pack_size,
                        strides=a_s_l2l1_pack_stride,
                    )
                    yield_([])

                for i in range_(0, k // tile_k):
                    for_iv_data_offset = affine_apply(for_iv_map_data, [i])
                    b_l2l1_pack_offset = [0, for_iv_data_offset, 0]
                    ChannelPut(
                        "bL2ToL1",
                        l2_b_data,
                        offsets=b_l2l1_pack_offset,
                        sizes=b_l2l1_pack_size,
                        strides=b_l2l1_pack_stride,
                    )
                    for_iv_scale_offset = affine_apply(for_iv_map_scale, [i])
                    b_s_l2l1_pack_offset = [0, for_iv_scale_offset, 0]
                    ChannelPut(
                        "bL2ToL1",
                        l2_b_scale,
                        offsets=b_s_l2l1_pack_offset,
                        sizes=b_s_l2l1_pack_size,
                        strides=b_s_l2l1_pack_stride,
                    )
                    yield_([])

                l2_c_data = AllocOp(l2MemrefTyC, [], [])
                ChannelGet(
                    "cL1ToL2",
                    l2_c_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )

                @herd(name="herd_0", sizes=[1, 1])
                def herd_body(_tx, _ty, _sx, _sy):

                    l1_c_data = AllocOp(l1MemrefTyC, [], [])
                    zero_const = ConstantOp(FloatAttr.get(xrt_dtype_out, 0.0), None)
                    zero_fill = CallOp(linalg_fill_func, [zero_const, l1_c_data])

                    for i in range_(0, k, tile_k):
                        l1_a_data = AllocOp(l1MemrefTyA, [], [])
                        l1_a_scale = AllocOp(l1MemrefTyAS, [], [])

                        ChannelGet(
                            "aL2ToL1",
                            l1_a_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )
                        ChannelGet(
                            "aL2ToL1",
                            l1_a_scale,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )

                        l1_b_data = AllocOp(l1MemrefTyB, [], [])
                        l1_b_scale = AllocOp(l1MemrefTyBS, [], [])

                        ChannelGet(
                            "bL2ToL1",
                            l1_b_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )
                        ChannelGet(
                            "bL2ToL1",
                            l1_b_scale,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )

                        vecmat = CallOp(
                            vecmat_func,
                            [l1_a_data, l1_a_scale, l1_b_data, l1_b_scale, l1_c_data],
                        )

                        DeallocOp(l1_a_data)
                        DeallocOp(l1_a_scale)
                        DeallocOp(l1_b_data)
                        DeallocOp(l1_b_scale)

                        yield_([])

                    ChannelPut(
                        "cL1ToL2",
                        l1_c_data,
                        offsets=[],
                        sizes=[],
                        strides=[],
                    )
                    DeallocOp(l1_c_data)

                herd_body.attributes["link_with"] = StringAttr.get("vm.o")

                ChannelPut(
                    "cL2ToL3",
                    l2_c_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )
                DeallocOp(l2_a_data)
                DeallocOp(l2_a_scale)
                DeallocOp(l2_b_data)
                DeallocOp(l2_b_scale)
                DeallocOp(l2_c_data)


if __name__ == "__main__":
    # Default values.
    M = 1
    K = 288
    N = 48
    BS = 32
    TILE_K = 96
    TILE_N = 48
    INPUT_DATATYPE = np.int8
    ACC_DATATYPE = np.int32
    OUTPUT_DATATYPE = np.float32

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
        "--k", type=int, default=K, help="K dimension size in a (1xK) * (KxN) matmul"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="N dimension size in a (1xK) * (KxN) matmul",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=BS,
        help="Block size, for blocked quantization",
    )
    parser.add_argument(
        "--tile-k", type=int, default=TILE_K, help="K dimension size of each L1 tile"
    )
    parser.add_argument(
        "--tile-n", type=int, default=TILE_N, help="N dimension size of each L1 tile"
    )
    args = parser.parse_args()

    mlir_module = build_module(
        args.k,
        args.n,
        args.bs,
        args.tile_k,
        args.tile_n,
        INPUT_DATATYPE,
        ACC_DATATYPE,
        OUTPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.random.randint(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=(args.k),
        dtype=INPUT_DATATYPE,
    )
    input_a_s = np.random.randn(
        args.k // args.bs,
    ).astype(OUTPUT_DATATYPE)
    input_b = np.random.randint(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=(args.k, args.n),
        dtype=INPUT_DATATYPE,
    )
    input_b_s = np.random.rand(
        args.k // args.bs,
        args.n,
    ).astype(OUTPUT_DATATYPE)
    output_c = np.zeros(shape=(args.n), dtype=OUTPUT_DATATYPE)
    for n1 in range(args.n):
        output_c[n1] = OUTPUT_DATATYPE(0.0)
        ival = ACC_DATATYPE(0.0)
        for g in range(0, args.k - args.bs + 1, args.bs):
            for k1 in range(args.bs):
                ival += ACC_DATATYPE(input_a[g + k1]) * ACC_DATATYPE(
                    input_b[g + k1][n1]
                )
            output_c[n1] += (
                OUTPUT_DATATYPE(ival)
                * input_a_s[g // args.bs]
                * input_b_s[g // args.bs][n1]
            )
            ival = 0

    runner = XRTRunner(verbose=args.verbose, omit_while_true_loop=False)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a, input_a_s, input_b, input_b_s],
            expected_outputs=[output_c],
        )
    )
