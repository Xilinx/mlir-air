# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
from math import cos, sin, sqrt, exp

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from ml_dtypes import bfloat16

range_ = for_


@module_builder
def build_module(
    k, n, tile_k, tile_n, seq_len, np_dtype_in, np_dtype_vm_acc, np_dtype_out, pos_host
):

    QKV_COUNT = 3  # Running the design 3 times for Q, K and V, respectively.
    KV_COUNT = 2

    assert k % tile_k == 0
    assert n % tile_n == 0
    a_size = [QKV_COUNT, k]
    b_size = [QKV_COUNT, k, n]
    q_size = [1, n]
    kv_cache_size = [seq_len, n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_vm_acc = type_mapper(np_dtype_vm_acc)
    xrt_dtype_out = type_mapper(np_dtype_out)

    # L3 MemRefTypes
    memrefTyA = MemRefType.get(a_size, xrt_dtype_in)
    memrefTyB = MemRefType.get(b_size, xrt_dtype_in)
    memrefTyQ = MemRefType.get(q_size, xrt_dtype_out)
    memrefTyKCache = MemRefType.get(kv_cache_size, xrt_dtype_out)
    memrefTyVCache = MemRefType.get(kv_cache_size, xrt_dtype_out)

    Channel("aL3ToL2")
    Channel("bL3ToL2")
    Channel("aL2ToL1")
    Channel("bL2ToL1")
    Channel("cL1ToL2")
    Channel("cL2ToL3")
    Channel("dL1ToL2")
    Channel("dL2ToL3")

    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

    a_l1_size = [tile_k]
    b_l1_size = [tile_k, tile_n]
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
    c_l1_size = [tile_n]
    l1MemrefTyC = MemRefType.get(
        shape=c_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyACC = MemRefType.get(
        shape=c_l1_size,
        element_type=xrt_dtype_vm_acc,
        memory_space=l1_mem_space,
    )
    l1MemrefTyThreeByFortyEightVec = MemRefType.get(
        shape=[3, 48],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    linalg_fill_func = FuncOp(
        "linalg_fill_bf16",
        ([T.i32(), l1MemrefTyThreeByFortyEightVec], []),
        visibility="private",
    )
    vecmat_func = FuncOp(
        "vecmat_bf16_bf16",
        ([T.i32(), l1MemrefTyA, l1MemrefTyB, l1MemrefTyThreeByFortyEightVec], []),
        visibility="private",
    )
    l1MemrefTyHSByTwo = MemRefType.get(
        shape=[24],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyVec = MemRefType.get(
        shape=[8],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    cosf_poly_func = FuncOp(
        "cosf_bf16_24_8",
        ([l1MemrefTyHSByTwo, l1MemrefTyHSByTwo], []),
        visibility="private",
    )
    sinf_poly_func = FuncOp(
        "sinf_bf16_24_8",
        ([l1MemrefTyHSByTwo, l1MemrefTyHSByTwo], []),
        visibility="private",
    )
    freq_pos_func = FuncOp(
        "freq_pos_bf16_24_8",
        ([T.i32(), l1MemrefTyHSByTwo], []),
        visibility="private",
    )
    shuffle_apply_rope_poly_func = FuncOp(
        "shuffle_apply_rope_bf16_48",
        (
            [
                T.i32(),
                l1MemrefTyHSByTwo,
                l1MemrefTyHSByTwo,
                l1MemrefTyThreeByFortyEightVec,
            ],
            [],
        ),
        visibility="private",
    )
    l1MemrefTyQKV = MemRefType.get(
        shape=[QKV_COUNT, tile_n],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTySharedL1BDBuf = MemRefType.get(
        shape=[48],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    vector_copy_func = FuncOp(
        "vector_copy",
        ([T.i32(), l1MemrefTySharedL1BDBuf, l1MemrefTyA], []),
        visibility="private",
    )

    # Launch 2 funcs

    q_l1_size = [1, n]
    k_l1_size = [1, n]
    v_l1_size = [1, n]
    xb_l1_size = [n]
    attn_l1_size = [seq_len]
    xb_size = xb_l1_size
    memrefTyXb = MemRefType.get(xb_size, xrt_dtype_out)
    l1MemrefTyQ = MemRefType.get(
        shape=q_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyK = MemRefType.get(
        shape=k_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyV = MemRefType.get(
        shape=v_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyXb = MemRefType.get(
        shape=xb_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    l1MemrefTyAttn = MemRefType.get(
        shape=attn_l1_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    attn_func = FuncOp(
        "attn_1",
        ([l1MemrefTyQ, l1MemrefTySharedL1BDBuf, T.i32(), l1MemrefTyAttn], []),
        visibility="private",
    )
    softmax_func = FuncOp(
        "softmax_bf16",
        ([l1MemrefTyAttn, T.i32(), l1MemrefTyAttn], []),
        visibility="private",
    )
    attn2_func = FuncOp(
        "attn_2",
        ([l1MemrefTyAttn, l1MemrefTySharedL1BDBuf, T.i32(), l1MemrefTyXb], []),
        visibility="private",
    )

    for func in [
        linalg_fill_func,
        vecmat_func,
        cosf_poly_func,
        sinf_poly_func,
        freq_pos_func,
        shuffle_apply_rope_poly_func,
        vector_copy_func,
        attn_func,
        softmax_func,
        attn2_func,
    ]:
        func.attributes["link_with"] = StringAttr.get("mha.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(
        memrefTyA, memrefTyB, memrefTyKCache, memrefTyVCache, memrefTyXb
    )
    def mha_bf16(arg0, arg1, arg2, arg3, arg4):

        launch_size = [1, 1]

        @launch(operands=[arg0, arg1, arg2, arg3, arg4], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a_data,
            l3_b_data,
            l3_k_cache_data,
            l3_v_cache_data,
            l3_xb_data,
        ):

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

            for mm_iter in range_(0, QKV_COUNT):

                ChannelPut(
                    "aL3ToL2",
                    l3_a_data,
                    offsets=[mm_iter, 0],
                    sizes=[1, k],
                    strides=[k, 1],
                )
                ChannelPut(
                    "bL3ToL2",
                    l3_b_data,
                    offsets=[mm_iter, 0, launch_offset_y],
                    sizes=[1, k, tile_n],
                    strides=[n * k, n, 1],
                )
                yield_([])

            ChannelGet(
                "cL2ToL3",
                l3_k_cache_data,
                offsets=[pos_host, launch_offset_y],
                sizes=[1, tile_n],
                strides=[n, 1],
            )

            ChannelGet(
                "cL2ToL3",
                l3_v_cache_data,
                offsets=[pos_host, launch_offset_y],
                sizes=[1, tile_n],
                strides=[n, 1],
            )

            # Launch 2 runtime

            for i in range_(0, pos_host + 1):
                ChannelPut(
                    "aL3ToL2",
                    l3_k_cache_data,
                    offsets=[i, 0],
                    sizes=[1, tile_n],
                    strides=[n, 1],
                )
                yield_([])
            for i in range_(0, pos_host + 1):
                ChannelPut(
                    "aL3ToL2",
                    l3_v_cache_data,
                    offsets=[i, 0],
                    sizes=[1, tile_n],
                    strides=[n, 1],
                )
                yield_([])

            ChannelGet(
                "dL2ToL3",
                l3_xb_data,
                offsets=[],
                sizes=[],
                strides=[],
            )

            @segment(name="vecmat_i8_0")
            def segment_body():
                # L2 MemRefTypes
                a_size_l2 = [48]  # Common integer factor of all shapes being moved
                b_size_l2 = [k, tile_n]
                c_size_l2 = [KV_COUNT, tile_n]
                l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
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
                l2_a_data = AllocOp(l2MemrefTyA, [], [])

                l2_b_data = AllocOp(l2MemrefTyB, [], [])
                l2_c_data = AllocOp(l2MemrefTyC, [], [])

                l1_c_data = AllocOp(l1MemrefTyQKV, [], [])

                ChannelGet(
                    "aL3ToL2",
                    l2_a_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )
                ChannelPut(
                    "aL2ToL1",
                    l2_a_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )

                for mm_iter in range_(0, QKV_COUNT):

                    ChannelGet(
                        "bL3ToL2",
                        l2_b_data,
                        offsets=[],
                        sizes=[],
                        strides=[],
                    )
                    b_l2l1_pack_size = [tile_k, tile_n]
                    b_l2l1_pack_stride = [tile_n, 1]

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

                    for i in range_(0, k // tile_k):
                        for_iv_data_offset = affine_apply(for_iv_map_data, [i])
                        b_l2l1_pack_offset = [for_iv_data_offset, 0]
                        ChannelPut(
                            "bL2ToL1",
                            l2_b_data,
                            offsets=b_l2l1_pack_offset,
                            sizes=b_l2l1_pack_size,
                            strides=b_l2l1_pack_stride,
                        )
                        yield_([])
                    yield_([])

                ChannelGet(
                    "cL1ToL2",
                    l2_c_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )

                l1_out_data = AllocOp(l1MemrefTyC, [], [])

                pos_c = arith.ConstantOp.create_index(pos_host)

                @herd(
                    name="herd_0",
                    sizes=[1, 1],
                    operands=[l1_c_data, l1_out_data, pos_c],
                )
                def herd_body_0(_tx, _ty, _sx, _sy, c_data, out_data, pos):

                    zero_const = ConstantOp(FloatAttr.get(xrt_dtype_vm_acc, 0), None)
                    q_offset_const = ConstantOp(IntegerAttr.get(T.i32(), 0), None)
                    k_offset_const = ConstantOp(IntegerAttr.get(T.i32(), tile_n), None)
                    v_offset_const = ConstantOp(
                        IntegerAttr.get(T.i32(), 2 * tile_n), None
                    )

                    k_offset_const = ConstantOp(IntegerAttr.get(T.i32(), tile_n), None)
                    v_offset_const = ConstantOp(
                        IntegerAttr.get(T.i32(), 2 * tile_n), None
                    )
                    l1_a_data = AllocOp(l1MemrefTyA, [], [])
                    l1_b_data = AllocOp(l1MemrefTyB, [], [])
                    l1_shared_bd_buf_data = AllocOp(l1MemrefTySharedL1BDBuf, [], [])
                    zero_fill = CallOp(linalg_fill_func, [q_offset_const, c_data])
                    for i in range_(0, k, tile_k):
                        for j in range_(0, tile_k, 48):
                            ChannelGet(
                                "aL2ToL1",
                                l1_shared_bd_buf_data,
                                offsets=[],
                                sizes=[],
                                strides=[],
                            )
                            vector_copy = CallOp(
                                vector_copy_func,
                                [
                                    arith.index_cast(T.i32(), j),
                                    l1_shared_bd_buf_data,
                                    l1_a_data,
                                ],
                            )
                            yield_([])
                        ChannelGet(
                            "bL2ToL1",
                            l1_b_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )
                        vecmat = CallOp(
                            vecmat_func,
                            [q_offset_const, l1_a_data, l1_b_data, c_data],
                        )
                        yield_([])
                    zero_fill = CallOp(linalg_fill_func, [k_offset_const, c_data])
                    for i in range_(0, k, tile_k):
                        for j in range_(0, tile_k, 48):
                            ChannelGet(
                                "aL2ToL1",
                                l1_shared_bd_buf_data,
                                offsets=[],
                                sizes=[],
                                strides=[],
                            )
                            vector_copy = CallOp(
                                vector_copy_func,
                                [
                                    arith.index_cast(T.i32(), j),
                                    l1_shared_bd_buf_data,
                                    l1_a_data,
                                ],
                            )
                            yield_([])
                        ChannelGet(
                            "bL2ToL1",
                            l1_b_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )
                        vecmat = CallOp(
                            vecmat_func,
                            [k_offset_const, l1_a_data, l1_b_data, c_data],
                        )
                        yield_([])
                    zero_fill = CallOp(linalg_fill_func, [v_offset_const, c_data])
                    for i in range_(0, k, tile_k):
                        for j in range_(0, tile_k, 48):
                            ChannelGet(
                                "aL2ToL1",
                                l1_shared_bd_buf_data,
                                offsets=[],
                                sizes=[],
                                strides=[],
                            )
                            vector_copy = CallOp(
                                vector_copy_func,
                                [
                                    arith.index_cast(T.i32(), j),
                                    l1_shared_bd_buf_data,
                                    l1_a_data,
                                ],
                            )
                            yield_([])
                        ChannelGet(
                            "bL2ToL1",
                            l1_b_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )
                        vecmat = CallOp(
                            vecmat_func,
                            [v_offset_const, l1_a_data, l1_b_data, c_data],
                        )
                        yield_([])
                    DeallocOp(l1_a_data)
                    DeallocOp(l1_b_data)

                    l1_freq_pos_data = AllocOp(l1MemrefTyHSByTwo, [], [])
                    zero_const = ConstantOp(IntegerAttr.get(T.i32(), 0), None)
                    one_const = ConstantOp(IntegerAttr.get(T.i32(), 1), None)
                    head_size_constindex = ConstantOp.create_index(48)
                    head_size_by_two_constindex = ConstantOp.create_index(24)
                    zero_constindex = ConstantOp.create_index(0)
                    one_constindex = ConstantOp.create_index(1)
                    two_constindex = ConstantOp.create_index(2)
                    eight_constindex = ConstantOp.create_index(8)

                    freq_pos_call = CallOp(
                        freq_pos_func,
                        [arith.index_cast(T.i32(), pos), l1_freq_pos_data],
                    )

                    l1_sinf_vec = AllocOp(l1MemrefTyHSByTwo, [], [])
                    l1_cosf_vec = AllocOp(l1MemrefTyHSByTwo, [], [])
                    sinf_poly_call = CallOp(
                        sinf_poly_func,
                        [l1_freq_pos_data, l1_sinf_vec],
                    )
                    cosf_poly_call = CallOp(
                        cosf_poly_func,
                        [l1_freq_pos_data, l1_cosf_vec],
                    )

                    # sq
                    rope_offset_0_const = ConstantOp(IntegerAttr.get(T.i32(), 0), None)
                    shuffle_apply_rope_call = CallOp(
                        shuffle_apply_rope_poly_func,
                        [rope_offset_0_const, l1_cosf_vec, l1_sinf_vec, c_data],
                    )
                    # sk
                    rope_offset_1_const = ConstantOp(
                        IntegerAttr.get(T.i32(), tile_n), None
                    )
                    shuffle_apply_rope_call = CallOp(
                        shuffle_apply_rope_poly_func,
                        [rope_offset_1_const, l1_cosf_vec, l1_sinf_vec, c_data],
                    )

                    DeallocOp(l1_sinf_vec)
                    DeallocOp(l1_cosf_vec)
                    DeallocOp(l1_freq_pos_data)

                    ChannelPut(
                        "cL1ToL2",
                        c_data,
                        offsets=[1, 0],
                        sizes=[KV_COUNT, tile_n],
                        strides=[tile_n, 1],
                    )

                    # launch 2 herd

                    pos_p1 = arith.addi(pos, arith.ConstantOp.create_index(1))

                    q_l1_data = AllocOp(l1MemrefTyQ, [], [])
                    attn_l1_data = AllocOp(l1MemrefTyAttn, [], [])
                    softmax_l1_data = AllocOp(l1MemrefTyAttn, [], [])

                    # # Zero fill
                    # TODO: exp(-inf) = 1.0. Why?
                    const_negInf = ConstantOp(FloatAttr.get(T.bf16(), -99), None)
                    zero_constindex = ConstantOp.create_index(0)
                    for i in range_(0, seq_len):
                        store(const_negInf, attn_l1_data, [i])
                        yield_([])

                    for y in range_(0, n):
                        inval = load(c_data, [zero_constindex, y])
                        store(inval, q_l1_data, [zero_constindex, y])
                        yield_([])

                    for i in range_(0, pos_p1):
                        ChannelGet(
                            "aL2ToL1",
                            l1_shared_bd_buf_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )

                        i32_iv = arith.index_cast(T.i32(), i)
                        attn_call = CallOp(
                            attn_func,
                            [q_l1_data, l1_shared_bd_buf_data, i32_iv, attn_l1_data],
                        )

                        yield_([])

                    # Softmax
                    softmax_call = CallOp(
                        softmax_func,
                        [
                            attn_l1_data,
                            arith.index_cast(T.i32(), pos_p1),
                            softmax_l1_data,
                        ],
                    )

                    xb_l1_data = AllocOp(l1MemrefTyXb, [], [])
                    const_zero = ConstantOp(FloatAttr.get(T.bf16(), 0), None)
                    for i in range_(0, n):
                        store(const_zero, xb_l1_data, [i])
                        yield_([])
                    for i in range_(0, pos_p1):
                        ChannelGet(
                            "aL2ToL1",
                            l1_shared_bd_buf_data,
                            offsets=[],
                            sizes=[],
                            strides=[],
                        )
                        i32_iv = arith.index_cast(T.i32(), i)
                        attn_call = CallOp(
                            attn2_func,
                            [
                                softmax_l1_data,
                                l1_shared_bd_buf_data,
                                i32_iv,
                                xb_l1_data,
                            ],
                        )

                        yield_([])

                    ChannelPut(
                        "dL1ToL2",
                        xb_l1_data,
                        offsets=[],
                        sizes=[],
                        strides=[],
                    )

                    DeallocOp(q_l1_data)
                    DeallocOp(attn_l1_data)
                    DeallocOp(softmax_l1_data)
                    DeallocOp(xb_l1_data)
                    DeallocOp(l1_shared_bd_buf_data)

                herd_body_0.attributes["link_with"] = StringAttr.get("mha.o")

                ChannelPut(
                    "cL2ToL3",
                    l2_c_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )

                xb_l2_size = xb_l1_size
                l2MemrefTyXb = MemRefType.get(
                    shape=xb_l2_size,
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                xb_l2_data = AllocOp(l2MemrefTyXb, [], [])
                ChannelGet(
                    "dL1ToL2",
                    xb_l2_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )
                ChannelPut(
                    "dL2ToL3",
                    xb_l2_data,
                    offsets=[],
                    sizes=[],
                    strides=[],
                )
                DeallocOp(xb_l2_data)
                DeallocOp(l1_c_data)
                DeallocOp(l2_a_data)
                DeallocOp(l2_b_data)
                DeallocOp(l2_c_data)


if __name__ == "__main__":
    # Default values.
    M = 1
    K = 288
    N = 48
    pos = 16
    seq_len = 256
    TILE_K = 96
    TILE_N = 48
    QKV_COUNT = 3  # Running the design 3 times for Q, K and V, respectively.
    KV_COUNT = 2
    INPUT_DATATYPE = bfloat16
    VM_ACC_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

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
        "--pos",
        type=int,
        default=pos,
        help="Position runtime variable",
    )
    parser.add_argument(
        "--tile-k", type=int, default=TILE_K, help="K dimension size of each L1 tile"
    )
    parser.add_argument(
        "--tile-n", type=int, default=TILE_N, help="N dimension size of each L1 tile"
    )
    parser.add_argument("--seq-len", type=int, default=seq_len, help="Sequence length")
    parser.add_argument(
        "--xclbin-kernel-name",
        dest="kernel_name",
        default="",
        help="Kernel name in xclbin file",
    )
    parser.add_argument(
        "--xclbin-instance-name",
        dest="instance_name",
        default="",
        help="Instance name in xclbin metadata",
    )
    parser.add_argument(
        "--xclbin-kernel-id",
        dest="kernel_id",
        default="",
        help="Kernel id in xclbin file",
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
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.k,
        args.n,
        args.tile_k,
        args.tile_n,
        args.seq_len,
        INPUT_DATATYPE,
        VM_ACC_DATATYPE,
        OUTPUT_DATATYPE,
        args.pos,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, QKV_COUNT * args.k, dtype=INPUT_DATATYPE).reshape(
        QKV_COUNT, args.k
    )
    input_b = np.arange(0, QKV_COUNT * args.k * args.n, dtype=INPUT_DATATYPE).reshape(
        QKV_COUNT, args.k, args.n
    )
    output_vm = np.zeros(shape=(QKV_COUNT, args.n), dtype=OUTPUT_DATATYPE)
    for qkv_iter in range(QKV_COUNT):
        output_vm[qkv_iter] = np.dot(
            input_a[qkv_iter].astype(VM_ACC_DATATYPE),
            input_b[qkv_iter].astype(VM_ACC_DATATYPE),
        ).astype(OUTPUT_DATATYPE)
    output_c = np.reshape(output_vm, (-1))
    for s in range(0, 48, 2):
        freq = 1.0 / pow(10000.0, float(s) / float(48))
        val = args.pos * freq

        fcr = cos(val)
        fci = sin(val)

        v0 = output_c[s]
        v1 = output_c[s + 1]

        output_c[s] = v0 * fcr - v1 * fci
        output_c[s + 1] = v0 * fci + v1 * fcr

        v0 = output_c[s + 48]
        v1 = output_c[s + 48 + 1]
        output_c[s + 48] = v0 * fcr - v1 * fci
        output_c[s + 48 + 1] = v0 * fci + v1 * fcr

    # Output buffers
    output_q = np.zeros(shape=(1, args.n), dtype=OUTPUT_DATATYPE)
    output_kc = np.zeros(shape=(args.seq_len, args.n), dtype=OUTPUT_DATATYPE)
    output_vc = np.zeros(shape=(args.seq_len, args.n), dtype=OUTPUT_DATATYPE)
    output_xb = np.zeros(shape=(args.n), dtype=OUTPUT_DATATYPE)
    softmax_output = np.zeros(shape=(args.seq_len), dtype=OUTPUT_DATATYPE)

    for i in range(0, args.n):
        output_q[0][i] = output_c[i]
        output_kc[args.pos][i] = output_c[args.n + i]
        output_vc[args.pos][i] = output_c[2 * args.n + i]

    # Launch 2

    # Attn 1
    for t in range(0, args.pos + 1):
        score = OUTPUT_DATATYPE(0)
        for i in range(0, args.n):
            score += output_q[0][i] * output_kc[t][i]
        score /= sqrt(args.n)
        softmax_output[t] = score

    # Softmax
    max_val = softmax_output[0]
    for i in range(1, args.pos + 1):
        if softmax_output[i] > max_val:
            max_val = softmax_output[i]
    sum_val = 0.0
    for i in range(args.pos + 1):
        softmax_output[i] = exp(softmax_output[i] - max_val)
        sum_val += softmax_output[i]
    for i in range(args.pos + 1):
        softmax_output[i] = softmax_output[i] / sum_val

    # Attn 2
    for t in range(0, args.pos + 1):
        for i in range(0, args.n):
            output_xb[i] += softmax_output[t] * output_vc[t][i]

    if args.compile_mode == "compile-and-run":
        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            output_format=args.output_format,
            instance_name="mha_bf16",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b, output_kc, output_vc],
                expected_outputs=[output_xb],
                rtol=1e0,
            )
        )

    elif args.compile_mode == "compile-only":
        ####### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            kernel_name=args.kernel_name,
            instance_name=args.instance_name,
            kernel_id=args.kernel_id,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
