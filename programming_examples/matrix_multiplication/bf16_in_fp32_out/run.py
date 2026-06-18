# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import math
import os
import sys
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.linalg import fill
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, ViewOp, load, store, subview
from air.dialects.func import CallOp, FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from air.extras import types as extrasT
from air.dialects.linalg.opdsl.lang import *
import air.dialects.linalg.opdsl.lang as linalg_lang

import numpy as np

np.random.seed(42)

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
    emit_external_call=False,
    drain_chunks=1,
):
    assert m % tile_m == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert n % tile_n == 0
    if emit_external_call:
        # Manual-CallOp external-mm.o path (also usable inside a fused multi-launch
        # ELF, where the whole-module air-linalg-to-func pass can't run). The L1 C
        # ACCUMULATOR is always f32 — FP32 partials across the whole K loop, the
        # registry/GPU standard — regardless of the requested OUTPUT dtype:
        #   - np_dtype_out == f32: drain the f32 accumulator straight out (no cast).
        #   - np_dtype_out == bf16: one on-chip f32->bf16 cast (f32_to_bf16_mn) in
        #     the drain herd into a bf16 L1 buffer, then DMA bf16 out. Numerically
        #     identical to direct-codegen-bf16 (f32 accumulate, single epilogue cast),
        #     but DDR output is half the bytes of f32-out.
        assert np_dtype_out in (
            np.float32,
            bfloat16,
        ), "emit_external_call supports f32 or bf16 output"
    # f32-accumulate-then-bf16-cast in the external path: C accumulator is f32,
    # output (drain) dtype is np_dtype_out.
    cast_out = emit_external_call and np_dtype_out == bfloat16
    np_dtype_acc = np.float32 if emit_external_call else np_dtype_out
    a_size = [m, k]
    b_size = [k, n]
    c_size = [m, n]
    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)
    # L1/L2 C ACCUMULATOR element type. In the external f32-accumulate path this is
    # f32 even when the final OUTPUT is bf16 (the drain herd casts once). Elsewhere
    # accumulator == output.
    xrt_dtype_acc = type_mapper(np_dtype_acc)

    # Architecture-specific matrix multiplication dimensions
    # aie2p uses 8x8x8 (using -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 if without direct-codegen, i.e. aie_api)
    # aie2 uses 4x8x4
    if arch == "aie2p":
        mmul_mkn = [8, 8, 8]  # For aie2p
    else:
        mmul_mkn = [4, 8, 4]  # For aie2

    # Chunked drain (bf16-out only): cast+DMA the C tile in `drain_chunks` (G)
    # segments along tile_n, so only a (tile_m·tile_n/G) bf16 buffer coexists with
    # the f32 accumulator — letting tile_m=64 fit L1 (the full-tile bf16 drain
    # overflows). G=1 is the original single-shot drain. The C tile is laid out
    # 6D [1,1,tile_n/n,tile_m/m,m,n] with tile_n/n outermost, so splitting that
    # dim into G groups gives G contiguous chunks of (tile_n/G)·tile_m elements.
    if cast_out:
        assert tile_n % drain_chunks == 0, "tile_n must be divisible by drain_chunks"
        assert (tile_n // drain_chunks) % mmul_mkn[
            2
        ] == 0, "tile_n/drain_chunks must be a multiple of mmul n"

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
        element_type=xrt_dtype_acc,
        memory_space=l1_mem_space,
        layout=layout,
    )
    l1MemrefTyCHerd = MemRefType.get(
        shape=c_herd_l1_size,
        element_type=xrt_dtype_acc,
        memory_space=l1_mem_space,
    )
    # bf16 drain buffer (only the external bf16-out path): per-PE contiguous tile,
    # allocated INSIDE the drain herd (herd-local, so it doesn't coexist with the
    # compute herd's A/B). The f32 accumulator subview is cast into it (in
    # `drain_chunks` segments along tile_n), then DMA'd out. Plain (non-strided)
    # layout. With drain_chunks=G the buffer is 1/G of the full tile.
    tile_n_chunk = tile_n // drain_chunks if cast_out else tile_n
    c_l1_chunk_size = [
        1,
        1,
        tile_n_chunk // mmul_mkn[2],
        tile_m // mmul_mkn[0],
        mmul_mkn[0],
        mmul_mkn[2],
    ]
    l1MemrefTyCDrain = MemRefType.get(
        shape=c_l1_chunk_size,
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )

    # Private FuncOp declarations for the external hand-tuned mm.o kernel.
    # Only emitted/used when emit_external_call=True. The C accumulator (l1MemrefTyC
    # / l1MemrefTyCHerd) is f32. Declared at module scope (before @FuncOp.from_py_func)
    # so the symbols are visible to the CallOps inside the herds, mirroring
    # int4_awq/matmul_int4_packed.py. For bf16 output, f32_to_bf16_func casts the
    # f32 accumulator once in the drain herd.
    if emit_external_call:
        zero_func = FuncOp(
            "zero_f32_mn",
            ([l1MemrefTyC], []),
            visibility="private",
        )
        zero_func.attributes["link_with"] = StringAttr.get("mm.o")
        zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        matmul_func = FuncOp(
            "op_has_no_registered_library_name",
            ([l1MemrefTyA, l1MemrefTyB, l1MemrefTyC], []),
            visibility="private",
        )
        matmul_func.attributes["link_with"] = StringAttr.get("mm.o")
        matmul_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        if cast_out:
            # f32 source chunk type (per-PE subview of the f32 accumulator, one
            # tile_n chunk). Matches l1MemrefTyCDrain's shape but f32.
            l1MemrefTyCChunkF32 = MemRefType.get(
                shape=c_l1_chunk_size,
                element_type=xrt_dtype_acc,
                memory_space=l1_mem_space,
                layout=layout,
            )
            if drain_chunks == 1:
                # f32_to_bf16_mn(float* src, bfloat16* dst): single full-tile cast.
                f32_to_bf16_func = FuncOp(
                    "f32_to_bf16_mn",
                    ([l1MemrefTyC, l1MemrefTyCDrain], []),
                    visibility="private",
                )
            else:
                # f32_to_bf16_n(float* src, bfloat16* dst, int n): cast one chunk of
                # n contiguous elements. Called G times in the drain herd.
                f32_to_bf16_func = FuncOp(
                    "f32_to_bf16_n",
                    (
                        [
                            l1MemrefTyCChunkF32,
                            l1MemrefTyCDrain,
                            IntegerType.get_signless(32),
                        ],
                        [],
                    ),
                    visibility="private",
                )
            f32_to_bf16_func.attributes["link_with"] = StringAttr.get("mm.o")
            f32_to_bf16_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

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
                # l1_a and l1_b are allocated inside the compute herd (the
                # only herd that uses them) so they have unambiguous per-PE
                # semantics.
                # C accumulator: f32 herd buffer in the external path (xrt_dtype_acc
                # is f32), else output-dtype. The bf16 drain buffer is allocated
                # INSIDE the drain herd (not here) so it doesn't coexist with the
                # compute herd's A/B ping-pong buffers — keeping each herd's L1
                # footprint under 64 KB.
                l1_c_data = AllocOp(l1MemrefTyCHerd, [], [])
                if cast_out and drain_chunks > 1:
                    # Exempt the f32 C accumulator from air-shrink-memref-sizes-by-access:
                    # the chunked drain DMAs the C tile in tile_n segments, so the pass
                    # (which sizes allocs from DMA access bounds, ignoring the matmul
                    # func.call that writes the full tile) would otherwise shrink this
                    # buffer to one chunk and break the matmul. Pre-setting the attr
                    # makes the pass skip it (AIRDependencyScheduleOpt.cpp:5389).
                    l1_c_data.operation.attributes["air.shrinkage"] = BoolAttr.get(
                        False
                    )

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
                    operands=[l1_c_data],
                    link_with="mm.o" if emit_external_call else None,
                )
                def herd_body(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _l1_c,
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
                    if emit_external_call:
                        CallOp(zero_func, [l1_c_subview])
                    else:
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
                            l1_c_data,
                            l2_a_data,
                            l2_b_data,
                        ],
                        link_with="mm.o" if emit_external_call else None,
                    )
                    def herd_body(
                        _tx,
                        _ty,
                        _sx,
                        _sy,
                        _l1_c,
                        _l2_a,
                        _l2_b,
                    ):
                        # L1 A/B allocated inside compute herd: unambiguous per-PE buffers
                        _l1_a = AllocOp(l1MemrefTyA, [], [])
                        _l1_b = AllocOp(l1MemrefTyB, [], [])
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
                            if emit_external_call:
                                CallOp(matmul_func, [_l1_a, _l1_b, l1_c_subview])
                            else:
                                matmul = block_matmul(_l1_a, _l1_b, outs=[l1_c_subview])
                            yield_([])

                        DeallocOp(_l1_a)
                        DeallocOp(_l1_b)

                    yield_([])

                # Drain herd: DMA the C tile L1->L2. For the external bf16-out path,
                # first cast the f32 accumulator into a per-PE bf16 L1 drain buffer
                # (one f32_to_bf16_mn call), then DMA the bf16 tile out.
                @herd(
                    name="herd_0",
                    sizes=[herd_m, herd_n],
                    operands=[l1_c_data, l2_a_data, l2_b_data, l2_c_data],
                    link_with="mm.o" if cast_out else None,
                )
                def herd_body(
                    _tx,
                    _ty,
                    _sx,
                    _sy,
                    _l1_c,
                    _l2_a,
                    _l2_b,
                    _l2_c,
                ):
                    if cast_out:
                        # Chunked drain: split the tile_n dim into `drain_chunks` (G)
                        # contiguous segments. Per chunk: subview the f32 accumulator,
                        # cast it into a small per-chunk bf16 buffer, DMA that chunk
                        # to L2. Only one (tile_m·tile_n/G) bf16 buffer is live at a
                        # time, so tile_m=64 fits L1. G=1 = original single-shot drain.
                        n_chunk_blk = (
                            tile_n_chunk // mmul_mkn[2]
                        )  # dim2 extent per chunk
                        chunk_elems = tile_m * tile_n_chunk
                        for g in range(drain_chunks):
                            c_acc_sub = subview(
                                _l1_c,
                                offsets=[_tx, _ty, g * n_chunk_blk, 0, 0, 0],
                                sizes=[
                                    1,
                                    1,
                                    n_chunk_blk,
                                    tile_m // mmul_mkn[0],
                                    mmul_mkn[0],
                                    mmul_mkn[2],
                                ],
                                strides=[1, 1, 1, 1, 1, 1],
                            )
                            l1_c_drain = AllocOp(l1MemrefTyCDrain, [], [])
                            if drain_chunks == 1:
                                CallOp(f32_to_bf16_func, [c_acc_sub, l1_c_drain])
                            else:
                                n_i32 = ConstantOp(
                                    IntegerType.get_signless(32), chunk_elems
                                )
                                CallOp(f32_to_bf16_func, [c_acc_sub, l1_c_drain, n_i32])
                            dma_memcpy_nd(
                                _l2_c,
                                l1_c_drain,
                                dst_offsets=[_tx, _ty, 0, g * tile_n_chunk],
                                dst_sizes=[1, 1, tile_m, tile_n_chunk],
                                dst_strides=[
                                    herd_n * tile_m * tile_n,
                                    tile_m * tile_n,
                                    tile_n,
                                    1,
                                ],
                                src_offsets=[0, 0, 0, 0, 0, 0],
                                src_sizes=[
                                    1,
                                    1,
                                    tile_m // mmul_mkn[0],
                                    mmul_mkn[0],
                                    n_chunk_blk,
                                    mmul_mkn[2],
                                ],
                                src_strides=[
                                    tile_m * tile_n_chunk,
                                    tile_m * tile_n_chunk,
                                    mmul_mkn[2] * mmul_mkn[0],
                                    mmul_mkn[2],
                                    tile_m * mmul_mkn[2],
                                    1,
                                ],
                            )
                            DeallocOp(l1_c_drain)
                    else:
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
                # l1_c_data is the C accumulator (f32 in external path, output-dtype
                # otherwise). The bf16 drain buffer is herd-local (alloc/dealloc inside
                # the drain herd), so nothing to free here.
                DeallocOp(l1_c_data)


# ---------------------------------------------------------------------------
# Strategy 2: fused external GEMM (f32-out, tile_m=64 full speed) + a separate
# f32->bf16 cast launch, in ONE module → bf16 output, FP32-accumulate precision.
#
# Unlike the in-GEMM bf16-drain path (which is capped at tile_m=32 because the
# bf16 drain buffer + f32 accumulator overflow L1), here the GEMM runs at full
# tile_m=64 (~9658 GFLOPS on Down) writing an f32 C scratch, then a memory-bound
# vectorized-truncf cast launch converts f32→bf16. The cast is cheap (Down 4.19M
# = 0.45 ms measured). Numerically identical to f32-out (single epilogue cast),
# mean_rel_L1 ≈ 9.3e-3.
#
# Implementation = text-stitch two separately-built launch bodies into one func
# (same mechanism as llms/llama32_1b o_ffn). Helpers are inlined here so this
# example stays self-contained (no cross-dir import).
# ---------------------------------------------------------------------------

import re as _re


def _stitch_extract_body(mlir_text):
    lines = mlir_text.split("\n")
    body_start = body_end = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            body_start = i + 1
    ret_re = _re.compile(r"^\s*return(\s|$|//|loc\()")
    for i in range(len(lines) - 1, body_start, -1):
        if ret_re.match(lines[i]):
            body_end = i
            break
    return "\n".join(lines[body_start:body_end])


def _stitch_affine_maps(mlir_text):
    return [
        l for l in mlir_text.split("\n") if l.startswith("#map") or l.startswith("#set")
    ]


def _stitch_private_funcs(mlir_text):
    return [l for l in mlir_text.split("\n") if "func.func private" in l]


def _stitch_rename(text, prefix, extern_funcs):
    affine_names = set(_re.findall(r"#map\d*", text)) | set(
        _re.findall(r"#set\d*", text)
    )
    for name in sorted(affine_names, key=len, reverse=True):
        text = _re.sub(_re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)
    for name in sorted(set(_re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = _re.sub(_re.escape(name) + r"(?!\w)", f"%{prefix}_{name[1:]}", text)
    for name in sorted(
        set(_re.findall(r"%\d+", text)), key=lambda x: int(x[1:]), reverse=True
    ):
        text = _re.sub(_re.escape(name) + r"(?!\d)", f"%{prefix}_n{name[1:]}", text)
    for name in sorted(set(_re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if name not in extern_funcs:
            text = text.replace(name, f"@{prefix}_{name[1:]}")
    return text


def _stitch_fix_args(text, prefix, arg_map):
    for orig_idx, combined_idx in arg_map.items():
        old_ref = f"%{prefix}_arg{orig_idx}"
        new_ref = f"%arg{combined_idx}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    return text


@module_builder
def _build_cast_module(m, n, np_dtype_in, tile_n=2048, herd_x=8, herd_y=1):
    """Standalone f32->bf16 cast: 2D in/out, collapse to 1D, contiguous-chunk
    partition (1-level DMA descriptor), vectorized truncf in L1. herd_x*herd_y
    tiles, each owns a contiguous chunk. Mirrors llama o_ffn's _build_cast_2d."""
    from air.dialects.memref import collapse_shape as memref_collapse_shape
    from air.dialects.vector import transfer_read, transfer_write
    from air.dialects import arith as _arith

    nelem = m * n
    f32 = type_mapper(np.float32)
    bf16 = type_mapper(bfloat16)
    total_tiles = herd_x * herd_y
    chunk_size = nelem // total_tiles
    assert chunk_size % tile_n == 0, (nelem, total_tiles, tile_n)
    l3_in_2d = MemRefType.get([m, n], f32)
    l3_in_1d = MemRefType.get([nelem], f32)
    l3_out_2d = MemRefType.get([m, n], bf16)
    l3_out_1d = MemRefType.get([nelem], bf16)
    l1_space = IntegerAttr.get(extrasT.i32(), MemorySpace.L1)
    l1_f32 = MemRefType.get([tile_n], f32, memory_space=l1_space)
    l1_bf16 = MemRefType.get([tile_n], bf16, memory_space=l1_space)
    vf = VectorType.get([16], f32)
    vb = VectorType.get([16], bf16)
    idmap = AffineMapAttr.get(AffineMap.get_identity(1))

    @FuncOp.from_py_func(l3_in_2d, l3_out_2d)
    def cast_f32_bf16(arg0, arg1):
        @launch(operands=[arg0, arg1])
        def cast_launch(l_in, l_out):
            in_1d = memref_collapse_shape(l3_in_1d, l_in, [[0, 1]])
            out_1d = memref_collapse_shape(l3_out_1d, l_out, [[0, 1]])

            @segment(name="cast_seg", operands=[in_1d, out_1d])
            def cast_seg(s_in, s_out):
                @herd(name="herd_0", sizes=[herd_x, herd_y], operands=[s_in, s_out])
                def herd_body(_tx, _ty, _sx, _sy, h_in, h_out):
                    li = AllocOp(l1_f32, [], [])
                    lo = AllocOp(l1_bf16, [], [])
                    c0 = _arith.ConstantOp.create_index(0)
                    cst0 = _arith.ConstantOp(f32, 0.0)
                    offmap = AffineMap.get(
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
                                    AffineConstantExpr.get(chunk_size),
                                ),
                            )
                        ],
                    )
                    for iv in range_(0, chunk_size, tile_n):
                        off = affine_apply(offmap, [iv, _tx, _ty])
                        dma_memcpy_nd(
                            li,
                            h_in,
                            src_offsets=[off],
                            src_sizes=[tile_n],
                            src_strides=[1],
                        )
                        for j in range_(0, tile_n, 16):
                            si = subview(li.result, [j], [16], [1])
                            so = subview(lo.result, [j], [16], [1])
                            v = transfer_read(vf, si, [c0], idmap, cst0, [True])
                            v_bf = _arith.TruncFOp(vb, v)
                            transfer_write(None, v_bf, so, [c0], idmap, [True])
                            yield_([])
                        dma_memcpy_nd(
                            h_out,
                            lo,
                            dst_offsets=[off],
                            dst_sizes=[tile_n],
                            dst_strides=[1],
                        )
                        yield_([])
                    DeallocOp(li)
                    DeallocOp(lo)


def build_module_gemm_cast(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
    arch="aie2p",
    cast_tile_n=2048,
):
    """Strategy 2: external GEMM (f32-out, full tile_m) + cast launch → bf16, in
    one module. func args: (A_bf16, B_bf16, C_f32_scratch, C_bf16_out).
      launch 0: GEMM A,B -> C_f32_scratch  (emit_external_call f32-out)
      launch 1: cast C_f32_scratch -> C_bf16_out
    """
    gemm_ir = str(
        build_module(
            m,
            k,
            n,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
            bfloat16,
            np.float32,
            arch=arch,
            direct_codegen=False,
            emit_external_call=True,
        )
    )
    cast_ir = str(_build_cast_module(m, n, np.float32, tile_n=cast_tile_n))

    # mm.o symbols the GEMM links — preserve across renaming.
    externs = {"@op_has_no_registered_library_name", "@zero_f32_mn"}

    # arg layout: 0=A, 1=B, 2=C_f32_scratch, 3=C_bf16_out
    bodies, maps_all = [], []
    for ir, prefix, arg_map in [
        (gemm_ir, "gm", {0: 0, 1: 1, 2: 2}),  # A,B -> C_f32
        (cast_ir, "ct", {0: 2, 1: 3}),  # C_f32 -> C_bf16
    ]:
        body = _stitch_extract_body(ir)
        maps = _stitch_affine_maps(ir)
        body = _stitch_rename(body, prefix, externs)
        maps = [_stitch_rename(mp, prefix, externs) for mp in maps]
        body = _stitch_fix_args(body, prefix, arg_map)
        bodies.append(body)
        maps_all.extend(maps)

    privates = set()
    for p in _stitch_private_funcs(gemm_ir):
        privates.add(p.strip())
    privates_str = "\n  ".join(sorted(privates))

    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @gemm_cast_bf16(
    %arg0: memref<{m}x{k}xbf16>,
    %arg1: memref<{k}x{n}xbf16>,
    %arg2: memref<{m}x{n}xf32>,
    %arg3: memref<{m}x{n}xbf16>
  ) {{
{bodies[0]}
{bodies[1]}
    return
  }}
}}
"""
    with Context() as ctx:
        try:
            module = Module.parse(combined, ctx)
        except Exception:
            with open("/tmp/debug_gemm_cast.mlir", "w") as f:
                f.write(combined)
            print("  PARSE ERROR: dumped to /tmp/debug_gemm_cast.mlir")
            raise
        return module


_DIRECT_F32_TRANSFORM = """
            module attributes {transform.with_named_sequence} {
              transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

                %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %func0 {
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                } : !transform.any_op
                %func_fold_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op


                %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op

                %inner_most_matmul, %vec_loops:3 =
                  transform.structured.tile_using_for %matmul tile_sizes [2, 2, 1, 0, 0, 0]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)  
                %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
                  transform.structured.tile_using_for %inner_most_matmul tile_sizes [1, 1, 0, 0, 0, 0]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)  
                transform.loop.unroll %vec_loops_to_unroll#1 {factor = 2} : !transform.any_op
                transform.loop.unroll %vec_loops_to_unroll#0 {factor = 2} : !transform.any_op

                %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %inner_most_fills, %vec_fill_loops:2 =
                  transform.structured.tile_using_for %linalg_fills tile_sizes [0, 0, 1, 1]
                  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

                %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op
                
                %herd1, %herd2, %herd3 = transform.split_handle %vectorized_herds : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                %scf_fors = transform.structured.match ops{["scf.for"]} in %herd2 : (!transform.any_op) -> !transform.any_op

                %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %func1 {
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.memref.fold_memref_alias_ops
                } : !transform.any_op
                %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

                // Eliminate redundant vector.transfer_read operations
                %func1_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func1_rematch : (!transform.any_op) -> !transform.any_op
                
                // Hoist loop-invariant vector transfers out of innermost loop
                %herds_1 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %vectorized_herds_1 = transform.air.herd_vectorize %herds_1 : (!transform.any_op) -> !transform.any_op
                %herd1_1, %herd2_1, %herd3_1 = transform.split_handle %vectorized_herds_1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
                
                %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
                %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
                
                %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %result11 = transform.air.vector_type_cast %vector_contracts {target_element_type = f32, input_indices = [2], output_indices = [0]} : (!transform.any_op) -> !transform.any_op

                // Hoist all accumulator transfer pairs from the innermost loop
                %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op

                %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
                %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

                %fors_to_hoist_ptrs = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
                %innermost_for1, %outer_fors1 = transform.split_handle %fors_to_hoist_ptrs {overflow_result = 1}: (!transform.any_op) -> (!transform.any_op, !transform.any_op)


                %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                transform.apply_patterns to %func2 {
                    transform.apply_patterns.linalg.tiling_canonicalization
                    transform.apply_patterns.scf.for_loop_canonicalization
                    transform.apply_patterns.canonicalization
                    transform.apply_patterns.memref.fold_memref_alias_ops
                } : !transform.any_op
                %func_fold_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
                %func_folded_3 = transform.air.fold_unit_extent_dims %func_fold_3 : (!transform.any_op) -> !transform.any_op
              transform.yield
            }
            }
"""


def build_module_lowered(
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
    arch="aie2p",
):
    """direct-codegen f32-out GEMM, fully lowered (transform applied).

    `build_module` (a `@module_builder`) returns a module whose matmul is still a
    `linalg.generic`; the `_DIRECT_F32_TRANSFORM` script must run on the finished
    module to vectorize + lower it. This wrapper does both so every caller shares
    ONE transform definition.
    """
    mlir_module = build_module(
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
        arch=arch,
        direct_codegen=True,
    )
    transform_ir = Module.parse(_DIRECT_F32_TRANSFORM, context=mlir_module.context)
    run_transform(transform_ir, mlir_module)
    return mlir_module


# ===========================================================================
# bf16-in / f32-out GEMM — contract entry point.
#
# Output is f32, so the accumulator IS the output: the K reduction is FP32 the
# whole way and there is NO intermediate bf16 truncation — this path is always
# "high precision" (no knob needed). Two code-generation backends produce the
# same math:
#   --codegen external (default): hand-tuned mm.o microkernel, ~9.6K GFLOPS on
#       large shapes. Needs mm.o pre-compiled (make compile-kernel).
#   --codegen direct: compiler-generated from a transform script (self-contained,
#       no .o), ~5.7K GFLOPS.
# Both: mean_rel_L1 ~9.3e-3 (BFP16-emulated 8x8x8 MMA + FP32 accumulate).
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="bf16-in / f32-out GEMM (always FP32-accumulate; external or direct codegen)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-k-l2", type=int, default=256)
    parser.add_argument("--tile-k-l1", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--herd-m", type=int, default=8)
    parser.add_argument("--herd-n", type=int, default=4)
    parser.add_argument(
        "--codegen",
        choices=["external", "direct"],
        default="external",
        help="external = hand-tuned mm.o microkernel (fast, needs mm.o); "
        "direct = compiler codegen (self-contained). Both are FP32-accumulate.",
    )
    parser.add_argument("--arch", type=str, choices=["aie2", "aie2p"], default="aie2p")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-xclbin", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument("--perf-iters", type=int, default=0, dest="perf_iters")
    args = parser.parse_args()

    direct = args.codegen == "direct"
    if direct and not os.environ.get("PEANO_INSTALL_DIR"):
        print(
            "Error: PEANO_INSTALL_DIR required for --codegen direct.", file=sys.stderr
        )
        sys.exit(1)

    if direct:
        # direct-codegen: build_module_lowered applies the f32-out vectorization
        # transform internally (single shared transform definition).
        mlir_module = build_module_lowered(
            args.m,
            args.k,
            args.n,
            args.tile_m,
            args.tile_k_l2,
            args.tile_k_l1,
            args.tile_n,
            args.herd_m,
            args.herd_n,
            bfloat16,
            np.float32,
            arch=args.arch,
        )
    else:
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
            bfloat16,
            np.float32,
            arch=args.arch,
            emit_external_call=True,
        )

    if args.print_module_only:
        print(mlir_module)
        sys.exit(0)

    scale = 1.0 / math.sqrt(args.k)
    input_a = (np.random.randn(args.m, args.k) * scale).astype(bfloat16)
    input_b = (np.random.randn(args.k, args.n) * scale).astype(bfloat16)

    if args.compile_mode == "compile-and-run":
        reference = (input_a.astype(np.float32) @ input_b.astype(np.float32)).astype(
            np.float32
        )
        runner_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
            "stack_size": 2048,
            "report_precision": True,
            "n_perf_iters": args.perf_iters,
            "perf_flops": (
                (2.0 * args.m * args.k * args.n) if args.perf_iters > 0 else None
            ),
        }
        runner = XRTRunner(**runner_kwargs, instance_name="matmul_bf16")
        # Tolerance tier A (FP32-accumulate). rtol anchors to PyTorch's bf16
        # standard (torch.testing.assert_close: bf16 rtol=1.6e-2) — the GEMM
        # *computes* in bf16 (bf16 inputs + BFP16-emulated MMA), so f32 storage
        # does NOT earn f32's 1.3e-6 rtol; the error floor is bf16's. atol=1.5e-3
        # encodes the FP32-accumulate tier: measured worst-case abs_err ~5.8e-4
        # (Down 2048x8192x2048), ~2.5x margin, and tight enough to REJECT the
        # bf16-truncation tier (low-precision abs_err ~2.4e-3).
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                expected_outputs=[reference],
                rtol=1.6e-2,
                atol=1.5e-3,
            )
        )
    else:
        backend_kwargs = {
            "verbose": args.verbose,
            "omit_while_true_loop": False,
            "runtime_loop_tiling_sizes": [2, 2],
            "stack_size": 2048,
            "output_format": (
                "xclbin" if args.compile_mode == "compile-and-xclbin" else "none"
            ),
        }
        backend = XRTBackend(**backend_kwargs, instance_name="matmul_bf16")
        backend.compile(mlir_module)
        backend.unload()
        print("Compile-only done.")
