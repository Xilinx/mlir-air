# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4-AWQ GEMV: D[M] = dequant(A_q, A_s, A_z)[M, K] @ B[K].
#
# Q, S, Z are prepacked into a single L3 BO with one contiguous slab per
# kernel-call tile:
#     [ Q :  M_TILE * K_CHUNK/2 bytes uint8 ]
#     [ S :  K_CHUNK/GS * M_TILE bf16        ]
#     [ Z :  K_CHUNK/GS * M_TILE uint8       ]
# Single packed BO collapses the per-tile shim DMA to a single BD per shim
# channel (vs three with separate Q/S/Z channels), which is what makes the
# inner loop's ping-pong viable without exceeding the 2-S2MM-per-tile cap.
# The kernel slices Q, S, Z back out via pointer arithmetic — offsets are
# 32-byte aligned for typical M_TILE/K_CHUNK/GS choices.

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineSymbolExpr,
    BF16Type,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    UnitAttr,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    Channel,
    ChannelGet,
    ChannelPut,
    MemorySpace,
    T,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.air import channel as channel_decl
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects import arith
from air.dialects.scf import for_, yield_
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

KERNEL_OBJ_NAME = "mv_int4_bf16.o"


def pack_inputs(A_q, A_s, A_z, M, K, GS, M_TILE, K_CHUNK, N_CORES, M_PER_LAUNCH):
    """Pack Q+S+Z per kernel-call tile into a single contiguous L3 buffer.

    Output: uint8 array shaped [total_tiles, TILE_BYTES] where total_tiles =
    N_LAUNCHES * N_CORES * (M_PER_LAUNCH / N_CORES / M_TILE) * (K / K_CHUNK).
    """
    n_gpc = K_CHUNK // GS
    q_bytes = M_TILE * (K_CHUNK // 2)
    s_bytes = n_gpc * M_TILE * 2
    z_bytes = n_gpc * M_TILE
    tile_bytes = q_bytes + s_bytes + z_bytes

    M_per_core_per_launch = M_PER_LAUNCH // N_CORES
    M_div = M_per_core_per_launch // M_TILE
    K_div = K // K_CHUNK
    N_LAUNCHES = M // M_PER_LAUNCH

    total_tiles = N_LAUNCHES * N_CORES * M_div * K_div
    packed = np.zeros((total_tiles, tile_bytes), dtype=np.uint8)

    tile_idx = 0
    for lj in range(N_LAUNCHES):
        for c in range(N_CORES):
            base_row = lj * M_PER_LAUNCH + c * M_per_core_per_launch
            for outer in range(M_div):
                row_off = base_row + outer * M_TILE
                for kc in range(K_div):
                    q_col = kc * (K_CHUNK // 2)
                    g_off = kc * n_gpc
                    q_tile = A_q[
                        row_off : row_off + M_TILE,
                        q_col : q_col + (K_CHUNK // 2),
                    ]
                    s_tile = A_s[g_off : g_off + n_gpc, row_off : row_off + M_TILE]
                    z_tile = A_z[g_off : g_off + n_gpc, row_off : row_off + M_TILE]
                    p = packed[tile_idx]
                    p[0:q_bytes] = (
                        np.ascontiguousarray(q_tile).view(np.uint8).reshape(-1)
                    )
                    p[q_bytes : q_bytes + s_bytes] = (
                        np.ascontiguousarray(s_tile).view(np.uint8).reshape(-1)
                    )
                    p[q_bytes + s_bytes :] = (
                        np.ascontiguousarray(z_tile).view(np.uint8).reshape(-1)
                    )
                    tile_idx += 1
    return packed


def build_module(M, K, GS=128, M_TILE=8, K_CHUNK=2048, N_CORES=8, M_PER_LAUNCH=None):
    if M_PER_LAUNCH is None:
        M_PER_LAUNCH = M  # 1 launch by default

    assert M % M_PER_LAUNCH == 0
    assert M_PER_LAUNCH % N_CORES == 0
    M_per_core_per_launch = M_PER_LAUNCH // N_CORES
    assert M_per_core_per_launch % M_TILE == 0
    M_div_m_per_core = M_per_core_per_launch // M_TILE
    assert K % K_CHUNK == 0
    assert K_CHUNK % GS == 0
    K_div_k = K // K_CHUNK
    n_groups_per_chunk = K_CHUNK // GS
    N_LAUNCHES = M // M_PER_LAUNCH
    total_tiles = N_LAUNCHES * N_CORES * M_div_m_per_core * K_div_k

    q_bytes = M_TILE * (K_CHUNK // 2)
    s_bytes = n_groups_per_chunk * M_TILE * 2
    z_bytes = n_groups_per_chunk * M_TILE
    tile_bytes = q_bytes + s_bytes + z_bytes

    assert q_bytes % 32 == 0
    assert (q_bytes + s_bytes) % 32 == 0

    @module_builder
    def build():
        bf16_ty = BF16Type.get()
        i8_ty = IntegerType.get_signless(8)

        packed_l3 = MemRefType.get([total_tiles, tile_bytes], i8_ty)
        B_l3 = MemRefType.get([K], bf16_ty)
        D_l3 = MemRefType.get([M], bf16_ty)

        l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

        packed_l2 = MemRefType.get([tile_bytes], i8_ty, memory_space=l2_ms)
        packed_l1 = MemRefType.get([tile_bytes], i8_ty, memory_space=l1_ms)
        B_l1 = MemRefType.get([K_CHUNK], bf16_ty, memory_space=l1_ms)
        D_l1 = MemRefType.get([M_TILE], bf16_ty, memory_space=l1_ms)

        channel_decl("inL3", size=[N_CORES])
        Channel("inB", size=[1, 1], broadcast_shape=[N_CORES, 1])
        channel_decl("aL2ToL1", size=[N_CORES])
        channel_decl("outD", size=[N_CORES])

        zero_func = FuncOp("zero_vectorized_bf16", ([D_l1], []), visibility="private")
        zero_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        matvec_func = FuncOp(
            "matvec_int4_bf16_packed",
            ([packed_l1, B_l1, D_l1], []),
            visibility="private",
        )
        matvec_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        matvec_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(packed_l3, B_l3, D_l3)
        def matvec_int4_packed(PACKED, B, D):
            @launch(sizes=[1, N_LAUNCHES], operands=[PACKED, B, D])
            def launch_body(li, lj, lsx, lsy, packed, b, d):
                launch_to_off = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(M_PER_LAUNCH),
                        )
                    ],
                )
                launch_off = affine_apply(launch_to_off, [lj])
                tiles_per_launch_per_core = M_div_m_per_core * K_div_k
                tile_base_per_launch = arith.muli(
                    lj,
                    arith.ConstantOp.create_index(N_CORES * tiles_per_launch_per_core),
                )

                for c in range(N_CORES):
                    c_idx = arith.ConstantOp.create_index(c)
                    c_tile_const = arith.ConstantOp.create_index(
                        c * tiles_per_launch_per_core
                    )
                    core_tile_base = arith.AddIOp(
                        tile_base_per_launch, c_tile_const
                    ).result
                    # Single multi-dim BD per core covers all (outer, kc)
                    # tiles for this launch — the inner ping-pong is on the
                    # tile loop, kept hidden behind the multi-dim wrap.
                    ChannelPut(
                        "inL3",
                        packed,
                        indices=[c_idx],
                        offsets=[core_tile_base, 0],
                        sizes=[tiles_per_launch_per_core, tile_bytes],
                        strides=[tile_bytes, 1],
                    )

                # B: replay each K-chunk across outer iters (outer-stride=0).
                ChannelPut(
                    "inB",
                    b,
                    offsets=[0, 0, 0],
                    sizes=[M_div_m_per_core, K_div_k, K_CHUNK],
                    strides=[0, K_CHUNK, 1],
                )

                @segment(name="seg")
                def segment_body():
                    for c in range(N_CORES):
                        c_idx_s = arith.ConstantOp.create_index(c)
                        for _ in for_(M_div_m_per_core):
                            for _ in for_(K_div_k):
                                l2_op = AllocOp(packed_l2, [], [])
                                ChannelGet("inL3", l2_op, indices=[c_idx_s])
                                ChannelPut("aL2ToL1", l2_op, indices=[c_idx_s])
                                DeallocOp(l2_op)
                                yield_([])
                            yield_([])

                    @herd(name="matvec_h", sizes=[N_CORES, 1])
                    def herd_body(tx, ty, _sx, _sy):
                        for _outer in for_(M_div_m_per_core):
                            l1_d_op = AllocOp(D_l1, [], [])
                            CallOp(zero_func, [l1_d_op])
                            for _ in for_(K_div_k):
                                l1_packed_op = AllocOp(packed_l1, [], [])
                                l1_b_op = AllocOp(B_l1, [], [])
                                ChannelGet("aL2ToL1", l1_packed_op, indices=[tx])
                                ChannelGet("inB", l1_b_op, indices=[tx, ty])
                                CallOp(
                                    matvec_func,
                                    [l1_packed_op, l1_b_op, l1_d_op],
                                )
                                DeallocOp(l1_packed_op)
                                DeallocOp(l1_b_op)
                                yield_([])
                            ChannelPut("outD", l1_d_op, indices=[tx])
                            DeallocOp(l1_d_op)
                            yield_([])

                    herd_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
                    herd_body.attributes["x_loc"] = IntegerAttr.get(T.i64(), 0)
                    herd_body.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                # Per-core D gets, placed AFTER @segment so source program
                # order encodes producer→consumer (#1671).
                for c in range(N_CORES):
                    c_idx = arith.ConstantOp.create_index(c)
                    c_row_const = arith.ConstantOp.create_index(
                        c * M_per_core_per_launch
                    )
                    core_launch_base = arith.AddIOp(launch_off, c_row_const).result
                    ChannelGet(
                        "outD",
                        d,
                        indices=[c_idx],
                        offsets=[core_launch_base],
                        sizes=[M_per_core_per_launch],
                        strides=[1],
                    )

    return build()


def cpu_reference(A_q, A_s, A_z, B):
    M_ = A_q.shape[0]
    K_ = B.shape[0]
    n_groups = A_s.shape[0]
    gs = K_ // n_groups
    D = np.zeros(M_, dtype=np.float32)
    Bf = B.astype(np.float32)
    A_s_f = A_s.astype(np.float32)
    A_z_i = A_z.astype(np.int32)
    for r in range(M_):
        for kk in range(K_):
            byte = int(A_q[r, kk // 2])
            nib = (byte & 0x0F) if (kk % 2 == 0) else ((byte >> 4) & 0x0F)
            g = kk // gs
            D[r] += (nib - A_z_i[g, r]) * A_s_f[g, r] * Bf[kk]
    return D.astype(bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="matvec_int4_packed.py",
        description="int4-AWQ GEMV: D = dequant(A) @ B",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--m-tile", type=int, default=8, dest="m_tile")
    parser.add_argument("--k-chunk", type=int, default=2048, dest="k_chunk")
    parser.add_argument("--n-cores", type=int, default=8, dest="n_cores")
    parser.add_argument("--m-per-launch", type=int, default=None, dest="m_per_launch")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-only"],
        default="compile-and-run",
        dest="compile_mode",
    )
    args = parser.parse_args()

    module = build_module(
        args.m,
        args.k,
        GS=args.gs,
        M_TILE=args.m_tile,
        K_CHUNK=args.k_chunk,
        N_CORES=args.n_cores,
        M_PER_LAUNCH=args.m_per_launch,
    )
    if args.print_module_only:
        print(module)
        exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="matvec_int4_packed",
            use_lock_race_condition_fix=False,
            stack_size=4096,
        )
        backend.compile(module)
        backend.unload()
        exit(0)

    np.random.seed(42)
    A_q_unp = np.random.randint(0, 16, size=(args.m, args.k), dtype=np.uint8)
    A_q = (A_q_unp[:, 0::2] | (A_q_unp[:, 1::2] << 4)).astype(np.uint8)
    n_groups = args.k // args.gs
    A_s = np.random.uniform(0.005, 0.02, size=(n_groups, args.m)).astype(bfloat16)
    A_z = np.random.randint(7, 9, size=(n_groups, args.m), dtype=np.uint8)
    B = np.random.randn(args.k).astype(bfloat16)

    D_ref = cpu_reference(A_q, A_s, A_z, B)
    m_per_launch = args.m_per_launch if args.m_per_launch is not None else args.m
    PACKED = pack_inputs(
        A_q,
        A_s,
        A_z,
        args.m,
        args.k,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        m_per_launch,
    )

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="matvec_int4_packed",
        use_lock_race_condition_fix=False,
        stack_size=4096,
    )
    exit(
        runner.run_test(
            module,
            inputs=[PACKED, B],
            expected_outputs=[D_ref],
            rtol=0.1,
            atol=0.05,
            min_correlation=0.999,
        )
    )
