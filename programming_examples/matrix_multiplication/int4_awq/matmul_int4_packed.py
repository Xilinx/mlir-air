# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4-AWQ GEMM (prefill): C[M, N] = A[M, K] @ dequant(W)[K, N].
#
# Weight is laid out as W_q[N, K/2] (output-major) so the per-tile packed
# layout matches the int4-AWQ GEMV. One packed L3 BO per tile:
#     [ Q :  N_TILE * K_CHUNK/2 bytes uint8 ]
#     [ S :  K_CHUNK/GS * N_TILE bf16        ]
#     [ Z :  K_CHUNK/GS * N_TILE uint8       ]
# Single multi-dim BD per shim channel keeps the 2-S2MM-per-tile budget
# (packed Q+S+Z on one S2MM, A on the other).
#
# Herd is 1D over N (sizes=[N_CORES, 1]). Each core owns N_per_core=N/N_CORES
# output columns and loops over (n_outer, m_outer, k_outer). M is handled by
# the serial M-outer loop in the core; multi-launch over M would extend this.

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


def pack_inputs(W_q, W_s, W_z, M, K, N, GS, M_TILE, N_TILE, K_CHUNK, N_CORES):
    """Pack per-(n_outer, k_outer) Q+S+Z tiles into a single L3 buffer.

    Output: uint8 [N_CORES * N_div * K_div, tile_bytes] where each core gets a
    contiguous slab of N_div*K_div tiles (n_outer is outermost within a core).
    W_q shape: [N, K/2] uint8 (output-major, K packed 2 nibbles per byte).
    W_s shape: [K/GS, N] bf16.
    W_z shape: [K/GS, N] uint8.
    """
    n_gpc = K_CHUNK // GS
    q_bytes = N_TILE * (K_CHUNK // 2)
    s_bytes = n_gpc * N_TILE * 2
    z_bytes = n_gpc * N_TILE
    tile_bytes = q_bytes + s_bytes + z_bytes

    N_per_core = N // N_CORES
    N_div = N_per_core // N_TILE
    K_div = K // K_CHUNK

    total_tiles = N_CORES * N_div * K_div
    packed = np.zeros((total_tiles, tile_bytes), dtype=np.uint8)

    tile_idx = 0
    for c in range(N_CORES):
        base_col = c * N_per_core
        for n_outer in range(N_div):
            col_off = base_col + n_outer * N_TILE
            for kc in range(K_div):
                q_col_byte = kc * (K_CHUNK // 2)
                g_off = kc * n_gpc
                q_tile = W_q[
                    col_off : col_off + N_TILE,
                    q_col_byte : q_col_byte + (K_CHUNK // 2),
                ]
                s_tile = W_s[g_off : g_off + n_gpc, col_off : col_off + N_TILE]
                z_tile = W_z[g_off : g_off + n_gpc, col_off : col_off + N_TILE]
                p = packed[tile_idx]
                p[0:q_bytes] = np.ascontiguousarray(q_tile).view(np.uint8).reshape(-1)
                p[q_bytes : q_bytes + s_bytes] = (
                    np.ascontiguousarray(s_tile).view(np.uint8).reshape(-1)
                )
                p[q_bytes + s_bytes :] = (
                    np.ascontiguousarray(z_tile).view(np.uint8).reshape(-1)
                )
                tile_idx += 1
    return packed


def build_module(M, K, N, GS=128, M_TILE=16, N_TILE=16, K_CHUNK=128, N_CORES=4):
    assert M % M_TILE == 0
    M_div = M // M_TILE
    assert N % N_CORES == 0
    N_per_core = N // N_CORES
    assert N_per_core % N_TILE == 0
    N_div = N_per_core // N_TILE
    assert K % K_CHUNK == 0
    assert K_CHUNK % GS == 0
    K_div = K // K_CHUNK
    n_gpc = K_CHUNK // GS

    total_tiles = N_CORES * N_div * K_div

    q_bytes = N_TILE * (K_CHUNK // 2)
    s_bytes = n_gpc * N_TILE * 2
    z_bytes = n_gpc * N_TILE
    tile_bytes = q_bytes + s_bytes + z_bytes

    assert q_bytes % 32 == 0
    assert (q_bytes + s_bytes) % 32 == 0

    tiles_per_core = N_div * K_div

    @module_builder
    def build():
        bf16_ty = BF16Type.get()
        i8_ty = IntegerType.get_signless(8)

        packed_l3 = MemRefType.get([total_tiles, tile_bytes], i8_ty)
        A_l3 = MemRefType.get([M, K], bf16_ty)
        C_l3 = MemRefType.get([M, N], bf16_ty)

        l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)

        packed_l1 = MemRefType.get([tile_bytes], i8_ty, memory_space=l1_ms)
        A_l1 = MemRefType.get([M_TILE, K_CHUNK], bf16_ty, memory_space=l1_ms)
        C_l1 = MemRefType.get([M_TILE, N_TILE], bf16_ty, memory_space=l1_ms)

        channel_decl("inL3", size=[N_CORES])
        Channel("inA", size=[1, 1], broadcast_shape=[N_CORES, 1])
        channel_decl("outC", size=[N_CORES])

        zero_func = FuncOp(
            "zero_vectorized_bf16_mn", ([C_l1], []), visibility="private"
        )
        zero_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        matmul_func = FuncOp(
            "matmul_int4_bf16_packed",
            ([packed_l1, A_l1, C_l1], []),
            visibility="private",
        )
        matmul_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        matmul_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # Launch's lj axis iterates m_outer; lj * M_TILE is the row offset
        # into A and C for this launch.
        lj_to_row_map = AffineMap.get(
            0,
            1,
            [
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(0),
                    AffineConstantExpr.get(M_TILE),
                )
            ],
        )

        @FuncOp.from_py_func(packed_l3, A_l3, C_l3)
        def matmul_int4_packed(PACKED, A, C):
            @launch(sizes=[1, M_div], operands=[PACKED, A, C])
            def launch_body(li, lj, lsx, lsy, packed, a, c):
                m_row_off = affine_apply(lj_to_row_map, [lj])
                for cc in range(N_CORES):
                    c_idx = arith.ConstantOp.create_index(cc)
                    c_tile_const = arith.ConstantOp.create_index(cc * tiles_per_core)
                    # Packed weight: each launch re-streams all N_div*K_div
                    # tiles for this core. No stride-0 dim — matches GEMV
                    # packed-put pattern.
                    ChannelPut(
                        "inL3",
                        packed,
                        indices=[c_idx],
                        offsets=[c_tile_const, 0],
                        sizes=[tiles_per_core, tile_bytes],
                        strides=[tile_bytes, 1],
                    )
                    # Output C: per launch, write M_TILE rows x N_per_core
                    # cols starting at row m_row_off, col cc*N_per_core.
                    c_n_const = arith.ConstantOp.create_index(cc * N_per_core)
                    ChannelGet(
                        "outC",
                        c,
                        indices=[c_idx],
                        offsets=[0, m_row_off, c_n_const],
                        sizes=[N_div, M_TILE, N_TILE],
                        strides=[N_TILE, N, 1],
                    )

                # A: per launch, broadcast the M_TILE-row band to all cores.
                # n_outer stride 0 = replay the same A tile for each n_outer.
                # Matches GEMV B-put stride-0 outer pattern.
                ChannelPut(
                    "inA",
                    a,
                    offsets=[0, 0, m_row_off, 0],
                    sizes=[N_div, K_div, M_TILE, K_CHUNK],
                    strides=[0, K_CHUNK, K, 1],
                )

                @segment(name="seg")
                def segment_body():
                    @herd(name="mm_h", sizes=[N_CORES, 1])
                    def herd_body(tx, ty, _sx, _sy):
                        for _n_outer in for_(N_div):
                            l1_c_op = AllocOp(C_l1, [], [])
                            CallOp(zero_func, [l1_c_op])
                            for _ in for_(K_div):
                                l1_p_op = AllocOp(packed_l1, [], [])
                                l1_a_op = AllocOp(A_l1, [], [])
                                ChannelGet("inL3", l1_p_op, indices=[tx])
                                ChannelGet("inA", l1_a_op, indices=[tx, ty])
                                CallOp(
                                    matmul_func,
                                    [l1_p_op, l1_a_op, l1_c_op],
                                )
                                DeallocOp(l1_p_op)
                                DeallocOp(l1_a_op)
                                yield_([])
                            ChannelPut("outC", l1_c_op, indices=[tx])
                            DeallocOp(l1_c_op)
                            yield_([])

                    herd_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
                    herd_body.attributes["x_loc"] = IntegerAttr.get(T.i64(), 0)
                    herd_body.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

    return build()


def cpu_reference(W_q, W_s, W_z, A):
    """W is stored as [N, K/2] uint8 (output-major). dequant(W)[k, n]."""
    N_ = W_q.shape[0]
    K_ = A.shape[1]
    M_ = A.shape[0]
    n_groups = W_s.shape[0]
    gs = K_ // n_groups
    Af = A.astype(np.float32)
    W_s_f = W_s.astype(np.float32)
    W_z_i = W_z.astype(np.int32)

    # Dequantize W into [K, N] f32.
    W_dq = np.zeros((K_, N_), dtype=np.float32)
    for n in range(N_):
        for kk in range(K_):
            byte = int(W_q[n, kk // 2])
            nib = (byte & 0x0F) if (kk % 2 == 0) else ((byte >> 4) & 0x0F)
            g = kk // gs
            W_dq[kk, n] = (nib - W_z_i[g, n]) * W_s_f[g, n]
    C = Af @ W_dq
    return C.astype(bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="matmul_int4_packed.py",
        description="int4-AWQ GEMM: C[M,N] = A[M,K] @ dequant(W)[K,N]",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--m-tile", type=int, default=16, dest="m_tile")
    parser.add_argument("--n-tile", type=int, default=16, dest="n_tile")
    parser.add_argument("--k-chunk", type=int, default=128, dest="k_chunk")
    parser.add_argument("--n-cores", type=int, default=4, dest="n_cores")
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
        args.n,
        GS=args.gs,
        M_TILE=args.m_tile,
        N_TILE=args.n_tile,
        K_CHUNK=args.k_chunk,
        N_CORES=args.n_cores,
    )
    if args.print_module_only:
        print(module)
        exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            output_format=args.output_format,
            instance_name="matmul_int4_packed",
            use_lock_race_condition_fix=True,
            stack_size=16384,
        )
        backend.compile(module)
        backend.unload()
        exit(0)

    np.random.seed(42)
    W_q_unp = np.random.randint(0, 16, size=(args.n, args.k), dtype=np.uint8)
    W_q = (W_q_unp[:, 0::2] | (W_q_unp[:, 1::2] << 4)).astype(np.uint8)
    n_groups = args.k // args.gs
    W_s = np.random.uniform(0.005, 0.02, size=(n_groups, args.n)).astype(bfloat16)
    W_z = np.random.randint(7, 9, size=(n_groups, args.n), dtype=np.uint8)
    A = np.random.randn(args.m, args.k).astype(bfloat16)

    C_ref = cpu_reference(W_q, W_s, W_z, A)
    PACKED = pack_inputs(
        W_q,
        W_s,
        W_z,
        args.m,
        args.k,
        args.n,
        args.gs,
        args.m_tile,
        args.n_tile,
        args.k_chunk,
        args.n_cores,
    )

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="matmul_int4_packed",
        use_lock_race_condition_fix=True,
        stack_size=16384,
    )
    exit(
        runner.run_test(
            module,
            inputs=[PACKED, A],
            expected_outputs=[C_ref],
            rtol=0.1,
            atol=0.05,
            min_correlation=0.999,
        )
    )
