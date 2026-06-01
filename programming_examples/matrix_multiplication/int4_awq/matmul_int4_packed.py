# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4-AWQ GEMM (prefill). 2D herd over (M, N), with K accumulated per-PE
# inside the herd; per-PE drain into a 4D L2 C [herd_m, herd_n, tile_m,
# tile_n]. Uses matmul_int4_bf16_packed from mv_int4_bf16.cc (packed
# Q+S+Z weight BO; bf16 activation and output).

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineSymbolExpr,
    BF16Type,
    F32Type,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ShapedType,
    StridedLayoutAttr,
    StringAttr,
    UnitAttr,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    MemorySpace,
    T,
    dma_memcpy_nd,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

KERNEL_OBJ_NAME = "mv_int4_bf16.o"


def packed_tile_bytes(n_tile, k_chunk, gs):
    n_gpc = k_chunk // gs
    q_bytes = n_tile * (k_chunk // 2)
    s_bytes = n_gpc * n_tile * 2
    z_bytes = n_gpc * n_tile
    return q_bytes, s_bytes, z_bytes, q_bytes + s_bytes + z_bytes


def pack_inputs(W_q, W_s, W_z, M, K, N, GS, N_TILE, K_CHUNK):
    """Pack per-(n_outer, k_outer) Q+S+Z tiles into [N_div, K_div, tile_bytes].

    W_q [N, K/2] u8 (output-major), W_s [K/GS, N] bf16, W_z [K/GS, N] u8.
    """
    n_gpc = K_CHUNK // GS
    q_bytes, s_bytes, _, tile_bytes = packed_tile_bytes(N_TILE, K_CHUNK, GS)
    N_div = N // N_TILE
    K_div = K // K_CHUNK
    packed = np.zeros((N_div, K_div, tile_bytes), dtype=np.uint8)
    for n_outer in range(N_div):
        col_off = n_outer * N_TILE
        for k_outer in range(K_div):
            q_col_byte = k_outer * (K_CHUNK // 2)
            g_off = k_outer * n_gpc
            q_tile = W_q[
                col_off : col_off + N_TILE,
                q_col_byte : q_col_byte + (K_CHUNK // 2),
            ]
            s_tile = W_s[g_off : g_off + n_gpc, col_off : col_off + N_TILE]
            z_tile = W_z[g_off : g_off + n_gpc, col_off : col_off + N_TILE]
            p = packed[n_outer, k_outer]
            p[0:q_bytes] = np.ascontiguousarray(q_tile).view(np.uint8).reshape(-1)
            p[q_bytes : q_bytes + s_bytes] = (
                np.ascontiguousarray(s_tile).view(np.uint8).reshape(-1)
            )
            p[q_bytes + s_bytes :] = (
                np.ascontiguousarray(z_tile).view(np.uint8).reshape(-1)
            )
    return packed


def cpu_reference(W_q, W_s, W_z, A):
    N_ = W_q.shape[0]
    K_ = A.shape[1]
    n_groups = W_s.shape[0]
    gs = K_ // n_groups
    Af = A.astype(np.float32)
    W_s_f = W_s.astype(np.float32)
    W_z_i = W_z.astype(np.int32)
    W_dq = np.zeros((K_, N_), dtype=np.float32)
    for n in range(N_):
        for kk in range(K_):
            byte = int(W_q[n, kk // 2])
            nib = (byte & 0x0F) if (kk % 2 == 0) else ((byte >> 4) & 0x0F)
            g = kk // gs
            W_dq[kk, n] = (nib - W_z_i[g, n]) * W_s_f[g, n]
    C = Af @ W_dq
    return C.astype(bfloat16)


@module_builder
def build_module(m, k, n, gs, tile_m, tile_k_l2, tile_k_l1, tile_n, herd_m, herd_n):
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0
    assert tile_k_l2 % tile_k_l1 == 0
    assert tile_k_l1 % gs == 0
    # Kernel-side static_assert constraints from mv_int4_bf16.cc:
    #   mm_int4_bf16_mmul_impl: tile_m/n/k_chunk % 8 (mmul dims), gs % R=32
    #   zero_vectorized_bf16_mn: (tile_m * tile_n) % VW=32
    assert (
        tile_m % 8 == 0 and tile_n % 8 == 0 and tile_k_l1 % 8 == 0
    ), "tile_m, tile_n, tile_k_l1 must each be multiples of 8 (mmul tile size)"
    assert gs % 32 == 0, "gs must be a multiple of dequant inner-vector width 32"
    assert (tile_m * tile_n) % 32 == 0, (
        f"tile_m*tile_n ({tile_m}*{tile_n}={tile_m * tile_n}) must be a multiple "
        f"of vector width 32 for zero_vectorized_bf16_mn"
    )

    _, _, _, tile_bytes = packed_tile_bytes(tile_n, tile_k_l1, gs)
    k_per_l2 = tile_k_l2 // tile_k_l1
    N_div = n // tile_n
    K_div = k // tile_k_l1

    bf16_ty = BF16Type.get()
    f32_ty = F32Type.get()
    u8_ty = IntegerType.get_signless(8)

    A_l3_ty = MemRefType.get([m, k], bf16_ty)
    B_l3_ty = MemRefType.get([N_div, K_div, tile_bytes], u8_ty)
    C_l3_ty = MemRefType.get([m, n], bf16_ty)

    l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

    A_l2_ty = MemRefType.get(
        [herd_m, 1, tile_m, tile_k_l2], bf16_ty, memory_space=l2_ms
    )
    B_l2_ty = MemRefType.get(
        [1, herd_n, k_per_l2, tile_bytes], u8_ty, memory_space=l2_ms
    )
    C_l2_ty = MemRefType.get(
        [herd_m, herd_n, tile_m, tile_n], bf16_ty, memory_space=l2_ms
    )

    A_l1_ty = MemRefType.get([tile_m, tile_k_l1], bf16_ty, memory_space=l1_ms)
    B_l1_ty = MemRefType.get([tile_bytes], u8_ty, memory_space=l1_ms)
    # L1 C accumulator: f32. Kept across the host K-chunk loop so partial sums
    # don't bf16-truncate between calls. Converted to bf16 once at the end.
    C_l1_acc_ty = MemRefType.get([tile_m, tile_n], f32_ty, memory_space=l1_ms)
    C_l1_drain_ty = MemRefType.get([tile_m, tile_n], bf16_ty, memory_space=l1_ms)

    zero_func = FuncOp(
        "zero_vectorized_f32_mn",
        ([C_l1_acc_ty], []),
        visibility="private",
    )
    zero_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
    zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    matmul_func = FuncOp(
        "matmul_int4_bf16_packed_f32",
        ([B_l1_ty, A_l1_ty, C_l1_acc_ty], []),
        visibility="private",
    )
    matmul_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
    matmul_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    f32_to_bf16_func = FuncOp(
        "f32_to_bf16_mn",
        ([C_l1_acc_ty, C_l1_drain_ty], []),
        visibility="private",
    )
    f32_to_bf16_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
    f32_to_bf16_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(A_l3_ty, B_l3_ty, C_l3_ty)
    def matmul_int4_packed(arg_a, arg_b, arg_c):
        launch_size = [m // tile_m // herd_m, n // tile_n // herd_n]

        @launch(operands=[arg_a, arg_b, arg_c], sizes=launch_size)
        def launch_body(li, lj, lsx, lsy, l3_a, l3_b, l3_c):
            @segment(name="seg", operands=[li, lj, l3_a, l3_b, l3_c])
            def segment_body(li_s, lj_s, l3_a_s, l3_b_s, l3_c_s):
                l2_a = AllocOp(A_l2_ty, [], [])
                l2_b = AllocOp(B_l2_ty, [], [])
                l2_c = AllocOp(C_l2_ty, [], [])

                ix_to_row = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_m),
                        )
                    ],
                )
                iy_to_n_outer = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(herd_n)
                        )
                    ],
                )
                row_off = affine_apply(ix_to_row, [li_s])
                n_outer_off = affine_apply(iy_to_n_outer, [lj_s])

                k_l2_to_k = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_k_l2)
                        )
                    ],
                )
                k_l2_to_chunk = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(k_per_l2)
                        )
                    ],
                )
                k_chunk_off_l1_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0), AffineConstantExpr.get(tile_k_l1)
                        )
                    ],
                )

                for i in for_(0, k // tile_k_l2):
                    k_l2_off = affine_apply(k_l2_to_k, [i])
                    k_chunk_off = affine_apply(k_l2_to_chunk, [i])

                    dma_memcpy_nd(
                        l2_a,
                        l3_a_s,
                        src_offsets=[0, 0, row_off, k_l2_off],
                        src_sizes=[herd_m, 1, tile_m, tile_k_l2],
                        src_strides=[k * tile_m, tile_k_l2, k, 1],
                    )
                    dma_memcpy_nd(
                        l2_b,
                        l3_b_s,
                        src_offsets=[0, n_outer_off, k_chunk_off, 0],
                        src_sizes=[1, herd_n, k_per_l2, tile_bytes],
                        src_strides=[
                            K_div * tile_bytes,
                            K_div * tile_bytes,
                            tile_bytes,
                            1,
                        ],
                    )

                    @herd(
                        name="herd_0",
                        sizes=[herd_m, herd_n],
                        operands=[l2_a, l2_b, l2_c],
                    )
                    def compute_body(_tx, _ty, _sx, _sy, _l2a, _l2b, _l2c):
                        _l1_a = AllocOp(A_l1_ty, [], [])
                        _l1_b = AllocOp(B_l1_ty, [], [])
                        _l1_c_acc = AllocOp(C_l1_acc_ty, [], [])
                        _l1_c_drain = AllocOp(C_l1_drain_ty, [], [])
                        CallOp(zero_func, [_l1_c_acc])
                        for j in for_(0, k_per_l2):
                            k1_off = affine_apply(k_chunk_off_l1_map, [j])
                            dma_memcpy_nd(
                                _l1_a,
                                _l2a,
                                src_offsets=[_tx, 0, 0, k1_off],
                                src_sizes=[1, 1, tile_m, tile_k_l1],
                                src_strides=[
                                    tile_m * tile_k_l2,
                                    tile_m * tile_k_l2,
                                    tile_k_l2,
                                    1,
                                ],
                            )
                            dma_memcpy_nd(
                                _l1_b,
                                _l2b,
                                src_offsets=[0, _ty, j, 0],
                                src_sizes=[1, 1, 1, tile_bytes],
                                src_strides=[
                                    herd_n * k_per_l2 * tile_bytes,
                                    k_per_l2 * tile_bytes,
                                    tile_bytes,
                                    1,
                                ],
                            )
                            CallOp(matmul_func, [_l1_b, _l1_a, _l1_c_acc])
                            yield_([])
                        # Convert f32 accumulator → bf16 once per launch.
                        CallOp(f32_to_bf16_func, [_l1_c_acc, _l1_c_drain])
                        dma_memcpy_nd(
                            _l2c,
                            _l1_c_drain,
                            dst_offsets=[_tx, _ty, 0, 0],
                            dst_sizes=[1, 1, tile_m, tile_n],
                            dst_strides=[
                                herd_n * tile_m * tile_n,
                                tile_m * tile_n,
                                tile_n,
                                1,
                            ],
                        )
                        DeallocOp(_l1_a)
                        DeallocOp(_l1_b)
                        DeallocOp(_l1_c_acc)
                        DeallocOp(_l1_c_drain)

                    yield_([])

                col_off_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_n * herd_n),
                        )
                    ],
                )
                col_off = affine_apply(col_off_map, [lj_s])
                dma_memcpy_nd(
                    l3_c_s,
                    l2_c,
                    dst_offsets=[row_off, col_off],
                    dst_sizes=[herd_m * tile_m, herd_n * tile_n],
                    dst_strides=[n, 1],
                    src_offsets=[0, 0, 0, 0],
                    src_sizes=[herd_m, tile_m, herd_n, tile_n],
                    src_strides=[
                        herd_n * tile_m * tile_n,
                        tile_n,
                        tile_m * tile_n,
                        1,
                    ],
                )

                DeallocOp(l2_a)
                DeallocOp(l2_b)
                DeallocOp(l2_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=16, dest="tile_m")
    parser.add_argument("--tile-k-l2", type=int, default=128, dest="tile_k_l2")
    parser.add_argument("--tile-k-l1", type=int, default=128, dest="tile_k_l1")
    parser.add_argument("--tile-n", type=int, default=16, dest="tile_n")
    parser.add_argument("--herd-m", type=int, default=2, dest="herd_m")
    parser.add_argument("--herd-n", type=int, default=4, dest="herd_n")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-and-run", "compile-only"],
        default="compile-and-run",
        dest="compile_mode",
    )
    args = parser.parse_args()

    module = build_module(
        args.m,
        args.k,
        args.n,
        args.gs,
        args.tile_m,
        args.tile_k_l2,
        args.tile_k_l1,
        args.tile_n,
        args.herd_m,
        args.herd_n,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    np.random.seed(42)
    W_q_unp = np.random.randint(0, 16, size=(args.n, args.k), dtype=np.uint8)
    W_q = (W_q_unp[:, 0::2] | (W_q_unp[:, 1::2] << 4)).astype(np.uint8)
    n_groups = args.k // args.gs
    W_s = np.random.uniform(0.005, 0.02, size=(n_groups, args.n)).astype(bfloat16)
    W_z = np.random.randint(7, 9, size=(n_groups, args.n), dtype=np.uint8)
    A = np.random.randn(args.m, args.k).astype(bfloat16)

    PACKED = pack_inputs(
        W_q, W_s, W_z, args.m, args.k, args.n, args.gs, args.tile_n, args.tile_k_l1
    )
    C_ref = cpu_reference(W_q, W_s, W_z, A)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="xclbin",
            runtime_loop_tiling_sizes=[2, 2],
            stack_size=16384,
        )
        backend.compile(module)
        backend.unload()
        sys.exit(0)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format="xclbin",
        runtime_loop_tiling_sizes=[2, 2],
        stack_size=16384,
    )
    sys.exit(
        runner.run_test(
            module,
            inputs=[A, PACKED],
            expected_outputs=[C_ref],
            rtol=0.1,
            atol=0.05,
            # bf16 floor: at large K and tight atol a small fraction of
            # elements land just outside atol while correlation stays > 0.9999.
            max_mismatch_percentage=0.05,
            min_correlation=0.999,
        )
    )
