# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Two-tile-per-column matvec with fused residual add: D[M] = A[M, K] @ B[K] + R[M].
# BF16 input/output. 4-arg func signature (A, B, R, D) so the host can keep
# A and R as separate L3 BOs (no pre-pack).
#
# Each column runs two stacked herds connected by an intra-column
# npu_cascade channel:
#   - matvec_h (north): streams A and B from L3, accumulates a partial
#     dot product into a 32-lane L1 buffer.
#   - addr_h (south):   receives the partial via cascade, adds the
#     corresponding slice of R, writes D back to L3.
#
# AIE2P cascade payloads are fixed 512 bits = vector<32xbf16>, so the L1
# partial buffer is widened to 32 lanes even though the kernel only writes
# the first `m` (=8 default). The cascade flows north→south on AIE2P, so
# matvec_h is pinned to a higher row than addr_h.

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    BF16Type,
    BoolAttr,
    IntegerAttr,
    MemRefType,
    StringAttr,
    UnitAttr,
)
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
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.memref import cast as memref_cast
from air.dialects.scf import for_, yield_
from air.dialects import arith
from air.backend.xrt_runner import XRTRunner

KERNEL_OBJ_NAME = "mv_bf16.o"


def build_module(M, K, m=8, k=512, n_cores=8):
    """Build the 2-tile-per-column matvec+add module.

    Args:
        M: output / row count of A.
        K: inner dimension (must be divisible by `k`).
        m: rows per matvec micro-tile (must divide M / n_cores).
        k: K-chunk per matvec micro-tile (must divide K).
        n_cores: column count of each herd.
    """
    assert M % (m * n_cores) == 0
    assert K % k == 0

    M_per_core = M // n_cores
    M_div_m_per_core = M_per_core // m
    K_div_k = K // k

    @module_builder
    def build():
        bf16_ty = BF16Type.get()

        A_l3 = MemRefType.get([M, K], bf16_ty)
        B_l3 = MemRefType.get([K], bf16_ty)
        R_l3 = MemRefType.get([M], bf16_ty)
        D_l3 = MemRefType.get([M], bf16_ty)

        l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

        CASCADE_WIDTH = 32  # AIE2P cascade payload, in bf16 lanes.
        A_chunk_l2 = MemRefType.get([m * k], bf16_ty, memory_space=l2_ms)
        A_chunk_l1 = MemRefType.get([m * k], bf16_ty, memory_space=l1_ms)
        B_l1 = MemRefType.get([k], bf16_ty, memory_space=l1_ms)
        R_full_l1 = MemRefType.get([M], bf16_ty, memory_space=l1_ms)
        partial_l1 = MemRefType.get([CASCADE_WIDTH], bf16_ty, memory_space=l1_ms)
        partial_slice_ty = MemRefType.get([m], bf16_ty, memory_space=l1_ms)
        D_l1 = MemRefType.get([m], bf16_ty, memory_space=l1_ms)

        channel_decl("memA", size=[n_cores])
        channel_decl("inA", size=[n_cores])
        Channel("inB", size=[1, 1], broadcast_shape=[n_cores, 1])
        Channel("inR", size=[1, 1], broadcast_shape=[n_cores, 1])
        channel_decl("partial_cas", size=[n_cores], channel_type="npu_cascade")
        channel_decl("outD", size=[n_cores])

        zero_func = FuncOp(
            "zero_vectorized_bf16", ([partial_slice_ty], []), visibility="private"
        )
        zero_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        zero_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        matvec_func = FuncOp(
            "matvec_vectorized_bf16",
            ([A_chunk_l1, B_l1, partial_slice_ty], []),
            visibility="private",
        )
        matvec_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        matvec_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        partial_plus_r_func = FuncOp(
            "partial_plus_r_bf16",
            ([partial_slice_ty, R_full_l1, T.i32(), D_l1], []),
            visibility="private",
        )
        partial_plus_r_func.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
        partial_plus_r_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(A_l3, B_l3, R_l3, D_l3)
        def matvec_2tile_add(A, B, R, D):
            @launch(sizes=[1, 1], operands=[A, B, R, D])
            def launch_body(li, lj, lsx, lsy, a, b, r, d):
                for i in range(n_cores):
                    c_col = arith.ConstantOp.create_index(i)
                    # A: stream (m × k) micro-tiles to col i's memtile.
                    # Outer-dim offset is in micro-tile units (stride m*K).
                    ChannelPut(
                        "memA",
                        a,
                        indices=[c_col],
                        offsets=[i * M_div_m_per_core, 0, 0, 0],
                        sizes=[M_div_m_per_core, K_div_k, m, k],
                        strides=[m * K, k, K, 1],
                    )
                # B: replay the same K-chunk stream per outer iter
                # (outer-dim stride=0). R: broadcast the full vector
                # once and reuse via per-iter offset inside addr_h.
                ChannelPut(
                    "inB",
                    b,
                    offsets=[0, 0, 0],
                    sizes=[M_div_m_per_core, K_div_k, k],
                    strides=[0, k, 1],
                )
                ChannelPut(
                    "inR",
                    r,
                    offsets=[0],
                    sizes=[M],
                    strides=[1],
                )

                @segment(name="seg")
                def segment_body():
                    for i in range(n_cores):
                        c_col_s = arith.ConstantOp.create_index(i)
                        for _ in for_(M_div_m_per_core * K_div_k):
                            l2_a_op = AllocOp(A_chunk_l2, [], [])
                            l2_a = l2_a_op.result
                            ChannelGet("memA", l2_a, indices=[c_col_s])
                            ChannelPut("inA", l2_a, indices=[c_col_s])
                            DeallocOp(l2_a_op)
                            yield_([])

                    @herd(name="matvec_h", sizes=[n_cores, 1])
                    def matvec_herd(tx, ty, sx, sy):
                        for _outer in for_(M_div_m_per_core):
                            # 32-lane buf for the cascade payload; the
                            # matvec/zero kernels only touch the first
                            # `m` lanes (rest is unused padding).
                            l1_part_op = AllocOp(partial_l1, [], [])
                            l1_part_op.attributes["air.shrinkage"] = BoolAttr.get(False)
                            l1_part = l1_part_op.result
                            l1_part_slice_strided = subview(
                                l1_part,
                                [0],
                                [m],
                                [1],
                            )
                            l1_part_slice = memref_cast(
                                partial_slice_ty,
                                l1_part_slice_strided,
                            )
                            CallOp(zero_func, [l1_part_slice])
                            for _kc in for_(K_div_k):
                                l1_b_op = AllocOp(B_l1, [], [])
                                l1_b = l1_b_op.result
                                ChannelGet("inB", l1_b, indices=[tx, ty])
                                l1_a_op = AllocOp(A_chunk_l1, [], [])
                                l1_a = l1_a_op.result
                                ChannelGet("inA", l1_a, indices=[tx])
                                CallOp(matvec_func, [l1_a, l1_b, l1_part_slice])
                                DeallocOp(l1_a_op)
                                DeallocOp(l1_b_op)
                                yield_([])
                            ChannelPut("partial_cas", l1_part, indices=[tx])
                            DeallocOp(l1_part_op)
                            yield_([])

                    matvec_herd.attributes["link_with"] = StringAttr.get(
                        KERNEL_OBJ_NAME
                    )
                    # Pin matvec_h north of addr_h; cascade flows N→S on AIE2P.
                    matvec_herd.attributes["x_loc"] = IntegerAttr.get(T.i64(), 0)
                    matvec_herd.attributes["y_loc"] = IntegerAttr.get(T.i64(), 3)

                    @herd(name="addr_h", sizes=[n_cores, 1])
                    def addr_herd(tx, ty, sx, sy):
                        M_per_core_c = arith.constant(T.i32(), M_per_core)
                        m_c = arith.constant(T.i32(), m)
                        tx_i32 = arith.index_cast(T.i32(), tx)
                        core_base = arith.muli(tx_i32, M_per_core_c)

                        # Pull R once and reuse across all outer iters.
                        l1_r_op = AllocOp(R_full_l1, [], [])
                        l1_r = l1_r_op.result
                        ChannelGet("inR", l1_r, indices=[tx, ty])

                        for outer in for_(M_div_m_per_core):
                            l1_part_op = AllocOp(partial_l1, [], [])
                            l1_part_op.attributes["air.shrinkage"] = BoolAttr.get(False)
                            l1_d_op = AllocOp(D_l1, [], [])
                            l1_part = l1_part_op.result
                            l1_d = l1_d_op.result
                            ChannelGet("partial_cas", l1_part, indices=[tx])
                            l1_part_slice_strided = subview(
                                l1_part,
                                [0],
                                [m],
                                [1],
                            )
                            l1_part_slice = memref_cast(
                                partial_slice_ty,
                                l1_part_slice_strided,
                            )
                            outer_i32 = arith.index_cast(T.i32(), outer)
                            iter_off = arith.muli(outer_i32, m_c)
                            offset = arith.addi(core_base, iter_off)
                            CallOp(
                                partial_plus_r_func,
                                [l1_part_slice, l1_r, offset, l1_d],
                            )
                            ChannelPut("outD", l1_d, indices=[tx])
                            DeallocOp(l1_d_op)
                            DeallocOp(l1_part_op)
                            yield_([])
                        DeallocOp(l1_r_op)

                    addr_herd.attributes["link_with"] = StringAttr.get(KERNEL_OBJ_NAME)
                    addr_herd.attributes["x_loc"] = IntegerAttr.get(T.i64(), 0)
                    addr_herd.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                # Per-core D gets, placed AFTER @segment so source program
                # order encodes producer→consumer (#1671).
                for i in range(n_cores):
                    c_col = arith.ConstantOp.create_index(i)
                    ChannelGet(
                        "outD",
                        d,
                        indices=[c_col],
                        offsets=[i * M_per_core],
                        sizes=[M_per_core],
                        strides=[1],
                    )

    return build()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="matvec_2tile_add.py",
        description="Two-tile-per-col BF16 matvec with fused residual add: "
        "D = A @ B + R",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--tile-m", type=int, default=8, dest="tile_m")
    parser.add_argument("--k-chunk", type=int, default=512, dest="k_chunk")
    parser.add_argument("--herd-cols", type=int, default=8, dest="herd_cols")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
    )
    args = parser.parse_args()

    module = build_module(args.m, args.k, args.tile_m, args.k_chunk, args.herd_cols)
    if args.print_module_only:
        print(module)
        exit(0)

    np.random.seed(42)
    A = (np.random.randn(args.m, args.k) * 0.02).astype(bfloat16)
    B = np.random.randn(args.k).astype(bfloat16)
    R = np.random.randn(args.m).astype(bfloat16)
    D_ref = (A.astype(np.float32) @ B.astype(np.float32) + R.astype(np.float32)).astype(
        bfloat16
    )

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="matvec_2tile_add",
        use_lock_race_condition_fix=False,
    )
    exit(
        runner.run_test(
            module,
            inputs=[A, B, R],
            expected_outputs=[D_ref],
            rtol=0.05,
            atol=2.0,
            min_correlation=0.99,
        )
    )
