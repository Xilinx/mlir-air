# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# sum-of-N + weighted RMS norm, in one herd.
#
#     out = hidden_norm( P_0 + P_1 + ... + P_{n-1} )
#
# WHY IT EXISTS. The int4 `fc` is several GEMM launches over slices of K (see
# dflash_int4_fc_builder.py for why it cannot be one), each writing its own
# [M, 2560] partial. Something has to add them, and the norm that follows
# already streams each row into L1, so the sum rides along for free.
#
# n_in IS CAPPED AT 2 IN PRACTICE. A herd tile takes at most two incoming L3
# streams, so `build_module(2, ..., with_norm=False)` (two partials in, sum out)
# and `build_module(1, ..., with_norm=True)` (sum in, weight in, out) are the
# only two shapes the fc builder instantiates. Three inputs compiles and
# silently races; five crashes aircc. The measurements are in the fc builder.
#
# WHY THE SUM IS bf16 AND NOT f32. The partials arrive as bf16: the GEMM's own
# f32 accumulator is truncated by `f32_to_bf16_mn` before it ever leaves L1, so
# an f32 accumulator here would be adding exactly-representable bf16 values and
# only sharpening the LAST rounding. Measured on random partials at this shape,
# a bf16 chain-sum sits 2.97e-03 from the f32 sum and 3.39e-03 from exact, where
# the f32 sum is 1.65e-03 from exact -- i.e. it buys ~1.7e-03 against a GEMM
# whose own error is 7.3e-03. Not worth a second L1 buffer and an f32-vector
# truncf that aievec may or may not legalize.
#
# Everything below the sum is weighted_rms_norm.py's single-tile path verbatim:
# f32 sum-of-squares, f32 rsqrt, bf16 elementwise epilogue, including the
# scratch round-trip that breaks the mulf->addf chain aievec rejects.

from air.ir import (
    AffineMap,
    AffineMapAttr,
    F32Type,
    IntegerAttr,
    MemRefType,
    VectorType,
)
from air.dialects.air import MemorySpace, T, dma_memcpy_nd, herd, module_builder
from air.dialects import arith, math as math_dialect
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper

range_ = for_

EPS = 1e-5


@module_builder
def build_module(n_in, M, N, np_dtype, vector_size=16, with_norm=True):
    """func(%arg0..%arg{n_in-1} in [M,N], [%arg{n_in} w [N],] out [M,N]).

    `with_norm=False` drops the weight argument and emits the bare sum, which
    is what the fold stages of the fc tail use.
    """
    xrt_dtype = type_mapper(np_dtype)
    assert N % vector_size == 0, (N, vector_size)
    assert n_in >= 1

    vecTy = VectorType.get([vector_size], xrt_dtype)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))
    f32 = F32Type.get()
    vecTyF32 = VectorType.get([vector_size], f32)

    l3MemrefTy = MemRefType.get([M, N], xrt_dtype)
    l3WeightTy = MemRefType.get([N], xrt_dtype)

    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1RowTy = MemRefType.get([N], xrt_dtype, memory_space=l1)
    l1VecTyF32 = MemRefType.get([vector_size], f32, memory_space=l1)
    l1SqTy = MemRefType.get([vector_size], xrt_dtype, memory_space=l1)

    sig = [l3MemrefTy] * n_in + ([l3WeightTy] if with_norm else []) + [l3MemrefTy]

    @FuncOp.from_py_func(*sig)
    def sum_rms_norm(*args):
        ins = list(args[:n_in])
        rest = list(args[n_in:])

        @herd(name="herd_0", sizes=[1, 1], operands=ins + rest)
        def herd_body(_tx, _ty, _sx, _sy, *ops):
            h_ins = list(ops[:n_in])
            h_w = ops[n_in] if with_norm else None
            h_out = ops[-1]

            l1_row = AllocOp(l1RowTy, [], [])
            # ONE STAGING ROW PER ADDEND, not one reused across them. Reusing a
            # single buffer is a WAR the dependency pass does not order: addend
            # k+1's DMA lands in it while addend k's vector add is still
            # reading. Two inputs happen to be safe (2.59e-03); three is not
            # (1.12, with garbage in the padded rows). 5 KB each at N=2560.
            l1_add = [AllocOp(l1RowTy, [], []) for _ in range(max(n_in - 1, 1))]
            # l1_out is NOT optional. Draining the accumulator (l1_row)
            # straight to L3 lets the next iteration's input DMA overwrite it
            # while the drain is still in flight -- a WAR the dependency pass
            # does not order, and the symptom is a run-to-run-varying output
            # whose rows are permutations of OTHER rows' data (measured: rel
            # 0.85 / 0.88 / 0.94 / 1.04 on repeated runs of the same pure-copy
            # module). Draining a separate buffer restores the discipline the
            # norm path had by accident.
            l1_out = AllocOp(l1RowTy, [], [])
            l1_weight = AllocOp(l1RowTy, [], []) if with_norm else None
            l1_acc = AllocOp(l1VecTyF32, [], []) if with_norm else None
            l1_sq = AllocOp(l1SqTy, [], []) if with_norm else None

            c0 = arith.ConstantOp.create_index(0)
            cst0 = arith.ConstantOp(xrt_dtype, 0.0)
            if with_norm:
                cst0_f32 = arith.ConstantOp(f32, 0.0)
                n_f = arith.ConstantOp(f32, float(N))
                eps_f = arith.ConstantOp(f32, EPS)
                v_zero_f32 = BroadcastOp(vecTyF32, cst0_f32)
                dma_memcpy_nd(l1_weight, h_w)

            for row in range_(M):
                dma_memcpy_nd(
                    l1_row,
                    h_ins[0],
                    src_offsets=[row, 0],
                    src_sizes=[1, N],
                    src_strides=[N, 1],
                )
                for a, src in enumerate(h_ins[1:]):
                    dma_memcpy_nd(
                        l1_add[a],
                        src,
                        src_offsets=[row, 0],
                        src_sizes=[1, N],
                        src_strides=[N, 1],
                    )
                    for j in range_(0, N, vector_size):
                        s_row = subview(l1_row.result, [j], [vector_size], [1])
                        s_add = subview(l1_add[a].result, [j], [vector_size], [1])
                        v_r = transfer_read(
                            vecTy, s_row, [c0], identity_map, cst0, [True]
                        )
                        v_a = transfer_read(
                            vecTy, s_add, [c0], identity_map, cst0, [True]
                        )
                        transfer_write(
                            None,
                            arith.addf(v_r, v_a),
                            s_row,
                            [c0],
                            identity_map,
                            [True],
                        )
                        yield_([])

                if not with_norm:
                    for j in range_(0, N, vector_size):
                        s_row = subview(l1_row.result, [j], [vector_size], [1])
                        s_out = subview(l1_out.result, [j], [vector_size], [1])
                        transfer_write(
                            None,
                            transfer_read(
                                vecTy, s_row, [c0], identity_map, cst0, [True]
                            ),
                            s_out,
                            [c0],
                            identity_map,
                            [True],
                        )
                        yield_([])
                    dma_memcpy_nd(
                        h_out,
                        l1_out,
                        dst_offsets=[row, 0],
                        dst_sizes=[1, N],
                        dst_strides=[N, 1],
                    )
                    yield_([])
                    continue

                # --- weighted_rms_norm, single-tile path, unchanged ---
                transfer_write(None, v_zero_f32, l1_acc, [c0], identity_map, [True])
                for j in range_(0, N, vector_size):
                    sub_row = subview(l1_row.result, [j], [vector_size], [1])
                    v_x = transfer_read(
                        vecTy, sub_row, [c0], identity_map, cst0, [True]
                    )
                    v_sq = arith.mulf(v_x, v_x)
                    transfer_write(None, v_sq, l1_sq, [c0], identity_map, [True])
                    v_sq_rd = transfer_read(
                        vecTy, l1_sq, [c0], identity_map, cst0, [True]
                    )
                    v_sq_f32 = arith.extf(vecTyF32, v_sq_rd)
                    v_acc = transfer_read(
                        vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                    )
                    transfer_write(
                        None,
                        arith.addf(v_acc, v_sq_f32),
                        l1_acc,
                        [c0],
                        identity_map,
                        [True],
                    )
                    yield_([])

                v_final = transfer_read(
                    vecTyF32, l1_acc, [c0], identity_map, cst0_f32, [True]
                )
                rms = arith.divf(vector_reduction(f32, "add", v_final), n_f)
                rstd = arith.truncf(
                    xrt_dtype, math_dialect.rsqrt(arith.addf(rms, eps_f))
                )
                v_rstd = BroadcastOp(vecTy, rstd)

                for j in range_(0, N, vector_size):
                    sub_row = subview(l1_row.result, [j], [vector_size], [1])
                    sub_w = subview(l1_weight.result, [j], [vector_size], [1])
                    sub_out = subview(l1_out.result, [j], [vector_size], [1])
                    v_x = transfer_read(
                        vecTy, sub_row, [c0], identity_map, cst0, [True]
                    )
                    v_w = transfer_read(vecTy, sub_w, [c0], identity_map, cst0, [True])
                    v_normed = arith.mulf(v_x, v_rstd)
                    transfer_write(
                        None,
                        arith.mulf(v_normed, v_w),
                        sub_out,
                        [c0],
                        identity_map,
                        [True],
                    )
                    yield_([])

                dma_memcpy_nd(
                    h_out,
                    l1_out,
                    dst_offsets=[row, 0],
                    dst_sizes=[1, N],
                    dst_strides=[N, 1],
                )
                yield_([])

            DeallocOp(l1_row)
            for b in l1_add:
                DeallocOp(b)
            DeallocOp(l1_out)
            if with_norm:
                DeallocOp(l1_weight)
                DeallocOp(l1_acc)
                DeallocOp(l1_sq)


def reference(parts, weight=None, eps=EPS):
    """CPU f32 reference: hidden_norm(sum(parts)), or just sum if weight is None."""
    import numpy as np

    x = sum(np.asarray(p, np.float32) for p in parts)
    if weight is None:
        return x
    rms = np.sqrt((x**2).mean(-1, keepdims=True) + eps)
    return (x / rms) * np.asarray(weight, np.float32)


if __name__ == "__main__":
    import sys
    from ml_dtypes import bfloat16

    m = build_module(5, 32, 2560, bfloat16)
    txt = str(m)
    print(f"[sumnorm] {len(txt.splitlines())} lines, {txt.count('air.herd')} herd(s)")
    sys.exit(0)
