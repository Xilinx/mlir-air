# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Minimal perf test for the attn_1_group inline blocker.
#
# attn_1_group computes, for each of GROUP_SIZE Q heads:
#   attn_out[g, pos] = dot(Q[g, :HEAD], K[:HEAD]) / sqrt(HEAD)
# using a 32-lane bf16 fma loop. The C++ form uses 4 independent acc
# chains (manually unrolled) for VLIW interleaving — that gave a 2x
# speedup over a naive sequential chain in an earlier session.
#
# This test runs ITERS calls per token to amortize DMA cost, then we
# read airrt.profile to compare:
#   --variant naive      : single acc chain (Peano default scheduling)
#   --variant unroll4    : 4 independent acc chains in inline MLIR
#
# If `unroll4` matches the C++ form's perf, we can inline. If not, the
# extern is justified.
import argparse
import time

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview, store as memref_store
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    fma as vector_fma,
    reduction as vector_reduction,
)
from air.dialects import arith as arith_dialect
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

GROUP_SIZE = 4
HEAD = 64
VL = 32  # 32-lane bf16 (matches C++ aie::vector<bfloat16, 32>)


@module_builder
def build_module(n_tokens, variant):
    bf16_t = type_mapper(bfloat16)
    vec_t = VectorType.get([VL], bf16_t)
    id_map = AffineMapAttr.get(AffineMap.get_identity(1))
    map_2d_to_1d = AffineMapAttr.get(AffineMap.get(2, 0, [AffineDimExpr.get(1)]))

    # Inputs: Q [GROUP_SIZE, HEAD] bf16; K_seq [n_tokens, HEAD] bf16.
    # Output: attn_out [GROUP_SIZE, n_tokens] bf16.
    # Pack inputs into one [1 + n_tokens, HEAD] L3 buffer (Q rows then K rows).
    qk_n = (GROUP_SIZE + n_tokens) * HEAD
    out_n = GROUP_SIZE * n_tokens
    l3_in_t = MemRefType.get([qk_n], bf16_t)
    l3_out_t = MemRefType.get([out_n], bf16_t)

    l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_q_t = MemRefType.get([GROUP_SIZE, HEAD], bf16_t, memory_space=l1_mem)
    l1_k_t = MemRefType.get([HEAD], bf16_t, memory_space=l1_mem)
    l1_out_t = MemRefType.get([GROUP_SIZE, n_tokens], bf16_t, memory_space=l1_mem)

    @FuncOp.from_py_func(l3_in_t, l3_out_t)
    def k(in_buf, out_buf):

        @herd(name="herd_0", sizes=[1, 1], operands=[in_buf, out_buf])
        def _(_tx, _ty, _sx, _sy, l3_in, l3_out):
            l1_q = AllocOp(l1_q_t, [], [])
            l1_k = AllocOp(l1_k_t, [], [])
            l1_out = AllocOp(l1_out_t, [], [])

            # DMA Q (GROUP_SIZE * HEAD elements) to l1_q.
            dma_memcpy_nd(
                l1_q,
                l3_in,
                src_offsets=[0],
                src_sizes=[GROUP_SIZE * HEAD],
                src_strides=[1],
            )

            c0 = ConstantOp(IndexType.get(), 0)
            c16 = ConstantOp(IndexType.get(), 16)
            c32 = ConstantOp(IndexType.get(), VL)
            cst0 = arith.ConstantOp(bf16_t, 0.0)
            cst0_v = arith.ConstantOp(
                vec_t,
                DenseElementsAttr.get(np.zeros(VL, dtype=bfloat16), type=vec_t),
            )

            # Hoist Q loads — one [VL] vector per (g, half).
            q_lo = []
            q_hi = []
            for g in range(GROUP_SIZE):
                sub = subview(l1_q.result, [g, 0], [1, HEAD], [1, 1])
                qg_lo = transfer_read(vec_t, sub, [c0, c0], map_2d_to_1d, cst0, [True])
                # Second half via offset-c32.
                qg_hi = transfer_read(vec_t, sub, [c0, c32], map_2d_to_1d, cst0, [True])
                q_lo.append(qg_lo)
                q_hi.append(qg_hi)

            n_t = ConstantOp(IndexType.get(), n_tokens)
            one_idx = ConstantOp(IndexType.get(), 1)

            for t in range_(c0, n_t, one_idx):
                # DMA the K row for this position.
                # Note: dynamic offset = (GROUP_SIZE + t) * HEAD. Build via ConstantOp + addi.
                gs_idx = ConstantOp(IndexType.get(), GROUP_SIZE)
                t_plus_gs = arith.addi(t, gs_idx)
                # offset = t_plus_gs * HEAD. Use affine_map for the dynamic stride.
                head_idx = ConstantOp(IndexType.get(), HEAD)
                k_off = arith.muli(t_plus_gs, head_idx)
                dma_memcpy_nd(
                    l1_k,
                    l3_in,
                    src_offsets=[k_off],
                    src_sizes=[HEAD],
                    src_strides=[1],
                )

                # Load K row in two 32-lane vectors.
                k0 = transfer_read(vec_t, l1_k.result, [c0], id_map, cst0, [True])
                k1 = transfer_read(vec_t, l1_k.result, [c32], id_map, cst0, [True])

                if variant == "unroll4":
                    # 4 independent acc chains, fully unrolled (mirrors C++).
                    accs = [cst0_v.result for _ in range(GROUP_SIZE)]
                    for g in range(GROUP_SIZE):
                        accs[g] = vector_fma(q_lo[g], k0, accs[g])
                    for g in range(GROUP_SIZE):
                        accs[g] = vector_fma(q_hi[g], k1, accs[g])
                elif variant == "naive":
                    # Single chain per g (compiler decides interleaving).
                    accs = []
                    for g in range(GROUP_SIZE):
                        a = vector_fma(q_lo[g], k0, cst0_v.result)
                        a = vector_fma(q_hi[g], k1, a)
                        accs.append(a)
                else:
                    raise ValueError(variant)

                # Reduce + store per-g result at out[g, t].
                for g in range(GROUP_SIZE):
                    sc = vector_reduction(bf16_t, "add", accs[g])
                    sub_o = subview(l1_out.result, [g, 0], [1, n_tokens], [1, 1])
                    memref_store(sc, sub_o, [c0, t])
                yield_([])

            dma_memcpy_nd(
                l3_out,
                l1_out,
                dst_offsets=[0],
                dst_sizes=[out_n],
                dst_strides=[1],
            )
            DeallocOp(l1_q)
            DeallocOp(l1_k)
            DeallocOp(l1_out)


def reference(qk_packed, n_tokens):
    Q = qk_packed[: GROUP_SIZE * HEAD].reshape(GROUP_SIZE, HEAD)
    Kseq = qk_packed[GROUP_SIZE * HEAD :].reshape(n_tokens, HEAD)
    out = np.zeros((GROUP_SIZE, n_tokens), dtype=np.float32)
    for g in range(GROUP_SIZE):
        for t in range(n_tokens):
            out[g, t] = (Q[g].astype(np.float32) * Kseq[t].astype(np.float32)).sum()
    return out.astype(bfloat16)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=2048)
    p.add_argument("--variant", choices=["unroll4", "naive"], default="unroll4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    qk_packed = rng.uniform(-0.5, 0.5, (GROUP_SIZE + args.n_tokens) * HEAD).astype(
        bfloat16
    )
    out_ref = reference(qk_packed, args.n_tokens).flatten()

    mod = build_module(args.n_tokens, args.variant)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format="xclbin",
        instance_name=f"attn1_{args.variant}",
    )
    exit(
        runner.run_test(
            mod,
            inputs=[qk_packed],
            expected_outputs=[out_ref],
            rtol=2e-1,
            atol=2e-1,
        )
    )
