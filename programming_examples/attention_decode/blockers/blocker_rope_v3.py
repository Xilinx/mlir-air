# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# v3: more closely mirror the attention_decode rope inline structure:
#  - Inputs packed [c_lo; c_hi; s_lo; s_hi] (same vectors reused across rows)
#  - State buffer is [n_rows, 64] (matches c_data shape)
#  - n_rows iterations; each row processes 2 chunks (i=0, i=16)
#  - Hoisted cos/sin loads (4 vector<16xbf16> values) BEFORE the row loop
#
# This exercises:
#  - hoisted reads from L1 buffers feeding all subsequent iterations
#  - many subf+fma chunks all consuming the same hoisted operands
#  - 2D subview writes back to the state buffer
import argparse

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    fma as vector_fma,
)
from air.dialects import arith as arith_dialect
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

VL = 16
HEAD = 64  # head_size; half = HEAD // 2 = 32
HALF = HEAD // 2


@module_builder
def build_module(n_rows, variant):
    bf16_t = type_mapper(bfloat16)
    vec_t = VectorType.get([VL], bf16_t)
    id_map = AffineMapAttr.get(AffineMap.get_identity(1))
    map_2d_to_1d = AffineMapAttr.get(AffineMap.get(2, 0, [AffineDimExpr.get(1)]))

    # Single packed L3 input: [state(n_rows*HEAD); cos(HEAD); sin(HEAD)] flat.
    # We'll DMA the state into [n_rows, HEAD] and the cos/sin into [HEAD] L1 bufs.
    state_n = n_rows * HEAD
    csin_n = 2 * HEAD  # cos then sin
    in_n = state_n + csin_n
    l3_in_t = MemRefType.get([in_n], bf16_t)
    l3_out_t = MemRefType.get([state_n], bf16_t)

    l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_state_t = MemRefType.get([n_rows, HEAD], bf16_t, memory_space=l1_mem)
    l1_cs_t = MemRefType.get([HEAD], bf16_t, memory_space=l1_mem)

    @FuncOp.from_py_func(l3_in_t, l3_out_t)
    def k(in_buf, out_buf):

        @herd(name="herd_0", sizes=[1, 1], operands=[in_buf, out_buf])
        def _(_tx, _ty, _sx, _sy, l3_in, l3_out):
            l1_state = AllocOp(l1_state_t, [], [])
            l1_cos = AllocOp(l1_cs_t, [], [])
            l1_sin = AllocOp(l1_cs_t, [], [])

            # Single DMA per buffer, no outer loop.
            dma_memcpy_nd(
                l1_state, l3_in, src_offsets=[0], src_sizes=[state_n], src_strides=[1]
            )
            dma_memcpy_nd(
                l1_cos, l3_in, src_offsets=[state_n], src_sizes=[HEAD], src_strides=[1]
            )
            dma_memcpy_nd(
                l1_sin,
                l3_in,
                src_offsets=[state_n + HEAD],
                src_sizes=[HEAD],
                src_strides=[1],
            )

            c0 = ConstantOp(IndexType.get(), 0)
            c16 = ConstantOp(IndexType.get(), 16)
            c32 = ConstantOp(IndexType.get(), 32)
            c48 = ConstantOp(IndexType.get(), 48)
            cst0 = arith.ConstantOp(bf16_t, 0.0)

            # Hoisted cos/sin loads — same operands shared across all rows.
            c_lo = transfer_read(vec_t, l1_cos.result, [c0], id_map, cst0, [True])
            c_hi = transfer_read(vec_t, l1_cos.result, [c16], id_map, cst0, [True])
            s_lo = transfer_read(vec_t, l1_sin.result, [c0], id_map, cst0, [True])
            s_hi = transfer_read(vec_t, l1_sin.result, [c16], id_map, cst0, [True])

            for row in range(n_rows):
                for i_lo_int, cv, sv in [(0, c_lo, s_lo), (16, c_hi, s_hi)]:
                    sub_x1 = subview(l1_state.result, [row, i_lo_int], [1, VL], [1, 1])
                    sub_x2 = subview(
                        l1_state.result, [row, i_lo_int + 32], [1, VL], [1, 1]
                    )
                    x1 = transfer_read(
                        vec_t, sub_x1, [c0, c0], map_2d_to_1d, cst0, [True]
                    )
                    x2 = transfer_read(
                        vec_t, sub_x2, [c0, c0], map_2d_to_1d, cst0, [True]
                    )
                    if variant == "subf_fma":
                        m_x1c = arith_dialect.mulf(x1, cv)
                        m_x2s = arith_dialect.mulf(x2, sv)
                        out1 = arith_dialect.subf(m_x1c, m_x2s)
                        m_x2c = arith_dialect.mulf(x2, cv)
                        out2 = vector_fma(x1, sv, m_x2c)
                    elif variant == "subf_only":
                        m_x1c = arith_dialect.mulf(x1, cv)
                        m_x2s = arith_dialect.mulf(x2, sv)
                        out1 = arith_dialect.subf(m_x1c, m_x2s)
                        out2 = arith_dialect.mulf(x2, cv)
                    elif variant == "fma_only":
                        out1 = arith_dialect.mulf(x1, cv)
                        m_x2c = arith_dialect.mulf(x2, cv)
                        out2 = vector_fma(x1, sv, m_x2c)
                    elif variant == "mul_only":
                        out1 = arith_dialect.mulf(x1, cv)
                        out2 = arith_dialect.mulf(x2, sv)
                    else:
                        raise ValueError(variant)

                    transfer_write(None, out1, sub_x1, [c0, c0], map_2d_to_1d, [True])
                    transfer_write(None, out2, sub_x2, [c0, c0], map_2d_to_1d, [True])

            # DMA the (rotated) state back to L3.
            dma_memcpy_nd(
                l3_out, l1_state, dst_offsets=[0], dst_sizes=[state_n], dst_strides=[1]
            )
            DeallocOp(l1_state)
            DeallocOp(l1_cos)
            DeallocOp(l1_sin)


def reference(state_in, cos_v, sin_v, variant):
    n_rows = state_in.shape[0]
    out = state_in.copy().astype(np.float32)
    c = cos_v.astype(np.float32)
    s = sin_v.astype(np.float32)
    for row in range(n_rows):
        for i_lo in [0, 16]:
            x1 = out[row, i_lo : i_lo + 16].copy()
            x2 = out[row, i_lo + 32 : i_lo + 48].copy()
            cv = c[i_lo : i_lo + 16]
            sv = s[i_lo : i_lo + 16]
            if variant == "subf_fma":
                out[row, i_lo : i_lo + 16] = x1 * cv - x2 * sv
                out[row, i_lo + 32 : i_lo + 48] = x1 * sv + x2 * cv
            elif variant == "subf_only":
                out[row, i_lo : i_lo + 16] = x1 * cv - x2 * sv
                out[row, i_lo + 32 : i_lo + 48] = x2 * cv
            elif variant == "fma_only":
                out[row, i_lo : i_lo + 16] = x1 * cv
                out[row, i_lo + 32 : i_lo + 48] = x1 * sv + x2 * cv
            elif variant == "mul_only":
                out[row, i_lo : i_lo + 16] = x1 * cv
                out[row, i_lo + 32 : i_lo + 48] = x2 * sv
    return out.astype(bfloat16)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=5)  # 4 Q + 1 K, like attn_decode
    p.add_argument(
        "--variant",
        choices=["subf_fma", "subf_only", "fma_only", "mul_only"],
        default="subf_fma",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    state = rng.uniform(-1.0, 1.0, (args.n_rows, HEAD)).astype(bfloat16)
    cos_v = rng.uniform(-1.0, 1.0, HEAD).astype(bfloat16)
    sin_v = rng.uniform(-1.0, 1.0, HEAD).astype(bfloat16)
    out_ref = reference(state, cos_v, sin_v, args.variant)

    in_packed = np.concatenate([state.flatten(), cos_v, sin_v]).astype(bfloat16)
    out_packed = out_ref.flatten()

    mod = build_module(args.n_rows, args.variant)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format="xclbin",
        instance_name=f"rope3_{args.variant}",
    )
    exit(
        runner.run_test(
            mod,
            inputs=[in_packed],
            expected_outputs=[out_packed],
            rtol=5e-2,
            atol=5e-2,
        )
    )
