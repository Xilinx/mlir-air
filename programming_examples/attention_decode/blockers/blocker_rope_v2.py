# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# v2: pack the 4 inputs into a single [4, N] L3 buffer so we stay under the
# 2-S2MM/tile limit. Output is a packed [2, N] buffer.
#
# Per-iter body (matches shuffle_apply_rope half-step):
#   out1 = x1 * c - x2 * s
#   out2 = x1 * s + x2 * c
#
# --variant selects which arithmetic body to emit:
#   subf_fma : the failing pattern (subf + fma combined)
#   subf_only: out1 only (passes)
#   fma_only : out2 only (passes)
#   mul_only : two muls back to back (passes)
#   addf     : addf-after-mulf (Peano llc fails to compile)
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


@module_builder
def build_module(n, tile_n, vl, variant):
    bf16_t = type_mapper(bfloat16)
    vec_t = VectorType.get([vl], bf16_t)
    id_map = AffineMapAttr.get(AffineMap.get_identity(1))
    map_2d_to_1d = AffineMapAttr.get(AffineMap.get(2, 0, [AffineDimExpr.get(1)]))

    l3_in_t = MemRefType.get([4, n], bf16_t)  # [x1; x2; c; s]
    l3_out_t = MemRefType.get([2, n], bf16_t)  # [out1; out2]

    l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)
    # L1: pack the 4 inputs into one 4×tile_n buffer (1 DMA flow).
    l1_in_t = MemRefType.get([4, tile_n], bf16_t, memory_space=l1_mem)
    l1_out_t = MemRefType.get([2, tile_n], bf16_t, memory_space=l1_mem)

    @FuncOp.from_py_func(l3_in_t, l3_out_t)
    def k(in_buf, out_buf):

        @herd(name="herd_0", sizes=[1, 1], operands=[in_buf, out_buf])
        def _(_tx, _ty, _sx, _sy, l3_in, l3_out):
            l1_in = AllocOp(l1_in_t, [], [])
            l1_out = AllocOp(l1_out_t, [], [])

            for ivx in range_(0, n, tile_n):
                # Single DMA: stage [4, tile_n] from L3.
                dma_memcpy_nd(
                    l1_in,
                    l3_in,
                    src_offsets=[0, ivx],
                    src_sizes=[4, tile_n],
                    src_strides=[n, 1],
                )

                c0 = ConstantOp(IndexType.get(), 0)
                cTile = ConstantOp(IndexType.get(), tile_n)
                cVl = ConstantOp(IndexType.get(), vl)
                cst0 = arith.ConstantOp(bf16_t, 0.0)

                for j in range_(c0, cTile, cVl):
                    sub_x1 = subview(l1_in.result, [0, 0], [1, vl], [1, 1])
                    sub_x2 = subview(l1_in.result, [1, 0], [1, vl], [1, 1])
                    sub_c = subview(l1_in.result, [2, 0], [1, vl], [1, 1])
                    sub_s = subview(l1_in.result, [3, 0], [1, vl], [1, 1])
                    sub_o1 = subview(l1_out.result, [0, 0], [1, vl], [1, 1])
                    sub_o2 = subview(l1_out.result, [1, 0], [1, vl], [1, 1])

                    # Use j as the dim-1 index so the read scans across tile_n.
                    x1 = transfer_read(
                        vec_t, sub_x1, [c0, j], map_2d_to_1d, cst0, [True]
                    )
                    x2 = transfer_read(
                        vec_t, sub_x2, [c0, j], map_2d_to_1d, cst0, [True]
                    )
                    cv = transfer_read(
                        vec_t, sub_c, [c0, j], map_2d_to_1d, cst0, [True]
                    )
                    sv = transfer_read(
                        vec_t, sub_s, [c0, j], map_2d_to_1d, cst0, [True]
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
                    elif variant == "addf":
                        m_x1s = arith_dialect.mulf(x1, sv)
                        m_x2c = arith_dialect.mulf(x2, cv)
                        out1 = arith_dialect.mulf(x1, cv)
                        out2 = arith_dialect.addf(m_x1s, m_x2c)
                    else:
                        raise ValueError(f"unknown variant {variant}")

                    transfer_write(None, out1, sub_o1, [c0, j], map_2d_to_1d, [True])
                    transfer_write(None, out2, sub_o2, [c0, j], map_2d_to_1d, [True])
                    yield_([])

                # Single DMA out: write the full [2, tile_n] block.
                dma_memcpy_nd(
                    l3_out,
                    l1_out,
                    dst_offsets=[0, ivx],
                    dst_sizes=[2, tile_n],
                    dst_strides=[n, 1],
                )
                yield_([])

            DeallocOp(l1_in)
            DeallocOp(l1_out)


def reference(x1, x2, c, s, variant):
    f = lambda a: a.astype(np.float32)
    if variant == "subf_fma":
        out1 = (f(x1) * f(c) - f(x2) * f(s)).astype(bfloat16)
        out2 = (f(x1) * f(s) + f(x2) * f(c)).astype(bfloat16)
    elif variant == "subf_only":
        out1 = (f(x1) * f(c) - f(x2) * f(s)).astype(bfloat16)
        out2 = (f(x2) * f(c)).astype(bfloat16)
    elif variant == "fma_only":
        out1 = (f(x1) * f(c)).astype(bfloat16)
        out2 = (f(x1) * f(s) + f(x2) * f(c)).astype(bfloat16)
    elif variant == "mul_only":
        out1 = (f(x1) * f(c)).astype(bfloat16)
        out2 = (f(x2) * f(s)).astype(bfloat16)
    elif variant == "addf":
        out1 = (f(x1) * f(c)).astype(bfloat16)
        out2 = (f(x1) * f(s) + f(x2) * f(c)).astype(bfloat16)
    return out1, out2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--vl", type=int, default=16)
    p.add_argument(
        "--variant",
        choices=["subf_fma", "subf_only", "fma_only", "mul_only", "addf"],
        default="subf_fma",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    x1 = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    x2 = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    c = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    s = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)

    in_packed = np.stack([x1, x2, c, s], axis=0)  # [4, n]
    out1_ref, out2_ref = reference(x1, x2, c, s, args.variant)
    out_packed = np.stack([out1_ref, out2_ref], axis=0)  # [2, n]

    mod = build_module(args.n, args.tile_n, args.vl, args.variant)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format="xclbin",
        instance_name=f"rope_{args.variant}",
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
