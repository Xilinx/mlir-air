# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Minimal e2e for the sinf/cosf inline blocker — the "vector mask"
# pieces only. Sin/cos polynomial inline needs a mask derived from the
# low bits of a per-lane integer. This test exercises just that primitive:
#
#   for lane j: out[j] = (int(x[j]) & 1 == 0) ? a[j] : b[j]
#
# That requires:
#   arith.fptosi   : vector<16xbf16> -> vector<16xi16>
#   arith.andi     : vector<16xi16> & 1 -> vector<16xi16>
#   arith.cmpi eq  : vector<16xi16> -> vector<16xi1>
#   arith.select   : (vector<16xi1>, vector<16xbf16>, vector<16xbf16>) -> vector<16xbf16>
#
# All four steps need to lower cleanly to AIE2P. Pass = blocker is mechanical.
# Fail = sinf/cosf inline is structurally infeasible without new conversion
# patterns.
import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp
from air.dialects import arith as arith_dialect
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

VL = 16


@module_builder
def build_module(n, tile_n, step):
    bf16_t = type_mapper(bfloat16)
    i16_t = T.i16()
    i32_t = T.i32()
    f32_t = T.f32()
    vec_bf_t = VectorType.get([VL], bf16_t)
    vec_i16_t = VectorType.get([VL], i16_t)
    vec_i32_t = VectorType.get([VL], i32_t)
    vec_f32_t = VectorType.get([VL], f32_t)
    id_map = AffineMapAttr.get(AffineMap.get_identity(1))
    map_2d_to_1d = AffineMapAttr.get(AffineMap.get(2, 0, [AffineDimExpr.get(1)]))

    # Pack 3 inputs (x, a, b) into [3, n] L3, write 1 output [n] L3.
    l3_in_t = MemRefType.get([3, n], bf16_t)
    l3_out_t = MemRefType.get([n], bf16_t)
    l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_in_t = MemRefType.get([3, tile_n], bf16_t, memory_space=l1_mem)
    l1_out_t = MemRefType.get([tile_n], bf16_t, memory_space=l1_mem)

    @FuncOp.from_py_func(l3_in_t, l3_out_t)
    def k(in_buf, out_buf):

        @herd(name="herd_0", sizes=[1, 1], operands=[in_buf, out_buf])
        def _(_tx, _ty, _sx, _sy, l3_in, l3_out):
            l1_in = AllocOp(l1_in_t, [], [])
            l1_out = AllocOp(l1_out_t, [], [])

            for ivx in range_(0, n, tile_n):
                dma_memcpy_nd(
                    l1_in,
                    l3_in,
                    src_offsets=[0, ivx],
                    src_sizes=[3, tile_n],
                    src_strides=[n, 1],
                )

                c0 = ConstantOp(IndexType.get(), 0)
                cTile = ConstantOp(IndexType.get(), tile_n)
                cVl = ConstantOp(IndexType.get(), VL)
                cst0 = arith.ConstantOp(bf16_t, 0.0)

                # Constants for the mask construction.
                if step in ("fptosi", "andi", "cmpi", "select", "select_via_f32"):
                    one_i32 = arith.ConstantOp(i32_t, 1)
                    zero_i32 = arith.ConstantOp(i32_t, 0)

                for j in range_(c0, cTile, cVl):
                    sub_x = subview(l1_in.result, [0, 0], [1, VL], [1, 1])
                    sub_a = subview(l1_in.result, [1, 0], [1, VL], [1, 1])
                    sub_b = subview(l1_in.result, [2, 0], [1, VL], [1, 1])
                    sub_o = subview(l1_out.result, [j], [VL], [1])

                    vx = transfer_read(
                        vec_bf_t, sub_x, [c0, j], map_2d_to_1d, cst0, [True]
                    )
                    va = transfer_read(
                        vec_bf_t, sub_a, [c0, j], map_2d_to_1d, cst0, [True]
                    )
                    vb = transfer_read(
                        vec_bf_t, sub_b, [c0, j], map_2d_to_1d, cst0, [True]
                    )

                    # Workaround: bf16 doesn't have a direct fptosi legalization
                    # in Peano on AIE2P (G_FPTOSI <16 x s16> -> <16 x s16> fails).
                    # Promote bf16 -> f32 first, then fptosi -> i32.
                    if step in ("fptosi", "andi", "cmpi", "select"):
                        vxf = arith_dialect.extf(vec_f32_t, vx)
                        vi = arith_dialect.fptosi(vec_i32_t, vxf)

                    if step == "fptosi":
                        # Round-trip: i32 -> f32 -> bf16. The trailing
                        # i32->f32->bf16 hits "G_SITOFP <16 x s32> -> <16 x s16>"
                        # legalization. Workaround: write the i32 buffer instead.
                        vf = arith_dialect.sitofp(vec_f32_t, vi)
                        out = arith_dialect.truncf(vec_bf_t, vf)
                    elif step == "andi":
                        vone = BroadcastOp(vec_i32_t, one_i32).result
                        vand = arith_dialect.andi(vi, vone)
                        vf = arith_dialect.sitofp(vec_f32_t, vand)
                        out = arith_dialect.truncf(vec_bf_t, vf)
                    elif step == "cmpi":
                        vone = BroadcastOp(vec_i32_t, one_i32).result
                        vand = arith_dialect.andi(vi, vone)
                        vzero = BroadcastOp(vec_i32_t, zero_i32).result
                        # cmpi predicate: 0 = eq
                        mask = arith_dialect.cmpi(0, vand, vzero)
                        one_bf = arith.ConstantOp(bf16_t, 1.0)
                        zero_bf_v = BroadcastOp(vec_bf_t, cst0).result
                        one_bf_v = BroadcastOp(vec_bf_t, one_bf).result
                        out = arith_dialect.select(mask, one_bf_v, zero_bf_v)
                    elif step == "select":
                        vone = BroadcastOp(vec_i32_t, one_i32).result
                        vand = arith_dialect.andi(vi, vone)
                        vzero = BroadcastOp(vec_i32_t, zero_i32).result
                        mask = arith_dialect.cmpi(0, vand, vzero)
                        out = arith_dialect.select(mask, va, vb)
                    else:
                        raise ValueError(step)

                    transfer_write(None, out, sub_o, [c0], id_map, [True])
                    yield_([])

                dma_memcpy_nd(
                    l3_out,
                    l1_out,
                    dst_offsets=[ivx],
                    dst_sizes=[tile_n],
                    dst_strides=[1],
                )
                yield_([])

            DeallocOp(l1_in)
            DeallocOp(l1_out)


def reference(x, a, b, step):
    xi = x.astype(np.float32).astype(np.int16)
    if step == "fptosi":
        return xi.astype(np.float32).astype(bfloat16)
    elif step == "andi":
        return (xi & 1).astype(np.float32).astype(bfloat16)
    elif step == "cmpi":
        return ((xi & 1) == 0).astype(np.float32).astype(bfloat16)
    elif step == "select":
        cond = (xi & 1) == 0
        return np.where(cond, a, b).astype(bfloat16)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument(
        "--step",
        choices=["fptosi", "andi", "cmpi", "select"],
        default="fptosi",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    # x in a small integer range so fptosi is unambiguous
    x = rng.integers(-7, 8, args.n).astype(np.float32).astype(bfloat16)
    a = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    b = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    in_packed = np.stack([x, a, b], axis=0)
    out_ref = reference(x, a, b, args.step)

    mod = build_module(args.n, args.tile_n, args.step)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format="xclbin",
        instance_name=f"sin_mask_{args.step}",
    )
    exit(
        runner.run_test(
            mod,
            inputs=[in_packed],
            expected_outputs=[out_ref],
            rtol=5e-2,
            atol=5e-2,
        )
    )
