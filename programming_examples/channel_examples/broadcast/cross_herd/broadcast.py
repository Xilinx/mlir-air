# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cross-herd broadcast: Herd A [1,1] puts data to a broadcast channel,
# Herd B [N,1] gets the same data on all tiles.
#
# This exercises the pattern needed for RMSNorm[1,1] → GEMV[N,1] broadcast:
# - Channel("bcast", size=[1, 1], broadcast_shape=[N, 1])
# - Herd A [1,1]: ChannelPut("bcast", buf)           — single put, no indices
# - Herd B [N,1]: ChannelGet("bcast", buf, [tx, ty]) — N gets via broadcast
#
# Test: Herd A reads input, adds 1, broadcasts.
#       Herd B receives broadcast, each tile writes to its section of output.
#       Output = N copies of (input + 1).

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.dialects.vector import transfer_read, transfer_write, BroadcastOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

VECTOR_LEN = 64       # Length of vector to broadcast (like head_dim or K)
HERD_N = 4            # Number of consumer tiles (like HERD_M in GEMV)
VEC_SIZE = 16         # SIMD width for bf16
DTYPE = bfloat16


def _make_mul_map(factor):
    return AffineMap.get(0, 1, [
        AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(factor))
    ])


@module_builder
def build_module():
    xrt = type_mapper(DTYPE)
    vecTy = VectorType.get([VEC_SIZE], xrt)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    # L3 types
    in_ty = MemRefType.get([VECTOR_LEN], xrt)
    out_ty = MemRefType.get([HERD_N * VECTOR_LEN], xrt)

    # Memory spaces
    l1s = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # L1 types
    l1_vec_ty = MemRefType.get([VECTOR_LEN], xrt, memory_space=l1s)

    # Cross-herd broadcast channel: 1 put from herd A, HERD_N gets in herd B
    Channel("bcast", size=[1, 1], broadcast_shape=[HERD_N, 1])

    @FuncOp.from_py_func(in_ty, out_ty)
    def cross_herd_broadcast(data_in, data_out):

        @launch(operands=[data_in, data_out])
        def launch_body(l_in, l_out):

            @segment(name="seg", operands=[l_in, l_out])
            def seg(s_in, s_out):

                # === Herd A [1,1]: producer — read, transform, broadcast ===
                @herd(name="producer", sizes=[1, 1], operands=[s_in])
                def producer(_tx, _ty, _sx, _sy, h_in):
                    l1_in = AllocOp(l1_vec_ty, [], [])
                    l1_out = AllocOp(l1_vec_ty, [], [])

                    # Load from L3
                    dma_memcpy_nd(l1_in, h_in)

                    # Add 1 to each element (vectorized)
                    c0 = ConstantOp(T.index(), 0)
                    cst0 = ConstantOp(xrt, 0.0)
                    one_scalar = ConstantOp(xrt, 1.0)
                    v_one = BroadcastOp(vecTy, one_scalar)
                    for j in range_(0, VECTOR_LEN, VEC_SIZE):
                        from air.dialects.memref import subview
                        sv_in = subview(l1_in.result, [j], [VEC_SIZE], [1])
                        sv_out = subview(l1_out.result, [j], [VEC_SIZE], [1])
                        v = transfer_read(vecTy, sv_in, [c0], identity_map, cst0, [True])
                        v_plus = arith.addf(v, v_one)
                        transfer_write(None, v_plus, sv_out, [c0], identity_map, [True])
                        yield_([])

                    # Broadcast to all consumer tiles
                    ChannelPut("bcast", l1_out)

                    DeallocOp(l1_in)
                    DeallocOp(l1_out)

                # === Herd B [HERD_N, 1]: consumer — receive broadcast, write output ===
                @herd(name="consumer", sizes=[HERD_N, 1], operands=[s_out])
                def consumer(_tx, _ty, _sx, _sy, h_out):
                    l1_buf = AllocOp(l1_vec_ty, [], [])

                    # Get broadcast data (each tile gets the same data)
                    ChannelGet("bcast", l1_buf, indices=[_tx, _ty])

                    # Write to output at tile-specific offset
                    mul_map = _make_mul_map(VECTOR_LEN)
                    out_off = affine_apply(mul_map, [_tx])
                    dma_memcpy_nd(h_out, l1_buf,
                        dst_offsets=[out_off],
                        dst_sizes=[VECTOR_LEN],
                        dst_strides=[1])

                    DeallocOp(l1_buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="broadcast.py",
        description="Cross-herd broadcast: [1,1] producer → [N,1] consumer via broadcast channel",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format", type=str, choices=["xclbin", "elf"],
        default="xclbin", dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Test data
    np.random.seed(42)
    data_in = (np.random.randn(VECTOR_LEN) * 2).astype(DTYPE)

    # Golden: each of HERD_N tiles gets (data_in + 1)
    expected_tile = (data_in.astype(np.float32) + 1.0).astype(DTYPE)
    expected_out = np.tile(expected_tile, HERD_N)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="cross_herd_broadcast",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[data_in],
            expected_outputs=[expected_out],
            rtol=0.01,
            atol=0.01,
        )
    )
