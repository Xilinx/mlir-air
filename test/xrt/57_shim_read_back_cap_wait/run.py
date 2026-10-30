# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Two jobs in one launch, the second reading back what the first drained.
#
# Job 0 reads x as M chunks on JobIn and drains their sum into row 0 of the
# arena. Job 1 waits for that drain, reads row 0 back on JobIn, then reads
# NTASK transfers of K chunks each on JobC, a channel only job 1 uses, and
# drains row 0 plus every JobC chunk into row 1.
#
# The core takes one chunk per receive and takes JobIn before JobC, so a JobC
# transfer finishes only after job 0 is done and job 1 has its read-back. With
# NTASK > 4, the host waits for the first JobC transfer before it starts the
# fifth. That wait has to come after job 0's last x chunk and job 1's read-back
# are issued, or the launch hangs.
#
# Each job's JobC transfers use descending offsets, so they stay separate
# tasks instead of being folded into one.

import argparse

import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

INOUT_DATATYPE = np.int32


@module_builder
def build_module(row, k, ntask, m):
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    x_ty = MemRefType.get([m * row], xrt_dtype)
    c_ty = MemRefType.get([ntask * k * row], xrt_dtype)
    arena_ty = MemRefType.get([2 * row], xrt_dtype)
    l1_ty = MemRefType.get(
        shape=[row],
        element_type=xrt_dtype,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    Channel("JobIn")
    Channel("JobC")
    Channel("JobOut")

    @FuncOp.from_py_func(x_ty, c_ty, arena_ty)
    def read_back(x, c, arena):

        @launch(operands=[x, c, arena])
        def launch_body(x_l, c_l, arena_l):
            ChannelGet("JobOut", arena_l, offsets=[0], sizes=[row], strides=[1])
            for n in range(m):
                ChannelPut(
                    "JobIn",
                    x_l,
                    offsets=[(m - 1 - n) * row],
                    sizes=[row],
                    strides=[1],
                )
            ChannelGet("JobOut", arena_l, offsets=[row], sizes=[row], strides=[1])
            ChannelPut("JobIn", arena_l, offsets=[0], sizes=[row], strides=[1])
            for n in range(ntask):
                ChannelPut(
                    "JobC",
                    c_l,
                    offsets=[(ntask - 1 - n) * k * row],
                    sizes=[k * row],
                    strides=[1],
                )

            @segment(name="seg")
            def segment_body():

                @herd(name="herd_0", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    tile_in = AllocOp(l1_ty, [], [])
                    tile_c = AllocOp(l1_ty, [], [])
                    acc = AllocOp(l1_ty, [], [])

                    # Job 0: sum the m chunks of x.
                    ChannelGet("JobIn", tile_in)
                    for i in range_(row):
                        store(load(tile_in, [i]), acc, [i])
                        yield_([])
                    for _ in range_(m - 1):
                        ChannelGet("JobIn", tile_in)
                        for i in range_(row):
                            store(
                                arith.addi(load(acc, [i]), load(tile_in, [i])),
                                acc,
                                [i],
                            )
                            yield_([])
                        yield_([])
                    ChannelPut("JobOut", acc)

                    # Job 1: the read-back first, then every JobC chunk.
                    ChannelGet("JobIn", tile_in)
                    for i in range_(row):
                        store(load(tile_in, [i]), acc, [i])
                        yield_([])
                    for _ in range_(ntask * k):
                        ChannelGet("JobC", tile_c)
                        for i in range_(row):
                            store(
                                arith.addi(load(acc, [i]), load(tile_c, [i])),
                                acc,
                                [i],
                            )
                            yield_([])
                        yield_([])
                    ChannelPut("JobOut", acc)

                    DeallocOp(tile_in)
                    DeallocOp(tile_c)
                    DeallocOp(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Read-back job followed by a paced channel of its own",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--row", type=int, default=256, help="elements per chunk")
    parser.add_argument("--k", type=int, default=8, help="chunks per JobC transfer")
    parser.add_argument("--ntask", type=int, default=6, help="JobC transfers")
    parser.add_argument("--m", type=int, default=6, help="x chunks job 0 reads")
    args = parser.parse_args()

    mlir_module = build_module(args.row, args.k, args.ntask, args.m)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    rng = np.random.default_rng(11)
    x = rng.integers(-100, 100, args.m * args.row).astype(INOUT_DATATYPE)
    c = rng.integers(-100, 100, args.ntask * args.k * args.row).astype(INOUT_DATATYPE)
    row0 = x.reshape(args.m, args.row).sum(0)
    row1 = row0 + c.reshape(args.ntask * args.k, args.row).sum(0)
    expected = np.concatenate([row0, row1]).astype(INOUT_DATATYPE)

    runner = XRTRunner(verbose=args.verbose, instance_name="read_back")
    exit(runner.run_test(mlir_module, inputs=[x, c], expected_outputs=[expected]))
