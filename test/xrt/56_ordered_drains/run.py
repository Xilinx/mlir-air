# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A launch that chains jobs through one host buffer: job k reads the row job
# k-1 drained into that buffer, adds 1 and drains row k. So the launch reads
# back, from host memory, what its own device-to-host drains wrote.
#
# air-to-std used to arm every drain up front and await them all at the launch
# terminator, which is only correct if the launch never reads its own drains.
# Here job k + 1's input needs drain k to have landed, and more than 4 drains on
# one shim channel do not fit its task queue, so such a launch hung
# (ERT_CMD_STATE_TIMEOUT) or ran out of buffer descriptors. The compiler now
# orders each drain against the input that reads it by the region each touches:
# drain k is awaited right before the input that reads row k, and the channels
# keep a bounded number of tasks in flight.
#
#   row 0 = x + 1, row 1 = x + 2, ... row N-1 = x + N

import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects import arith
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

ROW = 16  # int32 elements per job
DEFAULT_JOBS = 6  # more than the 4 drains a shim channel queues

INOUT_DATATYPE = np.int32


@module_builder
def build_module(num_jobs):
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    row_ty = MemRefType.get([ROW], xrt_dtype)
    arena_ty = MemRefType.get([num_jobs * ROW], xrt_dtype)
    l1_ty = MemRefType.get(
        shape=[ROW],
        element_type=xrt_dtype,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    Channel("JobIn")
    Channel("JobOut")

    @FuncOp.from_py_func(row_ty, arena_ty)
    def chain(x, arena):

        @launch(operands=[x, arena])
        def launch_body(x_l, arena_l):
            # Job k's drain is armed before its input is issued, as a builder that
            # chains jobs does: the input reads what drain k - 1 wrote.
            for k in range(num_jobs):
                ChannelGet(
                    "JobOut", arena_l, offsets=[k * ROW], sizes=[ROW], strides=[1]
                )
                if k == 0:
                    ChannelPut("JobIn", x_l)
                else:
                    ChannelPut(
                        "JobIn",
                        arena_l,
                        offsets=[(k - 1) * ROW],
                        sizes=[ROW],
                        strides=[1],
                    )

            @segment(name="seg")
            def segment_body():

                @herd(name="herd_0", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    tile_in = AllocOp(l1_ty, [], [])
                    tile_out = AllocOp(l1_ty, [], [])

                    # Unrolled: the core's job loop is not what this test is about.
                    for _job in range(num_jobs):
                        ChannelGet("JobIn", tile_in)
                        for i in range_(ROW):
                            val = load(tile_in, [i])
                            store(
                                arith.addi(val, ConstantOp(xrt_dtype, 1)), tile_out, [i]
                            )
                            yield_([])
                        ChannelPut("JobOut", tile_out)

                    DeallocOp(tile_in)
                    DeallocOp(tile_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Chain of jobs through one host buffer",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="jobs in the chain",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module(args.jobs)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    x = np.arange(ROW, dtype=INOUT_DATATYPE) + 100
    expected = np.concatenate([x + (k + 1) for k in range(args.jobs)]).astype(
        INOUT_DATATYPE
    )

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="chain",
    )
    exit(runner.run_test(mlir_module, inputs=[x], expected_outputs=[expected]))
