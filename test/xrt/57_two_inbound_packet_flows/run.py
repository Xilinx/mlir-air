# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Two inbound packet flows into one core.
#
# The host sends x and r, ROWS rows each, on two packet channels from two
# independent shim DMAs. The core takes one row of each per iteration and sends
# back x - r. A packet header routes an arrival to the tile, not to a BD, so the
# core's receive side must keep the two flows apart however their arrivals
# interleave.

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

ROWS = 32
LEN = 64

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    l3_type = MemRefType.get([ROWS * LEN], xrt_dtype)
    l1_type = MemRefType.get(
        [LEN], xrt_dtype, memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1)
    )

    for name in ("XIn", "RIn"):
        ch = Channel(name)
        ch.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
    Channel("YOut")

    @FuncOp.from_py_func(l3_type, l3_type, l3_type)
    def two_inbound_packet_flows(x, r, y):

        @launch(operands=[x, r, y])
        def launch_body(l3_x, l3_r, l3_y):
            ChannelPut("XIn", l3_x)
            ChannelPut("RIn", l3_r)
            for i in range_(ROWS):
                off = arith.muli(i, arith.ConstantOp.create_index(LEN))
                ChannelGet("YOut", l3_y, offsets=[off], sizes=[LEN], strides=[1])
                yield_([])

            @segment(name="seg")
            def segment_body():

                @herd(name="sub", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    bx = AllocOp(l1_type, [], [])
                    br = AllocOp(l1_type, [], [])
                    by = AllocOp(l1_type, [], [])
                    for _ in range_(ROWS):
                        ChannelGet("XIn", bx)
                        ChannelGet("RIn", br)
                        for j in range_(LEN):
                            d = arith.subi(load(bx, [j]), load(br, [j]))
                            store(d, by, [j])
                            yield_([])
                        ChannelPut("YOut", by)
                        yield_([])
                    DeallocOp(bx)
                    DeallocOp(br)
                    DeallocOp(by)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py", description="Two inbound packet flows into one core"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    x = np.arange(ROWS * LEN, dtype=INOUT_DATATYPE) * 3 + 100000
    r = np.arange(ROWS * LEN, dtype=INOUT_DATATYPE)
    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="two_inbound_packet_flows",
    )
    exit(runner.run_test(mlir_module, inputs=[x, r], expected_outputs=[x - r]))
