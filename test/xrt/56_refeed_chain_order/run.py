# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Two re-fed memtile buffers sharing one logical flow.
#
# The memtile holds x and z, and sends x NX times then z NZ times on one
# channel, each as a re-feed loop (air.refeed_count). The channel is a FIFO, so
# the core must receive x NX times and then z NZ times. It forwards each block
# it receives to the host, so the output is the sequence the flow delivered.
#
# By default the core fills z, with data b it got from the host, only after it
# has received every copy of x. The order of the sends is then also a
# dependency: a DMA chain that alternates x and z waits for z after the first
# x, and the core never gets the second x that would let it fill z.
#
# With --z-from-host the host fills z directly, so z is ready as early as x and
# nothing but the DMA chain orders the two.

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

LEN = 64
NX = 4
NZ = 3

INOUT_DATATYPE = np.int32


# Receive n blocks and send each on to the host. The block is copied out
# rather than sent from the buffer it arrived in, so the next arrival cannot
# overwrite it before it has been sent.
def forward(n, blk, sent):
    for _ in range_(n):
        ChannelGet("Out", blk)
        for i in range_(LEN):
            store(load(blk, [i]), sent, [i])
            yield_([])
        ChannelPut("YOut", sent)
        yield_([])


@module_builder
def build_module(z_from_host):
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    l3_in_type = MemRefType.get([LEN], xrt_dtype)
    l3_out_type = MemRefType.get([(NX + NZ) * LEN], xrt_dtype)
    l2_type = MemRefType.get(
        [LEN], xrt_dtype, memory_space=IntegerAttr.get(T.i32(), MemorySpace.L2)
    )
    l1_type = MemRefType.get(
        [LEN], xrt_dtype, memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1)
    )

    Channel("XIn")
    Channel("BIn")
    out = Channel("Out")
    out.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
    Channel("ZFill")
    Channel("YOut")

    @FuncOp.from_py_func(l3_in_type, l3_in_type, l3_out_type)
    def refeed_chain_order(a, b, y):

        @launch(operands=[a, b, y])
        def launch_body(l3_a, l3_b, l3_y):
            ChannelPut("XIn", l3_a)
            ChannelPut("ZFill" if z_from_host else "BIn", l3_b)
            for i in range_(NX + NZ):
                off = arith.muli(i, arith.ConstantOp.create_index(LEN))
                ChannelGet("YOut", l3_y, offsets=[off], sizes=[LEN], strides=[1])
                yield_([])

            @segment(name="seg")
            def segment_body():
                x = AllocOp(l2_type, [], [])
                z = AllocOp(l2_type, [], [])
                for buf in (x, z):
                    buf.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                        T.i32(), 0
                    )
                ChannelGet("XIn", x)
                for _ in range_(NX):
                    ChannelPut("Out", x)
                    yield_([])
                ChannelGet("ZFill", z)
                for _ in range_(NZ):
                    ChannelPut("Out", z)
                    yield_([])
                DeallocOp(x)
                DeallocOp(z)

                @herd(name="fwd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    blk = AllocOp(l1_type, [], [])
                    sent = AllocOp(l1_type, [], [])
                    if z_from_host:
                        forward(NX + NZ, blk, sent)
                    else:
                        held = AllocOp(l1_type, [], [])
                        ChannelGet("BIn", held)
                        forward(NX, blk, sent)
                        ChannelPut("ZFill", held)
                        DeallocOp(held)
                        forward(NZ, blk, sent)
                    DeallocOp(blk)
                    DeallocOp(sent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Two re-fed memtile buffers sharing one logical flow",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--z-from-host",
        action="store_true",
        help="fill z from the host instead of after the core has consumed x",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module(args.z_from_host)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    a = np.arange(LEN, dtype=INOUT_DATATYPE) + 1000
    b = np.arange(LEN, dtype=INOUT_DATATYPE) + 2000
    expected = np.concatenate([a] * NX + [b] * NZ)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="refeed_chain_order",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[a, b],
            expected_outputs=[expected],
        )
    )
