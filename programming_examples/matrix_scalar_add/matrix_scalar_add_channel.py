# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

from common import *

range_ = for_


def build_module():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())
            ChannelOp("ChanIn")
            ChannelOp("ChanOut")

            # We will send the image worth of data in and out
            @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
            def addone(arg0, arg1):

                # The arguments are the input and output
                @launch(operands=[arg0, arg1])
                def launch_body(a, b):
                    ChannelPut("ChanIn", a)
                    ChannelGet("ChanOut", b)

                    # The arguments are still the input and the output
                    @segment(name="seg")
                    def segment_body():

                        # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                        # We just need one compute core, so we ask for a 1x1 herd
                        @herd(name="addherd", sizes=[1, 1])
                        def herd_body(tx, ty, sx, sy):

                            # We want to store our data in L1 memory
                            mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                            # This is the type definition of the tile
                            tile_type = MemRefType.get(
                                shape=TILE_SIZE,
                                element_type=T.i32(),
                                memory_space=mem_space,
                            )

                            # Double forloop so we have one offset per file dimension, [t0, t1]
                            for t0 in range_(IMAGE_SIZE[0] // TILE_SIZE[0]):
                                for t1 in range_(IMAGE_SIZE[1] // TILE_SIZE[1]):
                                    # We must allocate a buffer of the tile size for the input/output
                                    tile_in = Alloc(tile_type)
                                    tile_out = Alloc(tile_type)

                                    # Calculate offsets
                                    m0 = arith.ConstantOp.create_index(TILE_SIZE[0])
                                    m1 = arith.ConstantOp.create_index(TILE_SIZE[1])
                                    o0 = arith.MulIOp(m0, t0)
                                    o1 = arith.MulIOp(m1, t1)

                                    # Input a tile
                                    ChannelGet(
                                        "ChanIn",
                                        tile_in,
                                        dst_offsets=[o0, o1],
                                        dst_strides=[m0, m1],
                                        dst_sizes=[1, 1],
                                    )

                                    # Copy the input tile into the output file while adding one
                                    for j in range_(TILE_HEIGHT):
                                        for i in range_(TILE_WIDTH):
                                            val0 = load(tile_in, [i, j])
                                            val1 = arith.addi(
                                                val0, arith.ConstantOp(T.i32(), 1)
                                            )
                                            store(val1, tile_out, [i, j])
                                            yield_([])
                                        yield_([])

                                    # Output the incremented tile
                                    ChannelPut(
                                        "ChanOut",
                                        tile_out,
                                        src_offsets=[o0, o1],
                                        src_strides=[m0, m1],
                                        src_sizes=[1, 1],
                                    )

                                    # Deallocate our L1 buffers
                                    Dealloc(tile_in)
                                    Dealloc(tile_out)
                                    yield_([])
                                yield_([])

                            # We are done - terminate all layers
                            HerdTerminatorOp()

                        SegmentTerminatorOp()

                    LaunchTerminatorOp()

        return module


if __name__ == "__main__":
    module = build_module()
    print(module)
