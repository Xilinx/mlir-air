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


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # Create two channels which will send/receive the
    # input/output data respectively
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # We will send the image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def addone(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # Read/write the data regions represented by the parameters
            # into/out of the respective channels.
            ChannelPut("ChanIn", a)
            ChannelGet("ChanOut", b)

            @segment(name="seg")
            def segment_body():

                # The herd sizes correspond to the dimensions of the
                # contiguous block of cores we are hoping to get.
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

                    # Double forloop so we have one offset per tile dimension, [tile_index0, tile_index1]
                    for tile_index0 in range_(IMAGE_SIZE[0] // TILE_SIZE[0]):
                        for tile_index1 in range_(IMAGE_SIZE[1] // TILE_SIZE[1]):
                            # We must allocate a buffer of the tile size for the input/output
                            tile_in = Alloc(tile_type)
                            tile_out = Alloc(tile_type)

                            # Convert the type of the tile size variable to the Index type
                            tile_size0 = arith.ConstantOp.create_index(TILE_SIZE[0])
                            tile_size1 = arith.ConstantOp.create_index(TILE_SIZE[1])

                            # Calculate the offset into the channel data, which is based on which tile index
                            # we are at using tile_index0 and tile_index1 (our loop vars).
                            # tile_index0 and tile_index1 are dynamic so we have to use specialized
                            # operations do to calculations on them
                            offset0 = arith.MulIOp(tile_size0, tile_index0)
                            offset1 = arith.MulIOp(tile_size1, tile_index1)

                            # Read in one tile worth of data to tile_in
                            ChannelGet(
                                # name of the channel
                                "ChanIn",
                                # the destination
                                tile_in,
                                # two dimensional offsets based on tile number in each dimension
                                dst_offsets=[
                                    offset0,
                                    offset1,
                                ],
                                # amount of data to read in each dimension per operation
                                dst_strides=[
                                    tile_size0,
                                    tile_size1,
                                ],
                                # number of operations/sections to read
                                dst_sizes=[
                                    1,
                                    1,
                                ],
                            )

                            # Copy the input tile into the output tile while adding one, one
                            # i32 values at a time.
                            for j in range_(TILE_HEIGHT):
                                for i in range_(TILE_WIDTH):
                                    # Load the input value from tile_in
                                    val_in = load(tile_in, [i, j])

                                    # Compute the output value
                                    val_out = arith.addi(
                                        val_in, arith.ConstantOp(T.i32(), 1)
                                    )

                                    # Store the output value in tile_out
                                    store(val_out, tile_out, [i, j])
                                    yield_([])
                                yield_([])

                            # Output the incremented tile using the same
                            # offset/stride/sizes as we used to get the data
                            ChannelPut(
                                "ChanOut",
                                tile_out,
                                src_offsets=[offset0, offset1],
                                src_strides=[tile_size0, tile_size1],
                                src_sizes=[1, 1],
                            )

                            # Deallocate our L1 buffers
                            Dealloc(tile_in)
                            Dealloc(tile_out)
                            yield_([])
                        yield_([])


if __name__ == "__main__":
    module = build_module()
    print(module)
