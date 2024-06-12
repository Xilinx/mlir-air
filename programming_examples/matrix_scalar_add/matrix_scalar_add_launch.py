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

            # Create two channels which will send/receive the
            # input/output data respectively
            ChannelOp("ChanIn")
            ChannelOp("ChanOut")

            # We will send the image worth of data in and out
            @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
            def addone(arg0, arg1):

                # The arguments are the input and output
                @launch(
                    sizes=[
                        IMAGE_SIZE[0] // TILE_SIZE[0],
                        IMAGE_SIZE[1] // TILE_SIZE[1],
                    ],
                    operands=[arg0, arg1],
                )
                def launch_body(tile_index0, _s0, tile_index1, _s1, a, b):

                    # Calculate the offset into the channel data, which is based on which tile index
                    # we are at using tile_index0 and tile_index1 (provided in the launch context).
                    # tile_index0 and tile_index1 are dynamic so we have to use specialized
                    # operations do to calculations on them
                    tile_size0 = arith.ConstantOp.create_index(TILE_SIZE[0])
                    tile_size1 = arith.ConstantOp.create_index(TILE_SIZE[1])
                    offset0 = arith.MulIOp(tile_size0, tile_index0)
                    offset1 = arith.MulIOp(tile_size1, tile_index1)
                    ChannelPut(
                        "ChanIn",
                        a,
                        src_offsets=[offset0, offset1],
                        src_strides=[tile_size0, tile_size1],
                        src_sizes=[1, 1],
                    )
                    ChannelGet(
                        "ChanOut",
                        b,
                        dst_offsets=[offset0, offset1],
                        dst_strides=[tile_size0, tile_size1],
                        dst_sizes=[1, 1],
                    )

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

                            # We must allocate a buffer of the tile size for the input/output
                            tile_in = Alloc(tile_type)
                            tile_out = Alloc(tile_type)

                            # Input a tile
                            ChannelGet("ChanIn", tile_in)

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

                            # Output the incremented tile
                            ChannelPut("ChanOut", tile_out)

                            # Deallocate our L1 buffers
                            Dealloc(tile_in)
                            Dealloc(tile_out)

                            # We are done - terminate all layers
                            HerdTerminatorOp()

                        SegmentTerminatorOp()

                    LaunchTerminatorOp()

        return module


if __name__ == "__main__":
    module = build_module()
    print(module)
