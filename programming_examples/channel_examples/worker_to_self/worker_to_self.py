# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = [TILE_WIDTH, TILE_HEIGHT]

assert IMAGE_WIDTH % TILE_WIDTH == 0
assert IMAGE_HEIGHT % TILE_HEIGHT == 0


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # Create an input/output channel pair per worker
    ChannelOp("ChanIn", size=[IMAGE_WIDTH // TILE_WIDTH, IMAGE_HEIGHT // TILE_HEIGHT])
    ChannelOp("ChanOut", size=[IMAGE_WIDTH // TILE_WIDTH, IMAGE_HEIGHT // TILE_HEIGHT])
    ChannelOp(
        "SwitchTiles", size=[IMAGE_WIDTH // TILE_WIDTH, IMAGE_HEIGHT // TILE_HEIGHT]
    )

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # Transfer one tile of data per worker
            for h in range(IMAGE_HEIGHT // TILE_HEIGHT):
                for w in range(IMAGE_WIDTH // TILE_WIDTH):
                    offset0 = IMAGE_HEIGHT * h
                    offset1 = IMAGE_HEIGHT * w

                    # Put data into the channel tile by tile
                    ChannelPut(
                        "ChanIn",
                        a,
                        indices=[w, h],
                        offsets=[offset0, offset1],
                        sizes=[TILE_HEIGHT, TILE_WIDTH],
                        strides=[IMAGE_WIDTH, 1],
                    )

            # Transfer one tile of data per worker
            for h in range(IMAGE_HEIGHT // TILE_HEIGHT):
                for w in range(IMAGE_WIDTH // TILE_WIDTH):
                    offset0 = IMAGE_HEIGHT * h
                    offset1 = IMAGE_HEIGHT * w

                    # Write data back out to the channel tile by tile
                    ChannelGet(
                        "ChanOut",
                        b,
                        indices=[w, h],
                        offsets=[offset0, offset1],
                        sizes=[TILE_HEIGHT, TILE_WIDTH],
                        strides=[IMAGE_WIDTH, 1],
                    )

            # The arguments are still the input and the output
            @segment(name="seg")
            def segment_body():

                @herd(
                    name="xaddherd",
                    sizes=[IMAGE_WIDTH // TILE_WIDTH, IMAGE_HEIGHT // TILE_HEIGHT],
                )
                def herd_body(th, tw, _sx, _sy):

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    tile_type = MemRefType.get(
                        shape=TILE_SIZE,
                        element_type=T.i32(),
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    tile_in = AllocOp(tile_type, [], [])
                    tile_out = AllocOp(tile_type, [], [])
                    tile_in2 = AllocOp(tile_type, [], [])
                    tile_out2 = AllocOp(tile_type, [], [])

                    # Copy a tile from the input image (a) into the L1 memory region (tile_in)
                    ChannelGet("ChanIn", tile_in, indices=[tw, th])

                    # Access every value in the tile
                    for j in range_(TILE_HEIGHT):
                        for i in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tile_in, [i, j])

                            # Store the output value in tile_out
                            store(val, tile_out, [i, j])
                            yield_([])
                        yield_([])

                    # Copy the output tile into the output
                    ChannelPut("SwitchTiles", tile_out, indices=[tw, th])
                    ChannelGet("SwitchTiles", tile_in2, indices=[tw, th])

                    # Access every value in the tile
                    for j in range_(TILE_HEIGHT):
                        for i in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tile_in2, [i, j])

                            # Store the output value in tile_out
                            store(val, tile_out2, [i, j])
                            yield_([])
                        yield_([])

                    ChannelGet("ChanOut", tile_out2, indices=[tw, th])

                    # Deallocate our L1 buffers
                    DeallocOp(tile_in)
                    DeallocOp(tile_out)
                    DeallocOp(tile_in2)
                    DeallocOp(tile_out2)


if __name__ == "__main__":
    module = build_module()
    print(module)
