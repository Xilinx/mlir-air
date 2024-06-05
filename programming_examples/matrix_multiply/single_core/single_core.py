# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.affine import load, store
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_

range_ = for_

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = [TILE_WIDTH, TILE_HEIGHT]


def build_module():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())
            ChannelOp("ChanA")
            ChannelOp("ChanB")
            ChannelOp("ChanC")

            # We will send the image worth of data in and out
            @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut, memrefTyInOut)
            def copy(arg0, arg1, arg2):

                # The arguments are the input and output
                @launch(operands=[arg0, arg1, arg2])
                def launch_body(a, b, c):
                    ChannelPut("ChanA", [], a)
                    ChannelPut("ChanB", [], b)
                    ChannelGet("ChanC", [], c)

                    # The arguments are still the input and the output
                    @segment(name="seg", operands=[a, b, c])
                    def segment_body(arg2, arg3, arg4):

                        # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                        # We just need one compute core, so we ask for a 1x1 herd
                        @herd(
                            name="xaddherd", sizes=[1, 1], operands=[arg2, arg3, arg4]
                        )
                        def herd_body(tx, ty, sx, sy, a, b, c):

                            # We want to store our data in L1 memory
                            mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                            # This is the type definition of the tile
                            tile_type = MemRefType.get(
                                shape=TILE_SIZE,
                                element_type=T.i32(),
                                memory_space=mem_space,
                            )

                            # Process one tile at a time until we are done
                            for _ in for_(IMAGE_HEIGHT // IMAGE_WIDTH):
                                # We must allocate a buffer of the tile size for the input/output
                                tile_a = AllocOp(tile_type, [], [])
                                tile_b = AllocOp(tile_type, [], [])
                                tile_c = AllocOp(tile_type, [], [])

                                # Input a tile
                                ChannelGet("ChanA", [], tile_a)
                                ChannelGet("ChanB", [], tile_b)

                                # Copy the input tile into the output file while adding one
                                for j in range_(TILE_HEIGHT):
                                    for i in range_(TILE_WIDTH):
                                        val_a = load(tile_a, [i, j])
                                        val_b = load(tile_a, [i, j])
                                        val_c = arith.addi(val_a, val_b)
                                        store(val_c, tile_c, [i, j])
                                        yield_([])
                                    yield_([])

                                # Output the incremented tile
                                ChannelPut("ChanC", [], tile_c)

                                # Deallocate our L1 buffers
                                DeallocOp(tile_a)
                                DeallocOp(tile_b)
                                DeallocOp(tile_c)
                                yield_([])

                            # We are done - terminate all layers
                            HerdTerminatorOp()

                        SegmentTerminatorOp()

                    LaunchTerminatorOp()

        return module


if __name__ == "__main__":
    module = build_module()
    print(module)
