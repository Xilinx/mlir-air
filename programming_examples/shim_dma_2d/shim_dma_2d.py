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
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = [TILE_HEIGHT, TILE_WIDTH]

assert IMAGE_HEIGHT % TILE_HEIGHT == 0
assert IMAGE_WIDTH % TILE_WIDTH == 0


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # The arguments are still the input and the output
            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="xaddherd", sizes=[1, 1], operands=[arg2, arg3])
                def herd_body(_tx, _ty, _sx, _sy, a, b):
                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # This is the type definition of the tile
                    tile_type = MemRefType.get(
                        shape=TILE_SIZE,
                        element_type=T.i32(),
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    tile_in = AllocOp(tile_type, [], [])
                    tile_out = AllocOp(tile_type, [], [])

                    # Copy a tile from the input image (a) into the L1 memory region (tile_in)
                    dma_memcpy_nd(
                        tile_in,
                        a,
                        src_offsets=[0, 0],
                        src_sizes=TILE_SIZE,
                        src_strides=[IMAGE_WIDTH, 1],
                    )

                    # Access every value in the tile
                    for i in range_(TILE_HEIGHT):
                        for j in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tile_in, [i, j])

                            # Store the output value in tile_out
                            store(val, tile_out, [i, j])
                            yield_([])
                        yield_([])

                    # Copy the output tile into the output
                    dma_memcpy_nd(
                        b,
                        tile_out,
                        dst_offsets=[0, 0],
                        dst_sizes=TILE_SIZE,
                        dst_strides=[IMAGE_WIDTH, 1],
                    )

                    # Deallocate our L1 buffers
                    DeallocOp(tile_in)
                    DeallocOp(tile_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
