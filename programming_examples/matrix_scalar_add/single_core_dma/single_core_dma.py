# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path  # if you haven't already done so

# Python paths are a bit complex. Taking solution from : https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_

range_ = for_

from common import *


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

                    # Loop over columns and rows of tiles
                    for tile_index0 in range_(IMAGE_HEIGHT // TILE_HEIGHT):
                        for tile_index1 in range_(IMAGE_WIDTH // TILE_WIDTH):

                            # We must allocate a buffer of tile size for the input/output
                            tile_in = AllocOp(tile_type, [], [])
                            tile_out = AllocOp(tile_type, [], [])

                            # Convert the type of the tile size variable to the Index type
                            tile_size0 = arith.ConstantOp.create_index(IMAGE_HEIGHT)
                            tile_size1 = arith.ConstantOp.create_index(IMAGE_HEIGHT)

                            # Calculate the offset into the channel data, which is based on our loop vars
                            offset0 = arith.MulIOp(tile_size0, tile_index0)
                            offset1 = arith.MulIOp(tile_size1, tile_index1)
                            tile_num = arith.MulIOp(
                                tile_index0,
                                arith.ConstantOp.create_index(
                                    IMAGE_HEIGHT // TILE_HEIGHT
                                ),
                            )
                            tile_num = arith.AddIOp(tile_num, tile_index1)

                            # Copy a tile from the input image (a) into the L1 memory region (tile_in)
                            dma_memcpy_nd(
                                tile_in,
                                a,
                                src_offsets=[offset0, offset1],
                                src_sizes=[TILE_HEIGHT, TILE_WIDTH],
                                src_strides=[IMAGE_WIDTH, 1],
                            )

                            # Access every value in the tile
                            for j in range_(TILE_HEIGHT):
                                for i in range_(TILE_WIDTH):
                                    # Load the input value from tile_in
                                    val_in = load(tile_in, [i, j])

                                    # Compute the output value
                                    val_out = arith.addi(
                                        val_in, arith.index_cast(T.i32(), tile_num)
                                    )

                                    # Store the output value in tile_out
                                    store(val_out, tile_out, [i, j])
                                    yield_([])
                                yield_([])

                            # Copy the output tile into the output
                            dma_memcpy_nd(
                                b,
                                tile_out,
                                dst_offsets=[offset0, offset1],
                                dst_sizes=[TILE_HEIGHT, TILE_WIDTH],
                                dst_strides=[IMAGE_WIDTH, 1],
                            )

                            # Deallocate our L1 buffers
                            DeallocOp(tile_in)
                            DeallocOp(tile_out)

                            yield_([])
                        yield_([])


if __name__ == "__main__":
    module = build_module()
    print(module)
