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
from air.dialects.affine import apply as affine_apply

range_ = for_

from common import *


@module_builder
def build_module():
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, T.i32())

    # Create two channels which will send/receive the
    # input/output data respectively
    ChannelOp("ChanIn")
    ChannelOp("ChanOut")

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(
            sizes=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH],
            operands=[arg0, arg1],
        )
        def launch_body(tile_index0, tile_index1, _launch_size_x, _launch_size_y, a, b):
            scaled_index_map = AffineMap.get(
                0,
                1,
                [
                    AffineExpr.get_mul(
                        AffineSymbolExpr.get(0),
                        AffineConstantExpr.get(IMAGE_HEIGHT),
                    )
                ],
            )
            offset0 = affine_apply(scaled_index_map, [tile_index0])
            offset1 = affine_apply(scaled_index_map, [tile_index1])

            # Put data into the channel tile by tile
            ChannelPut(
                "ChanIn",
                a,
                offsets=[offset0, offset1],
                sizes=[TILE_HEIGHT, TILE_WIDTH],
                strides=[IMAGE_WIDTH, 1],
            )

            # Write data back out to the channel tile by tile
            ChannelGet(
                "ChanOut",
                b,
                offsets=[offset0, offset1],
                sizes=[TILE_HEIGHT, TILE_WIDTH],
                strides=[IMAGE_WIDTH, 1],
            )

            # The arguments are still the input and the output
            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="xaddherd", sizes=[1, 1], operands=[arg2, arg3])
                def herd_body(tx, ty, sx, sy, a, b):

                    # Loop over columns and rows of tiles
                    for tile_num in range_(
                        (IMAGE_WIDTH // TILE_WIDTH) * (IMAGE_HEIGHT // TILE_HEIGHT)
                    ):

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
                        ChannelGet("ChanIn", tile_in)

                        # Access every value in the tile
                        for j in range_(TILE_HEIGHT):
                            for i in range_(TILE_WIDTH):
                                # Load the input value from tile_in
                                val_in = load(tile_in, [i, j])

                                # Compute the output value TODO(hunhoffe): this is not correct, not sure how to percolate launch info here
                                val_out = arith.addi(
                                    val_in, arith.index_cast(T.i32(), tile_num)
                                )

                                # Store the output value in tile_out
                                store(
                                    val_out,
                                    tile_out,
                                    [i, j],
                                )
                                yield_([])
                            yield_([])

                        # Copy the output tile into the output
                        ChannelPut("ChanOut", tile_out)

                        # Deallocate our L1 buffers
                        DeallocOp(tile_in)
                        DeallocOp(tile_out)

                        yield_([])


if __name__ == "__main__":
    module = build_module()
    print(module)
