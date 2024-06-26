# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.dialects.affine import apply as affine_apply

range_ = for_

from data_config import *


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
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # Transform data into contiguous tiles
            for tile_index0 in range_(IMAGE_HEIGHT // TILE_HEIGHT):
                for tile_index1 in range_(IMAGE_WIDTH // TILE_WIDTH):
                    # Convert the type of the tile size variable to the Index type
                    tile_size0 = arith.ConstantOp.create_index(IMAGE_HEIGHT)
                    tile_size1 = arith.ConstantOp.create_index(IMAGE_HEIGHT)

                    # Calculate the offset into the channel data, which is based on which tile index
                    # we are at using tile_index0 and tile_index1 (our loop vars).
                    # tile_index0 and tile_index1 are dynamic so we have to use specialized
                    # operations do to calculations on them
                    offset0 = arith.MulIOp(tile_size0, tile_index0)
                    offset1 = arith.MulIOp(tile_size1, tile_index1)

                    # Put data into the channel tile by tile
                    ChannelPut(
                        "ChanIn",
                        [],
                        a,
                        src_offsets=[offset0, offset1],
                        src_sizes=[
                            arith.ConstantOp.create_index(TILE_HEIGHT),
                            arith.ConstantOp.create_index(TILE_WIDTH),
                        ],
                        src_strides=[
                            arith.ConstantOp.create_index(IMAGE_WIDTH),
                            arith.ConstantOp.create_index(1),
                        ],
                    )
                    yield_([])
                yield_([])

            for tile_index0 in range_(IMAGE_HEIGHT // TILE_HEIGHT):
                for tile_index1 in range_(IMAGE_WIDTH // TILE_WIDTH):
                    # Convert the type of the tile size variable to the Index type
                    tile_size0 = arith.ConstantOp.create_index(IMAGE_HEIGHT)
                    tile_size1 = arith.ConstantOp.create_index(IMAGE_HEIGHT)

                    # Calculate the offset into the channel data, which is based on which tile index
                    # we are at using tile_index0 and tile_index1 (our loop vars).
                    # tile_index0 and tile_index1 are dynamic so we have to use specialized
                    # operations do to calculations on them
                    offset0 = arith.MulIOp(tile_size0, tile_index0)
                    offset1 = arith.MulIOp(tile_size1, tile_index1)

                    # Write data back out to the channel tile by tile
                    ChannelGet(
                        "ChanOut",
                        [],
                        b,
                        dst_offsets=[offset0, offset1],
                        dst_sizes=[
                            arith.ConstantOp.create_index(TILE_HEIGHT),
                            arith.ConstantOp.create_index(TILE_WIDTH),
                        ],
                        dst_strides=[
                            arith.ConstantOp.create_index(IMAGE_WIDTH),
                            arith.ConstantOp.create_index(1),
                        ],
                    )
                    yield_([])
                yield_([])

            # The arguments are still the input and the output
            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We are hoping to map each tile to a different compute core.
                @herd(
                    name="xaddherd",
                    sizes=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH],
                    operands=[arg2, arg3],
                )
                def herd_body(tx, ty, sx, sy, a, b):
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
                    create_tile_index = AffineMap.get(
                        0,
                        2,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineSymbolExpr.get(1),
                            )
                        ],
                    )
                    offset0 = affine_apply(scaled_index_map, [tx])
                    compute_tile_id = affine_apply(create_tile_index, [offset0, ty])

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
                    ChannelGet(
                        "ChanIn",
                        [],
                        tile_in,
                    )

                    # Access every value in the tile
                    for j in range_(TILE_HEIGHT):
                        for i in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val_in = load(tile_in, [i, j])

                            # Compute the output value
                            val_out = arith.addi(
                                val_in, arith.index_cast(T.i32(), compute_tile_id)
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
                    ChannelPut(
                        "ChanOut",
                        [],
                        tile_in,
                    )

                    # Deallocate our L1 buffers
                    DeallocOp(tile_in)
                    DeallocOp(tile_out)


if __name__ == "__main__":
    module = build_module()
    print(module)
