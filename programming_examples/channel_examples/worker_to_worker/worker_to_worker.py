# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.dialects.affine import apply as affine_apply
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

IMAGE_WIDTH = 12
IMAGE_HEIGHT = 4
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

TILE_WIDTH = 4
TILE_HEIGHT = 2
TILE_SIZE = [TILE_HEIGHT, TILE_WIDTH]

assert IMAGE_HEIGHT % TILE_HEIGHT == 0
assert IMAGE_WIDTH % TILE_WIDTH == 0

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, xrt_dtype)

    # Create an input/output channel pair per worker
    ChannelOp("ChanIn", size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH])
    ChannelOp("ChanOut", size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH])
    ChannelOp(
        "SwitchTiles", size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH]
    )

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
                    tile_size0 = arith.ConstantOp.create_index(TILE_HEIGHT)
                    tile_size1 = arith.ConstantOp.create_index(TILE_WIDTH)

                    # Calculate the offset into the channel data, which is based on which tile index
                    # we are at using tile_index0 and tile_index1 (our loop vars).
                    # tile_index0 and tile_index1 are dynamic so we have to use specialized
                    # operations do to calculations on them
                    offset0 = arith.MulIOp(tile_size0, tile_index0)
                    offset1 = arith.MulIOp(tile_size1, tile_index1)

                    # Put data into the channel tile by tile
                    ChannelPut(
                        "ChanIn",
                        a,
                        indices=[tile_index0, tile_index1],
                        offsets=[offset0, offset1],
                        sizes=TILE_SIZE,
                        strides=[IMAGE_WIDTH, 1],
                    )

                    # Write data back out to the channel tile by tile
                    ChannelGet(
                        "ChanOut",
                        b,
                        indices=[tile_index0, tile_index1],
                        offsets=[offset0, offset1],
                        sizes=TILE_SIZE,
                        strides=[IMAGE_WIDTH, 1],
                    )
                    yield_([])
                yield_([])

            # The arguments are still the input and the output
            @segment(name="seg")
            def segment_body():

                @herd(
                    name="xaddherd",
                    sizes=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH],
                )
                def herd_body(tile_height, tile_width, size_height, size_width):
                    """
                    tw_next = (tw + 1) % sw
                    th_next = (th + ((tw + 1 ) // sw)) % sh
                    """
                    width_next = AffineMap.get(
                        0,
                        2,
                        [
                            AffineExpr.get_mod(
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(1),
                                ),
                                AffineSymbolExpr.get(1),
                            )
                        ],
                    )
                    height_next = AffineMap.get(
                        0,
                        4,
                        [
                            AffineExpr.get_mod(
                                # (((tw + 1) // sw) + th) % sh
                                AffineExpr.get_add(
                                    # ((tw + 1) // sw) + th
                                    AffineExpr.get_floor_div(
                                        # (tw + 1) // sw
                                        AffineExpr.get_add(
                                            # tw + 1
                                            AffineSymbolExpr.get(0),
                                            AffineConstantExpr.get(1),
                                        ),
                                        AffineSymbolExpr.get(1),
                                    ),
                                    AffineSymbolExpr.get(2),
                                ),
                                AffineSymbolExpr.get(3),
                            )
                        ],
                    )
                    # th * sw + tw
                    get_tile_num = AffineMap.get(
                        0,
                        3,
                        [
                            AffineExpr.get_add(
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineSymbolExpr.get(1),
                                ),
                                AffineSymbolExpr.get(2),
                            )
                        ],
                    )
                    tile_num = affine_apply(
                        get_tile_num, [tile_height, size_width, tile_width]
                    )
                    tile_width_next = affine_apply(width_next, [tile_width, size_width])
                    tile_height_next = affine_apply(
                        height_next, [tile_width, size_width, tile_height, size_height]
                    )

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    tile_type = MemRefType.get(
                        shape=TILE_SIZE,
                        element_type=xrt_dtype,
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    tile_in = AllocOp(tile_type, [], [])
                    tile_in2 = AllocOp(tile_type, [], [])
                    tile_out = AllocOp(tile_type, [], [])
                    tile_out2 = AllocOp(tile_type, [], [])

                    # Copy a tile from the input image
                    ChannelGet("ChanIn", tile_in, indices=[tile_height, tile_width])

                    # Access every value in the tile
                    for i in range_(TILE_HEIGHT):
                        for j in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val_in = load(tile_in, [i, j])
                            val_out = arith.MulIOp(
                                arith.index_cast(xrt_dtype, tile_num), val_in
                            )

                            # Store the output value in tile_out
                            store(val_out, tile_out, [i, j])
                            yield_([])
                        yield_([])

                    # Copy the output tile into a channel for the "next" worker to get
                    ChannelPut(
                        "SwitchTiles",
                        tile_out,
                        indices=[tile_height_next, tile_width_next],
                    )

                    # Get an output tile from another worker
                    ChannelGet(
                        "SwitchTiles", tile_in2, indices=[tile_height, tile_width]
                    )

                    # Access every value in the tile
                    for i in range_(TILE_HEIGHT):
                        for j in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tile_in2, [i, j])

                            # Store the output value in tile_out
                            store(val, tile_out2, [i, j])
                            yield_([])
                        yield_([])

                    # Send the output tile to the output
                    ChannelPut("ChanOut", tile_out2, indices=[tile_height, tile_width])

                    # Deallocate our L1 buffers
                    DeallocOp(tile_in)
                    DeallocOp(tile_out)
                    DeallocOp(tile_in2)
                    DeallocOp(tile_out2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the channel worker_to_worker example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_matrix = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)
    # TODO: this check is NOT yet correct
    output_matrix = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)

    runner = XRTRunner(verbose=args.verbose, experimental_passes=False)
    exit(
        runner.run_test(
            mlir_module, inputs=[input_matrix], expected_outputs=[output_matrix]
        )
    )
