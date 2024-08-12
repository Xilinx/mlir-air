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
        "WorkerToWorker", size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH]
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
                    offset0 = TILE_HEIGHT * h
                    offset1 = TILE_WIDTH * w

                    # Put data into the channel tile by tile
                    ChannelPut(
                        "ChanIn",
                        a,
                        indices=[h, w],
                        offsets=[offset0, offset1],
                        sizes=TILE_SIZE,
                        strides=[IMAGE_WIDTH, 1],
                    )

            # Transfer one tile of data per worker
            for h in range(IMAGE_HEIGHT // TILE_HEIGHT):
                for w in range(IMAGE_WIDTH // TILE_WIDTH):
                    offset0 = TILE_HEIGHT * h
                    offset1 = TILE_WIDTH * w

                    # Write data back out to the channel tile by tile
                    ChannelGet(
                        "ChanOut",
                        b,
                        indices=[h, w],
                        offsets=[offset0, offset1],
                        sizes=TILE_SIZE,
                        strides=[IMAGE_WIDTH, 1],
                    )

            # The arguments are still the input and the output
            @segment(name="seg")
            def segment_body():

                @herd(
                    name="xaddherd",
                    sizes=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH],
                )
                def herd_body(th, tw, sh, sw):
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
                    tile_num = affine_apply(get_tile_num, [th, sw, tw])
                    get_tile_width_next = AffineMap.get(
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
                    tw_next = affine_apply(get_tile_width_next, [tw, sw])
                    # th_next = (th + ((tw + 1) // sw)) % sh
                    get_tile_height_next = AffineMap.get(
                        0,
                        4,
                        [
                            AffineExpr.get_mod(
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(2),
                                    AffineExpr.get_floor_div(
                                        AffineExpr.get_add(
                                            AffineSymbolExpr.get(0),
                                            AffineConstantExpr.get(1),
                                        ),
                                        AffineSymbolExpr.get(1),
                                    ),
                                ),
                                AffineSymbolExpr.get(3),
                            )
                        ],
                    )
                    th_next = affine_apply(get_tile_height_next, [tw, sw, th, sh])

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # This is the type definition of the tile
                    tile_type = MemRefType.get(
                        shape=TILE_SIZE,
                        element_type=xrt_dtype,
                        memory_space=mem_space,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    tile_in = AllocOp(tile_type, [], [])
                    tile_out = AllocOp(tile_type, [], [])

                    # Copy a tile from the input image (a) into the L1 memory region (tile_in)
                    ChannelGet("ChanIn", tile_in, indices=[th, tw])

                    # Access every value in the tile
                    for i in range_(TILE_HEIGHT):
                        for j in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val_in = load(tile_in, [i, j])

                            val_out = arith.AddIOp(
                                arith.index_cast(xrt_dtype, tile_num), val_in
                            )

                            # Store the output value in tile_out
                            store(val_out, tile_out, [i, j])
                            yield_([])
                        yield_([])

                    ChannelPut("WorkerToWorker", tile_out, indices=[th_next, tw_next])

                    tile_in2 = AllocOp(tile_type, [], [])
                    tile_out2 = AllocOp(tile_type, [], [])
                    ChannelGet("WorkerToWorker", tile_in2, indices=[th, tw])

                    for i in range_(TILE_HEIGHT):
                        for j in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tile_in2, [i, j])

                            # Store the output value in tile_out
                            store(val, tile_out2, [i, j])
                            yield_([])
                        yield_([])

                    # Copy the output tile into the output
                    ChannelPut("ChanOut", tile_out2, indices=[th, tw])

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
    output_matrix = np.full(IMAGE_SIZE, 0x5, dtype=INOUT_DATATYPE)

    def get_next_tile_num(tile_height, tile_width):
        # This should be the same math for how tile next values are calculated in the herd
        tw_next = (tile_width + 1) % (IMAGE_WIDTH // TILE_WIDTH)
        th_next = (tile_height + ((tile_width + 1) // (IMAGE_WIDTH // TILE_WIDTH))) % (
            IMAGE_HEIGHT // TILE_HEIGHT
        )
        assert tw_next >= 0 and tw_next <= (IMAGE_WIDTH // TILE_WIDTH)
        assert th_next >= 0 and th_next <= (IMAGE_HEIGHT // TILE_HEIGHT)
        print(f"{tile_height}, {tile_width} => {th_next}, {tw_next}")
        return th_next, tw_next

    # Create a map of current tile coordinates to the tile_num of the "previous" tile num
    tile_num_map = {}
    for tile_height in range(IMAGE_HEIGHT // TILE_HEIGHT):
        for tile_width in range(IMAGE_WIDTH // TILE_WIDTH):
            next_tile_coords = get_next_tile_num(tile_height, tile_width)
            tile_num_map[next_tile_coords] = (
                # This should be the same math for how tile_num is calculated in the herd
                tile_height * (IMAGE_WIDTH // TILE_WIDTH)
                + tile_width
            )

    # Add the tile num of the "previous" tile to each value in each tile in the output data
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            output_matrix[i, j] = (
                input_matrix[i, j] + tile_num_map[(i // TILE_HEIGHT, j // TILE_WIDTH)]
            )

    runner = XRTRunner(verbose=args.verbose)
    exit(
        runner.run_test(
            mlir_module, inputs=[input_matrix], expected_outputs=[output_matrix]
        )
    )
