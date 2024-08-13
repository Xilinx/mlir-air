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

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_HEIGHT, IMAGE_WIDTH]

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = [TILE_HEIGHT, TILE_WIDTH]

assert IMAGE_HEIGHT % TILE_HEIGHT == 0
assert IMAGE_WIDTH % TILE_WIDTH == 0

INOUT_DATATYPE = np.int32


@module_builder
def build_module():
    xrt_dtype = type_mapper(INOUT_DATATYPE)
    memrefTyInOut = MemRefType.get(IMAGE_SIZE, xrt_dtype)

    # Create an input/output channel pair per worker
    Channel("ChanIn", size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH])
    Channel("ChanOut", size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH])
    Channel(
        "SwitchTiles", size=[IMAGE_HEIGHT // TILE_HEIGHT, IMAGE_WIDTH // TILE_WIDTH]
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
                def herd_body(th, tw, _sx, _sy):
                    height_next = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mod(
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(1),
                                ),
                                IMAGE_HEIGHT // TILE_HEIGHT,
                            )
                        ],
                    )
                    width_next = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mod(
                                AffineExpr.get_add(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(1),
                                ),
                                IMAGE_WIDTH // TILE_WIDTH,
                            )
                        ],
                    )
                    tw_next = affine_apply(width_next, [tw])
                    th_next = affine_apply(height_next, [th])

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
                    ChannelGet("ChanIn", tile_in, indices=[th, tw])

                    # Access every value in the tile
                    for i in range_(TILE_HEIGHT):
                        for j in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tile_in, [i, j])

                            # Store the output value in tile_out
                            store(val, tile_out, [i, j])
                            yield_([])
                        yield_([])

                    # Copy the output tile into a channel for another worker to get
                    ChannelPut("SwitchTiles", tile_out, indices=[th, tw])

                    # Get an output tile from another worker
                    ChannelGet("SwitchTiles", tile_in2, indices=[th_next, tw_next])

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
                    ChannelPut("ChanOut", tile_out, indices=[th, tw])

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
    # TODO: this check is not fully correct, should show how data is transferred explicitly
    # Will probably want to use different input to implement a more rigorous check.
    output_matrix = input_matrix.copy()

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(
        runner.run_test(
            mlir_module, inputs=[input_matrix], expected_outputs=[output_matrix]
        )
    )
