# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.dialects.affine import apply as affine_apply
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_


@module_builder
def build_module(image_height, image_width, tile_height, tile_width, np_dtype):
    assert image_height % tile_height == 0
    assert image_width % tile_width == 0
    image_size = [image_height, image_width]
    tile_size = [tile_height, tile_width]
    xrt_dtype = type_mapper(np_dtype)

    memrefTyInOut = MemRefType.get(image_size, xrt_dtype)

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
                # We are hoping to map each tile to a different compute core.
                @herd(
                    name="xaddherd",
                    sizes=[image_height // tile_height, image_width // tile_width],
                    operands=[arg2, arg3],
                )
                def herd_body(tx, ty, _sx, _sy, a, b):
                    scaled_index_map_height = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(tile_height),
                            )
                        ],
                    )
                    scaled_index_map_width = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(tile_width),
                            )
                        ],
                    )
                    create_tile_index_height = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(image_width // tile_width),
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
                    offset0 = affine_apply(scaled_index_map_height, [tx])
                    offset1 = affine_apply(scaled_index_map_width, [ty])
                    tile_index_height = affine_apply(create_tile_index_height, [tx])
                    compute_tile_id = affine_apply(
                        create_tile_index, [tile_index_height, ty]
                    )

                    # We want to store our data in L1 memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # This is the type definition of the tile
                    tile_type = MemRefType.get(
                        shape=tile_size,
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
                        src_offsets=[offset0, offset1],
                        src_sizes=tile_size,
                        src_strides=[image_width, 1],
                    )

                    # Access every value in the tile
                    for i in range_(tile_height):
                        for j in range_(tile_width):
                            # Load the input value from tile_in
                            val_in = load(tile_in, [i, j])

                            # Compute the output value
                            val_out = arith.addi(
                                val_in, arith.index_cast(xrt_dtype, compute_tile_id)
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
                        dst_sizes=tile_size,
                        dst_strides=[image_width, 1],
                    )

                    # Deallocate our L1 buffers
                    DeallocOp(tile_in)
                    DeallocOp(tile_out)


if __name__ == "__main__":
    # Default values.
    IMAGE_WIDTH = 16
    IMAGE_HEIGHT = 32
    TILE_WIDTH = 8
    TILE_HEIGHT = 16
    INOUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
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
    parser.add_argument(
        "--image-height",
        type=int,
        default=IMAGE_HEIGHT,
        help="Height of the image data",
    )
    parser.add_argument(
        "--image-width", type=int, default=IMAGE_WIDTH, help="Width of the image data"
    )
    parser.add_argument(
        "--tile-height", type=int, default=TILE_HEIGHT, help="Height of the tile data"
    )
    parser.add_argument(
        "--tile-width", type=int, default=TILE_WIDTH, help="Width of the tile data"
    )
    args = parser.parse_args()

    mlir_module = build_module(
        args.image_height,
        args.image_width,
        args.tile_height,
        args.tile_width,
        INOUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.zeros(
        shape=(args.image_height, args.image_width), dtype=INOUT_DATATYPE
    )
    output_b = np.zeros(
        shape=(args.image_height, args.image_width), dtype=INOUT_DATATYPE
    )
    for i in range(args.image_height):
        for j in range(args.image_width):
            input_a[i, j] = i * args.image_height + j
            tile_num = (
                i // args.tile_height * (args.image_width // args.tile_width)
                + j // args.tile_width
            )
            output_b[i, j] = input_a[i, j] + tile_num

    runner = XRTRunner(verbose=args.verbose)
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
