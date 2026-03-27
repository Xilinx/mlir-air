# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import type_mapper, make_air_parser, run_on_npu

range_ = for_

IMAGE_WIDTH = 32
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

    # We will send an image worth of data in and out
    @FuncOp.from_py_func(memrefTyInOut, memrefTyInOut)
    def copy(arg0, arg1):

        # The arguments are the input and output
        @launch(operands=[arg0, arg1])
        def launch_body(a, b):

            # The arguments are still the input and the output
            @segment(name="seg", operands=[a, b])
            def segment_body(arg2, arg3):
                # This is the type definition of the tile
                tile_type_l2 = l2_memref_type(TILE_SIZE, xrt_dtype)

                # We must allocate a buffer of tile size for the input/output
                tile_in_l2 = AllocOp(tile_type_l2, [], [])

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="copyherd", sizes=[1, 1], operands=[arg2, arg3, tile_in_l2])
                def herd_body(tx, ty, sx, sy, a, b, my_l2_tile):

                    # This is the type definition of the tile
                    tile_type_l1 = l1_memref_type(TILE_SIZE, xrt_dtype)

                    # We must allocate a buffer of tile size for the input/output
                    tile_in_l1 = AllocOp(tile_type_l1, [], [])
                    tile_out_l1 = AllocOp(tile_type_l1, [], [])

                    dma_memcpy_nd(
                        my_l2_tile,
                        a,
                        src_offsets=[0, 0],
                        src_sizes=TILE_SIZE,
                        src_strides=[IMAGE_WIDTH, 1],
                    )

                    # Copy a tile from the input image (a) into the L1 memory region (tile_in)
                    dma_memcpy_nd(
                        tile_in_l1,
                        my_l2_tile,
                    )

                    # Access every value in the tile
                    for i in range_(TILE_HEIGHT):
                        for j in range_(TILE_WIDTH):
                            # Load the input value from tile_in
                            val = load(tile_in_l1, [i, j])

                            # Store the output value in tile_out
                            store(val, tile_out_l1, [i, j])
                            yield_([])
                        yield_([])

                    # Copy the output tile into the output
                    dma_memcpy_nd(
                        b,
                        tile_out_l1,
                        dst_offsets=[0, 0],
                        dst_sizes=TILE_SIZE,
                        dst_strides=[IMAGE_WIDTH, 1],
                    )

                    # Deallocate our L1 buffers
                    DeallocOp(tile_in_l1)
                    DeallocOp(tile_out_l1)


if __name__ == "__main__":
    parser = make_air_parser("Builds, runs, and tests the segment_alloc example")
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
    output_b = np.zeros(shape=IMAGE_SIZE, dtype=INOUT_DATATYPE)
    for h in range(TILE_HEIGHT):
        for w in range(TILE_WIDTH):
            output_b[h, w] = input_a[h, w]

    run_on_npu(
        args,
        mlir_module,
        inputs=[input_a],
        expected_outputs=[output_b],
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
