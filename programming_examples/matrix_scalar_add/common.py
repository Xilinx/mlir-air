# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
import air.backend.xrt as xrt_backend
import filelock

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = [IMAGE_WIDTH, IMAGE_HEIGHT]

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = [TILE_WIDTH, TILE_HEIGHT]

assert IMAGE_WIDTH % TILE_WIDTH == 0
assert IMAGE_HEIGHT % TILE_HEIGHT == 0


KERNEL_NAME = "MLIR_AIE"

INOUT_DATATYPE = np.uint32
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
INOUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
INOUT_SIZE_BYTES = INOUT_SIZE * INOUT_ELEM_SIZE


def print_matrix(matrix_array):
    for i in range(IMAGE_HEIGHT):
        row = matrix_array[i * IMAGE_WIDTH : (i + 1) * IMAGE_WIDTH]
        for val in row:
            val = val & 0xFFFF
            print(f"{val:04x}", end=" ")
        print("")


def test_main(build_module, verbose=False):
    mlir_module = build_module()

    input_a = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    input_b = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    for i in range(INOUT_SIZE):
        input_a[i] = i + 0x1000
        input_b[i] = 0x00DEFACED

    # TODO(hunhoffe): need to figure out why single-core-dma fails with experimental_passes=True
    backend = xrt_backend.XRTBackend(verbose=verbose, omit_while_true_loop=True)

    if verbose:
        print_matrix(input_b)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        addone = backend.compile_and_load(mlir_module)
        (_, output_b) = addone(input_a, input_b)

    backend.unload()

    if verbose:
        print_matrix(output_b)

    # check output, should have all values incremented
    errors = 0
    for i in range(INOUT_SIZE):
        rb = output_b[i]

        row = i // IMAGE_WIDTH
        col = i % IMAGE_WIDTH
        tile_num = (row // TILE_HEIGHT) * (IMAGE_HEIGHT // TILE_HEIGHT) + (
            col // TILE_WIDTH
        )

        # value should have been updated
        expected_value = 0x1000 + i + tile_num
        if not (rb == expected_value):
            """
            print(
                f"IM {i} [{col}, {row}] should be 0x{expected_value:x}, is 0x{rb:x}\n"
            )
            """
            errors += 1

    if errors == 0:
        print("PASS!")
        exit(0)
    else:
        print("failed. errors=", errors)
        exit(-1)
