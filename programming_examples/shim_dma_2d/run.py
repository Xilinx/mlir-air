# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import air.backend.xrt as xrt_backend
import os
import os.path
import filelock
from shim_dma_2d import build_module

KERNEL_NAME = "MLIR_AIE"

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 16
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

TILE_WIDTH = 16
TILE_HEIGHT = 8
TILE_SIZE = (TILE_WIDTH, TILE_HEIGHT)

INOUT_DATATYPE = np.uint32
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
INOUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
INOUT_SIZE_BYTES = INOUT_SIZE * INOUT_ELEM_SIZE

verbose = False


def main():
    mlir_module = build_module()

    input_a = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    output_b = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    for i in range(INOUT_SIZE):
        input_a[i] = i + 0x1000
        output_b[i] = 0x00DEFACED

    backend = xrt_backend.XRTBackend(verbose=verbose)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        mul = backend.compile_and_load(mlir_module)
        (_, output_b) = mul(input_a, output_b)

    backend.unload()

    # check output, should have the top left filled in
    errors = 0
    for i in range(INOUT_SIZE):
        rb = output_b[i]

        row = i / IMAGE_WIDTH
        col = i % IMAGE_WIDTH

        if row < TILE_HEIGHT and col < TILE_WIDTH:
            # value should have been updated
            if not (rb == 0x1000 + i):
                print(f"IM {i} [{col}, {row}] should be 0x{i:x}, is 0x{rb:x}\n")
                errors += 1
        else:
            # value should stay unchanged
            if rb != 0x00DEFACED:
                print(
                    f"IM {i} [{col}, {row}] should be 0xdefaced, is 0x{rb:x}\n",
                    i,
                    col,
                    row,
                    rb,
                )
                errors += 1

    if errors == 0:
        print("PASS!")
        exit(0)
    else:
        print("failed. errors=", errors)
        exit(-1)


if __name__ == "__main__":
    main()
