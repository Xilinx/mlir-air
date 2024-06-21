# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import air.backend.xrt as xrt_backend
import os
import os.path
import filelock
import argparse

from data_config import *
import single_core_dma
import multi_core_dma

KERNEL_NAME = "MLIR_AIE"

INOUT_DATATYPE = np.uint32
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
INOUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
INOUT_SIZE_BYTES = INOUT_SIZE * INOUT_ELEM_SIZE

verbose = False


def main(build_module):
    mlir_module = build_module()

    input_a = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    input_b = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    for i in range(INOUT_SIZE):
        input_a[i] = i + 0x1000
        input_b[i] = 0x00DEFACED

    # TODO(hunhoffe): need to omit the emit while loop option
    # TODO(hunhoffe): need to figure out why it fails with experimental_passes=True
    backend = xrt_backend.XRTBackend(verbose=verbose)

    for i in range(IMAGE_HEIGHT):
        row = input_b[i * IMAGE_WIDTH : (i + 1) * IMAGE_WIDTH]
        for val in row:
            val = val & 0xFFFF
            print(f"{val:04x}", end=" ")
        print("")

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        addone = backend.compile_and_load(mlir_module)
        (_, output_b) = addone(input_a, input_b)

    backend.unload()

    for i in range(IMAGE_HEIGHT):
        row = output_b[i * IMAGE_WIDTH : (i + 1) * IMAGE_WIDTH]
        for val in row:
            val = val & 0xFFFF
            print(f"{val:04x}", end=" ")
        print("")

    # check output, should have all values incremented
    errors = 0
    for i in range(INOUT_SIZE):
        rb = output_b[i]

        row = i / IMAGE_WIDTH
        col = i % IMAGE_WIDTH

        # value should have been updated
        if not (rb == 0x1000 + i + 1):
            # print(f"IM {i} [{col}, {row}] should be 0x{i:x}, is 0x{rb:x}\n")
            errors += 1

    if errors == 0:
        print("PASS!")
        exit(0)
    else:
        print("failed. errors=", errors)
        exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the matrix_scalar_add example",
    )

    parser.add_argument(
        "tile_method",
        type=str,
        choices=["single-core-dma", "multi-core-dma"],
    )
    args = parser.parse_args()

    if args.tile_method == "single-core-dma":
        build_module = single_core_dma.build_module
    elif args.tile_method == "multi-core-dma":
        build_module = multi_core_dma.build_module
    main(build_module)
