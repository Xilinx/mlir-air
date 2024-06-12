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

from common import *
import matrix_scalar_add_implicit
import matrix_scalar_add_channel
import matrix_scalar_add_launch
import matrix_scalar_add_herd
import matrix_scalar_add_segment

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

    backend = xrt_backend.XRTBackend(verbose=verbose)
    print(input_b)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        addone = backend.compile_and_load(mlir_module)
        (_, output_b) = addone(input_a, input_b)

    backend.unload()

    print(output_b)

    # check output, should have all values incremented
    errors = 0
    for i in range(INOUT_SIZE):
        rb = output_b[i]

        row = i / IMAGE_WIDTH
        col = i % IMAGE_WIDTH

        # value should have been updated
        if not (rb == 0x1000 + i + 1):
            print(f"IM {i} [{col}, {row}] should be 0x{i:x}, is 0x{rb:x}\n")
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
        choices=["implicit", "channel", "launch", "herd", "segment"],
    )
    args = parser.parse_args()

    if args.tile_method == "implicit":
        build_module = matrix_scalar_add_implicit.build_module
    elif args.tile_method == "channel":
        build_module = matrix_scalar_add_channel.build_module
    elif args.tile_method == "launch":
        build_module = matrix_scalar_add_launch.build_module
    elif args.tile_method == "herd":
        build_module = matrix_scalar_add_herd.build_module
    else:
        build_module = matrix_scalar_add_segment.build_module
    main(build_module)
