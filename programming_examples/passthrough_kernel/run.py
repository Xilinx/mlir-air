# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import numpy as np
import air.backend.xrt as xrt_backend
import os
import os.path
import filelock
from passthrough_kernel import *

KERNEL_NAME = "MLIR_AIE"

INOUT_DATATYPE = np.uint8
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize


def main(vector_size, verbose=False, experimental_passes=False):
    mlir_module = build_module(vector_size)

    input_a = np.arange(1, vector_size + 1, dtype=INOUT_DATATYPE)
    output_b = np.arange(1, vector_size + 1, dtype=INOUT_DATATYPE)
    for i in range(vector_size):
        input_a[i] = i % 0xFF
        output_b[i] = 0xFF

    backend = xrt_backend.XRTBackend(
        verbose=verbose,
        experimental_passes=experimental_passes,
        omit_while_true_loop=True,
    )

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        copy = backend.compile_and_load(mlir_module)
        (_, output_b) = copy(input_a, output_b)

    backend.unload()

    # check output, should have the top left filled in
    errors = 0
    for i in range(vector_size):
        rb = output_b[i]

        expected_value = i % 0xFF
        if rb != expected_value:
            print(f"IM {i} should be 0x{expected_value:x}, is 0x{rb:x}\n")
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
        description="Builds, runs, and tests the passthrough/kernel example",
    )
    parser.add_argument(
        "vector_size",
        type=int,
        default=4096,
        help="The size (in bytes) of the data vector to passthrough",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()
    main(args.vector_size, experimental_passes=True, verbose=args.verbose)
