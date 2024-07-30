# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np
import air.backend.xrt as xrt_backend
import filelock

from broadcast import *

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
    input_c = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    input_d = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)

    backend = xrt_backend.XRTBackend(
        verbose=verbose, experimental_passes=True, omit_while_true_loop=True
    )

    if verbose:
        print_matrix(input_b)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        broadcast = backend.compile_and_load(mlir_module)
        (_, output_b, output_c, output_d) = broadcast(
            input_a, input_b, input_c, input_d
        )

    backend.unload()

    if verbose:
        print("OUTPUT B")
        print_matrix(output_b)
        print("")
        print("OUTPUT C")
        print_matrix(output_c)
        print("")
        print("OUTPUT D")
        print_matrix(output_d)

    # check output, should have all values incremented
    errors = 0
    outputs = [output_b, output_c, output_d]
    for k in range(3):
        current_output = outputs[k]
        for i in range(INOUT_SIZE):
            rb = current_output[i]
            expected_value = input_a[i] + (k + 1)

            # value should have been updated
            if not (rb == expected_value):
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
        description="Builds, runs, and tests the channel_examples/broadcast/single_herd example",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()
    test_main(build_module, verbose=True)
