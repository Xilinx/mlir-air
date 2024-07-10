# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np
import air.backend.xrt as xrt_backend
import filelock

from multi_segment import *

INOUT_DATATYPE = np.uint32
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
OUT_SIZE = VECTOR_OUT_SIZE[0] * VECTOR_OUT_SIZE[1]
OUT_SIZE_BYTES = OUT_SIZE * INOUT_ELEM_SIZE
IN_SIZE = VECTOR_SIZE[0] * VECTOR_SIZE[1]
IN_SIZE_BYTES = IN_SIZE * INOUT_ELEM_SIZE


def test_main(build_module, verbose=False):
    mlir_module = build_module()

    input_a = np.arange(1, IN_SIZE + 1, dtype=INOUT_DATATYPE)
    input_b = np.arange(1, IN_SIZE + 1, dtype=INOUT_DATATYPE)
    input_c = np.arange(1, OUT_SIZE + 1, dtype=INOUT_DATATYPE)
    for i in range(IN_SIZE):
        input_a[i] = 0x2
        input_b[i] = 0x3
    for i in range(OUT_SIZE):
        input_c[i] = 0x00C0FFEE

    backend = xrt_backend.XRTBackend(
        verbose=verbose, experimental_passes=True, omit_while_true_loop=True
    )

    if verbose:
        print(input_a)
        print(input_b)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        addone = backend.compile_and_load(mlir_module)
        (_, _, output_c) = addone(input_a, input_b, input_c)

    backend.unload()

    if verbose:
        print(output_c)

    # check output, should have all values incremented
    errors = 0
    for i in range(OUT_SIZE):
        rb = output_c[i]

        # value should have been updated
        if i < VECTOR_LEN:
            expected_value = 5
        else:
            expected_value = 6
        if not (rb == expected_value):
            """
            print(
                f"IM {i} should be 0x{expected_value:x}, is 0x{rb:x}\n"
            )
            """
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
        description="Builds, runs, and tests the herd-to-herd multi-segment example",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()
    test_main(build_module, verbose=args.verbose)
