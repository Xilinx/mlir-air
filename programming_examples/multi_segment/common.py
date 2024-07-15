# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import numpy as np
import air.backend.xrt as xrt_backend
import filelock

VECTOR_LEN = 32
VECTOR_SIZE = [VECTOR_LEN, 1]

INOUT_DATATYPE = np.uint32
INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
INOUT_SIZE = VECTOR_SIZE[0] * VECTOR_SIZE[1]
INOUT_SIZE_BYTES = INOUT_SIZE * INOUT_ELEM_SIZE


def test_main(build_module, verbose=False):
    mlir_module = build_module()

    input_a = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    input_b = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    input_c = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    input_d = np.arange(1, INOUT_SIZE + 1, dtype=INOUT_DATATYPE)
    for i in range(INOUT_SIZE):
        input_a[i] = 0x2
        input_b[i] = 0x3
    for i in range(INOUT_SIZE):
        input_c[i] = 0x00C0FFEE
        input_d[i] = 0x0000CAFE

    backend = xrt_backend.XRTBackend(
        verbose=verbose, experimental_passes=True, omit_while_true_loop=True
    )

    if verbose:
        print(input_a)
        print(input_b)

    # run the module
    with filelock.FileLock("/tmp/npu.lock"):
        addone = backend.compile_and_load(mlir_module)
        (_, _, output_c, output_d) = addone(input_a, input_b, input_c, input_d)

    backend.unload()

    if verbose:
        print(output_c)
        print(output_d)

    # check output, should have all values incremented
    errors = 0
    for i in range(INOUT_SIZE):
        rb = output_c[i]

        # value should have been updated
        if not (rb == 12):
            """
            print(
                f"C - IM {i} should be 0x{expected_value:x}, is 0x{rb:x}\n"
            )
            """
            errors += 1

    for i in range(INOUT_SIZE):
        rb = output_d[i]

        # value should have been updated
        if not (rb == 13):
            """
            print(
                f"D - IM {i} should be 0x{expected_value:x}, is 0x{rb:x}\n"
            )
            """
            errors += 1

    if errors == 0:
        print("PASS!")
        exit(0)
    else:
        print("failed. errors=", errors)
        exit(-1)
