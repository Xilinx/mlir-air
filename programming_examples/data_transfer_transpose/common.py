# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
import air.backend.xrt as xrt_backend
import filelock

# TODO: check with different data types
# INOUT_ELEM_SIZE = np.dtype(INOUT_DATATYPE).itemsize
# INOUT_SIZE_BYTES = INOUT_SIZE * INOUT_ELEM_SIZE


def test_main(build_module, m, k, verbose=False, experimental_passes=False):
    mlir_module = build_module(m, k)

    matrix_shape = (m, k)
    matrix_shape_t = (k, m)
    # TODO: configure with different data types
    matrix_dtype = np.uint32

    # Generate a random matrix
    input_matrix = np.random.randint(
        low=0, high=2**32 - 1, size=matrix_shape, dtype=matrix_dtype
    )
    expected_output_matrix = np.transpose(input_matrix)
    actual_output_matrix = np.zeros(matrix_shape_t, dtype=matrix_dtype)
    assert expected_output_matrix.shape == actual_output_matrix.shape

    backend = xrt_backend.XRTBackend(verbose=verbose, omit_while_true_loop=True, experimental_passes=experimental_passes)

    if verbose:
        print(input_matrix)

    # Run the module
    with filelock.FileLock("/tmp/npu.lock"):
        transpose = backend.compile_and_load(mlir_module)
        (_, actual_output_matrix) = transpose(input_matrix, actual_output_matrix)
    backend.unload()

    actual_output_matrix = actual_output_matrix.reshape(matrix_shape_t)
    assert expected_output_matrix.shape == actual_output_matrix.shape

    if verbose:
        print("======== ORIGINAL ========")
        print(input_matrix)
        print("======== EXPECTED ========")
        print(expected_output_matrix)
        print("======== ACTUAL ==========")
        print(actual_output_matrix)

    # check output, should have all values incremented
    errors = 0
    for m_index in range(m):
        for k_index in range(k):
            expected_value = expected_output_matrix.item((k_index, m_index))
            actual_value = actual_output_matrix.item((k_index, m_index))

            if not (actual_value == expected_value):
                """
                print(
                    f"IM {i} [{m_index}, {k_index}] should be 0x{expected_value:x}, is 0x{actual_value:x}\n"
                )
                """
                errors += 1

    if errors == 0:
        print("PASS!")
        exit(0)
    else:
        print("failed. errors=", errors)
        exit(-1)
