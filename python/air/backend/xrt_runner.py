# ./python/air/backend/xrt_runner.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from .xrt import XRTBackend
from air.dialects.air import *
import filelock
from typing import List
from collections import defaultdict
from ml_dtypes import bfloat16

TYPE_MAP_DICT = defaultdict(
    lambda: None,
    {
        # Integer types
        np.int8: T.i8,
        np.int16: T.i16,
        np.int32: T.i32,
        np.int64: T.i64,
        # Unsigned Integer Types
        np.uint8: T.ui8,
        np.uint16: T.ui16,
        np.uint32: T.ui32,
        np.uint64: T.ui64,
        # Floating point types
        np.float16: T.f16,
        np.float32: T.f32,
        np.float64: T.f64,
        bfloat16: T.bf16,
    },
)


def type_mapper(np_dtype):
    """
    This function is meant to run within a module context (e.g., with a function wrapped with @build_module)
    args:
        np_dtype: the numpy data type to map
    return:
        The data type to run on the npu
    """
    xrt_dtype = TYPE_MAP_DICT[np_dtype]()

    if xrt_dtype is None:
        raise AirBackendError(f"numpy data type {np_dtype} has no default mapping")
    elif xrt_dtype.width / 8 != np.dtype(np_dtype).itemsize:
        # This is a sanity check on the TYPE_MAP_DICT rather than a check on the user input
        raise AirBackendError(
            f"Python data type has width {xrt_dtype.width / 8} but numpy data type has width {np.dtype(np_dtype).itemsize}"
        )
    return xrt_dtype


class XRTRunner:
    def __init__(
        self,
        verbose: bool = False,
        omit_while_true_loop: bool = True,
        omit_pingpong: bool = False,
        lower_linalg_to_func: bool = False,
        air_loop_fusion: bool = False,
        runtime_loop_tiling_sizes: list[int] = [4, 4],
        omit_auto_broadcast: bool = False,
        channel_multiplexing: list[str] = [],
        trace_offset: int = 0,
        trace_size: int = 0,
    ):
        self.verbose = verbose
        self.omit_while_true_loop = omit_while_true_loop
        self.omit_pingpong = omit_pingpong
        self.lower_linalg_to_func = lower_linalg_to_func
        self.air_loop_fusion = air_loop_fusion
        self.runtime_loop_tiling_sizes = runtime_loop_tiling_sizes
        self.omit_auto_broadcast = omit_auto_broadcast
        self.channel_multiplexing = channel_multiplexing
        self.trace_offset = trace_offset
        self.trace_size = trace_size

    def run_test(
        self,
        mlir_module: np.ndarray,
        inputs: List[np.ndarray],
        expected_outputs: List[np.ndarray],
        rtol: float = 1e-3,
    ):
        if self.verbose:
            print("Running module: ")
            print(mlir_module)

        backend = XRTBackend(
            verbose=self.verbose,
            omit_while_true_loop=self.omit_while_true_loop,
            omit_pingpong=self.omit_pingpong,
            lower_linalg_to_func=self.lower_linalg_to_func,
            air_loop_fusion=self.air_loop_fusion,
            runtime_loop_tiling_sizes=self.runtime_loop_tiling_sizes,
            omit_auto_broadcast=self.omit_auto_broadcast,
            channel_multiplexing=self.channel_multiplexing,
            trace_offset=self.trace_offset,
            trace_size=self.trace_size,
        )

        # run the module - slots are input/output for now, assume non-overlapping inputs/outputs
        expanded_inputs = inputs + [
            np.zeros(o.shape, o.dtype) for o in expected_outputs
        ]
        with filelock.FileLock("/tmp/npu.lock"):
            module_function = backend.compile_and_load(mlir_module)
            actual_outputs = module_function(*expanded_inputs)

        backend.unload()

        # Remove input slots from the received outputs
        actual_outputs = actual_outputs[len(inputs) :]

        if self._check_outputs(actual_outputs, expected_outputs, rtol=rtol):
            print("PASS!")
            return_code = 0
        else:
            print("failed.")
            return_code = -1
        return return_code

    def _check_outputs(
        self,
        actual_outputs: List[np.ndarray],
        expected_outputs: List[np.ndarray],
        rtol: float = 1e-3,
    ):
        assert len(actual_outputs) == len(
            expected_outputs
        ), f"Number of actual outputs ({len(actual_outputs)}) does not equal number of expected outputs ({len(expected_outputs)})"
        np.set_printoptions(formatter={"int": hex})

        for i, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs)):
            actual = np.reshape(actual, expected.shape)

            if self.verbose:
                print("Expected: ")
                if len(expected.shape) == 2:
                    print(np.asmatrix(expected))
                else:
                    print(expected)
                print("Actual: ")
                if len(actual.shape) == 2:
                    print(np.asmatrix(actual))
                else:
                    print(actual)

            if expected.dtype in [np.float16, np.float32, np.float64, bfloat16]:
                if expected.dtype == bfloat16:
                    expected = expected.astype(np.float64)
                    actual = actual.astype(np.float64)
                if not np.allclose(actual, expected, rtol=rtol):
                    print(f"ERROR: Output {i} does not meet expected output.")
                    print("Expected: ")
                    print(expected)
                    print("Actual: ")
                    print(actual)
                    return False
            else:
                if not np.array_equal(actual, expected):
                    print(f"ERROR: Output {i} does not meet expected output.")
                    print("Expected: ")
                    print(expected)
                    print("Actual: ")
                    print(actual)
                    return False

        return True
