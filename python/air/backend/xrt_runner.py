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
import timeit

from aie.utils.xrt import write_out_trace, extract_trace

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
        omit_pingpong: str = "",
        lower_linalg_to_func: bool = False,
        air_loop_fusion: bool = False,
        runtime_loop_tiling_sizes: list[int] = [4, 4],
        omit_auto_broadcast: bool = False,
        channel_multiplexing: list[str] = [],
        use_lock_race_condition_fix: bool = False,
        trace_offset: int = 0,
        trace_size: int = 0,
        output_format: str = "xclbin",
        kernel_name: str = "",
        instance_name: str = "",
        kernel_id: str = "",
        xclbin_input: str = "",
        trace_file: str = "trace_data.txt",
    ):
        """
        Args:
            verbose: verbose output
            omit_while_true_loop: configure aircc to omit the while true loop it traditionally emits.
            omit_pingpong: configure aircc to omit the generation of ping-pong buffering for specific memory levels. Supported values: "", "L1", "L2", "all". Empty string means no omission (default).
            lower_linalg_to_func: configure aircc to lower linalg.generic to function calls, or loops.
            air_loop_fusion: configure aircc to add air-loop-fusion experimental pass.
            runtime_loop_tiling_sizes: configure aircc to add extra runtime loop tiling using the experimental affine-loop-opt pass.
            omit_auto_broadcast: configure aircc to omit the detection and lowering of broadcast data movements.
            channel_multiplexing: configure aircc to perform air channel multiplexing on specified memroy spaces.
            use_lock_race_condition_fix: configure aircc to enable a fix for lock race condition which protects against race condition.
            trace_offset: configure aircc to stream out profiling traces at outputs, starting from the specified offset.
            trace_size: configure aircc to stream out profiling traces at outputs, with specified trace data size.
            output_format: configure aircc to produce output binary in to one of the following formats: [xclbin, txn].
            kernel_name: configure aircc to package the kernel with the specified name.
            instance_name: configure aircc to package the kernel with specified instance name in xclbin metadata.
            kernel_id: configure aircc to package the kernel with specified kernel id in xclbin file.
            xclbin_input: configure aircc to package the kernel into an existing xclbin with specified xclbin file name.
            trace_file: default filename for saving trace data.
        """
        self.verbose = verbose
        self.omit_while_true_loop = omit_while_true_loop
        # Support backward compatibility: convert True to "all", False to ""
        if isinstance(omit_pingpong, bool):
            self.omit_pingpong = "all" if omit_pingpong else ""
        else:
            self.omit_pingpong = omit_pingpong
        self.lower_linalg_to_func = lower_linalg_to_func
        self.air_loop_fusion = air_loop_fusion
        self.runtime_loop_tiling_sizes = runtime_loop_tiling_sizes
        self.omit_auto_broadcast = omit_auto_broadcast
        self.channel_multiplexing = channel_multiplexing
        self.use_lock_race_condition_fix = use_lock_race_condition_fix
        self.trace_offset = trace_offset
        self.trace_size = trace_size
        self.output_format = output_format
        self.kernel_name = kernel_name
        self.instance_name = instance_name
        self.kernel_id = kernel_id
        self.xclbin_input = xclbin_input
        self.trace_file = trace_file

    def run_test(
        self,
        mlir_module: np.ndarray,
        inputs: List[np.ndarray],
        expected_outputs: List[np.ndarray] = [],
        stochastic_expected_outputs: List[np.ndarray] = [],
        rtol: float = 1e-3,
        trace_file: str = None,
    ):
        """
        Args:
            mlir_module: input mlir module to test.
            inputs: input matrices.
            expected_outputs: expected output matrices.
            stochastic_expected_outputs: expected output matrices stored in sparse coordinates. Expect each matrix to be a dictionary containing "shape", "indices" and "values" fields.
            rtol: relative error tolerance.
            trace_file: optional override for trace data filename. If None, uses instance default.
        """
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
            use_lock_race_condition_fix=self.use_lock_race_condition_fix,
            trace_offset=self.trace_offset,
            trace_size=self.trace_size,
            output_format=self.output_format,
            kernel_name=self.kernel_name,
            instance_name=self.instance_name,
            kernel_id=self.kernel_id,
            xclbin_input=self.xclbin_input,
        )

        # Use per-test trace file if provided, otherwise use instance default
        active_trace_file = trace_file if trace_file is not None else self.trace_file

        # run the module - slots are input/output for now, assume non-overlapping inputs/outputs
        # Handle different scenarios for trace data
        if self.trace_size > 0:
            if expected_outputs:
                # Case 1: Both outputs and trace
                # Add trace_size bytes to first output
                total_bytes = expected_outputs[0].nbytes + self.trace_size
                first_output_with_trace = np.zeros(total_bytes, dtype=np.uint8)
                remaining_outputs = [
                    np.zeros(o.shape, o.dtype) for o in expected_outputs[1:]
                ]
                output_placeholders = [first_output_with_trace] + remaining_outputs
                if self.verbose:
                    print(
                        f"Allocated {total_bytes} bytes for first output + {self.trace_size} bytes for trace data"
                    )
                # Record the expected_outputs[0]'s shape and dtype, to be used to split actual outputs from trace.
                expected_outputs_0_shape = expected_outputs[0].shape
                expected_outputs_0_dtype = expected_outputs[0].dtype
            elif stochastic_expected_outputs:
                # Case 2: Stochastic outputs and trace
                first_output_elements = np.prod(stochastic_expected_outputs[0]["shape"])
                first_output_bytes = (
                    first_output_elements
                    * stochastic_expected_outputs[0]["values"][0].dtype.itemsize
                )
                total_bytes = first_output_bytes + self.trace_size
                first_output_with_trace = np.zeros(total_bytes, dtype=np.uint8)
                remaining_outputs = [
                    np.zeros(o["shape"], o["values"][0].dtype)
                    for o in stochastic_expected_outputs[1:]
                ]
                output_placeholders = [first_output_with_trace] + remaining_outputs
                if self.verbose:
                    print(
                        f"Allocated {first_output_bytes} bytes for first stochastic output + {self.trace_size} bytes for trace data"
                    )
                # Record the expected_outputs[0]'s shape and dtype, to be used to split actual outputs from trace.
                expected_outputs_0_shape = stochastic_expected_outputs[0]["shape"]
                expected_outputs_0_dtype = stochastic_expected_outputs[0][
                    "values"
                ].dtype
            else:
                # Case 3: Trace only, no expected outputs
                trace_only_output = np.zeros(self.trace_size, dtype=np.uint8)
                output_placeholders = [trace_only_output]
                if self.verbose:
                    print(
                        f"Trace-only mode: allocated {self.trace_size} bytes for trace data"
                    )
        else:
            # Case 4: No trace, original behavior
            if expected_outputs:
                output_placeholders = [
                    np.zeros(o.shape, o.dtype) for o in expected_outputs
                ]
            elif stochastic_expected_outputs:
                output_placeholders = [
                    np.zeros(o["shape"], o["values"][0].dtype)
                    for o in stochastic_expected_outputs
                ]
            else:
                assert (
                    False
                ), f"Expect one of 'expected_outputs' and 'stochastic_expected_outputs' to not be empty, or trace_size > 0."

        expanded_inputs = inputs + output_placeholders

        compiled_module = backend.compile(mlir_module)
        with filelock.FileLock("/tmp/npu.lock"):
            module_function = backend.load(compiled_module)
            actual_outputs = module_function(*expanded_inputs)

        backend.unload()

        # Remove input slots from the received outputs first
        actual_outputs = list(actual_outputs[len(inputs) :])

        # Handle trace data extraction and saving
        if self.trace_size > 0:
            actual_outputs[0], trace = extract_trace(
                actual_outputs[0],
                expected_outputs_0_shape,
                expected_outputs_0_dtype,
                self.trace_size,
            )
            write_out_trace(trace, active_trace_file)

            print(f"Trace data ({self.trace_size} bytes) saved to {active_trace_file}")

        # Perform result checking only if we have expected outputs
        if expected_outputs and actual_outputs:
            if self._check_outputs(
                actual_outputs=actual_outputs,
                expected_outputs=expected_outputs,
                rtol=rtol,
            ):
                print("PASS!")
                return_code = 0
            else:
                print("failed.")
                return_code = -1
        elif stochastic_expected_outputs and actual_outputs:
            if self._check_outputs_stochastic(
                actual_outputs=actual_outputs,
                stochastic_expected_outputs=stochastic_expected_outputs,
                rtol=rtol,
            ):
                print("PASS!")
                return_code = 0
            else:
                print("failed.")
                return_code = -1
        elif self.trace_size > 0 and not (
            expected_outputs or stochastic_expected_outputs
        ):
            # Trace-only case
            print("Trace data extracted successfully!")
            return_code = 0
        else:
            print("No outputs to validate.")
            return_code = 0

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

    def _check_outputs_stochastic(
        self,
        actual_outputs: List[np.ndarray],
        stochastic_expected_outputs: List[np.ndarray],
        rtol: float = 1e-3,
    ):
        assert len(actual_outputs) == len(
            stochastic_expected_outputs
        ), f"Number of actual outputs ({len(actual_outputs)}) does not equal number of expected outputs ({len(stochastic_expected_outputs)})"
        np.set_printoptions(formatter={"int": hex})

        for i, (actual, expected) in enumerate(
            zip(actual_outputs, stochastic_expected_outputs)
        ):
            actual = np.reshape(actual, expected["shape"])

            if self.verbose:
                print("Expected: ")
                if len(expected["shape"]) == 2:
                    print(np.asmatrix(expected))
                else:
                    print("Shape: ", expected["shape"])
                    print("Indices: ", expected["indices"])
                    print("Values: ", expected["values"])
                print("Actual: ")
                if len(actual.shape) == 2:
                    print(np.asmatrix(actual))
                else:
                    print(actual)

            if expected["values"][0].dtype in [
                np.float16,
                np.float32,
                np.float64,
                bfloat16,
            ]:
                if expected["values"][0].dtype == bfloat16:
                    expected["values"] = expected["values"].astype(np.float64)
                    actual = actual.astype(np.float64)
                actual_stochastic = actual[tuple(expected["indices"])]
                if not np.allclose(actual_stochastic, expected["values"], rtol=rtol):
                    print(f"ERROR: Output {i} does not meet expected output.")
                    print("Expected: ")
                    print(expected["values"])
                    print("Actual: ")
                    print(actual_stochastic)
                    return False
            else:
                actual_stochastic = actual[tuple(expected["indices"])]
                if not np.array_equal(actual_stochastic, expected["values"]):
                    print(f"ERROR: Output {i} does not meet expected output.")
                    print("Expected: ")
                    print(expected["values"])
                    print("Actual: ")
                    print(actual_stochastic)
                    return False

        return True
