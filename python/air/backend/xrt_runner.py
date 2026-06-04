# ./python/air/backend/xrt_runner.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import tempfile

import numpy as np
from .xrt import XRTBackend
from air.dialects.air import *
import filelock
from typing import List
from collections import defaultdict
from ml_dtypes import bfloat16
import timeit

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
        runtime_loop_tiling_sizes: list[int] = [],
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
        num_device_cols: int = 0,
        debug_ir: bool = False,
        bf16_emulation: bool = False,
        target_device: str = None,
        stack_size: int = 1024,
        n_perf_iters: int = 0,
        n_warmup_iters: int = 10,
        perf_flops: float = None,
        report_precision: bool = False,
    ):
        """
        Args:
            verbose: verbose output
            omit_while_true_loop: configure aircc to omit the while true loop it traditionally emits.
            omit_pingpong: configure aircc to omit the generation of ping-pong buffering for specific memory levels. Supported values: "", "L1", "L2", "all". Empty string means no omission (default).
            lower_linalg_to_func: configure aircc to lower linalg.generic to function calls, or loops.
            air_loop_fusion: configure aircc to add air-loop-fusion experimental pass.
            runtime_loop_tiling_sizes: tile sizes forwarded through XRTBackend to aircc as --air-runtime-loop-tiling-sizes, which the shim DMA BD optimization pass (air-opt-shim-dma-bds) consumes as shim-dma-tile-sizes. Omit or pass an empty list to skip tiling.
            omit_auto_broadcast: configure aircc to omit the detection and lowering of broadcast data movements.
            channel_multiplexing: configure aircc to perform air channel multiplexing on specified memroy spaces.
            use_lock_race_condition_fix: configure aircc to enable a fix for lock race condition which protects against race condition.
            trace_offset: configure aircc to stream out profiling traces at outputs, starting from the specified offset.
            trace_size: configure aircc to stream out profiling traces at outputs, with specified trace data size.
            output_format: configure aircc to produce output binary in to one of the following formats: [xclbin, txn, elf].
            kernel_name: configure aircc to package the kernel with the specified name.
            instance_name: configure aircc to package the kernel with specified instance name in xclbin metadata.
            kernel_id: configure aircc to package the kernel with specified kernel id in xclbin file.
            xclbin_input: configure aircc to package the kernel into an existing xclbin with specified xclbin file name.
            trace_file: default filename for saving trace data.
            num_device_cols: number of device columns to confine the design within (0 means entire device, default).
                For npu1 (4 columns total): valid values are 0 (entire device), 1, 2, 3
                For npu2 (8 columns total): valid values are 0 (entire device), 1, 2, 3, 4, 5, 6, 7
            debug_ir: enable debug mode to emit IR after each individual pass for fine-grained inspection.
                IRs are saved to <tmpdir>/debug_ir/ with sequence numbers.
            bf16_emulation: emulate f32 vector arithmetic using bf16 operations.
            target_device: specify target device explicitly ("npu1", "npu2", etc.). If None, will attempt auto-detection.
            stack_size: stack size in bytes per AIE core (default: 1024). Increase when
                kernels have deep call chains (e.g., scalar fdiv needs ~1152 bytes).
            n_perf_iters: when > 0, time the kernel over this many iterations (after
                n_warmup_iters warmup runs) and print the average Latency (us). Default
                0 disables timing, preserving the original single-shot behavior.
            n_warmup_iters: warmup iterations excluded from timing when n_perf_iters > 0.
            perf_flops: total floating-point op count (e.g. 2*M*K*N for GEMM/GEMV) used
                to additionally report Throughput in GFLOP/s. Leave None for kernels
                whose FLOP count is not meaningful (e.g. RMSNorm, RoPE, eltwise) — then
                only Latency is printed.
            report_precision: when True, the output check prints error statistics
                (mean relative L1, plus max/p99 of relative and absolute error) even
                when the test passes — useful for seeing how much tolerance margin a
                kernel actually has. Default False (no extra output).
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
        self.num_device_cols = num_device_cols
        self.debug_ir = debug_ir
        self.bf16_emulation = bf16_emulation
        self.target_device = target_device
        self.stack_size = stack_size
        self.n_perf_iters = n_perf_iters
        self.n_warmup_iters = n_warmup_iters
        self.perf_flops = perf_flops
        self.last_latency_us = None
        self.report_precision = report_precision

    def run_test(
        self,
        mlir_module: np.ndarray,
        inputs: List[np.ndarray],
        expected_outputs: List[np.ndarray] = [],
        stochastic_expected_outputs: List[np.ndarray] = [],
        rtol: float = 1e-3,
        atol: float = 1e-8,
        max_mismatch_percentage: float = 0,
        min_correlation: float = None,
        trace_file: str = None,
    ):
        """
        Args:
            mlir_module: input mlir module to test.
            inputs: input matrices.
            expected_outputs: expected output matrices.
            stochastic_expected_outputs: expected output matrices stored in sparse coordinates. Expect each matrix to be a dictionary containing "shape", "indices" and "values" fields.
            rtol: relative error tolerance.
            atol: absolute error tolerance.
            max_mismatch_percentage: max percentage (0-100) of elements allowed to exceed tolerance (0 = all must pass, 20 = 20% can fail).
            min_correlation: minimum Pearson correlation coefficient (0-1) between actual and expected outputs for floating-point data. None disables this check.
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
            num_device_cols=self.num_device_cols,
            debug_ir=self.debug_ir,
            bf16_emulation=self.bf16_emulation,
            target_device=self.target_device,
            stack_size=self.stack_size,
            n_perf_iters=self.n_perf_iters,
            n_warmup_iters=self.n_warmup_iters,
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
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            module_function = backend.load(compiled_module)
            actual_outputs = module_function(*expanded_inputs)

        # Surface timing collected by the backend invoker (if n_perf_iters > 0).
        self.last_latency_us = getattr(backend, "last_latency_us", None)
        if self.n_perf_iters > 0 and self.last_latency_us is not None:
            line = f"Latency (us): {self.last_latency_us:.1f}"
            if self.perf_flops is not None:
                gflops = self.perf_flops / (self.last_latency_us * 1e-6) / 1e9
                line += f" | Throughput: {gflops:.6e} GFLOP/s"
            print(line)

        backend.unload()

        # Remove input slots from the received outputs first
        actual_outputs = list(actual_outputs[len(inputs) :])

        # Handle trace data extraction and saving
        if self.trace_size > 0:
            # Import trace utilities only when needed for trace handling
            try:
                from aie.utils import TraceConfig, HostRuntime
            except ImportError:
                raise AirBackendError(
                    "Trace utilities (aie.utils) are not available. "
                    "Trace functionality requires mlir-aie to be installed. "
                    "Install mlir-aie to use trace_size parameter."
                )

            actual_outputs[0], trace = HostRuntime._extract_prefix(
                actual_outputs[0],
                expected_outputs_0_shape,
                expected_outputs_0_dtype,
            )
            trace = trace.view(np.uint32).reshape(self.trace_size // 4)
            trace_config = TraceConfig(
                trace_size=self.trace_size, trace_file=active_trace_file
            )
            trace_config.write_trace(trace)

            print(f"Trace data ({self.trace_size} bytes) saved to {active_trace_file}")

        # Perform result checking only if we have expected outputs
        if expected_outputs and actual_outputs:
            if self._check_outputs(
                actual_outputs=actual_outputs,
                expected_outputs=expected_outputs,
                rtol=rtol,
                atol=atol,
                max_mismatch_percentage=max_mismatch_percentage,
                min_correlation=min_correlation,
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
                atol=atol,
                max_mismatch_percentage=max_mismatch_percentage,
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

    def _print_precision(self, i, actual_f64, expected_f64, rtol, atol):
        """Print error statistics for one output (called when report_precision).

        mean_rel_L1 (mean|a-e| / mean|e|) is the robust headline metric — it does
        not blow up where the reference is near zero. The max of relative and
        absolute error gives the worst-case point. Both are cheap (mean/max only,
        no sorting) so this stays light even on full-size outputs.
        """
        abs_err = np.abs(actual_f64 - expected_f64)
        rel_err = abs_err / (np.abs(expected_f64) + 1e-30)
        mean_rel_l1 = abs_err.mean() / (np.abs(expected_f64).mean() + 1e-30)
        print(
            f"[precision] Output {i} ({abs_err.size} elements): "
            f"mean_rel_L1={mean_rel_l1:.3e} | "
            f"rel_err max={rel_err.max():.3e} | "
            f"abs_err max={abs_err.max():.3e} | "
            f"rtol={rtol:.1e} atol={atol:.1e}"
        )

    def _check_outputs(
        self,
        actual_outputs: List[np.ndarray],
        expected_outputs: List[np.ndarray],
        rtol: float = 1e-3,
        atol: float = 1e-8,
        max_mismatch_percentage: float = 0,
        min_correlation: float = None,
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

                if self.report_precision:
                    # bf16 is already upcast to f64 above; other float dtypes
                    # (f16/f32/f64) are fine for the error arithmetic as-is.
                    self._print_precision(i, actual, expected, rtol, atol)

                # Element-wise tolerance check
                elementwise_ok = True
                close_mask = np.isclose(actual, expected, rtol=rtol, atol=atol)
                mismatch_indices = np.where(~close_mask)
                num_mismatches = len(mismatch_indices[0])
                total_elements = expected.size
                max_acceptable = int(total_elements * max_mismatch_percentage / 100)
                if num_mismatches > max_acceptable:
                    elementwise_ok = False
                    print(f"ERROR: Output {i} does not meet expected output.")
                    print(f"Shape: {expected.shape}")
                    if total_elements > 0:
                        print(
                            f"Mismatches: {num_mismatches} / {total_elements} elements ({100*num_mismatches/total_elements:.2f}%)"
                        )
                    else:
                        print(
                            f"Mismatches: {num_mismatches} / {total_elements} elements (empty array)"
                        )
                    if max_acceptable > 0:
                        print(
                            f"Max acceptable: {max_acceptable} ({max_mismatch_percentage}%)"
                        )
                    # Show first N mismatches
                    max_display = 20
                    print(
                        f"First {min(max_display, num_mismatches)} mismatched locations:"
                    )
                    for j in range(min(max_display, num_mismatches)):
                        idx = tuple(dim[j] for dim in mismatch_indices)
                        print(
                            f"  Index {idx}: expected={expected[idx]}, actual={actual[idx]}, diff={abs(actual[idx] - expected[idx])}"
                        )
                    if num_mismatches > max_display:
                        print(
                            f"  ... and {num_mismatches - max_display} more mismatches"
                        )

                # Correlation check (parallel with element-wise)
                corr_ok = True
                if min_correlation is not None and total_elements > 0:
                    corr = float(
                        np.corrcoef(actual.flatten(), expected.flatten())[0, 1]
                    )
                    print(
                        f"Output {i} correlation: {corr:.6f} "
                        f"(threshold: {min_correlation})"
                    )
                    if not np.isfinite(corr) or corr < min_correlation:
                        corr_ok = False
                        print(
                            f"ERROR: Output {i} correlation {corr:.6f} "
                            f"below threshold {min_correlation}"
                        )

                if not elementwise_ok or not corr_ok:
                    return False
            else:
                if not np.array_equal(actual, expected):
                    print(f"ERROR: Output {i} does not meet expected output.")
                    # Find mismatched elements
                    mismatch_mask = actual != expected
                    mismatch_indices = np.where(mismatch_mask)
                    num_mismatches = len(mismatch_indices[0])
                    total_elements = expected.size
                    print(f"Shape: {expected.shape}")
                    if total_elements > 0:
                        print(
                            f"Mismatches: {num_mismatches} / {total_elements} elements ({100*num_mismatches/total_elements:.2f}%)"
                        )
                    else:
                        print(
                            f"Mismatches: {num_mismatches} / {total_elements} elements (empty array)"
                        )
                    # Show first N mismatches
                    max_display = 20
                    print(
                        f"First {min(max_display, num_mismatches)} mismatched locations:"
                    )
                    for j in range(min(max_display, num_mismatches)):
                        idx = tuple(dim[j] for dim in mismatch_indices)
                        print(
                            f"  Index {idx}: expected={expected[idx]}, actual={actual[idx]}"
                        )
                    if num_mismatches > max_display:
                        print(
                            f"  ... and {num_mismatches - max_display} more mismatches"
                        )
                    return False

        return True

    def _check_outputs_stochastic(
        self,
        actual_outputs: List[np.ndarray],
        stochastic_expected_outputs: List[np.ndarray],
        rtol: float = 1e-3,
        atol: float = 1e-8,
        max_mismatch_percentage: float = 0,
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
                if self.report_precision:
                    self._print_precision(
                        i,
                        np.asarray(actual_stochastic, dtype=np.float64),
                        np.asarray(expected["values"], dtype=np.float64),
                        rtol,
                        atol,
                    )
                close_mask = np.isclose(
                    actual_stochastic, expected["values"], rtol=rtol, atol=atol
                )
                mismatch_positions = np.where(~close_mask)[0]
                num_mismatches = len(mismatch_positions)
                total_elements = len(expected["values"])
                max_acceptable = int(total_elements * max_mismatch_percentage / 100)
                if num_mismatches > max_acceptable:
                    print(f"ERROR: Output {i} does not meet expected output.")
                    print(f"Shape: {expected['shape']}")
                    print(f"Stochastic check: {total_elements} sampled elements")
                    print(
                        f"Mismatches: {num_mismatches} / {total_elements} elements ({100*num_mismatches/total_elements:.2f}%)"
                    )
                    if max_acceptable > 0:
                        print(
                            f"Max acceptable: {max_acceptable} ({max_mismatch_percentage}%)"
                        )
                    # Show first N mismatches
                    max_display = 20
                    print(
                        f"First {min(max_display, num_mismatches)} mismatched locations:"
                    )
                    for j in range(min(max_display, num_mismatches)):
                        pos = mismatch_positions[j]
                        idx = tuple(dim[pos] for dim in expected["indices"])
                        exp_val = expected["values"][pos]
                        act_val = actual_stochastic[pos]
                        print(
                            f"  Index {idx}: expected={exp_val}, actual={act_val}, diff={abs(act_val - exp_val)}"
                        )
                    if num_mismatches > max_display:
                        print(
                            f"  ... and {num_mismatches - max_display} more mismatches"
                        )
                    return False
            else:
                actual_stochastic = actual[tuple(expected["indices"])]
                if not np.array_equal(actual_stochastic, expected["values"]):
                    print(f"ERROR: Output {i} does not meet expected output.")
                    # Find mismatched elements
                    mismatch_mask = actual_stochastic != expected["values"]
                    mismatch_positions = np.where(mismatch_mask)[0]
                    num_mismatches = len(mismatch_positions)
                    total_elements = len(expected["values"])
                    print(f"Shape: {expected['shape']}")
                    print(f"Stochastic check: {total_elements} sampled elements")
                    if total_elements > 0:
                        print(
                            f"Mismatches: {num_mismatches} / {total_elements} elements ({100*num_mismatches/total_elements:.2f}%)"
                        )
                    else:
                        print(
                            f"Mismatches: {num_mismatches} / {total_elements} elements (empty array)"
                        )
                    # Show first N mismatches
                    max_display = 20
                    print(
                        f"First {min(max_display, num_mismatches)} mismatched locations:"
                    )
                    for j in range(min(max_display, num_mismatches)):
                        pos = mismatch_positions[j]
                        idx = tuple(dim[pos] for dim in expected["indices"])
                        exp_val = expected["values"][pos]
                        act_val = actual_stochastic[pos]
                        print(f"  Index {idx}: expected={exp_val}, actual={act_val}")
                    if num_mismatches > max_display:
                        print(
                            f"  ... and {num_mismatches - max_display} more mismatches"
                        )
                    return False

        return True
