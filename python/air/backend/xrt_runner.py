# ./python/air/backend/xrt_runner.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
High-level runner helpers for mlir-air programming examples and tests.

Public API
----------
run_on_npu(args, mlir_module, inputs, instance_name, ...)
    Compile (and optionally run) an AIR module, dispatching on
    args.compile_mode.  Replaces the boilerplate if/elif block found
    in every example's __main__.

make_air_parser(description, prog)
    Return an ArgumentParser pre-populated with the four universal flags.

type_mapper(np_dtype)
    Map a numpy dtype to the corresponding MLIR type inside a module context.

TYPE_MAP_DICT
    The underlying defaultdict used by type_mapper.
"""

import filelock
import numpy as np
from collections import defaultdict
from ml_dtypes import bfloat16
from typing import List

from air.dialects.air import *

from .abc import AirBackendError
from .xrt import (
    compile_air,
    get_air_runtime,
    AirRuntime,
    XRTTensor,
    XRTBackend,
    XRTCompileArtifact,
)

try:
    import aie.utils as _aie_utils

    _tensor = _aie_utils.tensor
    _has_aie_utils = True
except ImportError:
    _has_aie_utils = False
    _tensor = None


# ---------------------------------------------------------------------------
# Type mapping helpers (unchanged — used by many callers)
# ---------------------------------------------------------------------------

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
    """Map a numpy dtype to the MLIR type for use inside a module context.

    Args:
        np_dtype: The numpy data type to map.

    Returns:
        The corresponding MLIR type.

    Raises:
        AirBackendError: If the dtype has no known mapping.
    """
    xrt_dtype = TYPE_MAP_DICT[np_dtype]()

    if xrt_dtype is None:
        raise AirBackendError(f"numpy data type {np_dtype} has no default mapping")
    elif xrt_dtype.width / 8 != np.dtype(np_dtype).itemsize:
        raise AirBackendError(
            f"Python data type has width {xrt_dtype.width / 8} but numpy data type "
            f"has width {np.dtype(np_dtype).itemsize}"
        )
    return xrt_dtype


# ---------------------------------------------------------------------------
# Argument parser factory (unchanged)
# ---------------------------------------------------------------------------


def make_air_parser(description, prog="run.py"):
    """Return an ArgumentParser pre-populated with the four universal flags.

    Flags added:
        -v / --verbose
        -p / --print-module-only
        --compile-mode  {compile-only, compile-and-run}
        --output-format {xclbin, elf}

    The caller adds any example-specific arguments afterwards.
    """
    import argparse

    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    return parser


# ---------------------------------------------------------------------------
# check_print_module helper (kept here for backward compat)
# ---------------------------------------------------------------------------


def check_print_module(mlir_module, args):
    """Print the MLIR module and exit if --print-module-only was passed."""
    if args.print_module_only:
        print(mlir_module)
        exit(0)


# ---------------------------------------------------------------------------
# run_on_npu() — the main dispatch helper
# ---------------------------------------------------------------------------


def run_on_npu(
    args,
    mlir_module,
    inputs,
    instance_name,
    expected_outputs=None,
    stochastic_expected_outputs=None,
    rtol: float = 1e-3,
    atol: float = 1e-8,
    runtime_loop_tiling_sizes=None,
    max_mismatch_percentage: float = 0.0,
    min_correlation=None,
    # Extra compile_air kwargs forwarded as-is
    **compile_kwargs,
) -> int:
    """Compile (and optionally run+verify) an AIR module.

    Dispatches on args.compile_mode:
      - "compile-only"    → compile, write artifacts, return 0
      - "compile-and-run" → compile, run on NPU, verify, return exit code

    Args:
        args: Parsed argparse namespace (must have .verbose, .compile_mode,
              .output_format, and optionally .print_module_only).
        mlir_module: MLIR module from build_module() / Module.parse().
        inputs: List of numpy input arrays.
        instance_name: xclbin instance name string.
        expected_outputs: List of numpy reference arrays (dense check).
        stochastic_expected_outputs: List of {"shape","indices","values"} dicts.
        rtol: Relative tolerance forwarded to AirRuntime.run_test().
        atol: Absolute tolerance forwarded to AirRuntime.run_test().
        runtime_loop_tiling_sizes: Tiling sizes (default [4, 4]).
        max_mismatch_percentage: Max % of elements allowed to mismatch.
        min_correlation: Minimum Pearson correlation (None = disabled).
        **compile_kwargs: Additional kwargs forwarded to compile_air().

    Returns:
        int: 0 = pass / compile-only success, -1 = failure.
    """
    if runtime_loop_tiling_sizes is None:
        runtime_loop_tiling_sizes = [4, 4]

    # --print-module-only support
    if getattr(args, "print_module_only", False):
        print(mlir_module)
        return 0

    npu_kernel = compile_air(
        mlir_module,
        verbose=args.verbose,
        output_format=args.output_format,
        omit_while_true_loop=False,
        instance_name=instance_name,
        runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
        **compile_kwargs,
    )

    if args.compile_mode == "compile-only":
        return 0

    # compile-and-run
    runtime = get_air_runtime()

    # Build io_args: inputs + zero-initialised output buffers.
    # Use aie.utils.tensor() so the correct tensor class is picked automatically.
    input_tensors = [_tensor(a) if _has_aie_utils else a for a in inputs]
    output_tensors = _make_output_tensors(
        expected_outputs or [], stochastic_expected_outputs or []
    )
    io_args = input_tensors + output_tensors

    # Build refs dict: map output buffer index → expected numpy array
    refs = {len(inputs) + i: exp for i, exp in enumerate(expected_outputs or [])}

    return runtime.run_test(
        npu_kernel,
        io_args,
        refs=refs,
        rtol=rtol,
        atol=atol,
        max_mismatch_percentage=max_mismatch_percentage,
        min_correlation=min_correlation,
        stochastic_refs=stochastic_expected_outputs or [],
    )


def _make_output_tensors(expected_outputs, stochastic_expected_outputs):
    """Allocate zero-filled tensors matching each expected output spec.

    Uses aie.utils.tensor() so the correct tensor class (XRTTensor when XRT
    is available, CPUOnlyTensor otherwise) is selected automatically.
    """
    tensors = []
    for exp in expected_outputs:
        if _has_aie_utils:
            tensors.append(_tensor(np.zeros(exp.shape, dtype=exp.dtype)))
        else:
            tensors.append(np.zeros(exp.shape, dtype=exp.dtype))
    for sref in stochastic_expected_outputs:
        dtype = sref["values"].dtype if hasattr(sref["values"], "dtype") else np.float32
        shape = sref["shape"]
        if isinstance(shape, int):
            shape = (shape,)
        if _has_aie_utils:
            tensors.append(_tensor(np.zeros(shape, dtype=dtype)))
        else:
            tensors.append(np.zeros(shape, dtype=dtype))
    return tensors


# ---------------------------------------------------------------------------
# Backward compatibility shim — XRTRunner
# ---------------------------------------------------------------------------


class XRTRunner:
    """
    Deprecated. Use compile_air() + get_air_runtime() or run_on_npu() instead.

    This shim preserves the old XRTRunner interface so existing call sites
    continue to work without modification during the migration period.
    """

    def __init__(
        self,
        verbose: bool = False,
        omit_while_true_loop: bool = True,
        omit_pingpong: str = "",
        lower_linalg_to_func=None,
        air_loop_fusion: bool = False,
        runtime_loop_tiling_sizes: list = None,
        omit_auto_broadcast: bool = False,
        channel_multiplexing: list = None,
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
    ):
        self.verbose = verbose
        # Support backward compatibility: convert bool omit_pingpong
        if isinstance(omit_pingpong, bool):
            self.omit_pingpong = "all" if omit_pingpong else ""
        else:
            self.omit_pingpong = omit_pingpong
        self.omit_while_true_loop = omit_while_true_loop
        self.lower_linalg_to_func = lower_linalg_to_func
        self.air_loop_fusion = air_loop_fusion
        self.runtime_loop_tiling_sizes = runtime_loop_tiling_sizes or []
        self.omit_auto_broadcast = omit_auto_broadcast
        self.channel_multiplexing = channel_multiplexing or []
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

    def run_test(
        self,
        mlir_module,
        inputs: List[np.ndarray],
        expected_outputs: List[np.ndarray] = None,
        stochastic_expected_outputs: List = None,
        rtol: float = 1e-3,
        atol: float = 1e-8,
        max_mismatch_percentage: float = 0,
        min_correlation=None,
        trace_file: str = None,
    ) -> int:
        """Compile, run and verify an AIR module.

        Args:
            mlir_module: MLIR module to test.
            inputs: Input numpy arrays.
            expected_outputs: Expected dense output arrays.
            stochastic_expected_outputs: Sparse reference dicts.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            max_mismatch_percentage: Max % of mismatches tolerated.
            min_correlation: Min Pearson correlation (None = disabled).
            trace_file: Override trace data filename.

        Returns:
            0 on pass, -1 on failure.
        """
        if expected_outputs is None:
            expected_outputs = []
        if stochastic_expected_outputs is None:
            stochastic_expected_outputs = []

        if self.verbose:
            print("Running module:")
            print(mlir_module)

        npu_kernel = compile_air(
            mlir_module,
            verbose=self.verbose,
            target_device=self.target_device,
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
        )

        # Handle trace mode separately (uses legacy raw-numpy path for now)
        if self.trace_size > 0:
            return self._run_with_trace(
                npu_kernel,
                inputs,
                expected_outputs,
                stochastic_expected_outputs,
                rtol,
                atol,
                max_mismatch_percentage,
                trace_file or self.trace_file,
            )

        # Standard (no-trace) path
        runtime = get_air_runtime()
        input_tensors = [_tensor(a) if _has_aie_utils else a for a in inputs]
        output_tensors = _make_output_tensors(
            expected_outputs, stochastic_expected_outputs
        )
        io_args = input_tensors + output_tensors

        # Build refs dict
        refs = {len(inputs) + i: exp for i, exp in enumerate(expected_outputs)}

        return runtime.run_test(
            npu_kernel,
            io_args,
            refs=refs,
            rtol=rtol,
            atol=atol,
            max_mismatch_percentage=max_mismatch_percentage,
            min_correlation=min_correlation,
            stochastic_refs=stochastic_expected_outputs,
        )

    def _run_with_trace(
        self,
        npu_kernel,
        inputs,
        expected_outputs,
        stochastic_expected_outputs,
        rtol,
        atol,
        max_mismatch_percentage,
        trace_file,
    ) -> int:
        """Handle the trace-enabled execution path."""
        try:
            from aie.utils import TraceConfig, HostRuntime
        except ImportError:
            raise AirBackendError(
                "Trace utilities (aie.utils) are not available. "
                "Install mlir-aie to use trace_size parameter."
            )

        runtime = get_air_runtime()

        # Build combined tensors for trace path
        if expected_outputs:
            total_bytes = expected_outputs[0].nbytes + self.trace_size
            first_out = np.zeros(total_bytes, dtype=np.uint8)
            rest_outs = [np.zeros(o.shape, o.dtype) for o in expected_outputs[1:]]
            output_placeholders = [first_out] + rest_outs
            expected_outputs_0_shape = expected_outputs[0].shape
            expected_outputs_0_dtype = expected_outputs[0].dtype
        elif stochastic_expected_outputs:
            first_output_elements = np.prod(stochastic_expected_outputs[0]["shape"])
            first_output_bytes = (
                first_output_elements
                * stochastic_expected_outputs[0]["values"][0].dtype.itemsize
            )
            total_bytes = first_output_bytes + self.trace_size
            first_out = np.zeros(total_bytes, dtype=np.uint8)
            rest_outs = [
                np.zeros(o["shape"], o["values"][0].dtype)
                for o in stochastic_expected_outputs[1:]
            ]
            output_placeholders = [first_out] + rest_outs
            expected_outputs_0_shape = stochastic_expected_outputs[0]["shape"]
            expected_outputs_0_dtype = stochastic_expected_outputs[0]["values"].dtype
        else:
            trace_only_output = np.zeros(self.trace_size, dtype=np.uint8)
            output_placeholders = [trace_only_output]
            expected_outputs_0_shape = None
            expected_outputs_0_dtype = None

        all_np = inputs + output_placeholders
        io_args = [_tensor(a) if _has_aie_utils else a for a in all_np]

        handle = runtime.load(npu_kernel)
        with filelock.FileLock("/tmp/npu.lock"):
            runtime.run(handle, io_args)

        # Extract numpy results
        actual_outputs_np = [t.numpy() for t in io_args[len(inputs) :]]

        # Extract trace data
        if expected_outputs_0_shape is not None:
            actual_outputs_np[0], trace = HostRuntime._extract_prefix(
                actual_outputs_np[0],
                expected_outputs_0_shape,
                np.dtype(expected_outputs_0_dtype),
            )
        else:
            trace = actual_outputs_np[0].view(np.uint8)

        trace = trace.view(np.uint32).reshape(self.trace_size // 4)
        trace_config = TraceConfig(trace_size=self.trace_size, trace_file=trace_file)
        trace_config.write_trace(trace)
        print(f"Trace data ({self.trace_size} bytes) saved to {trace_file}")

        # Verify results — wrap numpy arrays as lightweight objects with .numpy()
        class _NumpyWrap:
            def __init__(self, arr):
                self._arr = arr

            def numpy(self):
                return self._arr

        wrapped = [_NumpyWrap(a) for a in actual_outputs_np]

        if expected_outputs and actual_outputs_np:
            refs = {i: exp for i, exp in enumerate(expected_outputs)}
            errors = AirRuntime.verify_results(
                wrapped,
                refs=refs,
                rtol=rtol,
                atol=atol,
                max_mismatch_percentage=max_mismatch_percentage,
            )
            if errors == 0:
                print("PASS!")
                return 0
            else:
                print("failed.")
                return -1
        elif stochastic_expected_outputs and actual_outputs_np:
            errors = AirRuntime.verify_results(
                wrapped,
                refs={},
                rtol=rtol,
                atol=atol,
                max_mismatch_percentage=max_mismatch_percentage,
                stochastic_refs=stochastic_expected_outputs,
            )
            if errors == 0:
                print("PASS!")
                return 0
            else:
                print("failed.")
                return -1
        else:
            print("Trace data extracted successfully!")
            return 0
