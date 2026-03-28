# ./python/air/backend/xrt_backend.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
XRT backend for mlir-air.

Public API
----------
compile_air(air_module, ...)  -> NPUKernel
    Compile an AIR dialect MLIR module to an NPU kernel artifact.

AirRuntime                   (CachedXRTRuntime subclass)
    Richer verify_results() with rtol/atol, stochastic sampling,
    mismatch budget, and Pearson correlation.

get_air_runtime() -> AirRuntime
    Return the process-level singleton AirRuntime.

XRTTensor                    (re-exported from aie.utils)
    Numpy-backed buffer object for XRT.
"""

import air.ir
import air.passmanager

from .abc import AirBackendError

import air.compiler.util

# Register the AIR dialect so air.ir.Context() can parse AIR ops.
from air.dialects import air as _air_dialect  # noqa: F401

import numpy as np
import os
import shutil
import subprocess

from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# mlir-aie runtime imports
# ---------------------------------------------------------------------------
try:
    import aie.utils as _aie_utils
    from aie.utils import CachedXRTRuntime, NPUKernel
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

    _HAS_AIE_RUNTIME = True
    # Factory function that selects XRTTensor or CPUOnlyTensor depending on
    # whether pyxrt is importable. Used throughout instead of XRTTensor() directly.
    _tensor = _aie_utils.tensor
except ImportError:
    _HAS_AIE_RUNTIME = False
    CachedXRTRuntime = object  # fallback base so class definition succeeds
    NPUKernel = None
    XRTTensor = None
    _tensor = None


# ---------------------------------------------------------------------------
# compile_air() — replaces XRTBackend.compile()
# ---------------------------------------------------------------------------


def compile_air(
    air_module: air.ir.Module,
    *,
    verbose: bool = False,
    target_device: str = None,
    omit_while_true_loop: bool = False,
    omit_pingpong: str = "",
    lower_linalg_to_func=None,
    air_loop_fusion: bool = False,
    runtime_loop_tiling_sizes=None,
    omit_auto_broadcast: bool = False,
    channel_multiplexing=None,
    use_lock_race_condition_fix: bool = False,
    trace_offset: int = 0,
    trace_size: int = 0,
    output_format: str = "xclbin",
    xclbin_kernel_name: str = "",
    instance_name: str = "",
    kernel_id: str = "",
    xclbin_input: str = "",
    num_device_cols: int = 0,
    debug_ir: bool = False,
    bf16_emulation: bool = False,
    # Legacy aliases kept for backward compat
    kernel_name: str = "",
    output_binary_name: str = "air",
    insts: str = "air.insts.bin",
):
    """Compile an AIR dialect MLIR module to an NPUKernel artifact.

    Replaces ``XRTBackend(...).compile(air_module)``.

    Args:
        air_module: The MLIR module in AIR dialect.
        verbose: Verbose output.
        target_device: Explicit target device ("npu1", "npu2", etc.).
            If None, auto-detect via xrt-smi.
        omit_while_true_loop: Omit the while-true loop in generated code.
        omit_pingpong: Omit ping-pong buffering for given memory level.
            Values: "", "L1", "L2", "all".
        lower_linalg_to_func: Lower linalg.generic to function calls.
        air_loop_fusion: Enable air-loop-fusion pass.
        runtime_loop_tiling_sizes: Extra runtime loop tiling sizes.
        omit_auto_broadcast: Omit automatic broadcast detection.
        channel_multiplexing: Air channel multiplexing memory spaces.
        use_lock_race_condition_fix: Enable lock race condition fix.
        trace_offset: Trace output offset (bytes).
        trace_size: Trace output size (bytes).
        output_format: Output binary format: "xclbin", "elf", or "txn".
        xclbin_kernel_name: Kernel name embedded in xclbin metadata.
        instance_name: Instance name embedded in xclbin metadata.
        kernel_id: Kernel ID embedded in xclbin file.
        xclbin_input: Existing xclbin to embed the new kernel into.
        num_device_cols: Device columns to constrain the design to (0=all).
        debug_ir: Save IR after each pass to debug_ir/ directory.
        bf16_emulation: Emulate f32 vector arithmetic with bf16.
        kernel_name: Legacy alias for xclbin_kernel_name.
        output_binary_name: Base name for the output binary (without extension).
        insts: Instruction filename (for xclbin format).

    Returns:
        NPUKernel: Compiled kernel artifact with xclbin/insts paths.
    """
    if runtime_loop_tiling_sizes is None:
        runtime_loop_tiling_sizes = []
    if channel_multiplexing is None:
        channel_multiplexing = []

    # Support legacy kernel_name alias
    effective_kernel_name = xclbin_kernel_name or kernel_name

    # Support backward compatibility: convert bool omit_pingpong
    if isinstance(omit_pingpong, bool):
        omit_pingpong = "all" if omit_pingpong else ""

    # Determine target device
    if target_device is not None:
        if verbose:
            print(f"Using explicitly specified target device: {target_device}")
    else:
        target_device = "npu1"  # default fallback
        try:
            import re

            xrtsmi = "/opt/xilinx/xrt/bin/xrt-smi"
            result = subprocess.run(
                [xrtsmi, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            result_lines = result.stdout.decode("utf-8").split("\n")
            p = re.compile(r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU (\w+))\W*\|")
            for line in result_lines:
                m = p.match(line)
                if not m:
                    continue
                if verbose:
                    print("Found Ryzen AI device:", m.group(1))
                model = "unknown"
                if m.group(3):
                    model = str(m.group(3))
                if m.group(4):
                    model = str(m.group(4))
                if verbose:
                    print(f"\tmodel: '{model}'")
                if model in ["npu1", "Phoenix"]:
                    target_device = "npu1"
                elif model in ["npu4", "Strix"]:
                    target_device = "npu2"
                else:
                    print(f"WARNING: xrt-smi reported unknown NPU model '{model}'.")
                break
        except Exception as e:
            if verbose:
                print("Failed to run xrt-smi, using default target device")
                print(e)

    # Validate ELF format compatibility
    if output_format == "elf" and "npu1" in target_device:
        raise AirBackendError(
            f"output_format='elf' is not supported for {target_device} target. "
            "ELF output format is only supported on npu2 and later devices."
        )

    # Apply column configuration
    if num_device_cols > 0:
        max_cols = 4 if target_device == "npu1" else 8
        if num_device_cols > max_cols - 1:
            raise AirBackendError(
                f"Invalid num_device_cols value: {num_device_cols}. "
                f"For {target_device}, valid values are 0 (entire device) or 1-{max_cols-1}"
            )
        base_device = target_device
        target_device = f"{target_device}_{num_device_cols}col"
        if verbose:
            print(
                f"Confining design to {num_device_cols} column(s) of {base_device} device: {target_device}"
            )

    # Determine peano toolchain
    peano_package_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_package_dir and os.path.isdir(peano_package_dir):
        print(
            "compile_air: llvm-aie package detected via PEANO_INSTALL_DIR:",
            peano_package_dir,
        )

    # Determine output binary file name
    if output_format == "elf":
        output_binary = f"{output_binary_name}.elf"
    elif output_format == "txn":
        output_binary = f"{output_binary_name}.txn"
    else:  # xclbin (default)
        output_binary = f"{output_binary_name}.xclbin"

    with air.ir.Context():
        if verbose:
            print("AIR Module:")
            print(air_module)

        aircc_options = [
            "--device",
            target_device,
            "air.mlir",
        ]

        # Output file options
        if output_format == "elf":
            aircc_options += ["--elf-name", output_binary]
        else:
            aircc_options += ["-o", output_binary]
            aircc_options += ["-i", insts]

        for s in runtime_loop_tiling_sizes:
            aircc_options += [f"--air-runtime-loop-tiling-sizes={s}"]

        if verbose:
            aircc_options = aircc_options + ["-v"]

        if omit_while_true_loop:
            aircc_options += ["--omit-while-true-loop"]

        if omit_pingpong:
            pp_val = "all" if omit_pingpong is True else str(omit_pingpong)
            aircc_options += [f"--omit-ping-pong-transform={pp_val}"]

        if lower_linalg_to_func:
            aircc_options += ["--lower-linalg-to-func"]
            aircc_options += [lower_linalg_to_func]

        if air_loop_fusion:
            aircc_options += ["--air-loop-fusion"]

        if omit_auto_broadcast:
            aircc_options += ["--omit-auto-broadcast"]

        if len(channel_multiplexing) != 0:
            for ch in channel_multiplexing:
                aircc_options += [f"--air-channel-multiplexing={ch}"]

        if use_lock_race_condition_fix:
            aircc_options += ["--use-lock-race-condition-fix"]

        if trace_size != 0:
            aircc_options += ["-trace-size"]
            aircc_options += [str(trace_size)]
            aircc_options += ["-trace-offset"]
            aircc_options += [str(trace_offset)]

        if output_format != "":
            aircc_options += ["--output-format"]
            aircc_options += [output_format]
        if effective_kernel_name != "":
            aircc_options += ["--xclbin-kernel-name"]
            aircc_options += [effective_kernel_name]
        if instance_name != "":
            aircc_options += ["--xclbin-instance-name"]
            aircc_options += [instance_name]
        if kernel_id != "":
            aircc_options += ["--xclbin-kernel-id"]
            aircc_options += [kernel_id]
        if xclbin_input != "":
            aircc_options += ["--xclbin-input"]
            aircc_options += [xclbin_input]

        if peano_package_dir != "":
            aircc_options += ["--peano"]
            aircc_options += [peano_package_dir]
            aircc_options += ["--no-xchesscc"]
            aircc_options += ["--no-xbridge"]
        else:
            aircc_options += ["--xchesscc"]
            aircc_options += ["--xbridge"]

        if debug_ir:
            aircc_options += ["--debug-ir"]

        if bf16_emulation:
            aircc_options += ["--bf16-emulation"]

        if verbose:
            print("Running aircc with options:", " ".join(aircc_options))

        # Write module to disk for aircc
        with open("air.mlir", "w") as f:
            f.write(str(air_module))

        # Invoke aircc
        aircc_exe = shutil.which("aircc")
        if not aircc_exe:
            raise AirBackendError(
                "aircc binary not found in PATH. "
                "Ensure mlir-air is installed and aircc is on PATH."
            )
        result = subprocess.run(
            [aircc_exe] + aircc_options,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise AirBackendError(f"aircc compilation failed:\n{error_msg}")

    # Build kernel_name for NPUKernel
    if output_format == "elf" and instance_name != "":
        npu_kernel_name = f"main:{instance_name}"
    else:
        npu_kernel_name = effective_kernel_name if effective_kernel_name else "MLIR_AIE"

    if _HAS_AIE_RUNTIME:
        return NPUKernel(output_binary, insts, kernel_name=npu_kernel_name)
    else:
        # Fallback: return a simple namespace when aie.utils is unavailable
        import types

        kernel = types.SimpleNamespace(
            xclbin_path=output_binary,
            insts_path=insts,
            kernel_name=npu_kernel_name,
        )
        return kernel


# ---------------------------------------------------------------------------
# AirRuntime — CachedXRTRuntime with mlir-air's richer verification
# ---------------------------------------------------------------------------


class AirRuntime(CachedXRTRuntime):
    """
    mlir-aie's CachedXRTRuntime extended with mlir-air's richer verification.

    Inherits: device open, xclbin/ELF caching (32 contexts NPU2), run().
    Overrides: verify_results() with rtol/atol/stochastic/correlation checks.
    Adds: run_test() convenience method that compiles, runs, and verifies.
    """

    def run_test(
        self,
        npu_kernel,
        io_args,
        refs=None,
        rtol: float = 1e-3,
        atol: float = 1e-8,
        max_mismatch_percentage: float = 0.0,
        min_correlation=None,
        stochastic_refs=None,
        verbosity: int = 0,
        trace_file: str = "trace_data.txt",
    ) -> int:
        """
        Load, run, and verify an NPU kernel.

        Args:
            npu_kernel: NPUKernel from compile_air().
            io_args: List of XRTTensor objects (inputs + outputs).
            refs: dict mapping output index → expected numpy array (dense).
            rtol: Relative tolerance for floating-point checks.
            atol: Absolute tolerance for floating-point checks.
            max_mismatch_percentage: Max % of elements allowed to mismatch.
            min_correlation: Minimum Pearson correlation (None = disabled).
            stochastic_refs: List of {"shape", "indices", "values"} dicts.
            verbosity: Verbosity level.
            trace_file: Filename to save trace data (if trace_size > 0).

        Returns:
            0 on pass, -1 on failure.
        """
        import filelock

        handle = self.load(npu_kernel)
        with filelock.FileLock("/tmp/npu.lock"):
            self.run(handle, io_args)

        errors = self.verify_results(
            io_args,
            refs=refs or {},
            rtol=rtol,
            atol=atol,
            max_mismatch_percentage=max_mismatch_percentage,
            min_correlation=min_correlation,
            stochastic_refs=stochastic_refs,
            verbosity=verbosity,
        )
        if errors == 0:
            print("PASS!")
            return 0
        else:
            print("failed.")
            return -1

    @classmethod
    def verify_results(
        cls,
        io_args,
        refs=None,
        rtol: float = 1e-3,
        atol: float = 1e-8,
        max_mismatch_percentage: float = 0.0,
        min_correlation=None,
        stochastic_refs=None,
        verbosity: int = 0,
    ) -> int:
        """
        Verify kernel outputs against reference data.

        Args:
            io_args: List of XRTTensor (or numpy array) outputs.
            refs: dict {index: expected_np_array} for dense checks.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            max_mismatch_percentage: Max % of mismatches tolerated (0–100).
            min_correlation: Minimum Pearson correlation (None = disabled).
            stochastic_refs: List of {"shape","indices","values"} dicts.
            verbosity: Verbosity level.

        Returns:
            Number of errors found (0 = pass).
        """
        if refs is None:
            refs = {}

        errors = 0
        np.set_printoptions(formatter={"int": hex})

        for idx, expected in refs.items():
            raw = io_args[idx]
            actual = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
            actual = np.reshape(actual, expected.shape)

            if verbosity >= 1:
                print(f"Expected output [{idx}]:", expected)
                print(f"Actual output [{idx}]:", actual)

            errors += _check_dense(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                idx=idx,
                max_mismatch_percentage=max_mismatch_percentage,
                min_correlation=min_correlation,
            )

        if stochastic_refs:
            num_dense = len(refs)
            for i, sref in enumerate(stochastic_refs):
                raw = io_args[num_dense + i]
                actual = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
                actual = np.reshape(actual, sref["shape"])

                if verbosity >= 1:
                    print(f"Stochastic expected [{i}]: shape={sref['shape']}")
                    print(f"Stochastic actual [{i}]:", actual)

                errors += _check_stochastic(
                    actual,
                    sref,
                    rtol=rtol,
                    atol=atol,
                    idx=i,
                    max_mismatch_percentage=max_mismatch_percentage,
                )

        return errors


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_air_runtime = None


def get_air_runtime() -> AirRuntime:
    """Return the process-level AirRuntime singleton."""
    global _air_runtime
    if _air_runtime is None:
        if not _HAS_AIE_RUNTIME:
            raise AirBackendError(
                "aie.utils (mlir-aie) is not available. "
                "Install mlir-aie to use AirRuntime."
            )
        _air_runtime = AirRuntime()
    return _air_runtime


# ---------------------------------------------------------------------------
# Internal helpers for verification
# ---------------------------------------------------------------------------


def _check_dense(
    actual, expected, rtol, atol, idx, max_mismatch_percentage, min_correlation
):
    """Dense element-wise check. Returns number of errors (0 or 1)."""
    total_elements = expected.size

    if expected.dtype in [np.float16, np.float32, np.float64, bfloat16]:
        if expected.dtype == bfloat16:
            expected = expected.astype(np.float64)
            actual = actual.astype(np.float64)

        # Element-wise tolerance check
        close_mask = np.isclose(actual, expected, rtol=rtol, atol=atol)
        mismatch_indices = np.where(~close_mask)
        num_mismatches = len(mismatch_indices[0])
        max_acceptable = int(total_elements * max_mismatch_percentage / 100)

        elementwise_ok = num_mismatches <= max_acceptable
        if not elementwise_ok:
            print(f"ERROR: Output {idx} does not meet expected output.")
            print(f"Shape: {expected.shape}")
            if total_elements > 0:
                print(
                    f"Mismatches: {num_mismatches} / {total_elements} elements "
                    f"({100*num_mismatches/total_elements:.2f}%)"
                )
            if max_acceptable > 0:
                print(f"Max acceptable: {max_acceptable} ({max_mismatch_percentage}%)")
            _print_mismatches_dense(actual, expected, mismatch_indices, num_mismatches)

        # Correlation check
        corr_ok = True
        if min_correlation is not None and total_elements > 0:
            corr = float(np.corrcoef(actual.flatten(), expected.flatten())[0, 1])
            print(
                f"Output {idx} correlation: {corr:.6f} (threshold: {min_correlation})"
            )
            if not np.isfinite(corr) or corr < min_correlation:
                corr_ok = False
                print(
                    f"ERROR: Output {idx} correlation {corr:.6f} below threshold {min_correlation}"
                )

        return 0 if (elementwise_ok and corr_ok) else 1
    else:
        if not np.array_equal(actual, expected):
            print(f"ERROR: Output {idx} does not meet expected output.")
            mismatch_mask = actual != expected
            mismatch_indices = np.where(mismatch_mask)
            num_mismatches = len(mismatch_indices[0])
            print(f"Shape: {expected.shape}")
            if total_elements > 0:
                print(
                    f"Mismatches: {num_mismatches} / {total_elements} elements "
                    f"({100*num_mismatches/total_elements:.2f}%)"
                )
            _print_mismatches_dense(actual, expected, mismatch_indices, num_mismatches)
            return 1
        return 0


def _print_mismatches_dense(actual, expected, mismatch_indices, num_mismatches):
    max_display = 20
    print(f"First {min(max_display, num_mismatches)} mismatched locations:")
    for j in range(min(max_display, num_mismatches)):
        idx_t = tuple(dim[j] for dim in mismatch_indices)
        if np.issubdtype(expected.dtype, np.floating):
            print(
                f"  Index {idx_t}: expected={expected[idx_t]}, actual={actual[idx_t]}, "
                f"diff={abs(actual[idx_t] - expected[idx_t])}"
            )
        else:
            print(
                f"  Index {idx_t}: expected={expected[idx_t]}, actual={actual[idx_t]}"
            )
    if num_mismatches > max_display:
        print(f"  ... and {num_mismatches - max_display} more mismatches")


def _check_stochastic(actual, sref, rtol, atol, idx, max_mismatch_percentage):
    """Stochastic spot-check. Returns number of errors (0 or 1)."""
    if sref["values"][0].dtype in [np.float16, np.float32, np.float64, bfloat16]:
        values = sref["values"]
        if values[0].dtype == bfloat16:
            values = values.astype(np.float64)
            actual = actual.astype(np.float64)
        actual_stochastic = actual[tuple(sref["indices"])]
        close_mask = np.isclose(actual_stochastic, values, rtol=rtol, atol=atol)
        mismatch_positions = np.where(~close_mask)[0]
        num_mismatches = len(mismatch_positions)
        total_elements = len(values)
        max_acceptable = int(total_elements * max_mismatch_percentage / 100)
        if num_mismatches > max_acceptable:
            print(f"ERROR: Stochastic output {idx} does not meet expected output.")
            print(f"Shape: {sref['shape']}")
            print(f"Stochastic check: {total_elements} sampled elements")
            print(
                f"Mismatches: {num_mismatches} / {total_elements} elements "
                f"({100*num_mismatches/total_elements:.2f}%)"
            )
            if max_acceptable > 0:
                print(f"Max acceptable: {max_acceptable} ({max_mismatch_percentage}%)")
            max_display = 20
            print(f"First {min(max_display, num_mismatches)} mismatched locations:")
            for j in range(min(max_display, num_mismatches)):
                pos = mismatch_positions[j]
                idx_t = tuple(dim[pos] for dim in sref["indices"])
                exp_val = values[pos]
                act_val = actual_stochastic[pos]
                print(
                    f"  Index {idx_t}: expected={exp_val}, actual={act_val}, "
                    f"diff={abs(act_val - exp_val)}"
                )
            if num_mismatches > max_display:
                print(f"  ... and {num_mismatches - max_display} more mismatches")
            return 1
        return 0
    else:
        actual_stochastic = actual[tuple(sref["indices"])]
        if not np.array_equal(actual_stochastic, sref["values"]):
            print(f"ERROR: Stochastic output {idx} does not meet expected output.")
            mismatch_mask = actual_stochastic != sref["values"]
            mismatch_positions = np.where(mismatch_mask)[0]
            num_mismatches = len(mismatch_positions)
            total_elements = len(sref["values"])
            print(f"Shape: {sref['shape']}")
            print(f"Stochastic check: {total_elements} sampled elements")
            if total_elements > 0:
                print(
                    f"Mismatches: {num_mismatches} / {total_elements} elements "
                    f"({100*num_mismatches/total_elements:.2f}%)"
                )
            max_display = 20
            print(f"First {min(max_display, num_mismatches)} mismatched locations:")
            for j in range(min(max_display, num_mismatches)):
                pos = mismatch_positions[j]
                idx_t = tuple(dim[pos] for dim in sref["indices"])
                exp_val = sref["values"][pos]
                act_val = actual_stochastic[pos]
                print(f"  Index {idx_t}: expected={exp_val}, actual={act_val}")
            if num_mismatches > max_display:
                print(f"  ... and {num_mismatches - max_display} more mismatches")
            return 1
        return 0


# ---------------------------------------------------------------------------
# Backward compatibility shims
# ---------------------------------------------------------------------------


class XRTCompileArtifact:
    """
    Deprecated. Use NPUKernel from compile_air() instead.

    This shim wraps NPUKernel so existing code that unpacks
    .output_binary / .kernel / .insts still works.
    """

    def __init__(self, output_binary, kernel, insts):
        self.output_binary = output_binary
        self.kernel = kernel
        self.insts = insts


class XRTBackend:
    """
    Deprecated. Use compile_air() + get_air_runtime() instead.

    This shim delegates to compile_air() and get_air_runtime() so
    existing code continues to work without modification.
    """

    def __init__(
        self,
        verbose: bool = False,
        target_device: str = None,
        omit_while_true_loop: bool = False,
        omit_pingpong: str = "",
        lower_linalg_to_func=None,
        air_loop_fusion: bool = False,
        runtime_loop_tiling_sizes=None,
        omit_auto_broadcast: bool = False,
        channel_multiplexing=None,
        use_lock_race_condition_fix: bool = False,
        trace_offset: int = 0,
        trace_size: int = 0,
        output_format: str = "xclbin",
        kernel_name: str = "",
        instance_name: str = "",
        kernel_id: str = "",
        xclbin_input: str = "",
        num_device_cols: int = 0,
        debug_ir: bool = False,
        bf16_emulation: bool = False,
    ):
        self.verbose = verbose
        self.target_device = target_device
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
        self.num_device_cols = num_device_cols
        self.debug_ir = debug_ir
        self.bf16_emulation = bf16_emulation
        # Legacy attributes referenced by some callers
        self._npu_kernel = None
        self._handle = None
        self._runtime = None
        self.currently_loaded = False
        # These were set as side-effects of load()
        self.xclbin = None
        self.elf = None
        self.device = None
        self.context = None
        self.kernel = None
        self.bo_instr = None
        self.instr_v = None

    def __del__(self):
        self.unload()

    def compile(
        self,
        air_module: air.ir.Module,
        output_binary_name="air",
        kernel="MLIR_AIE",
        insts="air.insts.bin",
    ):
        """Compile an AIR module. Returns XRTCompileArtifact for compat."""
        npu_kernel = compile_air(
            air_module,
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
            output_binary_name=output_binary_name,
            insts=insts,
        )
        self._npu_kernel = npu_kernel
        # Build a compat artifact
        xclbin_path = getattr(npu_kernel, "xclbin_path", output_binary_name)
        kernel_name = getattr(npu_kernel, "kernel_name", kernel)
        insts_path = getattr(npu_kernel, "insts_path", insts)
        return XRTCompileArtifact(xclbin_path, kernel_name, insts_path)

    def load(self, artifact):
        """Load a compiled artifact. Returns an invoker callable."""
        if self.currently_loaded:
            raise AirBackendError(
                "Cannot load while an artifact is currently loaded. Call unload() first."
            )
        if self._npu_kernel is None:
            # Reconstruct NPUKernel from the artifact for the case where
            # compile() was called separately.
            if _HAS_AIE_RUNTIME:
                self._npu_kernel = NPUKernel(
                    artifact.output_binary,
                    artifact.insts,
                    kernel_name=artifact.kernel,
                )
        self._runtime = get_air_runtime()
        self._handle = self._runtime.load(self._npu_kernel)
        self.currently_loaded = True

        # Return a callable that mimics the old invoker interface.
        # Use _tensor() factory (selects XRTTensor or CPUOnlyTensor based on
        # pyxrt availability) rather than XRTTensor() directly.
        runtime = self._runtime
        handle = self._handle

        def invoker(*args):
            import filelock

            tensors = [_tensor(a) for a in args]
            with filelock.FileLock("/tmp/npu.lock"):
                runtime.run(handle, tensors)
            return tuple(t.numpy() for t in tensors)

        return invoker

    def compile_and_load(self, module):
        """Compile and load in one step."""
        c = self.compile(module)
        return self.load(c)

    def unload(self):
        """Unload any loaded module."""
        self._handle = None
        self._runtime = None
        self._npu_kernel = None
        self.currently_loaded = False
        # Clear legacy attributes
        self.kernel = None
        self.context = None
        self.xclbin = None
        self.elf = None
        self.device = None
        self.bo_instr = None
        self.instr_v = None
