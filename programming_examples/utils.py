# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared helpers for programming_examples.

All helpers are importable from any example directory via:
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils import ...
or, when the example is one level deep (e.g. relu/relu.py):
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils import ...
"""

import argparse
import numpy as np

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects import arith
from air.dialects.memref import subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects.air import MemorySpace
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend
from air.extras import types as T


# ---------------------------------------------------------------------------
# MLIR construct helpers (used inside @module_builder)
# ---------------------------------------------------------------------------


def make_l1_memref(shape, dtype):
    """MemRefType in L1 (per-core scratchpad) memory space."""
    return MemRefType.get(
        shape, dtype, memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1)
    )


def make_l2_memref(shape, dtype):
    """MemRefType in L2 (segment-shared) memory space."""
    return MemRefType.get(
        shape, dtype, memory_space=IntegerAttr.get(T.i32(), MemorySpace.L2)
    )


def make_vec_type(size, dtype):
    """1D VectorType of given length and element type."""
    return VectorType.get([size], dtype)


def identity_map_1d():
    """1D identity AffineMapAttr — the standard transfer_read/write permutation map."""
    return AffineMapAttr.get(AffineMap.get_identity(1))


def tiled_1d_offset(loop_var, tile_idx, tile_n):
    """
    Compute offset = loop_var + tile_idx * tile_n via affine_apply.

    Replaces the 12-line AffineMap.get / AffineExpr chain used in every
    1D vectorized example with a 1x2 herd.

    Args:
        loop_var:  outer loop induction variable (SSA Value or int)
        tile_idx:  herd tile index, e.g. _ty            (SSA Value or int)
        tile_n:    tile size in elements                 (Python int)
    """
    offset_map = AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineSymbolExpr.get(0),
                AffineExpr.get_mul(
                    AffineSymbolExpr.get(1),
                    AffineConstantExpr.get(tile_n),
                ),
            )
        ],
    )
    return affine_apply(offset_map, [loop_var, tile_idx])


def vec_read(buf, j, vec_size, c0, vec_ty, cst0, imap):
    """subview + transfer_read with the standard fixed call signature."""
    result = buf.result if hasattr(buf, "result") else buf
    sub = subview(result, [j], [vec_size], [1])
    return transfer_read(vec_ty, sub, [c0], imap, cst0, [True])


def vec_write(val, buf, j, vec_size, c0, imap):
    """subview + transfer_write with the standard fixed call signature."""
    result = buf.result if hasattr(buf, "result") else buf
    sub = subview(result, [j], [vec_size], [1])
    transfer_write(None, val, sub, [c0], imap, [True])


# ---------------------------------------------------------------------------
# Argument-parser factory
# ---------------------------------------------------------------------------


def make_air_parser(description, prog="run.py"):
    """
    Return an ArgumentParser pre-populated with the 4 universal flags:
        -v / --verbose
        -p / --print-module-only
        --compile-mode  {compile-only, compile-and-run}
        --output-format {xclbin, elf}

    The caller adds example-specific arguments (--n, --tile-n, etc.) after.
    """
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-p", "--print-module-only", action="store_true")
    p.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    p.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    return p


# ---------------------------------------------------------------------------
# Runner/backend factory helpers
# ---------------------------------------------------------------------------


def make_xrt_runner(args, instance_name, **kwargs):
    """XRTRunner with the standard fixed defaults (omit_while_true_loop=False, tiling=[4,4])."""
    return XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name=instance_name,
        runtime_loop_tiling_sizes=[4, 4],
        **kwargs,
    )


def make_xrt_backend(args, **kwargs):
    """XRTBackend with the standard fixed defaults."""
    return XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        runtime_loop_tiling_sizes=[4, 4],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Stochastic sampling helper
# ---------------------------------------------------------------------------


def stochastic_check(inputs, n, ref_fn, dtype, num_samples=100):
    """
    Build the stochastic_expected_outputs dict for 1D element-wise ops.

    Args:
        inputs:      list of numpy input arrays (same as passed to run_test)
        n:           total element count
        ref_fn:      scalar reference function, called as ref_fn(*scalars)
        dtype:       output numpy dtype
        num_samples: number of randomly sampled indices
    Returns:
        dict with "shape", "indices", "values" for XRTRunner.run_test()
    """
    sampled_indices = np.vstack([np.random.randint(0, n, num_samples)])
    sampled_values = np.array(
        [ref_fn(*[inp[i] for inp in inputs]) for i in zip(*sampled_indices)],
        dtype=dtype,
    )
    return {"shape": (n,), "indices": sampled_indices, "values": sampled_values}


# ---------------------------------------------------------------------------
# Print-module-only convenience
# ---------------------------------------------------------------------------


def check_print_module(mlir_module, args):
    """Print the MLIR module and exit if --print-module-only was passed."""
    if args.print_module_only:
        print(mlir_module)
        exit(0)
