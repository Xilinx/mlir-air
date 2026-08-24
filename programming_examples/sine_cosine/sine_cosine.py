# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Elementwise sine or cosine via a hand-written AIE kernel, on air.api.

    kernel = air.extern(
        "sinf_bf16_24_8" if mode == "sin" else "cosf_bf16_24_8",
        object="sine_cosine.o",
    )
    ...
    air.ops.load(a, A[i0 : i0 + tile_n])
    kernel(a, out)
    air.ops.store(out, C[i0 : i0 + tile_n])

Nothing is computed in the DSL here -- the whole kernel body is one call into
``sine_cosine.o``. That is the point: ``air.extern`` emits the private
``func.func`` declaration, stamps ``link_with`` on both the declaration and the
herd, and builds the ``func.call``, which the raw-bindings version this replaces
spelled out as a ``FuncOp`` per symbol plus two attribute assignments plus a
``CallOp``.

Both symbols live in one object file, and only one of them is called per build
-- ``--mode`` picks at trace time, so the other is never declared. A herd links
against a single object, which is the constraint ``air.extern`` enforces; two
kernels from the *same* object are fine.

Note how much of this example's interface the object file pins.
``sinf_bf16_24_8`` instantiates ``sinf_cosf_poly_bf16<24, 8, true>``: both the
element type and the 24-element extent are compiled into the object rather than
read from the memref it is handed, so a kernel called with any other tile size
would read and write 24 elements regardless. ``--tile-n`` and the dtype are
therefore checked below rather than accepted and silently miscomputed.

Unchanged from the predecessor except that the herd is [herd_n, 1] rather than
[1, herd_n] -- a 1-D air.api herd is laid out along x, the orientation that
places on both generations -- and the tile grid is strip-mined onto those cores
by the DSL rather than by a hand-written outer loop with an AffineMap for the
offset.
"""

import argparse
from math import cos, sin

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

# Load-bearing: the inputs below are drawn from np.random, and the tolerance is
# loose enough (rtol=1e0 against a bf16 sine) that an unseeded draw fails
# intermittently rather than never.
np.random.seed(42)


KERNEL_TILE_N = 24  # the 24 in sinf_cosf_poly_bf16<24, 8, ...>


def build_module(n, tile_n, herd_n, mode, np_dtype_in):
    assert n % (tile_n * herd_n) == 0
    # Checked, not assumed. The object file exports exactly two symbols, and
    # each one bakes in the element type and the extent -- neither is read from
    # the memref at the call. Getting either wrong links and runs, and computes
    # the wrong answer, so refuse here where the cause can still be named.
    if mode not in ("sin", "cos"):
        raise ValueError(f"mode must be 'sin' or 'cos', got {mode!r}")
    if np_dtype_in is not bfloat16:
        raise ValueError(
            f"sine_cosine.o exports only bf16 kernels (sinf_bf16_24_8 / "
            f"cosf_bf16_24_8), so np_dtype_in must be ml_dtypes.bfloat16, "
            f"got {np_dtype_in!r}"
        )
    if tile_n != KERNEL_TILE_N:
        raise ValueError(
            f"tile_n must be {KERNEL_TILE_N}: the extent is a template "
            f"argument of sinf_cosf_poly_bf16<{KERNEL_TILE_N}, 8, ...>, "
            f"compiled into sine_cosine.o, not taken from the memref. "
            f"got tile_n={tile_n}"
        )
    dt = bf16

    A = air.tensor([n], dt)
    C = air.tensor([n], dt)

    # Declared per build, not both: --mode selects one, and the unused symbol is
    # never emitted. Both come from the same object file, which is what lets a
    # single herd link against them.
    kernel = air.extern(
        "sinf_bf16_24_8" if mode == "sin" else "cosf_bf16_24_8",
        object="sine_cosine.o",
    )

    with air.launch(name="sine_cosine") as launch:

        @launch.body
        def _():
            with air.herd([range(0, n, tile_n)], name="herd_0", shape=(herd_n,)) as h:

                @h.body
                def _(tx):
                    # tx counts tiles, not elements.
                    i0 = tx * tile_n
                    a = air.alloc([tile_n], dt, scope=h.private())
                    out = air.alloc([tile_n], dt, scope=h.private())

                    air.ops.load(a, A[i0 : i0 + tile_n])
                    kernel(a, out)
                    air.ops.store(out, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    N = 48
    TILE_N = 24
    HERD_N = 2
    SIN_OR_COS = "sin"
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the sine_cosine example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Total number of elements",
    )
    parser.add_argument(
        "--tile-n",
        type=int,
        default=TILE_N,
        help=f"Tile size. Pinned to {KERNEL_TILE_N} by sine_cosine.o, which "
        f"compiles the extent in as a template argument; other values are "
        f"rejected rather than silently miscomputed",
    )
    parser.add_argument(
        "--herd-n",
        type=int,
        default=HERD_N,
        help="Number of L1 tiles along the N dimension",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=SIN_OR_COS,
        choices=["sin", "cos"],
        help="Sine or cosine mode (must be one of [sin, cos])",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )

    args = parser.parse_args()

    launch = build_module(
        args.n,
        args.tile_n,
        args.herd_n,
        args.mode,
        INPUT_DATATYPE,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    inputs = np.random.randn(
        args.n,
    ).astype(INPUT_DATATYPE)
    outputs = np.zeros(shape=(args.n), dtype=INPUT_DATATYPE)
    for n1 in range(args.n):
        if args.mode == "sin":
            outputs[n1] = INPUT_DATATYPE(sin(inputs[n1]))
        elif args.mode == "cos":
            outputs[n1] = INPUT_DATATYPE(cos(inputs[n1]))
        else:
            raise AssertionError

    if args.compile_mode == "compile-and-run":

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="sine_cosine",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[inputs],
                expected_outputs=[outputs],
                rtol=1e0,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
