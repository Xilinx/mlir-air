# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""RMS normalisation of an [M, N] tile, on air.api.

Row by row::

    acc  = sum(x * x)            over the row
    rstd = rsqrt(acc / N + eps)
    y    = x * rstd

Three lines, and each is one line of DSL:

    acc[:]  = ops.reduce_add(row[:] * row[:])
    rstd[:] = ops.cast(ops.rsqrt(ops.cast(acc[:], f32) * (1 / N) + EPS), dtype)
    out[:]  = row[:] * rstd[:]

The last one is the interesting one. ``rstd`` is a ``[1, 1]`` buffer and ``row``
is ``[1, N]``, so multiplying them is numpy broadcasting along the innermost
axis: the DSL pins the broadcast axis at 0 and emits ``memref.load`` +
``vector.broadcast`` inside the vector loop. The predecessor had to reduce to a
scalar, build a ``vector.broadcast`` by hand and thread it through the second
loop nest; here the shapes say it.

``ops.reduce_add`` reads the whole innermost axis as one vector, which removes
the predecessor's accumulator buffer entirely -- and with it the comment about
writing the squared vector to a temporary and reading it back to break the
``mulf``->``addf`` def-use chain that the aievec lowering rejected. There is no
such chain left: the multiply feeds a ``vector.reduction``, not an add.

Two things carried over deliberately:

* **The reciprocal square root is computed in f32.** ``math.rsqrt`` on a scalar
  ``bf16`` is not legalised by Peano, and AIE has no ``sqrt`` at all -- which is
  why this is ``rsqrt`` and a multiply rather than ``sqrt`` and a divide. The
  two ``ops.cast`` calls are that constraint, not a rounding preference. The
  mean and the epsilon are now added in f32 as well, which is strictly more
  accurate than the predecessor's bf16 arithmetic and well inside the tolerance
  the test already used.
* **The row loop is ``air.sequential``**, so it stays one ``scf.for`` over M
  rather than unrolling M copies of the body at trace time.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

EPS = 1e-5

M_DEFAULT = 32
N_DEFAULT = 64
VECTOR_SIZE = 16
INPUT_DATATYPE = bfloat16


def build_module(M, N, dtype=bf16, vector=VECTOR_SIZE):
    if vector and N % vector:
        raise ValueError(
            f"N ({N}) must be a multiple of the vector width ({vector}): this "
            "kernel has no partial vectors, so the last one would run past the "
            "end of the row."
        )

    x = air.tensor([M, N], dtype)
    y = air.tensor([M, N], dtype)

    with air.launch(name="rms_norm") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="herd_0", shape=(1,)) as h:

                @h.body
                def _(tx):
                    # One row at a time. The leading 1 is what makes `rstd`
                    # a [1, 1] that broadcasts along the row.
                    row = air.alloc([1, N], dtype, scope=h.private(), vector=vector)
                    out = air.alloc([1, N], dtype, scope=h.private(), vector=vector)
                    acc = air.alloc([1, 1], dtype, scope=h.private(), vector=vector)
                    rstd = air.alloc([1, 1], dtype, scope=h.private(), vector=vector)

                    for r in air.sequential(M):
                        ops.load(row, x[r : r + 1, :])

                        acc[:] = ops.reduce_add(row[:] * row[:])
                        rstd[:] = ops.cast(
                            ops.rsqrt(ops.cast(acc[:], f32) * (1.0 / N) + EPS), dtype
                        )

                        out[:] = row[:] * rstd[:]
                        ops.store(out, y[r : r + 1, :])

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="rms_norm.py",
        description="Builds, runs, and tests the RMS normalization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_DEFAULT, help="M dimension (rows)")
    parser.add_argument("--N", type=int, default=N_DEFAULT, help="N dimension (cols)")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        dest="vector",
        help="compute vector width in lanes; 0 forces a scalar loop",
    )
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
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    launch = build_module(args.M, args.N, bf16, args.vector)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        return 0

    np.random.seed(0)
    x_input = np.random.rand(args.M, args.N).astype(INPUT_DATATYPE)

    # Reference: RMS normalization without weight/bias.
    rms = np.sqrt(
        np.mean(x_input.astype(np.float32) ** 2, axis=-1, keepdims=True) + EPS
    )
    y_expected = (x_input / rms).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        backend.compile(mlir_module)
        backend.unload()
        return 0

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="rms_norm",
        runtime_loop_tiling_sizes=[4, 4],
        target_device=launch.target,
    )
    return runner.run_test(
        mlir_module,
        inputs=[x_input],
        expected_outputs=[y_expected],
        rtol=5e-2,
        atol=5e-1,
    )


if __name__ == "__main__":
    exit(main())
