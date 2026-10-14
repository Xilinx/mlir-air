# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Vectorized erf primitive, on air.api.

    c[:] = air.ops.erf(a[:])

bf16 or f32, over a [n] vector cut into [tile_n] tiles across NUM_TILES cores.
``--vector-size 0`` takes the emitter's scalar path instead. npu2 only.
"""

import argparse
import math

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2
DTYPES = {"bf16": (bf16, bfloat16), "f32": (f32, np.float32)}


def build_module(n, tile_n, dtype=bf16, vector_size=16):
    assert n % (tile_n * NUM_TILES) == 0
    assert vector_size == 0 or tile_n % vector_size == 0

    A = air.tensor([n], dtype)
    C = air.tensor([n], dtype)

    with air.launch(name="vector_erf") as launch:

        @launch.body
        def _():
            with air.herd(
                [range(0, n, tile_n)], name="herd_0", shape=(NUM_TILES,)
            ) as h:

                @h.body
                def _(tx):
                    i0 = tx * tile_n
                    a = air.alloc(
                        [tile_n], dtype, scope=h.private(), vector=vector_size
                    )
                    c = air.alloc(
                        [tile_n], dtype, scope=h.private(), vector=vector_size
                    )

                    air.ops.load(a, A[i0 : i0 + tile_n])

                    c[:] = air.ops.erf(a[:])

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vectorized erf example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations; 0 runs the scalar path",
    )
    parser.add_argument(
        "--dtype", type=str, choices=sorted(DTYPES), default="bf16", help="Data type"
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

    args = parser.parse_args()
    dtype, np_dtype = DTYPES[args.dtype]

    launch = build_module(args.n, args.tile_n, dtype, args.vector_size)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        rng = np.random.default_rng(42)
        input_a = rng.uniform(-3.0, 3.0, args.n).astype(np_dtype)
        expected = np.vectorize(math.erf)(input_a.astype(np.float64)).astype(np_dtype)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="vector_erf",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        # bf16-level tolerances for both dtypes.
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                expected_outputs=[expected],
                rtol=1.6e-2,
                atol=1e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
