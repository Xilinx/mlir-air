# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-tile softmax via a hand-written AIE kernel, on air.api.

    softmax = air.extern("softmax_bf16", link_with="softmax.o", scalars=[i32])
    ...
    air.ops.load(a, A[i0 : i0 + tile_n])
    softmax(a, tile_n - 1, out)
    air.ops.store(out, C[i0 : i0 + tile_n])

Each core runs the reduction over its own [tile_n] tile, so the softmax is
per-tile rather than over the whole vector -- which is what the reference in
``__main__`` computes too, tile by tile.

The DSL computes nothing here; the kernel body is one call into ``softmax.o``.
``air.extern`` emits the private ``func.func`` declaration, stamps ``link_with``
on both it and the herd, and builds the ``func.call``.

Note where the scalar sits. ``softmax_bf16`` takes ``(memref, i32, memref)`` --
the length argument is *between* the two buffers, not appended after them.
``air.extern`` walks the call's arguments in order and takes the next entry from
``scalars=`` each time it meets a non-buffer, so the declaration comes out with
the operands in the order the C symbol expects; ``scalars=[i32]`` only says what
type that one non-buffer argument has, not where it goes.

The scalar is the *only* part of the shape the kernel reads at runtime. The
element type and the tile extent are compiled into ``softmax.o``:
``softmax_bf16`` opens with ``zero_vectorized<bfloat16, 1, 256, 16>(out)``,
which clears 256 elements whatever the memref says. Both are therefore checked
below rather than accepted and silently miscomputed.

Unchanged from the raw-bindings version this replaces except that the herd is
[herd_n, 1] rather than [1, herd_n] -- a 1-D air.api herd is laid out along x,
the orientation that places on both generations -- and the tile grid is
strip-mined onto those cores by the DSL rather than by a hand-written outer loop
with an AffineMap for the offset.
"""

import argparse
from math import exp

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)


KERNEL_TILE_N = 256  # the 256 in zero_vectorized<bfloat16, 1, 256, 16>


def build_module(n, tile_n, herd_n, np_dtype_in):
    assert n % (tile_n * herd_n) == 0
    # Checked, not assumed. softmax.o exports one symbol, and it bakes in both
    # the element type and the extent -- `softmax_bf16` opens by calling
    # zero_vectorized<bfloat16, 1, 256, 16>(out), which clears 256 elements
    # whatever the memref says. A smaller tile is written past its end; a larger
    # one is left partly dirty. Both link and run.
    if np_dtype_in is not bfloat16:
        raise ValueError(
            f"softmax.o exports only a bf16 kernel (softmax_bf16), so "
            f"np_dtype_in must be ml_dtypes.bfloat16, got {np_dtype_in!r}"
        )
    if tile_n != KERNEL_TILE_N:
        raise ValueError(
            f"tile_n must be {KERNEL_TILE_N}: the extent is a template "
            f"argument of the zero_vectorized call inside softmax_bf16, "
            f"compiled into softmax.o, not taken from the memref. "
            f"got tile_n={tile_n}"
        )
    dt = bf16

    A = air.tensor([n], dt)
    C = air.tensor([n], dt)

    softmax = air.extern("softmax_bf16", link_with="softmax.o", scalars=[i32])

    with air.launch(name="softmax") as launch:

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
                    softmax(a, tile_n - 1, out)
                    air.ops.store(out, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    N = 1024
    TILE_N = 256
    HERD_N = 4
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the softmax example",
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
        help=f"Tile size. Pinned to {KERNEL_TILE_N} by softmax.o, which "
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
        INPUT_DATATYPE,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Softmax
    num_tiles = args.n // args.tile_n
    inputs = np.random.randn(num_tiles, args.tile_n).astype(INPUT_DATATYPE)
    outputs = np.zeros(shape=(num_tiles, args.tile_n), dtype=INPUT_DATATYPE)

    max_val = np.max(inputs)
    for j in range(num_tiles):
        sum_val = 0.0
        for i in range(args.tile_n):
            outputs[j][i] = exp(inputs[j][i] - max_val)
            sum_val += outputs[j][i]
        for i in range(args.tile_n):
            outputs[j][i] = outputs[j][i] / sum_val

    if args.compile_mode == "compile-and-run":

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="softmax",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[inputs],
                expected_outputs=[outputs],
                rtol=1e-1,
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
