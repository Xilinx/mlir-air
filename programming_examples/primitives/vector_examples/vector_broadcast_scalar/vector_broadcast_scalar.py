# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Broadcast one scalar per row across a whole row, on air.api.

    c[:] = a[:]

with ``a`` shaped ``[tile_m, 1]`` and ``c`` shaped ``[tile_m, n]``. That is the
entire kernel. The right-hand side is a plain copy and the shapes are what make
it a broadcast: numpy's rule, right-aligned, an operand axis either matches the
destination's or is 1 and gets stretched.

The predecessor said the same thing as a per-row ``scf.for`` containing two
``memref.subview``s, two ``memref.collapse_shape``s with an explicit dynamic
``StridedLayoutAttr``, a ``memref.load`` and a ``vector.broadcast``, plus a
``vector.transfer_write``. The emitted arithmetic is unchanged -- when the
broadcast axis is the *innermost* one there is no contiguous run to read, so the
emitter loads the single element and splats it, which is exactly the
``memref.load`` + ``vector.broadcast`` pair above.

The subviews and collapse_shapes were only ever there to get a rank-1 view the
vector ops would accept; indexing a rank-2 buffer at ``[j, 0]`` needs neither.

Two differences from the predecessor worth naming, both shared with the
``average_pool`` conversion this example is the mirror image of:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's, so each core gets a contiguous run of tiles where
  the predecessor's hand-built AffineMap interleaved them. Every tile is still
  written exactly once by exactly one core, and the rows are independent.

``--target`` is new and defaults to detecting the installed part, which is what
the predecessor did implicitly by having no device flag at all.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

# The rows checked against the reference are drawn at random, so the seed is
# what makes a failure reproducible: without it a mismatch reports row indices
# that the next run will not look at.
np.random.seed(42)

NUM_TILES = 2


def build_module(m, n, tile_m):
    assert m % (tile_m * NUM_TILES) == 0

    A = air.tensor([m], bf16)
    C = air.tensor([m, n], bf16)

    with air.launch(name="vector_broadcast_scalar") as launch:

        @launch.body
        def _():
            # The iteration space is every tile of rows; shape= pins the core
            # count to what the predecessor asked for, and the DSL strip-mines
            # the rest into a loop on each core.
            with air.herd(
                [range(0, m, tile_m)], name="herd_0", shape=(NUM_TILES,)
            ) as h:

                @h.body
                def _(tx):
                    # tx is a tile *index*, not a row offset: the herd's
                    # iteration space counts tiles, and h.tile_sizes carries the
                    # step. Multiply to get the window into L3.
                    i0 = tx * tile_m
                    # One value per row. The trailing 1 is the axis that gets
                    # stretched, and ops.load ignores it against the rank-1 L3
                    # region -- both describe the same tile_m contiguous
                    # elements.
                    a = air.alloc([tile_m, 1], bf16, scope=h.private())
                    # A whole row per value, so the vector width is the row
                    # width -- the predecessor's vector<Nxbf16>.
                    c = air.alloc([tile_m, n], bf16, scope=h.private(), vector=n)

                    air.ops.load(a, A[i0 : i0 + tile_m])

                    # The whole example. [tile_m, 1] against [tile_m, n]: the
                    # trailing axis is stretched, and because it is the
                    # innermost one the read is a scalar load and a splat.
                    c[:] = a[:]

                    air.ops.store(c, C[i0 : i0 + tile_m, :])

    return launch


if __name__ == "__main__":
    # Default values.
    M = 65536
    N = 16
    TILE_M = 256
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vector_broadcast_scalar example",
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
        "--m",
        type=int,
        default=M,
        help="Output size (dimension M)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Output size (dimension N, the broadcasted dimension)",
    )
    parser.add_argument("--tile-m", type=int, default=TILE_M, help="Tile size M")
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

    launch = build_module(args.m, args.n, args.tile_m)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, (args.m), dtype=INPUT_DATATYPE).reshape(args.m, 1)

    if args.compile_mode == "compile-and-run":

        # Stochastically sample num_sample results, and pass to XRTRunner backend
        # for verification.
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.m, num_samples)])

        # Compute reference results for sampled indices
        sampled_values = np.array(
            [np.broadcast_to(input_a[i], (args.n,)) for i in zip(*sampled_indices)],
            dtype=INPUT_DATATYPE,
        )

        # Store as a dictionary
        sampled_data = {
            "shape": (args.m, args.n),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="vector_broadcast_scalar",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
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
