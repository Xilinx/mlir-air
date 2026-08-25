# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Average pool over the rows of an [M, N] array, on air.api.

    c[:] = air.ops.reduce_add(a[:] * (1.0 / n))

One line of compute. ``ops.reduce_add`` collapses the innermost dimension, so an
``[tile_m, n]`` tile reduces to an ``[tile_m, 1]`` buffer -- a column of row
averages -- which is what the predecessor built by hand out of a per-row
``scf.for`` containing two ``memref.subview``s, two ``memref.collapse_shape``s
with an explicit dynamic ``StridedLayoutAttr``, a ``vector.transfer_read`` with
an identity permutation map and a padding constant, a ``vector.broadcast``, an
``arith.mulf`` and a ``vector.reduction``.

The emitted arithmetic is unchanged: broadcast 1/n to a vector, multiply, then
``vector.reduction <add>``.

**The scale happens before the reduction, and that is load-bearing.** Writing it
the other way round -- reduce, then scale the result -- is one scalar bf16
multiply per row instead of one vector multiply per row, and the predecessor
carries a comment saying a scalar bf16 multiply "can produce corrupted output on
AIE2". Putting the multiply inside the reduce keeps it on the vector, matching
both the predecessor's IR and its reference, which also scales each element
before summing. In bf16 those two orders do not agree to the last bit even in
exact arithmetic, so this is not only about the AIE2 hazard.

The ``[tile_m, 1]`` shape of the output tile is what a reduction produces, while
the L3 array it belongs in is ``[m]``. ``ops.store`` accepts the pair: it
already ignored a *leading* unit axis, which is how a per-core L2 staging tile
is spelled, and now ignores a trailing one for the symmetric reason. Both
describe the same tile_m contiguous elements.

Two differences from the predecessor worth naming:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's, so each core gets a contiguous run of tiles where
  the predecessor's hand-built AffineMap interleaved them. Every tile is still
  computed exactly once by exactly one core, and the rows are independent.

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
# that the next run will not look at. Set at import, before any draw, which is
# where the predecessor and the rest of this directory set it -- so the sampled
# indices are not merely stable but the same ones the predecessor checked.
np.random.seed(42)

NUM_TILES = 2


def build_module(m, n, tile_m):
    assert n > 0, "Pool width N must be positive"
    assert m % (tile_m * NUM_TILES) == 0

    A = air.tensor([m, n], bf16)
    C = air.tensor([m], bf16)

    with air.launch(name="average_pool") as launch:

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
                    # The reduction runs across a whole row, so the vector width
                    # is the pool width -- the predecessor's vector<Nxbf16>.
                    a = air.alloc([tile_m, n], bf16, scope=h.private(), vector=n)
                    # One value per row, and scalar: the reduction writes a
                    # single element per row, as the predecessor's store does.
                    c = air.alloc([tile_m, 1], bf16, scope=h.private(), vector=0)

                    air.ops.load(a, A[i0 : i0 + tile_m, :])

                    # Scale inside the reduce, not after it -- see the module
                    # docstring. This keeps the multiply on the vector.
                    c[:] = air.ops.reduce_add(a[:] * (1.0 / n))

                    air.ops.store(c, C[i0 : i0 + tile_m])

    return launch


if __name__ == "__main__":
    # Default values.
    M = 65536
    N = 16
    TILE_M = 256
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the AveragePool example",
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
        help="Input size (dimension M)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Input size (dimension N, pool width)",
    )
    parser.add_argument("--tile-m", type=int, default=TILE_M, help="Tile size M")
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

    launch = build_module(args.m, args.n, args.tile_m)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, (args.m * args.n), dtype=INPUT_DATATYPE).reshape(
        args.m, args.n
    )

    if args.compile_mode == "compile-and-run":

        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.m, num_samples)])

        # AveragePool reference: sum of (each element * 1/N) per row. The scale
        # is applied per element rather than to the sum, matching the kernel --
        # see the module docstring.
        inv_n_bf16 = INPUT_DATATYPE(1.0 / args.n)
        sampled_values = np.array(
            [np.sum(input_a[i] * inv_n_bf16) for i in zip(*sampled_indices)],
            dtype=INPUT_DATATYPE,
        )

        sampled_data = {
            "shape": (args.m,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="average_pool",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
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
