# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Row-wise maximum reduction, on air.api.

    out[m] = np.max(a[m, :])

The predecessor hand-rolled the row loop: an ``scf.for`` over the tile, two
``memref.subview``s and two ``memref.collapse_shape``s per trip to turn a
``[1, n]`` slice into a rank-1 ``memref<n>``, a ``vector.transfer_read`` with an
explicit identity map and padding constant, ``vector.reduction``, and a scalar
store. Here that is ``air.ops.reduce_max``: the emitter walks the rows, reads each
one as a vector and reduces it, and the subviews and collapse_shapes are not
needed at all -- air.api reads at an offset directly.

A reduction is the DSL's only shape-changing operation, so it has to be the
whole right-hand side. Its *operand* can still be an expression, which is how
``ops.reduce_add(a[:] * b[:])`` would give a row-wise dot product; this example
reduces a bare tile.

The destination may keep the reduced axis (``[tile_m, 1]``, numpy's
keepdims=True) or drop it (``[tile_m]``). This example drops it, so the L1 tile
has the same rank as the ``[m]`` L3 output and the store is a plain transfer.
The predecessor kept it in L1 and dropped it in the DMA instead, which is the
same buffer either way.

The whole row is read as one vector of extent n, exactly as the predecessor
did. That is deliberate rather than incidental: stepping the row in
vector-width chunks would need a loop-carried vector accumulator, which is the
construct ``ops.dot`` documents as failing to legalize on AIE2. The cost is
that n has to be a vector length the backend accepts -- 32 here, and 16 and
32 are both known to work on npu1.

Two differences from the predecessor worth naming:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's. The predecessor asked for a [1, NUM_TILES] herd
  and then wrote the outer loop itself through a hand-built AffineMap.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import dtype_of
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

NUM_TILES = 2


def build_module(m, n, tile_m, np_dtype_in):
    assert m % (tile_m * NUM_TILES) == 0
    dt = dtype_of(np_dtype_in)
    if dt is None:
        raise ValueError(
            f"unsupported element type {np_dtype_in!r}; air.api knows "
            f"float32, float16, bfloat16, int8/16/32 and uint8/16/32"
        )

    A = air.tensor([m, n], dt)
    OUT = air.tensor([m], dt)

    with air.launch(name="vector_reduce_max") as launch:

        @launch.body
        def _():
            with air.herd(
                [range(0, m, tile_m)], name="herd_0", shape=(NUM_TILES,)
            ) as h:

                @h.body
                def _(tx):
                    # tx is a tile *index*: the herd's iteration space counts
                    # tiles and h.tile_sizes carries the step.
                    i0 = tx * tile_m
                    a = air.alloc([tile_m, n], dt, scope=h.private())
                    out = air.alloc([tile_m], dt, scope=h.private())

                    air.ops.load(a, A[i0 : i0 + tile_m, :])
                    out[:] = air.ops.reduce_max(a[:])
                    air.ops.store(out, OUT[i0 : i0 + tile_m])

    return launch


if __name__ == "__main__":
    M = 65536
    N = 32
    TILE_M = 256
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vector_reduce_max example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=M, help="Input size (dimension M)")
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Input size (dimension N), the axis reduced over. Read as a single "
        "vector, so it must be a vector length the backend accepts.",
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

    launch = build_module(args.m, args.n, args.tile_m, INPUT_DATATYPE)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, (args.m * args.n), dtype=INPUT_DATATYPE).reshape(
        args.m, args.n
    )

    if args.compile_mode == "compile-and-run":
        # Stochastically sample num_sample results, and pass to XRTRunner
        # backend for verification.
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.m, num_samples)])

        sampled_values = np.array(
            [np.max(input_a[i]) for i in zip(*sampled_indices)],
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
            instance_name="vector_reduce_max",
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
