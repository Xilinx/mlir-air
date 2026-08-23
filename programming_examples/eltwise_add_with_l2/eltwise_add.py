# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Element-wise add staged through L2, on air.api.

The sibling ``eltwise_add`` streams tiles from L3 straight into the cores. This
one lands the whole vector in a memtile first, and the cores read their tiles
from there:

    l2_a = air.alloc([n], f32, scope=seg.private())   a memtile buffer
      air.ops.load(l2_a, A)                           L3 -> L2, whole vector
      ...
      air.ops.load(a, l2_a[i0 : i0 + tile_n])         L2 -> L1, one tile
      c[:] = a[:] + b[:]
      air.ops.store(c, l2_c[i0 : i0 + tile_n])        L1 -> L2
    air.ops.store(l2_c, C)                            L2 -> L3, whole vector

``seg.private()`` is what makes the staging buffers memtile-resident, and the
herd nested in the segment body carries them in as operands automatically --
``air.herd`` is ``IsolatedFromAbove``, so the raw-bindings version this replaces
had to list them in ``operands=[...]`` by hand and pick them back up as body
parameters.

Two differences from the predecessor worth naming:

* The compute is written ``c[:] = a[:] + b[:]`` rather than a scalar
  ``load``/``addf``/``store`` loop over every element. The tile is 1024 f32, so
  that vectorises to ``<16 x f32>`` (512-bit), which is the width f32 has to use
  -- 8 lanes does not legalize on either generation. The predecessor moved one
  element per iteration.
* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES], and the tile grid is
  strip-mined onto those cores by the DSL rather than by a hand-written outer
  loop with an AffineMap for the offset. A 1-D air.api herd is laid out along x,
  the orientation that places on both generations.
"""

import argparse

import numpy as np

from air import api as air
from air.api import f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

NUM_TILES = 2


def build_module(n, tile_n, np_dtype_in):
    assert n % (tile_n * NUM_TILES) == 0
    dt = f32

    A = air.tensor([n], dt)
    B = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="eltwise_add") as launch:

        @launch.body
        def _():
            with air.segment(name="segment_0") as seg:

                @seg.body
                def _():
                    # L2: the whole vector, staged in a memtile. Passed into the
                    # herd for us.
                    l2_a = air.alloc([n], dt, scope=seg.private())
                    l2_b = air.alloc([n], dt, scope=seg.private())
                    l2_c = air.alloc([n], dt, scope=seg.private())

                    air.ops.load(l2_a, A)
                    air.ops.load(l2_b, B)

                    with air.herd(
                        [range(0, n, tile_n)], name="herd_0", shape=(NUM_TILES,)
                    ) as h:

                        @h.body
                        def _(tx):
                            # tx counts tiles, not elements; multiply by the
                            # tile size to get the window into L2.
                            i0 = tx * tile_n
                            a = air.alloc([tile_n], dt, scope=h.private())
                            b = air.alloc([tile_n], dt, scope=h.private())
                            c = air.alloc([tile_n], dt, scope=h.private())

                            air.ops.load(a, l2_a[i0 : i0 + tile_n])
                            air.ops.load(b, l2_b[i0 : i0 + tile_n])

                            c[:] = a[:] + b[:]

                            air.ops.store(c, l2_c[i0 : i0 + tile_n])

                    air.ops.store(l2_c, C)

    return launch


if __name__ == "__main__":
    # Default values.
    N = 16384
    TILE_N = 1024
    INPUT_DATATYPE = np.float32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the eltwise_add_with_l2 example",
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
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
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
        INPUT_DATATYPE,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, args.n, dtype=np.int64).reshape(args.n)
    input_a = input_a.astype(INPUT_DATATYPE)
    input_b = np.arange(0, args.n, dtype=np.int64).reshape(args.n)
    input_b = input_b.astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":

        # Stochastically sample num_sample results, and pass to XRTRunner backend for verification.
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, args.n, num_samples),  # i indices
            ]
        )

        # Compute reference results for sampled indices
        sampled_values = np.array(
            [input_a[i] + input_b[i] for i in zip(*sampled_indices)],
            dtype=INPUT_DATATYPE,
        )

        # Store as a dictionary
        sampled_data = {
            "shape": (args.n),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="eltwise_add",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-3,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_auto_broadcast=True,
            output_format=args.output_format,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
