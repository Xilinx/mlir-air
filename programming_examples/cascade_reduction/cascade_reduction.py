# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A cascade reduction across a row of cores, on air.api.

Four cores in a row, chained by a cascade channel. Each adds one, so the
result is ``input + 4``:

    launch                      put   -> chan_in
    tile 0    chan_in     -> get, +1, put -> chan_cascade[0]
    tile 1    cascade[0]  -> get, +1, put -> chan_cascade[1]
    tile 2    cascade[1]  -> get, +1, put -> chan_cascade[2]
    tile 3    cascade[2]  -> get, +1, put -> chan_out
    launch                      get   <- chan_out

The point of the example is that the four cores are *not* interchangeable: the
first reads from L3, the last writes to L3, and the middle two only forward. A
herd body is traced once for the whole herd, so telling them apart is
``ops.branch`` -- an ``scf.if`` on the tile coordinate -- and not a Python ``if``,
which would have to pick one branch for every core. ``tx == 0`` builds the
condition; the comparison is emitted as ``arith.cmpi`` where the region opens.

``chan_cascade`` is ``channel_type="npu_cascade"``: a direct core-to-core
connection between neighbouring tiles rather than a DMA stream. The herd is
therefore a row -- ``NUM_TILES`` columns by one -- so the cascade flows
west-to-east between adjacent columns.

Two differences from the predecessor:

* ``local[:] = recv[:] + local[:]`` replaces a ``linalg.add`` with the output
  aliasing an input. Same accumulate, and the DSL vectorises it rather than
  handing the pipeline a named op to fuse.
* The neighbour index is ``tx - 1`` rather than a hand-built ``arith.subi``, so
  it reaches the channel as one ``affine.apply`` like every other index
  expression in the DSL.
"""

import argparse

import numpy as np

from air import api as air
from air.api.types import i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

NUM_TILES = 4
DATA_SIZE = 2048
SHAPE = [1, 1, DATA_SIZE]


def build_module():
    src = air.tensor(SHAPE, i32)
    dst = air.tensor(SHAPE, i32)

    # chan_in / chan_out are ordinary DMA links to L3; chan_cascade is the
    # core-to-core connection between adjacent columns.
    chan_in = air.channel("chan_in", size=[1])
    chan_cascade = air.channel(
        "chan_cascade", size=[NUM_TILES], channel_type="npu_cascade"
    )
    chan_out = air.channel("chan_out", size=[1])

    with air.launch(name="cascade_reduce") as launch:

        @launch.body
        def _():
            chan_in.put(src)

            with air.segment(name="segment_0") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(NUM_TILES)], name="herd_0", shape=(NUM_TILES,)
                    ) as herd:

                        @herd.body
                        def _(tx):
                            # Every core contributes the same 1; the reduction
                            # is the chain, not the operand.
                            local = air.alloc(SHAPE, i32, scope=herd.private())
                            air.ops.fill(local, 1)
                            recv = air.alloc(SHAPE, i32, scope=herd.private())

                            with air.ops.branch(tx == 0) as head:
                                chan_in.get(recv)
                                local[:] = recv[:] + local[:]
                                chan_cascade.put(local, indices=[tx])

                            with head.otherwise():
                                chan_cascade.get(recv, indices=[tx - 1])
                                local[:] = recv[:] + local[:]

                                with air.ops.branch(tx == NUM_TILES - 1) as tail:
                                    chan_out.put(local)
                                with tail.otherwise():
                                    chan_cascade.put(local, indices=[tx])

            chan_out.get(dst)

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="cascade_reduction.py",
        description="Builds, runs, and tests the cascade reduction example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
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

    launch = build_module()
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        return 0

    input_a = np.arange(0, DATA_SIZE, dtype=np.int32).reshape(*SHAPE)

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
        instance_name="cascade_reduce",
        runtime_loop_tiling_sizes=[4, 4],
        target_device=launch.target,
    )

    # The output is 2048 elements of input + NUM_TILES; a sample of 100 of them
    # is what the predecessor checked, and it is enough to catch a broken link
    # in the chain (a missing +1 shifts every element).
    num_samples = 100
    sampled_indices = np.vstack(
        [
            np.zeros(num_samples, dtype=int),
            np.zeros(num_samples, dtype=int),
            np.random.randint(0, DATA_SIZE, num_samples),
        ]
    )
    sampled_values = np.array(
        [input_a[i, j, k] + NUM_TILES for i, j, k in zip(*sampled_indices)],
        dtype=np.int32,
    )

    return runner.run_test(
        mlir_module,
        inputs=[input_a],
        stochastic_expected_outputs=[
            {
                "shape": tuple(SHAPE),
                "indices": sampled_indices,
                "values": sampled_values,
            }
        ],
    )


if __name__ == "__main__":
    exit(main())
