# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Two cascade columns per segment, stamped out by segment unroll, on air.api.

Sixteen L3 tiles are handed straight to sixteen cores -- no memtile, no
broadcast -- and reduced along ``tx`` into four results::

    output[seg, ty] = sum over tx of input[seg, tx, ty]

Three iteration spaces are in play at once, and the example exists to show that
they are three different things:

* **``air.launch``** is gridless: the whole kernel runs once.
* **``air.segment([range(NUM_SEGMENTS)])``** gives the *segment* an iteration
  space of its own, which prints as ``unroll(...)``. Each point is a physical
  copy of the segment body that ``air-to-aie`` lays out on its own columns, so
  ``NUM_SEGMENTS`` is a replication factor, not a loop trip count.
* **``air.herd([range(NUM_TILES), range(NUM_COLS)])``** is the 4x2 array of
  cores inside each copy.

The segment coordinate reaches the herd body by ordinary Python closure. The
predecessor had to pass it as ``operands=[seg_x]`` because ``air.herd`` is
``IsolatedFromAbove``; the DSL threads every enclosing coordinate in as an
operand and rebinds it, so ``sx`` inside the herd body is the herd's own block
argument. That is what makes ``chan_in.get(in_buf, indices=[sx, tx, ty])``
legal: a 3-D channel index whose outermost component comes from a scope two
levels up.

The three roles along a cascade chain are the same head/middle/tail split as
``cascade_reduction``, so they are the same nested ``ops.branch``. A herd body
is traced once for all eight cores of a copy, so a Python ``if`` would pick one
role for every one of them.

Two differences from the predecessor:

* ``acc[:] = in_buf[:]`` and ``acc[:] = in_buf[:] + acc[:]`` replace
  ``linalg.copy`` and a ``linalg.add`` whose output aliases an input. Same
  values; the DSL vectorises them rather than handing the pipeline named ops to
  fuse.
* The upstream neighbour is ``tx - 1`` rather than a hand-built ``arith.subi``,
  so it reaches the channel as one ``affine.apply`` like every other index in
  the DSL.
"""

import argparse

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

NUM_TILES = 4  # cascade depth (cores per column)
NUM_COLS = 2  # cascade columns per segment
NUM_SEGMENTS = 2  # segment unroll factor
DATA_SIZE = 1024

TOTAL_IN = NUM_SEGMENTS * NUM_TILES * NUM_COLS * DATA_SIZE  # 16384
TOTAL_OUT = NUM_SEGMENTS * NUM_COLS * DATA_SIZE  # 4096

INOUT_DATATYPE = np.int32


def build_module():
    src = air.tensor([TOTAL_IN], i32)
    dst = air.tensor([TOTAL_OUT], i32)

    # One slot per core: L3 straight to L1, no staging and no broadcast.
    chan_in = air.channel("chan_in", size=[NUM_SEGMENTS, NUM_TILES, NUM_COLS])
    # The core-to-core links along each column.
    chan_cascade = air.channel(
        "chan_cascade",
        size=[NUM_SEGMENTS, NUM_TILES, NUM_COLS],
        channel_type="npu_cascade",
    )
    # One shim channel per finished column.
    chan_out = air.channel("chan_out", size=[NUM_SEGMENTS, NUM_COLS])

    with air.launch(name="channel_3d_segment_unroll") as launch:

        @launch.body
        def _():
            # Unrolled at trace time: each put names a different channel slot,
            # so there is nothing here for a loop to carry.
            for seg in range(NUM_SEGMENTS):
                for tx in range(NUM_TILES):
                    for ty in range(NUM_COLS):
                        lo = (
                            seg * NUM_TILES * NUM_COLS + tx * NUM_COLS + ty
                        ) * DATA_SIZE
                        chan_in.put(src[lo : lo + DATA_SIZE], indices=[seg, tx, ty])

            with air.segment([range(NUM_SEGMENTS)], name="segment_0") as seg_ctx:

                @seg_ctx.body
                def _(sx):
                    with air.herd(
                        [range(NUM_TILES), range(NUM_COLS)],
                        name="herd_0",
                        shape=(NUM_TILES, NUM_COLS),
                    ) as herd:

                        @herd.body
                        def _(tx, ty):
                            in_buf = air.alloc([DATA_SIZE], i32, scope=herd.private())
                            chan_in.get(in_buf, indices=[sx, tx, ty])

                            acc = air.alloc([DATA_SIZE], i32, scope=herd.private())

                            with ops.branch(tx == 0) as head:
                                # Head of the chain: nothing upstream to add.
                                acc[:] = in_buf[:]
                                chan_cascade.put(acc, indices=[sx, tx, ty])

                            with head.otherwise():
                                chan_cascade.get(acc, indices=[sx, tx - 1, ty])
                                acc[:] = in_buf[:] + acc[:]

                                with ops.branch(tx == NUM_TILES - 1) as tail:
                                    chan_out.put(acc, indices=[sx, ty])
                                with tail.otherwise():
                                    chan_cascade.put(acc, indices=[sx, tx, ty])

            for seg in range(NUM_SEGMENTS):
                for ty in range(NUM_COLS):
                    lo = (seg * NUM_COLS + ty) * DATA_SIZE
                    chan_out.get(dst[lo : lo + DATA_SIZE], indices=[seg, ty])

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="channel_3d_segment_unroll.py",
        description="Builds, runs, and tests the 3D channel with segment unroll example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    # Both lits are npu2-only, so on an npu1 box compiling is the whole gate
    # this example can offer. cascade_reduction carries the same flag.
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

    # Sixteen unique tiles, one per core: core (seg, tx, ty) reads the tile at
    # (seg * NUM_TILES * NUM_COLS + tx * NUM_COLS + ty).
    input_a = np.arange(0, TOTAL_IN, dtype=INOUT_DATATYPE)

    expected_output = np.zeros(TOTAL_OUT, dtype=INOUT_DATATYPE)
    for seg in range(NUM_SEGMENTS):
        for ty in range(NUM_COLS):
            out_start = (seg * NUM_COLS + ty) * DATA_SIZE
            for tx in range(NUM_TILES):
                in_start = (seg * NUM_TILES * NUM_COLS + tx * NUM_COLS + ty) * DATA_SIZE
                expected_output[out_start : out_start + DATA_SIZE] += input_a[
                    in_start : in_start + DATA_SIZE
                ]

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="channel_3d_segment_unroll",
        runtime_loop_tiling_sizes=[4, 4],
        target_device=launch.target,
    )
    return runner.run_test(
        mlir_module,
        inputs=[input_a],
        expected_outputs=[expected_output],
    )


if __name__ == "__main__":
    exit(main())
