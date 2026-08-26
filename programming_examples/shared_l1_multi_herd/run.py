# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Two herds passing a tile through a shared L1 buffer, on air.api.

The AIE array lets neighbouring tiles address the same L1 memory, so a herd can
hand a tile to another herd without going back out to a memtile. What makes a
buffer shared is where it is allocated: not in a herd body, where it lives and
dies with one entry, but in the segment, with the segment's lifetime.

    shared = air.alloc([m, n], bf16, scope=seg.shared())   L1, segment lifetime

      herd_producer     shared[:] = a[:] + 1.0             writes it
      herd_consumer     c[:] = shared[:] + 2.0             reads it back

    output = input + 3

``seg.shared()`` is the whole of it. Both herds are nested in the segment body,
and each carries the buffer in as an operand automatically -- ``air.herd`` is
``IsolatedFromAbove``, so the raw-bindings version this replaces had to list it
in ``operands=[shared_l1_buffer.result]`` on both herds and pick it back up as a
body parameter on both. Nothing marks the buffer as shared beyond being
allocated where it is; ``seg.private()`` next to it would be a memtile instead.

For the intra-herd, core-to-core counterpart see ``shared_l1_single_herd``.

Two differences from the predecessor worth naming:

* Each ``+ 1.0`` and ``+ 2.0`` is one line rather than a doubly-nested loop over
  16-element strips, each strip built from a ``memref.subview``, a
  ``memref.collapse_shape`` to erase the leading unit dimension, a
  ``vector.transfer_read``, a ``vector.BroadcastOp`` for the constant and a
  ``vector.transfer_write``. The DSL picks the vector width; 64 bf16 across the
  minor axis vectorises to 16 lanes, the same width the predecessor spelled out.
* The launch grid is written ``air.launch([range(m // 64), range(n // 64)])``
  and the tile origin falls out of the coordinates, rather than two hand-built
  ``AffineMap``s applied to the launch ivs and threaded through every transfer.
  At the shipped 64x64 that grid is 1x1; 128x128, 256x64 and 64x256 were run on
  npu1 as well, because the predecessor accepted them and no lit covers them.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

# The tile both herds work on, and the shape of every L1 and L2 buffer here.
TILE_M = 64
TILE_N = 64


def build_module(m, n):
    assert m % TILE_M == 0, f"m={m} is not a multiple of the {TILE_M}-row tile"
    assert n % TILE_N == 0, f"n={n} is not a multiple of the {TILE_N}-column tile"

    A = air.tensor([m, n], bf16)
    C = air.tensor([m, n], bf16)

    to_producer = air.channel("InputToProducer")
    from_consumer = air.channel("ConsumerToOutput")

    # The tile grid goes on the launch, where the predecessor put it. A launch
    # point is a replay of everything inside -- the L2 and shared L1 buffers
    # below are refilled at each one -- and it is the only one of the three
    # iteration spaces here that is 2-D on hardware: a *segment* grid is a
    # spatial copy of the body across columns, and air-to-aie refuses a
    # row-wise one with "AIE only supports column-wise device slicing".
    with air.launch([range(m // TILE_M), range(n // TILE_N)], name="func1") as launch:

        @launch.body
        def _(ix, iy):
            with air.segment(name="segment_0") as seg:

                @seg.body
                def _():
                    r0, c0 = ix * TILE_M, iy * TILE_N

                    l2_in = air.alloc([TILE_M, TILE_N], bf16, scope=seg.private())
                    l2_out = air.alloc([TILE_M, TILE_N], bf16, scope=seg.private())

                    # L1, allocated here rather than in a herd body, so it
                    # outlives each herd and both can reach it.
                    shared = air.alloc([TILE_M, TILE_N], bf16, scope=seg.shared())

                    air.ops.load(l2_in, A[r0 : r0 + TILE_M, c0 : c0 + TILE_N])
                    to_producer.put(l2_in)

                    with air.herd([range(1)], name="herd_producer", shape=(1,)) as hp:

                        @hp.body
                        def _(tx):
                            a = air.alloc([TILE_M, TILE_N], bf16, scope=hp.private())
                            to_producer.get(a)
                            shared[:] = a[:] + 1.0

                    with air.herd([range(1)], name="herd_consumer", shape=(1,)) as hc:

                        @hc.body
                        def _(tx):
                            c = air.alloc([TILE_M, TILE_N], bf16, scope=hc.private())
                            c[:] = shared[:] + 2.0
                            from_consumer.put(c)

                    from_consumer.get(l2_out)
                    air.ops.store(l2_out, C[r0 : r0 + TILE_M, c0 : c0 + TILE_N])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the shared L1 multi-herd example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-p",
        "--print-ir",
        action="store_true",
        help="Print MLIR IR and exit",
    )
    parser.add_argument(
        "--m-size", type=int, default=TILE_M, help="Number of rows (M dimension)"
    )
    parser.add_argument(
        "--n-size", type=int, default=TILE_N, help="Number of columns (N dimension)"
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

    launch = build_module(args.m_size, args.n_size)
    mlir_module = launch.build(target=args.target)
    if args.print_ir:
        print(mlir_module)
        exit(0)

    # output = input + 1 (producer) + 2 (consumer)
    A = np.random.rand(args.m_size, args.n_size).astype(bfloat16)
    C = (A + 3.0).astype(bfloat16)

    runner = XRTRunner(
        omit_while_true_loop=False,
        verbose=args.verbose,
        runtime_loop_tiling_sizes=[1, 1],
        output_format=args.output_format,
        instance_name="func1",
        target_device=launch.target,
        report_precision=True,
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[A],
            expected_outputs=[C],
            rtol=1e-2,
        )
    )
