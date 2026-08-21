# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A broadcast that crosses herds, on air.api.

    air.channel @bcast [1, 1] {broadcast_shape = [4, 1]}

One producer herd computes a vector and puts it once; a second herd of four
cores each get the same data and write it to its own slice of the output. So
this is the third shape of the same attribute: ``broadcast/single_herd`` fans
out to the cores of one herd, ``broadcast/multi_herd`` to three separate 1x1
herds, and this one from a herd to the cores of *another* herd. The producer
and the consumer share no buffer and no operand -- only the symbol.

Unchanged from the raw-bindings version this replaces, except that the
producer's "add one" is written ``l1_out[:] = l1_in[:] + 1.0`` rather than as a
hand-rolled loop over ``memref.subview`` + ``vector.transfer_read`` /
``arith.addf`` / ``vector.transfer_write``. bf16 vectorises at 16 lanes, which
is what the predecessor's ``VEC_SIZE`` picked by hand, and the DSL picks the
same width for a 64-element buffer.
"""

import argparse
from ml_dtypes import bfloat16
import numpy as np

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import bf16

VECTOR_LEN = 64  # Length of vector to broadcast (like head_dim or K)
HERD_N = 4  # Number of consumer tiles (like HERD_M in GEMV)
DTYPE = bfloat16


def build_module():
    A = air.tensor([VECTOR_LEN], bf16)
    B = air.tensor([HERD_N * VECTOR_LEN], bf16)

    bcast = air.channel("bcast", size=[1, 1], broadcast_shape=[HERD_N, 1])

    with air.launch(name="cross_herd_broadcast") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():

                    with air.herd([range(1)], name="producer", shape=(1,)) as p:

                        @p.body
                        def _(tx):
                            l1_in = air.alloc([VECTOR_LEN], bf16, scope=p.private())
                            l1_out = air.alloc([VECTOR_LEN], bf16, scope=p.private())

                            air.ops.load(l1_in, A)
                            l1_out[:] = l1_in[:] + 1.0
                            bcast.put(l1_out)

                    with air.herd(
                        [range(HERD_N), range(1)], name="consumer", shape=(HERD_N, 1)
                    ) as c:

                        @c.body
                        def _(tx, ty):
                            l1_buf = air.alloc([VECTOR_LEN], bf16, scope=c.private())

                            bcast.get(l1_buf, indices=[tx, ty])

                            off = tx * VECTOR_LEN
                            air.ops.store(l1_buf, B[off : off + VECTOR_LEN])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="broadcast.py",
        description="Cross-herd broadcast: [1,1] producer → [N,1] consumer via broadcast channel",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
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

    launch = build_module()
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Test data
    np.random.seed(42)
    data_in = (np.random.randn(VECTOR_LEN) * 2).astype(DTYPE)

    # Golden: each of HERD_N tiles gets (data_in + 1)
    expected_tile = (data_in.astype(np.float32) + 1.0).astype(DTYPE)
    expected_out = np.tile(expected_tile, HERD_N)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="cross_herd_broadcast",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[data_in],
            expected_outputs=[expected_out],
            rtol=0.01,
            atol=0.01,
        )
    )
