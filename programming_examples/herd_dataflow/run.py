# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A three-herd pipeline wired with channels, on air.api.

Each stage is a separate herd of NUM_COLUMNS cores, and a tile flows through
all three without ever going back to L2:

    herd_0    a, b from L2        c = a + b        -> L1ToL1Chan1
    herd_1    a from herd_0       c = a            -> L1ToL1Chan2  (cascade)
    herd_2    a from herd_1       add_3_bf16(a,c)  -> L1ToL2Chan1

    output = input_a + input_b + 3

The middle link is ``channel_type="npu_cascade"``: a direct core-to-core
connection between neighbouring tiles rather than a DMA stream. The two ends of
it are two different herds, which is the point of the example -- a channel is a
module-level symbol, so a herd finds what another herd sent by name alone, with
nothing threaded through the segment.

Three differences from the predecessor worth naming:

* ``c[:] = a[:] + b[:]`` and ``c[:] = a[:]`` replace, respectively, a
  doubly-nested loop building each 16-element strip out of two
  ``memref.subview``s and a ``memref.collapse_shape``, and a scalar
  ``memref.load``/``memref.store`` loop over all 4096 elements. The copy in
  herd_1 is now vectorised at the same 16 lanes as the add, where the
  predecessor moved one element per iteration.
* The per-column L2 staging is a Python ``for`` rather than an ``scf.forall``
  with two hand-built ``AffineMap``s per iteration. Both end up as
  NUM_COLUMNS independent transfers; writing it out is what lets the offsets be
  plain arithmetic on the loop variable.
* The tile origin comes from the launch coordinates directly, instead of
  ``affine_apply``-ing ``mul_m_l2_map`` and ``mul_n_l2_map`` to them and
  threading the results down.

``air.mlir``, which ``--mlir-source file`` parses instead of building, was
regenerated from this builder so the two sources stay the same design. It is
only valid at the default M=N=256.
"""

import argparse
import os

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

# One core's tile, and the number of cores each herd spans.
L1_M = 64
L1_N = 64
NUM_COLUMNS = 4
# The L2 staging buffer holds one row of tiles: every column's tile side by side.
L2_M = L1_M
L2_N = L1_N * NUM_COLUMNS

KERNEL_OBJ_NAME = "extern_func.o"


def build_module(m, n):
    assert m % L2_M == 0, f"m={m} is not a multiple of {L2_M}"
    assert n % L2_N == 0, f"n={n} is not a multiple of {L2_N}"

    A = air.tensor([m, n], bf16)
    B = air.tensor([m, n], bf16)
    C = air.tensor([m, n], bf16)

    # Named links between the stages. L1ToL1Chan2 is the cascade: a
    # point-to-point connection between neighbouring compute tiles, not a DMA
    # stream, which is what the middle stage exists to exercise.
    l2_to_l1_a = air.channel("L2ToL1Chan1", size=[NUM_COLUMNS, 1])
    l2_to_l1_b = air.channel("L2ToL1Chan2", size=[NUM_COLUMNS, 1])
    stage0_to_1 = air.channel("L1ToL1Chan1", size=[NUM_COLUMNS, 1])
    stage1_to_2 = air.channel(
        "L1ToL1Chan2", size=[NUM_COLUMNS, 1], channel_type="npu_cascade"
    )
    l1_to_l2 = air.channel("L1ToL2Chan1", size=[NUM_COLUMNS, 1])

    add_3 = air.extern("add_3_bf16", link_with=KERNEL_OBJ_NAME)

    with air.launch([range(m // L2_M), range(n // L2_N)], name="func1") as launch:

        @launch.body
        def _(ix, iy):
            r0, c0 = ix * L2_M, iy * L2_N

            with air.segment() as seg:

                @seg.body
                def _():
                    l2_a = air.alloc([L2_M, L2_N], bf16, scope=seg.private())
                    l2_b = air.alloc([L2_M, L2_N], bf16, scope=seg.private())
                    l2_c = air.alloc([L2_M, L2_N], bf16, scope=seg.private())

                    # L3 -> L2, then memtile -> cores, a column at a time.
                    # air.parallel, not a Python loop: the trips share one set
                    # of buffer descriptors, and the trip index is a channel
                    # bundle slot, which a temporal loop may not supply.
                    for col in air.parallel(NUM_COLUMNS):
                        lo, hi = col * L1_N, col * L1_N + L1_N
                        air.ops.load(
                            l2_a[:, lo:hi], A[r0 : r0 + L2_M, c0 + lo : c0 + hi]
                        )
                        air.ops.load(
                            l2_b[:, lo:hi], B[r0 : r0 + L2_M, c0 + lo : c0 + hi]
                        )

                    for col in air.parallel(NUM_COLUMNS):
                        lo, hi = col * L1_N, col * L1_N + L1_N
                        l2_to_l1_a.put(l2_a[:, lo:hi], indices=[col, 0])
                        l2_to_l1_b.put(l2_b[:, lo:hi], indices=[col, 0])

                    # Stage 1: add.
                    with air.herd(
                        [range(NUM_COLUMNS)], name="herd_0", shape=(NUM_COLUMNS,)
                    ) as h0:

                        @h0.body
                        def _(tx):
                            a = air.alloc([L1_M, L1_N], bf16, scope=h0.private())
                            b = air.alloc([L1_M, L1_N], bf16, scope=h0.private())
                            c = air.alloc([L1_M, L1_N], bf16, scope=h0.private())
                            l2_to_l1_a.get(a, indices=[tx, 0])
                            l2_to_l1_b.get(b, indices=[tx, 0])
                            c[:] = a[:] + b[:]
                            stage0_to_1.put(c, indices=[tx, 0])

                    # Stage 2: copy, and hand it over the cascade.
                    with air.herd(
                        [range(NUM_COLUMNS)], name="herd_1", shape=(NUM_COLUMNS,)
                    ) as h1:

                        @h1.body
                        def _(tx):
                            a = air.alloc([L1_M, L1_N], bf16, scope=h1.private())
                            c = air.alloc([L1_M, L1_N], bf16, scope=h1.private())
                            stage0_to_1.get(a, indices=[tx, 0])
                            c[:] = a[:]
                            stage1_to_2.put(c, indices=[tx, 0])

                    # Stage 3: the hand-written kernel, c = a + 3.
                    with air.herd(
                        [range(NUM_COLUMNS)],
                        name="herd_2",
                        shape=(NUM_COLUMNS,),
                        link_with=KERNEL_OBJ_NAME,
                    ) as h2:

                        @h2.body
                        def _(tx):
                            a = air.alloc([L1_M, L1_N], bf16, scope=h2.private())
                            c = air.alloc([L1_M, L1_N], bf16, scope=h2.private())
                            stage1_to_2.get(a, indices=[tx, 0])
                            add_3(a, c)
                            l1_to_l2.put(c, indices=[tx, 0])

                    for col in air.parallel(NUM_COLUMNS):
                        lo, hi = col * L1_N, col * L1_N + L1_N
                        l1_to_l2.get(l2_c[:, lo:hi], indices=[col, 0])
                        air.ops.store(
                            l2_c[:, lo:hi], C[r0 : r0 + L2_M, c0 + lo : c0 + hi]
                        )

    return launch


def parse_args():
    parser = argparse.ArgumentParser(description="AIR Herd Dataflow Example")
    parser.add_argument(
        "--m-size",
        type=int,
        default=256,
        help="Number of rows (M dimension) for L2 buffer",
    )
    parser.add_argument(
        "--n-size",
        type=int,
        default=256,
        help="Number of columns (N dimension) for L2 buffer",
    )
    parser.add_argument(
        "-p", "--print-ir", action="store_true", help="Print MLIR IR and exit"
    )
    parser.add_argument(
        "--mlir-source",
        choices=["python", "file"],
        default="python",
        help="How to obtain the MLIR module: 'python' (build with air.api) or "
        "'file' (load and parse air.mlir; NOTE: air.mlir is only valid for "
        "M=N=256)",
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
    return parser.parse_args()


def main():
    args = parse_args()
    M_SIZE = args.m_size
    N_SIZE = args.n_size

    # Either build the module here or parse the one checked in beside this
    # script. The file is the same design; it is regenerated from the builder.
    target = None
    if args.mlir_source == "python":
        launch = build_module(M_SIZE, N_SIZE)
        mlir_module = launch.build(target=args.target)
        target = launch.target
    else:
        import air.ir
        from air.ir import Module

        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "air.mlir"), "r") as f:
            mlir_text = f.read()
        mlir_module = Module.parse(mlir_text, context=air.ir.Context())

    if args.print_ir:
        print(str(mlir_module))
        return

    A = np.random.rand(M_SIZE, N_SIZE).astype(bfloat16)
    B = np.random.rand(M_SIZE, N_SIZE).astype(bfloat16)
    C = (A + B + 3.0).astype(bfloat16)

    runner = XRTRunner(
        omit_while_true_loop=False,
        verbose=False,
        runtime_loop_tiling_sizes=[2, 2],
        output_format=args.output_format,
        instance_name="func1",
        report_precision=True,
        **({"target_device": target} if target else {}),
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[A, B],
            expected_outputs=[C],
            rtol=1e-2,
        )
    )


if __name__ == "__main__":
    main()
