# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Cascade GEMV with a fused residual add, on air.api: D[M] = A[M,K] @ B[K] + R[M].

The cascade is ``matvec_cascade.py``'s -- ``n_cascade`` cores down a column,
each dotting its own K chunk and passing the running sum south -- with one
change: the residual is added at the *head* rather than the tail. R therefore
seeds the accumulator that travels the cascade instead of being folded into the
result at the end, which costs nothing: the head has to write the payload
buffer anyway, and the tail's work is unchanged.

A and R share one L3 -> L2 channel per column, ``ar_L3toL2``, and one
L2 -> L1 channel per (column, cascade row), ``ar_L2toL1``. They share the
channel but not the buffer -- each fill targets its own L2 allocation -- so the
two transfers never contend for the same lock pair. On the L2 -> L1 side R goes
only to the head's slot while A is sliced by K, one put per cascade row.

Two departures from the predecessor's IR.

``ops.branch`` emits ``arith.cmpi`` plus ``scf.if`` where the predecessor hand-
built an ``IntegerSet`` and an ``affine.if`` for the head-only R receive. The
condition is the same one; the DSL has a single spelling for "this core only",
and it is the one the rest of the cascade already used.

The partial sum goes through ``scratch`` rather than an SSA value, for the
reason ``matvec_cascade.py`` documents: a reduction is the whole right-hand side
of a statement, so what it produces cannot be added to R -- or to the value
arriving from the north -- in the same expression.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

VEC = 16
CASCADE_WIDTH = 16


def build_module(m, k, tile_m, m_input, herd_cols, n_cascade):
    assert (
        n_cascade >= 2
    ), f"n_cascade ({n_cascade}) must be >= 2 for a cascade pipeline"
    k_chunk = k // n_cascade
    assert (
        m % (tile_m * herd_cols) == 0
    ), f"M ({m}) must be divisible by tile_m * herd_cols ({tile_m * herd_cols})"
    assert (
        tile_m % m_input == 0
    ), f"tile_m ({tile_m}) must be divisible by m_input ({m_input})"
    assert k % n_cascade == 0, f"K ({k}) must be divisible by n_cascade ({n_cascade})"
    assert (
        k_chunk % 64 == 0
    ), f"k_chunk ({k_chunk}) must be divisible by 64 (vector width)"
    # R reaches L1 as one bulk transfer of tile_m bf16, and an AIE DMA moves
    # whole 4-byte words.
    assert (
        tile_m * 2 >= 4
    ), f"tile_m ({tile_m}) * sizeof(bf16) must be >= 4 bytes (AIE DMA alignment)"

    # L2 budget: one bulk A buffer and one R buffer per column, plus one bulk D.
    l2_per_col = tile_m * k * 2 + tile_m * 2
    d_l2_bytes = herd_cols * tile_m * 2
    L2_CAPACITY = 512 * 1024
    assert herd_cols * l2_per_col + d_l2_bytes <= L2_CAPACITY, (
        f"L2 capacity exceeded: per-col={l2_per_col}B x {herd_cols} cols "
        f"+ D={d_l2_bytes}B = {herd_cols * l2_per_col + d_l2_bytes}B "
        f"> {L2_CAPACITY}B."
    )

    cascade_len = (
        (max(tile_m, CASCADE_WIDTH) + CASCADE_WIDTH - 1) // CASCADE_WIDTH
    ) * CASCADE_WIDTH
    head = n_cascade - 1

    A = air.tensor([m, k], bf16)
    B = air.tensor([k], bf16)
    R = air.tensor([m], bf16)
    D = air.tensor([m], bf16)

    ar_l3_l2 = air.channel("ar_L3toL2", size=[herd_cols])
    ar_l2_l1 = air.channel("ar_L2toL1", size=[herd_cols, n_cascade])
    cascade = air.channel(
        "chan_cascade", size=[herd_cols, n_cascade - 1], channel_type="npu_cascade"
    )

    with air.launch(
        [range(m // tile_m // herd_cols), range(1)], name="matvec_cascade_add_bf16"
    ) as launch:

        @launch.body
        def _(li, lj):
            band = li * tile_m * herd_cols

            # R first, then A: the segment reads them back in the same order.
            for col in range(herd_cols):
                lo = band + col * tile_m
                ar_l3_l2.put(R[lo : lo + tile_m], indices=[col])
                ar_l3_l2.put(A[lo : lo + tile_m, :], indices=[col])

            with air.segment(name="matvec_cascade_add_seg") as seg:

                @seg.body
                def _():
                    a_l2 = [
                        air.alloc([tile_m, k], bf16, scope=seg.private())
                        for _ in range(herd_cols)
                    ]
                    r_l2 = [
                        air.alloc([tile_m], bf16, scope=seg.private())
                        for _ in range(herd_cols)
                    ]
                    l2_d = air.alloc([herd_cols, tile_m], bf16, scope=seg.private())

                    for col in range(herd_cols):
                        ar_l3_l2.get(r_l2[col], indices=[col])
                        # R is only wanted by the core that seeds the cascade.
                        ar_l2_l1.put(r_l2[col], indices=[col, head])
                        ar_l3_l2.get(a_l2[col], indices=[col])
                        for row in range(n_cascade):
                            ar_l2_l1.put(
                                a_l2[col][:, row * k_chunk : (row + 1) * k_chunk],
                                indices=[col, row],
                            )

                    l1_a = air.alloc([tile_m, k_chunk], bf16, scope=seg.per_core())
                    l1_b = air.alloc([k_chunk], bf16, scope=seg.per_core())
                    l1_d = air.alloc([tile_m], bf16, scope=seg.per_core())
                    l1_r = air.alloc([tile_m], bf16, scope=seg.per_core())
                    scratch = air.alloc([cascade_len], f32, scope=seg.per_core())
                    recv = air.alloc([cascade_len], f32, scope=seg.per_core())

                    with air.herd(
                        [range(herd_cols), range(n_cascade)],
                        name="herd_0",
                        shape=(herd_cols, n_cascade),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            k0 = ty * k_chunk
                            ops.load(l1_b, B[k0 : k0 + k_chunk])

                            # One bulk receive each per launch iteration,
                            # outside the row loop that consumes them.
                            with ops.branch(ty == head):
                                ar_l2_l1.get(l1_r, indices=[tx, ty])
                            ar_l2_l1.get(l1_a, indices=[tx, ty])

                            acc = air.alloc([VEC], f32, scope=h.private())

                            def dot_into_scratch(row, at):
                                """scratch[row] = A[at, :] . B[:], in f32."""
                                acc[:] = 0.0
                                for jk in air.sequential(0, k_chunk, VEC):
                                    acc[:] = ops.fma(
                                        ops.cast(l1_a[at, jk : jk + VEC], f32),
                                        ops.cast(l1_b[jk : jk + VEC], f32),
                                        acc[:],
                                    )
                                scratch[row : row + 1] = ops.reduce_add(acc[:])

                            for jm in air.sequential(0, tile_m // m_input):
                                j0 = jm * m_input

                                with ops.branch(ty == head) as top:
                                    # Seed the cascade with the residual.
                                    for row in air.sequential(0, m_input):
                                        at = j0 + row
                                        dot_into_scratch(row, at)
                                        scratch[row : row + 1] = scratch[
                                            row : row + 1
                                        ] + ops.cast(l1_r[at : at + 1], f32)
                                    cascade.put(scratch, indices=[tx, ty - 1])

                                with top.otherwise():
                                    with ops.branch(ty == 0) as tail:
                                        # R was already folded in at the head.
                                        cascade.get(recv, indices=[tx, ty])
                                        for row in air.sequential(0, m_input):
                                            at = j0 + row
                                            dot_into_scratch(row, at)
                                            l1_d[at : at + 1] = ops.cast(
                                                recv[row : row + 1]
                                                + scratch[row : row + 1],
                                                bf16,
                                            )

                                    with tail.otherwise():
                                        cascade.get(recv, indices=[tx, ty])
                                        for row in air.sequential(0, m_input):
                                            dot_into_scratch(row, j0 + row)
                                            scratch[row : row + 1] = (
                                                recv[row : row + 1]
                                                + scratch[row : row + 1]
                                            )
                                        cascade.put(scratch, indices=[tx, ty - 1])

                            with ops.branch(ty == 0):
                                ops.store(l1_d, l2_d[tx, :])

                    ops.store(
                        l2_d.reshape(herd_cols * tile_m),
                        D[band : band + herd_cols * tile_m],
                    )

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="matvec_cascade_add.py",
        description="Cascade BF16 GEMV with fused residual add: D = A @ B + R",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--tile-m", type=int, default=2, dest="tile_m")
    parser.add_argument("--m-input", type=int, default=1, dest="m_input")
    parser.add_argument("--herd-cols", type=int, default=8, dest="herd_cols")
    parser.add_argument("--n-cascade", type=int, default=4, dest="n_cascade")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-and-xclbin"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    parser.add_argument("--debug-ir", action="store_true", dest="debug_ir")
    return parser.parse_args()


def main():
    args = parse_args()

    launch = build_module(
        args.m, args.k, args.tile_m, args.m_input, args.herd_cols, args.n_cascade
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        return 0

    # Each column's ar_L3toL2 carries one R and one A transfer per launch
    # iteration, and the shim DMA BD queue holds 4, so the runtime loop tiles
    # by 2.
    if args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[2, 2],
            output_format=args.output_format,
            use_lock_race_condition_fix=True,
            target_device=launch.target,
        )
        backend.compile(mlir_module)
        backend.unload()
        return 0

    np.random.seed(42)
    input_a = (np.random.randn(args.m, args.k) * 4).astype(bfloat16)
    input_b = (np.random.randn(args.k) * 4).astype(bfloat16)
    input_r = (np.random.randn(args.m) * 4).astype(bfloat16)
    output_d = (
        np.dot(input_a.astype(np.float32), input_b.astype(np.float32))
        + input_r.astype(np.float32)
    ).astype(bfloat16)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[2, 2],
        output_format=args.output_format,
        instance_name="matvec_cascade_add_bf16",
        debug_ir=args.debug_ir,
        use_lock_race_condition_fix=True,
        target_device=launch.target,
    )
    return runner.run_test(
        mlir_module,
        inputs=[input_a, input_b, input_r],
        expected_outputs=[output_d],
        rtol=0.04,
        atol=1e-3,
    )


if __name__ == "__main__":
    exit(main())
