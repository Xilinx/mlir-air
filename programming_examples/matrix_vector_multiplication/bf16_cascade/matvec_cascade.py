# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Cascade matrix-vector multiplication (GEMV) on air.api: C[M] = A[M,K] @ B[K].

bf16 in and out, accumulated in f32.

The K reduction is split across ``n_cascade`` cores stacked in one column, and
``herd_cols`` such columns run side by side, each owning ``tile_m`` output rows.
Every core dots its own K chunk; the partial sums travel *down* the column over
a cascade channel, which is a direct core-to-core link rather than a DMA
stream::

    ty == n_cascade-1  (northernmost)   dot            -> put
    ty == 1..n_cascade-2  (middle)      get -> dot, += -> put
    ty == 0            (southernmost)   get -> dot, += -> L1 -> L2 -> L3

A and C stage through L2; B goes L3 -> L1 directly, since each core reads a
different slice of it.

Three things about this port are worth knowing.

**The three cascade stages are one traced body, not three.** A herd body is
traced once for every core, so the stage a core plays has to be an ``scf.if``
on its coordinate -- ``ops.branch(ty == ...)`` -- and not a Python ``if``,
which would pick one stage for all of them. That is the same shape
``programming_examples/cascade_reduction`` uses, one level deeper.

**The dot product is written out, not called.** ``ops.dot`` accumulates through
a loop-carried vector, which does not legalize on AIE2; the predecessor
therefore hand-rolled a bf16 -> f32 widening FMA over a 16-lane f32
accumulator, and so does this. ``ops.fma`` and ``ops.cast`` emit exactly the
``vector.fma`` and ``arith.extf`` it emitted, and ``ops.reduce_add`` the closing
``vector.reduction <add>``.

**The partial sum goes through ``scratch`` instead of an SSA value.** A
reduction is the whole right-hand side of a statement -- it is the one op whose
result shape differs from its operand's -- so it cannot be added to the value
arriving from the north in the same expression. It lands in ``scratch`` first
and is folded in on the next line, which the middle stage wanted anyway (that
is the buffer it forwards) and which costs the last stage one store and one
load per row. That is the only departure from the predecessor's IR.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

# f32 lanes per FMA: 16 x 32 bits is the full 512-bit SIMD width.
VEC = 16
# The AIE2P cascade bus is 512 bits wide, so a cascade payload is a whole
# multiple of 16 floats however few rows a column actually carries.
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

    # L2 capacity guard.
    a_l2_bytes = herd_cols * tile_m * k * 2
    c_l2_bytes = herd_cols * tile_m * 2
    L2_CAPACITY = 512 * 1024
    assert a_l2_bytes + c_l2_bytes <= L2_CAPACITY, (
        f"L2 capacity exceeded: A={a_l2_bytes}B + C={c_l2_bytes}B = "
        f"{a_l2_bytes + c_l2_bytes}B > {L2_CAPACITY}B. "
        f"Reduce herd_cols ({herd_cols}), tile_m ({tile_m}), or k ({k})."
    )

    cascade_len = (
        (max(tile_m, CASCADE_WIDTH) + CASCADE_WIDTH - 1) // CASCADE_WIDTH
    ) * CASCADE_WIDTH

    A = air.tensor([m, k], bf16)
    B = air.tensor([k], bf16)
    C = air.tensor([m], bf16)

    # One link per adjacent pair in each column, so n_cascade-1 per column.
    cascade = air.channel(
        "chan_cascade", size=[herd_cols, n_cascade - 1], channel_type="npu_cascade"
    )

    # One launch iteration per band of herd_cols * tile_m output rows.
    with air.launch(
        [range(m // tile_m // herd_cols), range(1)], name="matvec_cascade_bf16"
    ) as launch:

        @launch.body
        def _(li, lj):

            with air.segment(name="matvec_cascade_0") as seg:

                @seg.body
                def _():
                    l2_a = air.alloc([herd_cols, tile_m, k], bf16, scope=seg.private())
                    l2_c = air.alloc([herd_cols, tile_m], bf16, scope=seg.private())

                    # A's rows split into bands of tile_m, so the band index is
                    # the outer axis of the reshape and this launch iteration
                    # takes herd_cols consecutive bands.
                    ops.load(
                        l2_a,
                        A.reshape(m // tile_m, tile_m, k)[
                            li * herd_cols : li * herd_cols + herd_cols, :, :
                        ],
                    )

                    # L1, but allocated here: a core's stage buffers are read
                    # and written across the whole herd, and per_core gives each
                    # core its own whole copy rather than a slab of one.
                    l1_a = air.alloc([m_input, k_chunk], bf16, scope=seg.per_core())
                    l1_b = air.alloc([k_chunk], bf16, scope=seg.per_core())
                    l1_c = air.alloc([tile_m], bf16, scope=seg.per_core())
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

                            acc = air.alloc([VEC], f32, scope=h.private())

                            def dot_into_scratch(row):
                                """scratch[row] = A[row, :] . B[:], in f32."""
                                acc[:] = 0.0
                                for jk in air.sequential(0, k_chunk, VEC):
                                    acc[:] = ops.fma(
                                        ops.cast(l1_a[row, jk : jk + VEC], f32),
                                        ops.cast(l1_b[jk : jk + VEC], f32),
                                        acc[:],
                                    )
                                scratch[row : row + 1] = ops.reduce_add(acc[:])

                            for jm in air.sequential(0, tile_m // m_input):
                                j0 = jm * m_input
                                ops.load(
                                    l1_a,
                                    l2_a[tx, j0 : j0 + m_input, k0 : k0 + k_chunk],
                                )

                                with ops.branch(ty == n_cascade - 1) as head:
                                    # Northernmost: nothing to add in.
                                    for row in air.sequential(0, m_input):
                                        dot_into_scratch(row)
                                    cascade.put(scratch, indices=[tx, ty - 1])

                                with head.otherwise():
                                    with ops.branch(ty == 0) as tail:
                                        # Southernmost: fold in and narrow.
                                        cascade.get(recv, indices=[tx, ty])
                                        for row in air.sequential(0, m_input):
                                            dot_into_scratch(row)
                                            o = j0 + row
                                            l1_c[o : o + 1] = ops.cast(
                                                recv[row : row + 1]
                                                + scratch[row : row + 1],
                                                bf16,
                                            )

                                    with tail.otherwise():
                                        # Middle: fold in and forward south.
                                        cascade.get(recv, indices=[tx, ty])
                                        for row in air.sequential(0, m_input):
                                            dot_into_scratch(row)
                                            scratch[row : row + 1] = (
                                                recv[row : row + 1]
                                                + scratch[row : row + 1]
                                            )
                                        cascade.put(scratch, indices=[tx, ty - 1])

                            # Only the southernmost core of each column holds a
                            # finished result.
                            with ops.branch(ty == 0):
                                ops.store(l1_c, l2_c[tx, :])

                    band = li * herd_cols * tile_m
                    ops.store(
                        l2_c.reshape(herd_cols * tile_m),
                        C[band : band + herd_cols * tile_m],
                    )

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="matvec_cascade.py",
        description="Builds, runs, and tests the cascade bf16 matrix-vector "
        "multiplication (GEMV) example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--m", type=int, default=2048, help="M dimension (matrix rows / output size)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8192,
        help="K dimension (matrix columns / vector length)",
    )
    parser.add_argument(
        "--tile-m",
        type=int,
        default=2,
        dest="tile_m",
        help="Number of output rows per tile per column",
    )
    parser.add_argument(
        "--m-input",
        type=int,
        default=1,
        help="Number of matrix rows per inner loop iteration",
    )
    parser.add_argument(
        "--herd-cols",
        type=int,
        default=8,
        dest="herd_cols",
        help="Number of AIE columns (parallel compute tiles along M dimension)",
    )
    parser.add_argument(
        "--n-cascade",
        type=int,
        default=4,
        dest="n_cascade",
        help="Number of cascade tiles per column (K-reduction depth)",
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
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-and-xclbin"],
        dest="compile_mode",
        default="compile-and-run",
        help="compile-and-run (default): compile and validate; "
        "compile-and-xclbin: generate xclbin only",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    parser.add_argument(
        "--debug-ir",
        action="store_true",
        dest="debug_ir",
        help="Emit IR after each pass into debug_ir/ directory",
    )
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

    if args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[4, 4],
            use_lock_race_condition_fix=True,
            target_device=launch.target,
        )
        backend.compile(mlir_module)
        backend.unload()
        return 0

    np.random.seed(42)
    input_a = (np.random.randn(args.m, args.k) * 4).astype(bfloat16)
    input_b = (np.random.randn(args.k) * 4).astype(bfloat16)
    output_c = np.dot(input_a.astype(np.float32), input_b.astype(np.float32)).astype(
        bfloat16
    )

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[4, 4],
        output_format=args.output_format,
        instance_name="matvec_cascade_bf16",
        debug_ir=args.debug_ir,
        use_lock_race_condition_fix=True,
        target_device=launch.target,
    )
    return runner.run_test(
        mlir_module,
        inputs=[input_a, input_b],
        expected_outputs=[output_c],
        rtol=0.04,
        atol=1e-3,
    )


if __name__ == "__main__":
    exit(main())
