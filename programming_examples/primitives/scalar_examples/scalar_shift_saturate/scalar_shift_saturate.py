# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Scalar shift + saturate (fixed-point quantization) primitive, on air.api.

    v = ops.minimum(ops.maximum(a[:] >> shift, -128), 127)
    c[:] = ops.cast(ops.cast(v, i8), i32)                    with vector=0

    output[i] = clip(input[i] >> shift_amount, -128, 127)

**The shape of this expression is the point of the example, not incidental to
it.** The scalar ``shrsi + maxsi + minsi + trunci`` chain is what mlir-aie's
``LowerScalarShiftClampTruncToSRS`` pattern (PR #2894) matches, fusing it into a
vectorized SRS -- shift-round-saturate -- of the form
``broadcast_scalar -> cast(isResAcc) -> srs(narrowed) -> ext_elem``. So this
example exists to check that a *decomposed* chain still gets recognised, and the
conversion has to emit the same five ops in the same order for it to keep
testing that. It does: the emitted arith sequence is identical to the
predecessor's, op for op.

**Where the rounding comes from.** The kernel shifts with a bare ``shrsi`` and
never rounds. The reference rounds -- ``(x + (1 << (shift - 1))) >> shift``.
They agree because ``AIECoreToStandard`` sets rounding mode 9 (positive_inf) for
integer SRS, so the rounding is supplied by the fused hardware instruction and
not by the IR. That is exactly what makes this test load-bearing, and why it
checks at ``rtol=0, atol=0``: if the pattern failed to match, the shift would
happen without rounding and the answers would differ by one.

The ``trunci(i8)`` then ``extsi(i32)`` round-trip is likewise deliberate. It is
what completes the pattern, while leaving both DMA endpoints i32 so the L3
buffers stay one type.

**The narrowing cast.** ``air.api.ops.cast`` normally refuses i32 -> i8: on AIE
a vectorised ``arith.trunci`` saturates where the scalar one wraps, so the same
source would compute different things depending on the tile shape. That
objection does not apply here and air.api can tell, structurally -- the operand
is a clamp to constant bounds that fit i8, so no value the two paths disagree
about can reach the cast. The clamp has to genuinely be in the expression tree;
an unclamped narrow, or one clamped to bounds wider than the target, is still
refused.

Three differences from the predecessor, shared with the rest of this directory:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's, so each core gets a contiguous run of tiles where
  the predecessor's hand-built AffineMap interleaved them. Every tile is still
  computed exactly once by exactly one core, and the tiles are independent.
* The L1 buffers are allocated and freed together. The predecessor allocated
  both outside its temporal loop and called ``DeallocOp`` on them *inside* it,
  freeing each buffer once per trip.
"""

import argparse

import numpy as np

from air import api as air
from air.api.types import i8, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2

# The i8 saturation bounds. Named because they appear twice -- once in the
# kernel and once in the reference -- and because ops.cast reads them back off
# the expression tree to decide the narrowing cast is safe.
I8_MIN, I8_MAX = -128, 127


def build_module(n, tile_n, shift_amount=4):
    assert n % (tile_n * NUM_TILES) == 0
    if not 1 <= shift_amount < 32:
        raise ValueError(
            f"shift_amount must be in [1, 32) for an i32 shift, got "
            f"{shift_amount}. The reference rounds with 1 << (shift - 1), "
            f"which needs at least 1"
        )
    dt = i32

    A = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="scalar_shift_saturate") as launch:

        @launch.body
        def _():
            # The iteration space is every tile; shape= pins the core count to
            # what the predecessor asked for, and the DSL strip-mines the rest
            # into a loop on each core.
            with air.herd(
                [range(0, n, tile_n)], name="herd_0", shape=(NUM_TILES,)
            ) as h:

                @h.body
                def _(tx):
                    # tx is a tile *index*, not an element offset: the herd's
                    # iteration space counts tiles, and h.tile_sizes carries the
                    # step. Multiply to get the window into L3.
                    i0 = tx * tile_n
                    # vector=0 is the scalar path, which is what the SRS pattern
                    # this example exercises matches against.
                    a = air.alloc([tile_n], dt, scope=h.private(), vector=0)
                    c = air.alloc([tile_n], dt, scope=h.private(), vector=0)

                    air.ops.load(a, A[i0 : i0 + tile_n])

                    # shrsi, then maxsi, then minsi, then trunci/extsi. The
                    # order is the pattern's; see the module docstring.
                    clamped = air.ops.minimum(
                        air.ops.maximum(a[:] >> shift_amount, I8_MIN), I8_MAX
                    )
                    c[:] = air.ops.cast(air.ops.cast(clamped, i8), i32)

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    SHIFT_AMOUNT = 4
    INPUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the scalar shift+saturate example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--shift-amount",
        type=int,
        default=SHIFT_AMOUNT,
        help="Right shift amount (quantization scale factor)",
    )
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

    launch = build_module(args.n, args.tile_n, args.shift_amount)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(42)
    # Use a range where shifted values span the i8 output range:
    # With shift=4, input range [-2048, 2047] produces shifted values in [-128, 127].
    # Extend slightly beyond to also exercise saturation clamping.
    max_val = (I8_MAX << args.shift_amount) + (1 << args.shift_amount)
    input_a = np.random.randint(-max_val, max_val, args.n, dtype=INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])

        # Reference: SRS (Shift-Round-Saturate) with positive_inf rounding.
        # AIECoreToStandard sets rounding mode 9 (positive_inf) for integer SRS,
        # which rounds toward positive infinity at the midpoint.
        def ref_shift_saturate(x, shift):
            shifted = (x + (1 << (shift - 1))) >> shift
            return np.clip(shifted, I8_MIN, I8_MAX).astype(np.int8).astype(np.int32)

        sampled_values = np.array(
            [
                ref_shift_saturate(input_a[i], args.shift_amount)
                for i in zip(*sampled_indices)
            ],
            dtype=INPUT_DATATYPE,
        )
        sampled_data = {
            "shape": (args.n,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="scalar_shift_saturate",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=0,
                atol=0,
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
