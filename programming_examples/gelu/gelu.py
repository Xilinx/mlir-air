# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GELU, written with the air.api DSL.

Computes out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) over a 1-D [N] bf16 vector.

The whole compute body is one line:

    out_buf[:] = air.ops.gelu(x_buf[:])

`air.ops.gelu` is a composition over `air.ops.tanh`, written the same way the
raw-bindings kernel it replaces wrote it. Operand order differs in
places, but only where multiplication is commutative, which is exact in IEEE --
so the emitted arithmetic is the predecessor's, op for op.

**npu2 only, and bf16 only.** Both limits are inherited, not introduced:

  - On npu1, bf16 `math.tanh` lowers to a C call via `emitc.include`, which the
    peano path cannot translate ("missing LLVMTranslationDialectInterface
    registration ... for op: emitc.include"). The predecessor fails identically
    on npu1; its lit was already `REQUIRES: ryzen_ai_npu2`.
  - f32 `math.tanh` does not legalize on either generation, at any vector width
    including scalar ("unable to legalize instruction: G_FTANH"). So there is no
    f32 path to offer, unlike the other air.api examples.
  - Scalar bf16 tanh does not legalize either ("s16 G_FTANH"), so unlike every
    other air.api example there is no scalar fallback to drop to. The builder
    rejects a non-positive `--vector-size`, and rejects a tile that is not a
    multiple of the vector width, since that would reach the same scalar path
    indirectly.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

# Two tiles live in L1 and the pipeline ping-pongs them, so the figure that has
# to fit a 64 KB compute tile is about twice what is declared.
L1_BYTES = 65536
LIVE_BUFFERS = 2
PINGPONG = 2


def max_tile():
    """Largest tile whose ping-ponged bf16 buffers fit L1."""
    return L1_BYTES // (LIVE_BUFFERS * PINGPONG * bf16.itemsize)


def gelu_reference(x):
    """GELU reference, evaluated in fp32 and rounded back to the input dtype."""
    xf = x.astype(np.float32)
    return (
        0.5
        * xf
        * (
            1.0
            + np.tanh(
                air.ops.GELU_SQRT_2_OVER_PI * (xf + air.ops.GELU_BETA * xf * xf * xf)
            )
        )
    ).astype(x.dtype)


def build_gelu(n, tile_n, herd_shape=None, vector=None):
    """Build a GELU launch over a 1-D vector of length n."""
    if n % tile_n:
        raise ValueError(f"n ({n}) must be divisible by tile_n ({tile_n})")
    # math.tanh exists only as a vector lowering: scalar bf16 tanh fails with
    # "unable to legalize instruction: s16 G_FTANH" on npu2 too. That makes the
    # emitter's usual scalar fallback a trap here rather than a safety net, so
    # both routes into it are refused up front.
    width = bf16.default_vector_width if vector is None else int(vector)
    if width <= 0:
        raise ValueError(
            f"--vector-size must be positive, got {width}: this kernel has no "
            "scalar form, as math.tanh only lowers as a vector op"
        )
    if tile_n % width:
        raise ValueError(
            f"tile_n {tile_n} must be a multiple of the vector width {width}; "
            "otherwise the emitter falls back to a scalar loop, and math.tanh "
            "has no scalar lowering"
        )
    if tile_n > max_tile():
        raise ValueError(
            f"tile_n {tile_n} does not fit L1: the ping-ponged buffers of that "
            f"size exceed {L1_BYTES // 1024} KB (max tile is {max_tile()})"
        )

    x = air.tensor([n], bf16)
    out = air.tensor([n], bf16)

    tile = air.symbol(hint=tile_n, name="tile_n")

    # A 2-D herd_shape asks for a rectangular herd rather than a row of cores.
    # It matters: a 1-D herd is emitted as sizes=[P, 1], so it is bounded by the
    # column count -- 8 on npu2 -- while (8, 2) fills 16. SmolVLA's vision
    # encoder wants the latter, so the shape is honoured rather than flattened.
    grid_2d = herd_shape is not None and len(herd_shape) == 2

    def compute(h, base, tn):
        """One tile: stage in, apply gelu, stage out."""
        window = slice(base, base + tn)
        x_buf = air.alloc([tn], bf16, scope=h.private(), vector=vector)
        out_buf = air.alloc([tn], bf16, scope=h.private(), vector=vector)
        air.ops.load(x_buf, x[window])
        out_buf[:] = air.ops.gelu(x_buf[:])
        air.ops.store(out_buf, out[window])

    with air.launch(name="gelu") as launch:

        @launch.body
        def _():
            if grid_2d:
                cols, rows = int(herd_shape[0]), int(herd_shape[1])
                cores = cols * rows
                if n % (tile_n * cores):
                    raise ValueError(
                        f"n ({n}) must be divisible by tile_n * cores "
                        f"({tile_n} * {cores}); every core takes the same number "
                        "of tiles and air.api has no partial trips"
                    )

                with air.herd([range(cols), range(rows)], shape=(cols, rows)) as h:

                    @h.body
                    def _(tx, ty):
                        # Each core owns every cores'th tile, as the predecessor
                        # did: linear index tx*rows + ty, stepped by the whole
                        # herd. Allocations sit above the loop so the herd's
                        # deallocs still dominate them.
                        lin = tx * rows + ty
                        x_buf = air.alloc(
                            [tile_n], bf16, scope=h.private(), vector=vector
                        )
                        out_buf = air.alloc(
                            [tile_n], bf16, scope=h.private(), vector=vector
                        )
                        for iv in air.sequential(0, n, tile_n * cores):
                            base = iv + lin * tile_n
                            window = slice(base, base + tile_n)
                            air.ops.load(x_buf, x[window])
                            out_buf[:] = air.ops.gelu(x_buf[:])
                            air.ops.store(out_buf, out[window])

            else:
                with air.herd(range(0, n, tile), shape=herd_shape) as h:

                    @h.body
                    def _(tx):
                        compute(h, tx * h.tile_sizes[0], h.tile_sizes[0])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="air.api GELU example")
    parser.add_argument("--n", type=int, default=65536, help="vector length")
    parser.add_argument("--tile-n", type=int, default=1024, help="tile size")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=None,
        dest="vector",
        help="compute vector width in lanes (default: 16, the bf16 width). "
        "0 is rejected: math.tanh has no scalar lowering.",
    )
    parser.add_argument(
        "--herd-shape",
        type=int,
        nargs="+",
        default=None,
        metavar="EXTENT",
        help="override the physical herd shape (default: chosen per target)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        choices=["auto", "npu1", "npu2"],
        help="NPU generation to size the herd for and compile against "
        "(default: auto). This kernel needs npu2: bf16 math.tanh does not "
        "compile through the peano path on npu1.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="xclbin",
        choices=["xclbin", "elf"],
        dest="output_format",
        help="binary format aircc produces (default: xclbin)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="if >0, time the kernel over this many iterations (after warmup) "
        "and print Latency alongside the correctness check",
    )
    parser.add_argument("--print-ir", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")

    launch = build_gelu(
        n=args.n,
        tile_n=args.tile_n,
        herd_shape=args.herd_shape,
        vector=args.vector,
    )

    if args.print_ir:
        print(launch.mlir())
        raise SystemExit(0)

    if args.compile_only:
        # Build first: it resolves --target auto, and the backend has to compile
        # for the same generation the herd was sized for.
        module = launch.build(target=args.target)
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="gelu",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        backend.compile(module)
        backend.unload()
        print("Compiled successfully.")
        print("Search space:", launch.search_space)
        raise SystemExit(0)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(args.n, dtype=np.float32).astype(bfloat16)

    # The predecessor's tolerance, carried unchanged, and already justified there
    # from measurement: rtol is the canonical bf16 1.6e-2, and atol=5e-2 is
    # sized to the measured worst-case element on npu2 (abs_err max ~1.56e-2),
    # which a chained tanh LUT plus several bf16 roundings produces. The
    # predecessor already compared the full array, so this is unchanged.
    ref = gelu_reference(x_np)
    rtol, atol = 1.6e-2, 5e-2

    module = launch.build(target=args.target)
    print(f"GELU n={args.n} tile_n={args.tile_n} bf16 on {launch.target}")
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="gelu",
        target_device=launch.target,
        runtime_loop_tiling_sizes=[4, 4],
        report_precision=True,
        n_perf_iters=args.perf_iters,
    )
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[x_np],
            expected_outputs=[ref],
            rtol=rtol,
            atol=atol,
        )
    )
