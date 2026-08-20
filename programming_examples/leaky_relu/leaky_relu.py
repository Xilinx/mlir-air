# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Leaky ReLU, written with the air.api DSL.

Computes out = x if x >= 0 else alpha * x, over a 1-D [N] vector.

The raw-bindings implementation this replaces built that with a comparison and a
select:

    cmp      = arith.CmpFOp(OGE, x, 0)
    result   = arith.SelectOp(cmp, x, alpha * x)

air.api has no comparison or select, and this example does not add them. It uses
the identity

    leaky_relu(x, alpha) = max(x, 0) + alpha * min(x, 0)

which needs only operators the DSL already has. The two forms agree **bit for
bit**, not merely to a tolerance: for x >= 0 the max contributes x and the min
contributes zero, and for x < 0 the max contributes zero and the min
contributes x. Verified exactly equal over 200k unit-normal samples plus the
awkward inputs (+0, -0, +/-1e-30, +/-1) in both bf16 and f32, at alpha in
{0.01, 0.1, 0.2, 0.5}.

The cost is one extra op per element -- max, min, mul, add instead of cmp,
select, mul. The A/B in the PR shows what that is worth in practice.

f32 runs scalar: mlir-aie's convert-vector-to-aievec has no f32 max
("aievec.max conversion fails due to unsupported element data type"), and
separately a chained f32 multiply-then-add does not legalize at any vector
width. bf16 vectorises the whole expression at 16 lanes.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

DTYPES = {"bf16": (bf16, bfloat16), "f32": (f32, np.float32)}

# See the module docstring: f32 has neither a vector max nor a vector
# multiply-add, so it runs scalar. None means "the dtype's default" (16 lanes).
VECTOR_BY_DTYPE = {"bf16": None, "f32": 0}

L1_BYTES = 65536
LIVE_BUFFERS = 2
PINGPONG = 2


def max_tile(dtype):
    """Largest tile whose ping-ponged buffers fit L1."""
    return L1_BYTES // (LIVE_BUFFERS * PINGPONG * dtype.itemsize)


def build_leaky_relu(n, tile_n, alpha, dtype=bf16, herd_shape=None, vector=None):
    """Build a leaky-ReLU launch over a 1-D vector of length n."""
    if n % tile_n:
        raise ValueError(f"n ({n}) must be divisible by tile_n ({tile_n})")
    if tile_n > max_tile(dtype):
        raise ValueError(
            f"tile_n {tile_n} does not fit L1 for {dtype}: the ping-ponged "
            f"buffers of that size exceed {L1_BYTES // 1024} KB "
            f"(max tile is {max_tile(dtype)})"
        )

    x = air.tensor([n], dtype)
    out = air.tensor([n], dtype)

    tile = air.symbol(hint=tile_n, name="tile_n")

    with air.launch(name="leaky_relu") as launch:

        @launch.body
        def _():
            with air.herd(range(0, n, tile), shape=herd_shape) as h:

                @h.body
                def _(tx):
                    tn = h.tile_sizes[0]
                    window = slice(tx * tn, tx * tn + tn)

                    x_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    out_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)

                    air.ops.load(x_buf, x[window])

                    # max(x, 0) + alpha * min(x, 0) -- see the module docstring.
                    out_buf[:] = air.ops.maximum(
                        x_buf[:], 0.0
                    ) + alpha * air.ops.minimum(x_buf[:], 0.0)

                    air.ops.store(out_buf, out[window])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="air.api leaky ReLU example")
    parser.add_argument("--n", type=int, default=65536, help="vector length")
    parser.add_argument("--tile-n", type=int, default=1024, help="tile size")
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="slope for x < 0 (default: 0.01)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=sorted(DTYPES),
        help="element type (default: bf16)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=None,
        dest="vector",
        help="compute vector width in lanes; 0 forces a scalar loop. Default is "
        "16 for bf16 and 0 (scalar) for f32: aievec has no f32 vector max, and "
        "the chained f32 multiply-add does not legalize either.",
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
        "(default: auto, i.e. whichever one xrt-smi reports). Naming one "
        "explicitly is for compiling off-device: a binary built for a "
        "generation that is not installed still loads, and computes nothing.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="xclbin",
        choices=["xclbin", "elf"],
        dest="output_format",
        help="binary format aircc produces (default: xclbin). "
        "elf is npu2-only -- npu1 does not support the full-ELF flow.",
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

    dtype, np_dtype = DTYPES[args.dtype]
    vector = VECTOR_BY_DTYPE[args.dtype] if args.vector is None else args.vector
    launch = build_leaky_relu(
        n=args.n,
        tile_n=args.tile_n,
        alpha=args.alpha,
        dtype=dtype,
        herd_shape=args.herd_shape,
        vector=vector,
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
            instance_name="leaky_relu",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        backend.compile(module)
        backend.unload()
        print("Compiled successfully.")
        print("Search space:", launch.search_space)
        raise SystemExit(0)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(args.n, dtype=np.float32).astype(np_dtype)

    # Reference in the predecessor's own form (select, not the identity), so the
    # check is against the semantics being replaced rather than against the
    # rewrite.
    #
    # bf16 carries the predecessor's gate unchanged (rtol=1e-2 with the runner's
    # default atol), applied to the full array instead of 100 sampled indices.
    # Its measured rel_err max of 7.8e-03 is one bf16 ULP and comes from alpha
    # rounding to bf16, so the 1e-2 bar is genuinely needed there.
    #
    # f32 is new coverage with no predecessor tolerance to inherit, and it is
    # bit-exact (measured 0.000e+00), so it takes the runner's own default
    # instead of borrowing the bf16 number -- that would leave the f32 path an
    # order of magnitude slacker than the library standard for no reason.
    xf = x_np.astype(np.float32)
    ref = np.where(xf >= 0, xf, args.alpha * xf).astype(np_dtype)
    rtol, atol = (1e-2, 1e-8) if args.dtype == "bf16" else (1e-3, 1e-8)

    module = launch.build(target=args.target)
    print(
        f"Leaky ReLU n={args.n} tile_n={args.tile_n} alpha={args.alpha} "
        f"dtype={args.dtype} on {launch.target}"
    )
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="leaky_relu",
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
