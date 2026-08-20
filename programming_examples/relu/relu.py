# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ReLU, written with the air.api DSL.

Computes out = max(x, 0) over a 1-D [N] vector.

The DSL emits the herd, the L1 allocations, the affine tile offsets, the DMAs,
and the vectorised compute loop; the compute itself is one line:

    out_buf[:] = ops.relu(x_buf[:])

`ops.relu` is `ops.maximum(x, 0.0)`, which lowers to the same `arith.maximumf`
against a broadcast zero that the raw-bindings implementation built by hand.

f32 runs scalar, and that is not a tuning choice: mlir-aie's
`convert-vector-to-aievec` rejects a vector `maximumf` on f32 outright --

    aievec.max conversion fails due to unsupported element data type
    error: failed to legalize operation 'aievec.max'

-- on both generations. Scalar f32 max is fine, and bf16 vectorises at 16 lanes.
Hence VECTOR_BY_DTYPE below.

The predecessor pinned a 2-core herd and cycled tiles across it. Here the
logical tile grid (N // tile_n) is strip-mined onto whatever the target
provides -- 4 cores on npu1, 8 on npu2, which is the device column count -- with
each core owning a contiguous block of tiles.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

DTYPES = {"bf16": (bf16, bfloat16), "f32": (f32, np.float32)}

# Default compute vector width per dtype, overridable with --vector-size. None
# means "use the dtype's own default" (16 lanes). f32 is pinned to 0 (scalar)
# because aievec has no f32 max -- see the module docstring.
VECTOR_BY_DTYPE = {"bf16": None, "f32": 0}

# Two tiles live in L1, and the pipeline ping-pongs them, so the figure that has
# to fit a 64 KB compute tile is about twice what is declared.
L1_BYTES = 65536
LIVE_BUFFERS = 2
PINGPONG = 2


def max_tile(dtype):
    """Largest tile whose ping-ponged buffers fit L1."""
    return L1_BYTES // (LIVE_BUFFERS * PINGPONG * dtype.itemsize)


def build_relu(n, tile_n, dtype=bf16, herd_shape=None, vector=None):
    """Build a ReLU launch over a 1-D vector of length n."""
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

    with air.launch(name="relu") as launch:

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

                    out_buf[:] = ops.relu(x_buf[:])

                    air.ops.store(out_buf, out[window])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="air.api ReLU example")
    parser.add_argument("--n", type=int, default=65536, help="vector length")
    parser.add_argument("--tile-n", type=int, default=1024, help="tile size")
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
        "16 for bf16 and 0 (scalar) for f32: aievec has no f32 vector max.",
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
    launch = build_relu(
        n=args.n,
        tile_n=args.tile_n,
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
            instance_name="relu",
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

    # ReLU is exact in any float format -- it either passes a value through or
    # emits zero, so no rounding is introduced and the result is bit-exact
    # against the reference. Measured 0.000e+00 error on every config.
    #
    # bf16 carries the predecessor's gate unchanged (rtol=1e-2 with the runner's
    # default atol), so it is the same per-element bar applied to all N elements
    # instead of the 100 random indices the predecessor sampled. f32 is new
    # coverage with no predecessor tolerance to inherit, so it takes the runner's
    # own default rather than borrowing bf16's -- inheriting the looser bf16
    # number would leave the f32 path an order of magnitude slacker than the
    # library standard for no reason.
    ref = np.maximum(x_np.astype(np.float32), 0.0).astype(np_dtype)
    rtol, atol = (1e-2, 1e-8) if args.dtype == "bf16" else (1e-3, 1e-8)

    module = launch.build(target=args.target)
    print(f"ReLU n={args.n} tile_n={args.tile_n} dtype={args.dtype} on {launch.target}")
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="relu",
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
