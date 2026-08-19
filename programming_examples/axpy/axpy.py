# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AXPY, written with the air.api DSL.

Computes out = alpha * x + y over a 1-D [N] vector.

The DSL emits the herd, the L1 allocations, the affine tile offsets, the DMAs,
and the vectorised compute loop; what it hands to the backend is an ordinary AIR
module. `alpha * x_buf[:] + y_buf[:]` is a lazy expression tree that is lowered
once, as a single vectorised loop, when it is assigned into out_buf.

Two differences from the raw-bindings implementation this replaces, both
deliberate:

  - It emits arith.mulf + arith.addf where the predecessor hand-built a
    vector.fma.
  - The predecessor pinned a 2-core herd and cycled tiles across it. Here the
    logical tile grid (N // tile_n) is strip-mined onto whatever the target
    provides -- 4 cores on npu1, 8 on npu2 -- with each core owning a
    contiguous block of tiles.

f32 runs scalar, and that is not a tuning choice. A *chained* f32
multiply-then-add on 512-bit vectors does not legalize in the AIE2 backend:

    LLVM ERROR: unable to legalize instruction: <16 x s32> = G_FMUL

Measured on npu1 and npu2 at 8, 16 and 32 lanes; all three fail. Each op on its
own is fine -- f32 vector add, sub, mul and div all compile at 16 lanes, and so
does `2.0 * x[:]` by itself -- so it is specifically the mul feeding an add.
bf16 has no such problem and vectorises the whole expression. (bf16 *divide*
does not legalize, but axpy does not divide.) Hence VECTOR_BY_DTYPE below.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops  # registers air.ops  # noqa: F401
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

DTYPES = {"bf16": (bf16, bfloat16), "f32": (f32, np.float32)}

# Default compute vector width per dtype, overridable with --vector-size. None
# means "use the dtype's own default" (16 lanes). f32 is pinned to 0 (scalar)
# because the chained multiply-add does not legalize at any vector width -- see
# the module docstring. This is the example's choice, not the DSL's: the DSL
# cannot see that the expression is a mul feeding an add.
VECTOR_BY_DTYPE = {"bf16": None, "f32": 0}

# This kernel keeps three tiles live in L1, and the pipeline ping-pongs them, so
# the figure that has to fit a 64 KB compute tile is about twice what is
# declared.
L1_BYTES = 65536
LIVE_BUFFERS = 3
PINGPONG = 2


def max_tile(dtype):
    """Largest tile whose three ping-ponged buffers fit L1."""
    return L1_BYTES // (LIVE_BUFFERS * PINGPONG * dtype.itemsize)


def build_axpy(n, tile_n, alpha, dtype=bf16, herd_shape=None, vector=None):
    """Build an AXPY launch over a 1-D vector of length n."""
    if n % tile_n:
        raise ValueError(f"n ({n}) must be divisible by tile_n ({tile_n})")
    if tile_n > max_tile(dtype):
        raise ValueError(
            f"tile_n {tile_n} does not fit L1 for {dtype}: three ping-ponged "
            f"buffers of that size exceed {L1_BYTES // 1024} KB "
            f"(max tile is {max_tile(dtype)})"
        )

    x = air.tensor([n], dtype)
    y = air.tensor([n], dtype)
    out = air.tensor([n], dtype)

    # Compile-time tile size. v1 resolves this and reports the binding in
    # launch.search_space; it does not search over it.
    tile = air.symbol(hint=tile_n, name="tile_n")

    with air.launch(name="axpy") as launch:

        @launch.body
        def _():
            with air.herd(range(0, n, tile), shape=herd_shape) as h:

                @h.body
                def _(tx):
                    tn = h.tile_sizes[0]
                    window = slice(tx * tn, tx * tn + tn)

                    x_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    y_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    out_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)

                    air.ops.load(x_buf, x[window])
                    air.ops.load(y_buf, y[window])

                    out_buf[:] = alpha * x_buf[:] + y_buf[:]

                    air.ops.store(out_buf, out[window])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="air.api AXPY example")
    parser.add_argument("--n", type=int, default=65536, help="vector length")
    parser.add_argument("--tile-n", type=int, default=1024, help="tile size")
    parser.add_argument("--alpha", type=float, default=2.0, help="scalar multiplier")
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
        "16 for bf16 and 0 (scalar) for f32: the chained f32 multiply-add does "
        "not legalize in the AIE2 backend at any vector width.",
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
    launch = build_axpy(
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
            instance_name="axpy",
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
    y_np = rng.standard_normal(args.n, dtype=np.float32).astype(np_dtype)

    # Reference in fp32, rounded back to the kernel's dtype.
    #
    # bf16 carries the predecessor's exact gate -- rtol=1e-2 with the runner's
    # default atol -- so this is strictly stronger than what it replaces: the
    # same per-element tolerance, applied to all N elements instead of the 100
    # random indices the predecessor sampled. Measured worst case across the
    # full array is rel_err max 7.8e-03, so there is real headroom and no need
    # to loosen. f32 takes the runner's default rtol (measured 3.6e-05).
    ref = (args.alpha * x_np.astype(np.float32) + y_np.astype(np.float32)).astype(
        np_dtype
    )
    rtol, atol = (1e-2, 1e-8) if args.dtype == "bf16" else (1e-3, 1e-8)

    # Checked through XRTRunner rather than a bare np.allclose: it is the same
    # harness every other example in this tree uses, so an air.api kernel gets
    # the same error statistics (report_precision) and the same PASS!/failed.
    # reporting that the lit harnesses grep for.
    module = launch.build(target=args.target)
    print(
        f"AXPY n={args.n} tile_n={args.tile_n} alpha={args.alpha} "
        f"dtype={args.dtype} on {launch.target}"
    )
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="axpy",
        target_device=launch.target,
        runtime_loop_tiling_sizes=[4, 4],
        report_precision=True,
        n_perf_iters=args.perf_iters,
    )
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[x_np, y_np],
            expected_outputs=[ref],
            rtol=rtol,
            atol=atol,
        )
    )
