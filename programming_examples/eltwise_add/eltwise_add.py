# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Element-wise Add, written with the air.api DSL.

Computes C = A + B, over either a 1-D [N] or a 2-D [M, N] bf16 array. The two
ranks share one body: the herd iteration space, the tile offsets, and the slice
handed to the DMAs are all built from the shape, so `--shape 65536` and
`--shape 1024 1024` differ only in how many dimensions get tiled.

This is the same kernel as programming_examples/eltwise_add, expressed through
air.api instead of the raw dialect bindings: the DSL emits the herd, the L1
allocations, the affine tile offsets, the DMAs, and the vectorised compute loop.

Concepts:
  - a herd built from a range (1-D) or a product of ranges (2-D); a logical tile
    grid larger than the physical array is strip-mined onto it automatically
  - air.symbol() as a compile-time tile size, reported in launch.search_space
  - whole-tile elementwise assignment (c[:] = a[:] + b[:]), which lowers to a
    vector.transfer_read / arith.addf / transfer_write loop
"""

import argparse
from itertools import product

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops  # registers air.ops  # noqa: F401
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

DTYPES = {"bf16": (bf16, bfloat16), "f32": (f32, np.float32)}

# This kernel keeps three tiles live in L1, and the pipeline ping-pongs them, so
# the figure that has to fit a 64 KB compute tile is about twice what is
# declared. Derive the default tile from that budget rather than hardcoding it:
# an f32 tile has to be half the width of a bf16 one.
L1_BYTES = 65536
LIVE_BUFFERS = 3
PINGPONG = 2


def default_tile(rank, dtype, cap=1024):
    """Largest power-of-two tile whose 3 ping-ponged buffers fit L1."""
    budget = L1_BYTES // (LIVE_BUFFERS * PINGPONG * dtype.itemsize)
    tile = 1
    while (2 * tile) ** rank <= budget and 2 * tile <= cap:
        tile *= 2
    return tile


def build_eltwise_add(shape, dtype=bf16, tile=None, herd_shape=None, vector=None):
    """Build an elementwise-add launch over a 1-D or 2-D shape."""
    rank = len(shape)
    if rank not in (1, 2):
        raise ValueError(f"shape must be 1-D or 2-D, got {rank}-D: {shape}")
    tile = default_tile(rank, dtype) if tile is None else tile

    A = air.tensor(shape, dtype)
    B = air.tensor(shape, dtype)
    C = air.tensor(shape, dtype)

    # Compile-time tile sizes, one per tiled dimension. v1 resolves these to
    # `tile` and reports the binding in launch.search_space; it does not search.
    tiles = [air.symbol(hint=tile, name=f"tile_{d}") for d in range(rank)]

    def tile_body(h, coords):
        """One tile: stage both operands into L1, add, stage the result out."""
        sizes = h.tile_sizes
        window = tuple(slice(c * s, c * s + s) for c, s in zip(coords, sizes))
        a_buf = air.alloc(sizes, dtype, scope=h.private(), vector=vector)
        b_buf = air.alloc(sizes, dtype, scope=h.private(), vector=vector)
        c_buf = air.alloc(sizes, dtype, scope=h.private(), vector=vector)

        air.ops.load(a_buf, A[window])
        air.ops.load(b_buf, B[window])

        c_buf[:] = a_buf[:] + b_buf[:]

        air.ops.store(c_buf, C[window])

    with air.launch(name="eltwise_add") as launch:

        @launch.body
        def _():
            axes = [range(0, extent, t) for extent, t in zip(shape, tiles)]
            grid = axes[0] if rank == 1 else product(*axes)

            with air.herd(grid, shape=herd_shape) as h:
                if rank == 1:

                    @h.body
                    def _(tx):
                        tile_body(h, [tx])

                else:

                    @h.body
                    def _(tx, ty):
                        tile_body(h, [tx, ty])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="air.api element-wise add example")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 1024],
        metavar="EXTENT",
        help="1-D (N) or 2-D (M N) problem shape, e.g. --shape 65536 "
        "or --shape 1024 1024 (default: 1024 1024)",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help="tile size per tiled dimension (default: the largest power of two "
        "whose ping-ponged buffers fit L1, which depends on rank and dtype)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=None,
        dest="vector",
        help="compute vector width in lanes; 0 forces a scalar loop. Default is "
        "the dtype's (16 for bf16, 8 for f32). Note f32 at 8 lanes does not "
        "legalize on npu1 -- use 0 there, as the hand-written example does.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=sorted(DTYPES),
        help="element type (default: bf16)",
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
    launch = build_eltwise_add(
        shape=args.shape,
        dtype=dtype,
        tile=args.tile,
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
            instance_name="eltwise_add",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        backend.compile(module)
        backend.unload()
        print("Compiled successfully.")
        print("Search space:", launch.search_space)
        raise SystemExit(0)

    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(args.shape, dtype=np.float32).astype(np_dtype)
    b_np = rng.standard_normal(args.shape, dtype=np.float32).astype(np_dtype)

    # Reference in fp32, rounded back to the kernel's dtype -- the same standard
    # the raw-bindings version (eltwise_add_dialect.py) checks against.
    # Tolerances are taken from it: bf16 add is exact to a single bf16 round, so
    # atol is sized to the worst-case round; f32 is effectively exact.
    ref = (a_np.astype(np.float32) + b_np.astype(np.float32)).astype(np_dtype)
    rtol, atol = (1.6e-2, 5e-2) if args.dtype == "bf16" else (1e-3, 1e-5)

    # Checked through XRTRunner rather than a bare np.allclose: it is the same
    # harness every other example in this tree uses, so an air.api kernel gets
    # the same error statistics (report_precision) and the same PASS!/failed.
    # reporting that the lit harnesses grep for. air.api's job ends at
    # launch.build() -- what it hands over is an ordinary AIR module.
    # Build first: it resolves --target auto, and the runner has to compile for
    # the same generation the herd was sized for.
    module = launch.build(target=args.target)
    print(
        f"Eltwise add shape={tuple(args.shape)} dtype={args.dtype} on {launch.target}"
    )
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="eltwise_add",
        target_device=launch.target,
        runtime_loop_tiling_sizes=[4, 4],
        report_precision=True,
        n_perf_iters=args.perf_iters,
    )
    raise SystemExit(
        runner.run_test(
            module,
            inputs=[a_np, b_np],
            expected_outputs=[ref],
            rtol=rtol,
            atol=atol,
        )
    )
