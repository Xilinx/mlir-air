# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Scalar inverse square root (1/sqrt(x)) primitive, on air.api.

    c[:] = air.ops.rsqrt(a[:])        with vector=0

One line of compute, and the ``vector=0`` beside it is the whole point of this
example. It is what sends air.api's emitter down its scalar path: a plain
``scf.for`` over the tile in steps of one, ``memref.load``, ``math.rsqrt`` on an
f32 scalar, ``memref.store`` -- precisely what the raw-bindings predecessor
wrote by hand.

**npu2 only, and that is inherited, not new.** ``math.rsqrt`` on a plain f32 is
an instruction on npu2 and is not one on npu1, where the AIE lowering instead
calls ``getRsqrtBf16`` out of an object file -- a bf16 entry point, with nothing
to offer an f32 scalar. The predecessor's lit is already
``REQUIRES: ryzen_ai_npu2`` for that reason. What is new is that building for
npu1 now says so, rather than producing a module that fails several passes later
in the backend. ``vector_examples/vector_rsqrt --version 2`` is the same
computation and refuses npu1 the same way; ``--version 1`` there is the route
that works on npu1, in bf16 and against that object file.

Two differences from the predecessor worth naming:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's, and it distributes differently. The predecessor
  wrote the outer loop itself and gave core ``ty`` the tiles at
  ``_l_ivx + ty * tile_n`` -- an interleaved assignment, every other tile. The
  DSL hands each core a contiguous run of tiles instead. Every tile is still
  computed exactly once by exactly one core, and the tiles are independent, so
  the result is unchanged.
* The L1 buffers are allocated and freed together. The predecessor allocated
  both outside its temporal loop and called ``DeallocOp`` on them *inside* it,
  which frees each buffer once per trip -- 32 times, at the default size. Here
  the pair is scoped to the loop body and balances.

``--target`` is new and defaults to detecting the installed part, which is what
the predecessor did implicitly by having no device flag at all. Naming it makes
``-p`` reproducible, and gives the npu1 refusal above something to key off.
"""

import argparse

import numpy as np

from air import api as air
from air.api.types import f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2


def build_module(n, tile_n, target="auto"):
    assert n % (tile_n * NUM_TILES) == 0
    resolved = air.resolve_target(target)
    if resolved == "npu1":
        raise ValueError(
            "scalar f32 rsqrt is npu2-only: on npu1 math.rsqrt lowers to a call "
            "into getRsqrtBf16, which takes bf16 and has no f32 scalar form. "
            "The predecessor's lit was REQUIRES: ryzen_ai_npu2 for the same "
            "reason. For npu1, see vector_examples/vector_rsqrt --version 1."
        )
    dt = f32

    A = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="scalar_invsqrt") as launch:

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
                    # vector=0 is the scalar path -- see the module docstring.
                    a = air.alloc([tile_n], dt, scope=h.private(), vector=0)
                    c = air.alloc([tile_n], dt, scope=h.private(), vector=0)

                    air.ops.load(a, A[i0 : i0 + tile_n])

                    c[:] = air.ops.rsqrt(a[:])

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = np.float32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the scalar inverse square root (1/sqrt(x)) example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Total number of elements",
    )
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device) or npu2. This example is npu2-only -- see the module docstring",
    )
    args = parser.parse_args()

    launch = build_module(args.n, args.tile_n, args.target)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Generate random input vector with fixed seed for reproducibility
    # Use a safe range [0.1, 10] to avoid division by zero or very small numbers
    np.random.seed(37)
    input_a = np.random.uniform(0.1, 10.0, args.n).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":

        # Stochastically sample num_sample results, and pass to XRTRunner backend for verification.
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, args.n, num_samples),  # i indices
            ]
        )

        # Compute reference results for sampled indices: 1.0 / sqrt(x)
        sampled_values = np.array(
            [1.0 / np.sqrt(input_a[i]) for i in sampled_indices[0]],
            dtype=INPUT_DATATYPE,
        )

        # Store as a dictionary
        sampled_data = {
            "shape": (args.n,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        ###### Compile and test
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
