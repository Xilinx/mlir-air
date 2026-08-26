# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Vectorized exp primitive, on air.api.

    c[:] = air.ops.exp(a[:])

One line of compute over a [n] bf16 vector cut into [tile_n] tiles across
NUM_TILES cores. The predecessor wrote the same expression as a hand-rolled
loop: an ``scf.for`` over the tile in steps of VECTOR_SIZE, two
``memref.subview``s per trip, a ``vector.transfer_read`` with an explicit
identity permutation map and a padding constant, ``math.exp``, and a
``transfer_write``. Here the emitter builds that loop and the subviews are not
needed -- air.api reads at an offset directly. The emitted compute op is
unchanged: ``math.exp`` on ``vector<VECTOR_SIZExbf16>``.

**The object file is the interesting part of this example.** On npu1 ``math.exp``
is not an instruction: the AIE lowering turns it into a call to ``getExpBf16``,
which lives in ``extern_func.o`` -- the ``extern_func.cc`` beside this file,
compiled by the Makefile against ``lut_based_ops.h``. So the herd has to carry
``link_with``, and nothing at trace time emits a call to hang it off, which is
what ``air.herd(link_with=...)`` is for. On npu2 ``exp`` is native and no object is
linked; passing one there would link an aie2 object into an aie2p build.

That branch keys off the *resolved* target rather than ``--arch``, so
``--target auto`` stays the single source of truth for the generation. The
predecessor keyed it off ``--arch`` and had no ``--target`` at all, which left
the two able to disagree.

Two other differences from the predecessor:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's. The predecessor asked for a [1, NUM_TILES] herd
  and then wrote the outer loop itself, computing ``_l_ivx + _ty * tile_n``
  through a hand-built AffineMap.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2

# The object the npu1 lowering of math.exp calls into. Named once: the Makefile
# compiles extern_func.cc to exactly this name in the build directory.
EXTERN_OBJECT = "extern_func.o"


def needs_extern_object(target):
    """Whether ``math.exp`` on this generation lowers to a call we must link.

    npu1 (aie2) has no exp instruction and calls getExpBf16; npu2 (aie2p) does
    it natively.
    """
    return target == "npu1"


def build_module(n, tile_n, vector_size=16, target="auto"):
    assert n % (tile_n * NUM_TILES) == 0
    assert tile_n % vector_size == 0
    resolved = air.resolve_target(target)

    A = air.tensor([n], bf16)
    C = air.tensor([n], bf16)

    with air.launch(name="vector_exp") as launch:

        @launch.body
        def _():
            with air.herd(
                [range(0, n, tile_n)],
                name="herd_0",
                shape=(NUM_TILES,),
                link_with=EXTERN_OBJECT if needs_extern_object(resolved) else None,
            ) as h:

                @h.body
                def _(tx):
                    # tx is a tile *index*: the herd's iteration space counts
                    # tiles and h.tile_sizes carries the step.
                    i0 = tx * tile_n
                    a = air.alloc([tile_n], bf16, scope=h.private(), vector=vector_size)
                    c = air.alloc([tile_n], bf16, scope=h.private(), vector=vector_size)

                    air.ops.load(a, A[i0 : i0 + tile_n])

                    c[:] = air.ops.exp(a[:])

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vectorized exp example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2",
        help="Accepted for Makefile compatibility and otherwise unused: the "
        "generation comes from --target, and whether extern_func.o is linked "
        "follows from that. Inherited from the predecessor, where this flag "
        "chose the object file",
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

    launch = build_module(args.n, args.tile_n, args.vector_size, args.target)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Inputs in [-5, 5] keep exp(x) inside the bfloat16 range.
    np.random.seed(42)
    input_a = np.random.uniform(-5, 5, args.n).astype(bfloat16)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])
        sampled_values = np.array(
            [np.exp(input_a[i]) for i in zip(*sampled_indices)],
            dtype=bfloat16,
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
            instance_name="vector_exp",
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
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
