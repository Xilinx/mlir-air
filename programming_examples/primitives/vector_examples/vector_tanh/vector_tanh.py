# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Vectorized tanh primitive, on air.api.

    c[:] = air.ops.tanh(a[:])

One line of compute, over a [n] bf16 vector cut into [tile_n] tiles across
NUM_TILES cores. The predecessor built the same expression as a hand-rolled
vector loop -- an ``scf.for`` over the tile in steps of VECTOR_SIZE, two
``memref.subview``s per trip, a ``vector.transfer_read`` with an explicit
identity permutation map and a padding constant, ``math.tanh``, and a
``transfer_write``. Here the emitter builds that loop, and the subviews are not
needed at all: air.api reads at an offset directly.

The emitted op is unchanged: ``math.tanh`` on ``vector<VECTOR_SIZExbf16>``,
lowering ``math.tanh -> aievec.tanh -> xllvm.intr.aie2p.tanh``.

Three constraints here are hardware, not style, and the example must not drift
off them:

* **npu2 only.** On npu1 ``math.tanh`` lowers to a C call via ``emitc.include``,
  which the peano path cannot translate. Both lits are already
  ``REQUIRES: ryzen_ai_npu2`` and the predecessor fails on npu1 identically --
  an inherited limit, not a regression.
* **bf16 only, and vector only.** Scalar bf16 tanh does not legalize either
  (``s16 G_FTANH``), so the emitter's usual scalar fallback is the *unsafe*
  direction here rather than a safety net. ``tile_n`` must stay a multiple of
  the vector width or the fallback turns a working kernel into a build failure,
  which is why that is asserted below rather than left to chance.
* f32 tanh does not legalize at all, so ``np_dtype_in`` is checked rather than
  honoured -- the opposite call from the arithmetic siblings in this directory,
  and for a reason that is about the backend, not the object file.

Two differences from the predecessor worth naming:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's. The predecessor asked for a [1, NUM_TILES] herd
  and then wrote the outer loop itself, computing ``_l_ivx + _ty * tile_n``
  through a hand-built AffineMap. Here the herd's iteration space is the whole
  tile grid and air.api strip-mines it onto NUM_TILES cores.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2


def build_module(n, tile_n, np_dtype_in, vector_size=16):
    assert n % (tile_n * NUM_TILES) == 0
    # Not a stylistic assert: a tile that is not a multiple of the width sends
    # the emitter down its scalar fallback, and scalar bf16 tanh does not
    # legalize -- so the fallback fails to build rather than running slowly.
    assert tile_n % vector_size == 0
    if np_dtype_in is not bfloat16:
        raise ValueError(
            f"math.tanh legalizes only for bf16 vectors on AIE2P (f32 fails "
            f"with G_FTANH), so np_dtype_in must be ml_dtypes.bfloat16, "
            f"got {np_dtype_in!r}"
        )
    dt = bf16

    A = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="vector_tanh") as launch:

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
                    a = air.alloc([tile_n], dt, scope=h.private(), vector=vector_size)
                    c = air.alloc([tile_n], dt, scope=h.private(), vector=vector_size)

                    air.ops.load(a, A[i0 : i0 + tile_n])

                    c[:] = air.ops.tanh(a[:])

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vectorized tanh example",
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
        default="aie2p",
        help="Accepted for Makefile compatibility and otherwise unused: the "
        "vector width comes from --vector-size, which the lits always set, and "
        "the device from --target. Inherited from the predecessor, which also "
        "ignored it",
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

    launch = build_module(args.n, args.tile_n, INPUT_DATATYPE, args.vector_size)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(42)
    input_a = np.random.uniform(-4.0, 4.0, args.n).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])

        # Reference: compute tanh in f32 precision
        sampled_values = np.array(
            [np.tanh(input_a[i].astype(np.float32)) for i in zip(*sampled_indices)],
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
            instance_name="vector_tanh",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
                atol=5e-2,
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
