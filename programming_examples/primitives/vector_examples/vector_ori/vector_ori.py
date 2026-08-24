# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Elementwise bitwise OR primitive, on air.api.

    c[:] = a[:] | b[:]

One line of compute. The predecessor built the same expression as a hand-rolled
vector loop -- an ``scf.for`` over the tile in steps of VECTOR_SIZE, three
``memref.subview``s per trip, two ``vector.transfer_read``s with an explicit
identity permutation map and a padding constant, ``arith.ori``, and a
``transfer_write``. Here the emitter builds that loop, and the subviews are not
needed at all: air.api reads at an offset directly.

The emitted arithmetic is unchanged: a bare ``arith.ori`` on
``vector<VECTOR_SIZExi32>``.

Bitwise operators are integer-only, and that is a property of MLIR rather than
of this example: there is ``arith.ori`` and no floating-point counterpart, so
air.api refuses ``&``, ``|`` and ``^`` on a float buffer by name instead of
letting them fall through a generic "unsupported operator". i32 is checked below
for the same reason -- see the sibling ``vector_andi / vector_xori`` examples, which differ
from this one only in that operator.

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

from air import api as air
from air.api.types import dtype_of
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2


def build_module(n, tile_n, np_dtype_in, vector_size=16):
    assert n % (tile_n * NUM_TILES) == 0
    dt = dtype_of(np_dtype_in)
    if dt is None:
        raise ValueError(
            f"unsupported element type {np_dtype_in!r}; air.api knows "
            f"float32, float16, bfloat16, int8/16/32 and uint8/16/32"
        )
    # Checked rather than documented: `|` lowers to arith.ori, which has no
    # floating-point form, and an unsigned type cannot reach arith at all.
    if dt.is_float or dt.is_unsigned:
        raise ValueError(
            f"the bitwise operator '|' is integer-only and needs a signless "
            f"integer element type; got {np_dtype_in!r}"
        )

    A = air.tensor([n], dt)
    B = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="vector_ori") as launch:

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
                    b = air.alloc([tile_n], dt, scope=h.private(), vector=vector_size)
                    c = air.alloc([tile_n], dt, scope=h.private(), vector=vector_size)

                    air.ops.load(a, A[i0 : i0 + tile_n])
                    air.ops.load(b, B[i0 : i0 + tile_n])

                    c[:] = a[:] | b[:]

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16
    INPUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Per-element vector arith.ori over int32 tiles on the AIE.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations (16 lanes of i32 = 512 bits)",
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

    rng = np.random.default_rng(42)
    input_a = rng.integers(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=args.n,
        dtype=INPUT_DATATYPE,
    )
    input_b = rng.integers(
        np.iinfo(INPUT_DATATYPE).min,
        np.iinfo(INPUT_DATATYPE).max,
        size=args.n,
        dtype=INPUT_DATATYPE,
    )

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])
        sampled_values = np.array(
            [input_a[i] | input_b[i] for i in zip(*sampled_indices)],
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
            instance_name="vector_ori",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
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
        backend.compile(mlir_module)
        backend.unload()
