# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Elementwise select primitive, on air.api.

    c[:] = air.ops.select(a[:] >= b[:], a[:], b[:])

One line of compute. The predecessor built the same expression as a hand-rolled
vector loop -- an ``scf.for`` over the tile in steps of VECTOR_SIZE, three
``memref.subview``s per trip, two ``vector.transfer_read``s with an explicit
identity permutation map and a padding constant, ``arith.cmpf`` with
``CmpFPredicate.OGE``, ``arith.select``, and a ``transfer_write``. Here the
emitter builds that loop, and the subviews are not needed at all: air.api reads
at an offset directly.

The emitted arithmetic is unchanged: ``arith.cmpf oge`` producing a
``vector<VECTOR_SIZExi1>`` predicate, feeding ``arith.select``.

A comparison is the first thing in air.api's expression language whose result
type is *not* the element type, which is why ``select`` is a function rather
than an operator: the i1 a comparison yields has exactly one legal consumer.
The ordering comparisons are operators (``<``, ``<=``, ``>``, ``>=``) and map to
the *ordered* float predicates -- ``oge``, not ``uge``, so the result is false
when either operand is NaN, matching C and matching what the predecessor named.
Equality is ``ops.equal`` / ``ops.not_equal`` rather than ``==`` / ``!=``,
because overloading ``__eq__`` would make every expression unhashable and would
change what ``expr == expr`` means for ordinary Python.

Note what this kernel computes: ``select(a >= b, a, b)`` is ``max(a, b)``, and
the reference in ``__main__`` is written as ``max()``. ``ops.maximum`` would
lower it to a single ``arith.maximumf``. Both spellings are kept because
``vector_max`` and this example exist to be compared -- writing this one as a
maximum would erase the distinction the pair is for.

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
from air.api.types import dtype_of
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

NUM_TILES = 2


def build_module(n, tile_n, np_dtype_in, vector_size=16):
    assert n % (tile_n * NUM_TILES) == 0
    dt = dtype_of(np_dtype_in)
    if dt is None:
        raise ValueError(
            f"unsupported element type {np_dtype_in!r}; air.api knows "
            f"float32, float16, bfloat16, int8/16/32 and uint8/16/32"
        )

    A = air.tensor([n], dt)
    B = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="vector_select") as launch:

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

                    # cmpf + select on purpose, not ops.maximum -- vector_max is
                    # the single-op half of this pair. See the module docstring.
                    c[:] = air.ops.select(a[:] >= b[:], a[:], b[:])

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vector_select example",
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
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector size for SIMD operations",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
        help="Configure to whether to run after compile",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--bf16-emulation",
        dest="bf16_emulation",
        default=False,
        action="store_true",
        help="Use f32 input data type and emulate f32 vector arithmetic using bf16 operations.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )

    args = parser.parse_args()

    if args.bf16_emulation:
        INPUT_DATATYPE = np.float32
    bf16_emulation = args.bf16_emulation

    launch = build_module(
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
        args.vector_size,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.random.uniform(-100.0, 100.0, args.n).astype(INPUT_DATATYPE)
    input_b = np.random.uniform(-100.0, 100.0, args.n).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":

        # Stochastically sample num_sample results, and pass to XRTRunner backend for verification.
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, args.n, num_samples),  # i indices
            ]
        )

        # Compute reference results for sampled indices
        sampled_values = np.array(
            [max(input_a[i], input_b[i]) for i in zip(*sampled_indices)],
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
            output_format=args.output_format,
            instance_name="vector_select",
            bf16_emulation=bf16_emulation,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=5e-2 if bf16_emulation else 1e-3,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            bf16_emulation=bf16_emulation,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
