# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Elementwise i32 -> f32 conversion primitive, on air.api.

    c[:] = air.api.ops.cast(a[:], air.api.f32)

The predecessor built this as a hand-rolled vector loop -- an ``scf.for`` over
the tile in steps of VECTOR_SIZE, two ``memref.subview``s per trip, a
``vector.transfer_read`` with an explicit identity permutation map and a padding
constant, ``arith.sitofp``, and a ``transfer_write``. Here the emitter builds
that loop, and the subviews are not needed at all: air.api reads at an offset.

The emitted arithmetic is unchanged: a bare ``arith.sitofp`` from
``vector<VECTOR_SIZExi32>`` to ``vector<VECTOR_SIZExf32>``.

**This is the first example where the two L1 tiles have different element
types.** Every air.api expression until now was written in one type, and the
emitter carried a single element type, arith table and vector type down the
whole tree. ``ops.cast`` is the node where those change: everything below it is
read and computed in the source type, and only the conversion lands in the
destination's. That distinction is visible, and load-bearing:
``cast(a[:] * 2.0, i32)`` doubles in f32 and converts once, while
``cast(a[:], i32) * 2`` converts first and doubles in i32.

Which ``arith`` op a cast becomes follows from the two types, and each was run
on npu1 against an exact numpy reference before being allowed -- vectorised and
scalar, because compiling is a weaker claim than computing. Two results of that
sweep are worth knowing:

* ``fptosi`` and ``sitofp`` between f32 and i32, the pair this example uses,
  are **exact on both routes**, so the scalar fallback is safe here. That is not
  automatic: ``ops.tanh`` has no working scalar form at all, and its example has
  to refuse both routes into the fallback up front.
* Converting *to* a narrower float rounds toward negative infinity rather than
  to nearest, so a bf16 destination differs from numpy by up to one ULP. That
  is a property of the hardware and not of this op -- a plain
  ``c[:] = a[:] + b[:]`` on bf16 buffers differs the same way.

``ops.cast`` refuses narrowing between two integer types, which is why there is
no ``vector_trunci`` sibling: measured on npu1, the vectorised ``arith.trunci``
saturates while the scalar one wraps, and which one the emitter picks depends on
the tile size.

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


def build_module(n, tile_n, np_dtype_in, np_dtype_out, vector_size=16):
    assert n % (tile_n * NUM_TILES) == 0
    # Both element types are honoured rather than advertised and ignored: the
    # signature takes two, so passing np.int8 for the output has to either work
    # or say why not. `ops.cast` itself rejects the pairs that do not compute,
    # so there is no second copy of that rule here.
    #
    # Note that air.api.f16 is not usable on AIE2 whichever way it is reached:
    # the backend reads f16 data as bf16, so even a plain `c[:] = a[:] + b[:]`
    # on f16 buffers returns garbage. That is a property of the element type,
    # not of the conversion.
    types = []
    for np_dtype in (np_dtype_in, np_dtype_out):
        dt = dtype_of(np_dtype)
        if dt is None:
            raise ValueError(
                f"unsupported element type {np_dtype!r}; air.api knows "
                f"float32, float16, bfloat16, int8/16/32 and uint8/16/32"
            )
        types.append(dt)
    dt_in, dt_out = types

    A = air.tensor([n], dt_in)
    C = air.tensor([n], dt_out)

    with air.launch(name="vector_sitofp") as launch:

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
                    a = air.alloc(
                        [tile_n], dt_in, scope=h.private(), vector=vector_size
                    )
                    c = air.alloc(
                        [tile_n], dt_out, scope=h.private(), vector=vector_size
                    )

                    air.ops.load(a, A[i0 : i0 + tile_n])

                    c[:] = air.ops.cast(a[:], dt_out)

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    VECTOR_SIZE = 16
    INPUT_DATATYPE = np.int32
    OUTPUT_DATATYPE = np.float32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Per-element vector arith.sitofp (i32 -> f32) on the AIE.",
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

    launch = build_module(
        args.n, args.tile_n, INPUT_DATATYPE, OUTPUT_DATATYPE, args.vector_size
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Sample within +/-2^20 so each i32 converts to an exactly representable
    # f32; the host reference matches bit-for-bit on every backend.
    rng = np.random.default_rng(42)
    input_a = rng.integers(-(1 << 20), 1 << 20, size=args.n, dtype=INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])
        sampled_values = np.array(
            [input_a[i].astype(OUTPUT_DATATYPE) for i in zip(*sampled_indices)],
            dtype=OUTPUT_DATATYPE,
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
            instance_name="vector_sitofp",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
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
