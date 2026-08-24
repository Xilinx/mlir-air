# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Elementwise division primitive, on air.api.

    c[:] = a[:] / b[:]

One line of compute. The predecessor built the same expression as a hand-rolled
vector loop -- an ``scf.for`` over the tile in steps of VECTOR_SIZE, three
``memref.subview``s per trip, two ``vector.transfer_read``s with an explicit
identity permutation map and a padding constant, ``arith.divf``, and a
``transfer_write``. Here the emitter builds that loop, and the subviews are not
needed at all: air.api reads at an offset directly.

The emitted arithmetic is unchanged: a bare ``arith.divf`` on
``vector<VECTOR_SIZExf32>``, with no broadcast -- both operands are buffers.
That is the difference from the sibling ``vector_reciprocal``, whose numerator
is a scalar and so carries a ``vector.broadcast``.

f32 is deliberate and load-bearing: bf16 division does *not* legalize vectorised
on either generation, so this primitive only exists in f32.

Two differences from the predecessor worth naming:

* The herd is [NUM_TILES, 1] rather than [1, NUM_TILES]. A 1-D air.api herd is
  laid out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's. The predecessor asked for a [1, NUM_TILES] herd
  and then wrote the outer loop itself, computing ``_l_ivx + _ty * tile_n``
  through a hand-built AffineMap. Here the herd's iteration space is the whole
  tile grid and air.api strip-mines it onto NUM_TILES cores.

``--arch`` still selects only the vector width, exactly as before; it does not
pin the device. The generation comes from ``--target``, which defaults to
detecting the installed part.
"""

import argparse

import numpy as np

from air import api as air
from air.api.types import dtype_of
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2

# Vector width per architecture. Selects the SIMD width only -- it does not pin
# which device the design is built for. Note aie2p is 64 here where the sibling
# vector_reciprocal uses 32; both are the widths their lits exercise.
ARCH_VECTOR_SIZES = {"aie2": 16, "aie2p": 64}

# f32 scalar fdiv needs a deeper stack than the 1024-byte default.
STACK_SIZE = 2048


def build_module(n, tile_n, np_dtype_in, arch="aie2"):
    assert n % (tile_n * NUM_TILES) == 0
    # f32 only, and checked rather than merely documented: division is the one
    # operator in this directory where no other element type legalizes. bf16,
    # f16 and i32 all die in the AIE backend's legalizer, on both generations
    # and at every width tried, e.g. for bf16 at 16 lanes:
    #     LLVM ERROR: unable to legalize instruction:
    #     %42:_(<16 x s16>) = G_FDIV %37:_, %41:_ (in function: core_0_2)
    # Accepting them here would turn a one-line contract violation into a
    # backend crash several passes downstream.
    if np_dtype_in is not np.float32:
        raise ValueError(
            f"division legalizes only for f32 on AIE (bf16/f16/i32 all fail "
            f"the backend legalizer with G_FDIV), so np_dtype_in must be "
            f"np.float32, got {np_dtype_in!r}"
        )
    dt = dtype_of(np_dtype_in)
    vector_size = ARCH_VECTOR_SIZES.get(arch, 16)  # default to 16 if unknown

    A = air.tensor([n], dt)
    B = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="vector_div") as launch:

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

                    c[:] = a[:] / b[:]

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = np.float32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vector division example",
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
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2",
        help="AIE architecture whose vector width to use (aie2 or aie2p). "
        "Selects the SIMD width only; use --target to choose the device",
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
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )

    args = parser.parse_args()

    launch = build_module(
        args.n,
        args.tile_n,
        INPUT_DATATYPE,
        args.arch,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Generate random input vectors with fixed seed for reproducibility
    np.random.seed(37)
    # Use a safe range [1, 10] for input_b to avoid division by zero
    input_a = np.random.uniform(0.1, 10.0, args.n).astype(INPUT_DATATYPE)
    input_b = np.random.uniform(1.0, 10.0, args.n).astype(INPUT_DATATYPE)

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
            [input_a[i] / input_b[i] for i in zip(*sampled_indices)],
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
            instance_name="vector_div",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
            stack_size=STACK_SIZE,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        ###### Compile only
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
            # The predecessor set stack_size on the runner but not here, so a
            # compile-only build got the 1024-byte default while the tested path
            # got 2048. No lit exercises compile-only, so the mismatch was
            # dormant; made consistent rather than carried forward.
            stack_size=STACK_SIZE,
        )
        module_function = backend.compile(mlir_module)

        backend.unload()
