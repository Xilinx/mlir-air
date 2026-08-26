# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Vector multiply primitive, on air.api.

    c[:] = a[:] * b[:]

A member of the vector_examples family; see ``vector_add`` for what the DSL is
doing here and why the raw-bindings version this replaces was forty lines of
hand-rolled ``memref.subview`` plus ``vector.transfer_read``/``transfer_write``
to say one thing. Only the operator differs, and the vector width is selected by
``--arch`` rather than given directly -- see below.

The herd is [NUM_TILES, 1] rather than the predecessor's [1, NUM_TILES] -- a 1-D
air.api herd is laid out along x, the orientation that places on both
generations -- and the tile grid is strip-mined onto those cores by the DSL
rather than by a hand-written outer loop.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

NUM_TILES = 2


# The predecessor selects its vector width from the target architecture rather
# than taking it directly. Note this only picks a width -- it does not pin the
# device, which stays `auto` via --target, so a module built here still runs on
# whichever generation is installed.
ARCH_VECTOR_SIZES = {"aie2": 16, "aie2p": 64}


def build_module(n, tile_n, np_dtype_in, arch="aie2"):
    assert n % (tile_n * NUM_TILES) == 0
    vector_size = ARCH_VECTOR_SIZES.get(arch, 16)
    dt = bf16 if np_dtype_in is bfloat16 else f32

    A = air.tensor([n], dt)
    B = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="vector_mul") as launch:

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

                    c[:] = a[:] * b[:]

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    # Default values.
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vector_mul example",
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
        help="Target AIE architecture (aie2 or aie2p); selects the vector width",
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
        args.arch,
    )
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(0, args.n, dtype=np.int64).reshape(args.n)
    input_a = input_a.astype(INPUT_DATATYPE)
    input_b = np.arange(0, args.n, dtype=np.int64).reshape(args.n)
    input_b = input_b.astype(INPUT_DATATYPE)

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
            [input_a[i] * input_b[i] for i in zip(*sampled_indices)],
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
            instance_name="vector_mul",
            bf16_emulation=bf16_emulation,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                stochastic_expected_outputs=[sampled_data],
                rtol=5e-2 if bf16_emulation else 1e-2,
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
