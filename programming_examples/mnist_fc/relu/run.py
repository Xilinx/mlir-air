# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""2D ReLU on air.api: C[i, j] = max(A[i, j], 0).

Element-wise ReLU on an [M, N] matrix. The host interface is f32; the compare
and select run in bf16, because AIE2P has no f32 vector cmp/sel. That is one
expression here:

    tile_out[:] = ops.cast(ops.relu(ops.cast(tile_in[:], bf16)), f32)

`ops.cast` opens a region: everything below it is read and computed in the
source type, so the ReLU is bf16 and only the conversions cross. The
predecessor spelled the same thing as truncf, a *round trip through an L1 bf16
scratch buffer*, cmpf, select and extf. The scratch buffer is gone -- the value
stays in registers between the two conversions -- which gives back
`tile_m * tile_n * 2` bytes of L1.

MNIST context: op #3 in the GGML MNIST-FC pipeline. Default dimensions are
M=500, N=500 (the hidden layer activation), neither of which is tile-aligned,
so `build_module` is given padded extents and `air.actual_sizes` is stamped on
the launch afterwards -- exactly as the predecessor did it. `launch.build()`
returns the module, so that poke is unchanged.

Two levels of tiling: the launch grid walks [M, N] in `tile * herd` steps, and
the herd covers `herd_m x herd_n` tiles within each step. The tile offset is
plain arithmetic on the two sets of coordinates,

    (lx * herd_m + tx) * tile_m

where the predecessor built one arith.muli at segment scope and a two-symbol
AffineMap in the herd.
"""

import argparse
import math

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, f32
from air.ir import DenseI64ArrayAttr
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)


def build_module(m, n, tile_m, tile_n, herd_m, herd_n, vector=16):
    """Build the 2D ReLU launch over padded (tile-aligned) m, n."""
    if m % (tile_m * herd_m) or n % (tile_n * herd_n):
        raise ValueError(
            f"padded shape ({m}, {n}) must be a multiple of tile x herd "
            f"({tile_m * herd_m}, {tile_n * herd_n}): there is no partial-tile "
            "path, so the caller pads and stamps air.actual_sizes."
        )
    if vector and tile_n % vector:
        raise ValueError(
            f"tile_n ({tile_n}) must be a multiple of the vector width " f"({vector})."
        )

    A = air.tensor([m, n], f32)
    OUT = air.tensor([m, n], f32)

    with air.launch(
        [range(m // (tile_m * herd_m)), range(n // (tile_n * herd_n))],
        name="relu",
    ) as launch:

        @launch.body
        def _(lx, ly):
            with air.segment(name="relu_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as herd:

                        @herd.body
                        def _(tx, ty):
                            tile_in = air.alloc(
                                [tile_m, tile_n],
                                f32,
                                scope=herd.private(),
                                vector=vector,
                            )
                            tile_out = air.alloc(
                                [tile_m, tile_n],
                                f32,
                                scope=herd.private(),
                                vector=vector,
                            )

                            mo = (lx * herd_m + tx) * tile_m
                            no = (ly * herd_n + ty) * tile_n
                            window = (slice(mo, mo + tile_m), slice(no, no + tile_n))

                            ops.load(tile_in, A[window])
                            # AIE2P has no f32 vector cmp/sel, so the max runs
                            # in bf16 and only the conversions cross.
                            tile_out[:] = ops.cast(
                                ops.relu(ops.cast(tile_in[:], bf16)), f32
                            )
                            ops.store(tile_out, OUT[window])

    return launch


if __name__ == "__main__":
    M_ACTUAL = 500
    N_ACTUAL = 500
    TILE_M = 64
    TILE_N = 32
    HERD_M = 1
    HERD_N = 4
    VECTOR_SIZE = 16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="2D ReLU: C[i,j] = max(A[i,j], 0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_ACTUAL, help="Number of rows")
    parser.add_argument("--N", type=int, default=N_ACTUAL, help="Number of columns")
    parser.add_argument("--tile-m", type=int, default=TILE_M)
    parser.add_argument("--tile-n", type=int, default=TILE_N)
    parser.add_argument("--herd-m", type=int, default=HERD_M)
    parser.add_argument("--herd-n", type=int, default=HERD_N)
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )

    args = parser.parse_args()

    M_actual = args.M
    N_actual = args.N
    TILE_M = args.tile_m
    TILE_N = args.tile_n
    HERD_M = args.herd_m
    HERD_N = args.herd_n
    VECTOR_SIZE = args.vector_size

    # Pad to tile-aligned dimensions
    M_padded = math.ceil(M_actual / (TILE_M * HERD_M)) * (TILE_M * HERD_M)
    N_padded = math.ceil(N_actual / (TILE_N * HERD_N)) * (TILE_N * HERD_N)

    if args.verbose:
        print(f"M_actual={M_actual}, N_actual={N_actual}")
        print(f"M_padded={M_padded}, N_padded={N_padded}")
        print(f"TILE_M={TILE_M}, TILE_N={TILE_N}, HERD_M={HERD_M}, HERD_N={HERD_N}")

    launch = build_module(
        M_padded, N_padded, TILE_M, TILE_N, HERD_M, HERD_N, VECTOR_SIZE
    )
    # build() resolves --target auto to the installed generation, so it has to
    # run before launch.target is read below.
    mlir_module = launch.build(target=args.target)

    # Add actual_sizes attribute for device-side padding
    needs_padding = (M_actual != M_padded) or (N_actual != N_padded)
    if needs_padding:
        with mlir_module.context:
            for op in mlir_module.body.operations:
                for inner_op in op.body.blocks[0].operations:
                    if inner_op.name == "air.launch":
                        inner_op.attributes["air.actual_sizes"] = DenseI64ArrayAttr.get(
                            [M_actual, N_actual, 1]
                        )
                        break

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Host data: mix of positive and negative values
    input_a = np.zeros((M_padded, N_padded), dtype=np.float32)
    input_a[:M_actual, :N_actual] = (np.random.randn(M_actual, N_actual) * 4).astype(
        np.float32
    )

    if args.compile_mode == "compile-and-run":
        # Golden reference: max(x, 0)
        num_samples = 100
        sampled_indices = np.vstack(
            [
                np.random.randint(0, M_actual, num_samples),
                np.random.randint(0, N_actual, num_samples),
            ]
        )

        # Add boundary samples
        boundary_m = list(
            set(
                [
                    min(M_actual - 1, m)
                    for m in [M_actual - 1, M_actual - TILE_M + 1, 0]
                    if m >= 0
                ]
            )
        )
        boundary_n = list(
            set(
                [
                    min(N_actual - 1, n)
                    for n in [N_actual - 1, N_actual - TILE_N + 1, 0]
                    if n >= 0
                ]
            )
        )
        boundary_indices = np.array([[m, n] for m in boundary_m for n in boundary_n]).T
        sampled_indices = np.hstack([sampled_indices, boundary_indices])

        # Golden: truncate f32 to bf16 (matching hardware), relu in bf16,
        # then extend back to f32
        input_a_bf16 = input_a.astype(bfloat16)
        sampled_values = np.array(
            [max(float(input_a_bf16[i, j]), 0.0) for i, j in zip(*sampled_indices)],
            dtype=np.float32,
        )

        sampled_data = {
            "shape": (M_padded, N_padded),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf" if needs_padding else "xclbin",
            instance_name="relu",
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        # bf16 truncation introduces rounding; use bf16-appropriate tolerance
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf" if needs_padding else "xclbin",
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
