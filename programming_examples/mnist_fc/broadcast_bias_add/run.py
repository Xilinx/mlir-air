# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Broadcast bias add, on air.api: ``C[row, col] = A[row, col] + bias[col]``.

Ops #2 and #5 of the GGML MNIST-FC pipeline. GGML's layout is ``[ne0, ne1]``
with ``ne0`` contiguous, so in numpy row-major the matrix is ``(ne1, ne0)`` and
the bias is ``(ne0,)`` -- one value per column, reused down every row. All f32.

The whole kernel is one line:

    out[:] = a[:] + bias[:]        [tile_m, tile_n] + [tile_n]

which is numpy's broadcast rule, right-aligned: ``bias`` is short of an axis, so
it is stretched along the one it does not have. The predecessor wrote the same
thing as a doubly-nested loop that, for every 16-wide strip of every row,
re-took a ``memref.subview`` of the bias and re-read it with a
``vector.transfer_read`` -- the bias load is loop-invariant but was spelled
inside both loops.

Two things this does *not* change, because they are the example's contract:

* ``M`` and ``N`` are padded up to a whole number of tiles, and when padding
  happens the module carries ``air.actual_sizes`` on ``air.launch`` so the
  runtime knows the real extent. ``launch.build()`` hands back the module, so
  that attribute is set exactly where the predecessor set it.
* ``--vector-size`` still selects the width, through ``air.alloc(vector=...)``.
  16 is the default and the only width f32 has: 8 lanes does not legalize on
  either generation.
"""

import argparse
import math

import numpy as np

from air import api as air
from air.api.types import f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)


def build_module(m, n, tile_m, tile_n, herd_m, herd_n, vector_size=16):
    """Build the broadcast bias add launch.

    ``m`` = ne1 (rows, padded), ``n`` = ne0 (cols, contiguous, padded). The bias
    has length ``n`` and is indexed by column.
    """
    assert m % (tile_m * herd_m) == 0
    assert n % (tile_n * herd_n) == 0
    assert tile_n % vector_size == 0

    A = air.tensor([m, n], f32)
    BIAS = air.tensor([n], f32)
    OUT = air.tensor([m, n], f32)

    grid = [range(m // tile_m // herd_m), range(n // tile_n // herd_n)]

    with air.launch(grid, name="broadcast_bias_add") as launch:

        @launch.body
        def _(lx, ly):
            with air.segment(name="bias_add_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            # Where this core's tile starts. The launch point
                            # picks the block of tiles; the tile coordinate
                            # picks one inside it.
                            m0 = lx * (tile_m * herd_m) + tx * tile_m
                            n0 = ly * (tile_n * herd_n) + ty * tile_n

                            # Allocated in the predecessor's order -- tile in,
                            # tile out, bias -- so the aie.buffer numbering
                            # comes out the same and the two designs can be
                            # compared as generated code, not just as results.
                            a = air.alloc(
                                [tile_m, tile_n],
                                f32,
                                scope=h.private(),
                                vector=vector_size,
                            )
                            out = air.alloc(
                                [tile_m, tile_n],
                                f32,
                                scope=h.private(),
                                vector=vector_size,
                            )
                            bias = air.alloc(
                                [tile_n], f32, scope=h.private(), vector=vector_size
                            )

                            air.ops.load(a, A[m0 : m0 + tile_m, n0 : n0 + tile_n])
                            air.ops.load(bias, BIAS[n0 : n0 + tile_n])

                            out[:] = a[:] + bias[:]

                            air.ops.store(out, OUT[m0 : m0 + tile_m, n0 : n0 + tile_n])

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
        description="Broadcast bias add: C[row,col] = A[row,col] + bias[col]",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--M", type=int, default=M_ACTUAL, help="Number of rows (ne1)")
    parser.add_argument(
        "--N", type=int, default=N_ACTUAL, help="Number of columns (ne0)"
    )
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
        print(f"M_actual={M_actual} (ne1), N_actual={N_actual} (ne0)")
        print(f"M_padded={M_padded}, N_padded={N_padded}")
        print(f"TILE_M={TILE_M}, TILE_N={TILE_N}, HERD_M={HERD_M}, HERD_N={HERD_N}")

    launch = build_module(
        M_padded, N_padded, TILE_M, TILE_N, HERD_M, HERD_N, VECTOR_SIZE
    )
    mlir_module = launch.build(target=args.target)

    # Add actual_sizes attribute for device-side padding
    needs_padding = (M_actual != M_padded) or (N_actual != N_padded)
    if needs_padding:
        from air.ir import DenseI64ArrayAttr

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

    # Host data: matrix (ne1 x ne0), bias (ne0,)
    input_a = np.zeros((M_padded, N_padded), dtype=np.float32)
    input_a[:M_actual, :N_actual] = (np.random.randn(M_actual, N_actual) * 4).astype(
        np.float32
    )
    # Bias along ne0 (columns, contiguous dimension)
    input_bias = np.zeros(N_padded, dtype=np.float32)
    input_bias[:N_actual] = (np.random.randn(N_actual) * 2).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        # Golden: C[row,col] = A[row,col] + bias[col]
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

        # bias indexed by column (ne0)
        sampled_values = np.array(
            [input_a[i, j] + input_bias[j] for i, j in zip(*sampled_indices)],
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
            instance_name="broadcast_bias_add",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_bias],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-6,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="elf" if needs_padding else "xclbin",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
