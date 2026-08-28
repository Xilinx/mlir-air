# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Row-wise argmax on air.api: out[row] = argmax_col(A[row, col]).

One line of compute:

    out[:] = ops.argmax(tile[:, 0:ne0])

`ops.argmax` answers "which one" where `ops.reduce_max` answers "how big", so
its destination is an integer buffer whatever the operand's type is, and ties go
to the lowest index (a strict `>`), which is numpy's rule and the classifier
convention. It is the one reduction that is a scalar loop rather than a
`vector.reduction`: the running maximum and the index that produced it have to
travel together, and no vector reduction carries an index. The pair rides in the
loop's `iter_args`, which is fine because they are scalars -- it is a
loop-carried *vector* that AIE2 will not legalize.

The `[:, 0:ne0]` region is how the padding is handled. Columns are padded to a
multiple of 16 for DMA alignment, and reducing the padded tail would let a zero
outrank a genuinely negative logit; the predecessor kept a separate `ne0_actual`
bound on its scalar loop, and here it is a slice of the operand, which is what
numpy would write.

MNIST context: GGML argmax [10, 500] -> [500, 1]. ne0=10 (classes, contiguous,
reduced), ne1=500 (batch, rows, tiled across the herd). In numpy row-major the
input is (500, 10) and the output (500,) i32.
"""

import argparse
import math

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import f32, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)


def build_module(num_rows, num_cols, tile_rows, herd_n, ne0_actual, target="npu2"):
    """Row-wise argmax over the first `ne0_actual` of `num_cols` padded columns."""
    if num_rows % (tile_rows * herd_n):
        raise ValueError(
            f"padded rows ({num_rows}) must be a multiple of tile_rows * herd_n "
            f"({tile_rows * herd_n}); the caller pads."
        )
    if not 0 < ne0_actual <= num_cols:
        raise ValueError(
            f"ne0_actual ({ne0_actual}) must be in 1..num_cols ({num_cols})"
        )

    A = air.tensor([num_rows, num_cols], f32)
    OUT = air.tensor([num_rows], i32)

    with air.launch(
        [range(1), range(num_rows // (tile_rows * herd_n))], name="argmax"
    ) as launch:

        @launch.body
        def _(lx, ly):
            with air.segment(name="argmax_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(1), range(herd_n)], name="herd_0", shape=(1, herd_n)
                    ) as herd:

                        @herd.body
                        def _(tx, ty):
                            tile = air.alloc(
                                [tile_rows, num_cols], f32, scope=herd.private()
                            )
                            out = air.alloc([tile_rows], i32, scope=herd.private())

                            row = (ly * herd_n + ty) * tile_rows
                            ops.load(tile, A[row : row + tile_rows, :])

                            # Only the real columns: the padded tail is zero, and
                            # a zero would outrank a negative logit.
                            out[:] = ops.argmax(tile[:, 0:ne0_actual])

                            ops.store(out, OUT[row : row + tile_rows])

    return launch.build(target=target)


if __name__ == "__main__":
    # GGML [10, 500]: ne0=10 (classes), ne1=500 (batch)
    # numpy: (500, 10)
    NE0_ACTUAL = 10  # cols (classes, reduced)
    NE1_ACTUAL = 500  # rows (batch, tiled)
    TILE_ROWS = 32
    HERD_N = 4

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Row-wise argmax: out[row] = argmax_col(A[row,col])",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--ne0", type=int, default=NE0_ACTUAL, help="ne0 (cols, classes, reduced)"
    )
    parser.add_argument(
        "--ne1", type=int, default=NE1_ACTUAL, help="ne1 (rows, batch, tiled)"
    )
    parser.add_argument("--tile-rows", type=int, default=TILE_ROWS)
    parser.add_argument("--herd-n", type=int, default=HERD_N)
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )

    args = parser.parse_args()

    ne0_actual = args.ne0
    ne1_actual = args.ne1
    TILE_ROWS = args.tile_rows
    HERD_N = args.herd_n

    # Pad ne0 (cols) to multiple of 16 for DMA alignment
    ne0_padded = math.ceil(ne0_actual / 16) * 16
    # Pad ne1 (rows) to tile-aligned
    ne1_padded = math.ceil(ne1_actual / (TILE_ROWS * HERD_N)) * (TILE_ROWS * HERD_N)

    if args.verbose:
        print(f"ne0_actual={ne0_actual} (cols), ne1_actual={ne1_actual} (rows)")
        print(f"ne0_padded={ne0_padded}, ne1_padded={ne1_padded}")
        print(f"TILE_ROWS={TILE_ROWS}, HERD_N={HERD_N}")

    mlir_module = build_module(ne1_padded, ne0_padded, TILE_ROWS, HERD_N, ne0_actual)

    # Host-side padding (no air.actual_sizes needed; scalar loop uses ne0_actual)
    needs_padding = False

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Host data: (ne1, ne0) = (rows, cols) in numpy row-major
    input_a = np.zeros((ne1_padded, ne0_padded), dtype=np.float32)
    input_a[:ne1_actual, :ne0_actual] = (
        np.random.randn(ne1_actual, ne0_actual) * 4
    ).astype(np.float32)

    if args.compile_mode == "compile-and-run":
        # Golden: argmax over ne0 (axis=1, cols) for each row
        argmax_golden = np.argmax(input_a[:ne1_actual, :ne0_actual], axis=1).astype(
            np.int32
        )

        argmax_golden_padded = np.zeros(ne1_padded, dtype=np.int32)
        argmax_golden_padded[:ne1_actual] = argmax_golden

        # Sample indices
        num_samples = min(100, ne1_actual)
        sampled_row_indices = np.random.choice(ne1_actual, num_samples, replace=False)
        boundary_rows = [0, ne1_actual - 1]
        if ne1_actual - TILE_ROWS + 1 > 0:
            boundary_rows.append(ne1_actual - TILE_ROWS + 1)
        sampled_row_indices = np.unique(
            np.concatenate([sampled_row_indices, boundary_rows])
        )

        sampled_indices = np.vstack([sampled_row_indices])
        sampled_values = argmax_golden_padded[sampled_row_indices]

        sampled_data = {
            "shape": (ne1_padded,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="xclbin",
            instance_name="argmax",
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=0,
                atol=0,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format="xclbin",
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
