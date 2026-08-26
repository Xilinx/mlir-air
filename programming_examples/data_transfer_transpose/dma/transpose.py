# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Transpose an [M, K] matrix on the way out of L1, on air.api.

    air.ops.load(t, A[:, :])
    air.ops.store(t.transpose(1, 0), B[:, :])

There is no compute here at all. The transpose is a property of the *access
pattern* the second DMA walks: the tile is read back with its two axes swapped,
so the descriptor carries sizes [k, m] and strides [1, k] where the buffer's own
are [m, k] and [k, 1]. The hand-written predecessor spelled that by passing
``src_sizes=[1, k, m], src_strides=[1, 1, k]`` to ``dma_memcpy_nd`` as literal
lists; ``transpose`` derives the identical descriptor from the permutation,
which is the same thing said in terms of what it means rather than what it
computes to.

``transpose`` takes a full permutation, numpy-style, rather than being a bare
``.T``: on an nd machine reversing every axis is rarely what is meant, and
naming the permutation is what makes a rank-3 or rank-4 version of this readable
instead of a puzzle.

The L1 tile is ``[m, k]`` rather than the predecessor's flat ``[m * k]``. It
holds the same bytes and the same transfers move them; the shape is now the
shape of the thing in it, which is what lets the permutation be written at all.

Both element types the Makefile exercises still work -- ``uint32`` via
``run_int`` and ``float32`` via ``run_float``. Nothing here is arithmetic, so
the unsigned type is carried by the transfer without being computed on, which is
the one place air.api accepts one.
"""

import argparse

import numpy as np

from air import api as air
from air.api.types import dtype_of
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

dtype_map = {
    "uint32": np.uint32,
    "float32": np.float32,
}
DEFAULT_DTYPE = "uint32"


def build_module(m, k, np_dtype):
    dt = dtype_of(np_dtype)

    A = air.tensor([m, k], dt)
    B = air.tensor([k, m], dt)

    with air.launch(name="transpose") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="herd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            t = air.alloc([m, k], dt, scope=h.private())
                            air.ops.load(t, A[:, :])
                            # The whole example: the same tile, walked with its
                            # axes swapped. A view, not a copy -- nothing moves
                            # until the store's descriptor walks it.
                            air.ops.store(t.transpose(1, 0), B[:, :])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the data_transfer_transpose/dma example",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=64,
        help="The matrix to transpose will be of size M x K, this parameter sets the M value",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=32,
        help="The matrix to transpose will be of size M x K, this parameter sets the k value",
    )
    parser.add_argument(
        "-t",
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=dtype_map.keys(),
        help="The data type of the matrix",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
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

    np_dtype = dtype_map[args.dtype]
    launch = build_module(args.m, args.k, np_dtype)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Generate a random matrix
    matrix_shape = (args.m, args.k)
    if np.issubdtype(np_dtype, np.floating):
        for np_type in dtype_map.values():
            if not np.issubdtype(np_type, np.floating):
                if np_type.nbytes == np_dtype.nbytes:
                    int_type_substitution = np_type
        input_matrix = np.random.randint(
            low=np.iinfo(int_type_substitution).min,
            high=np.iinfo(int_type_substitution).max,
            size=matrix_shape,
            dtype=int_type_substitution,
        ).astype(np_dtype)
    else:
        input_matrix = np.random.randint(
            low=np.iinfo(np_dtype).min,
            high=np.iinfo(np_dtype).max,
            size=matrix_shape,
            dtype=np_dtype,
        )
    expected_output_matrix = np.transpose(input_matrix)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="transpose",
        target_device=launch.target,
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_matrix],
            expected_outputs=[expected_output_matrix],
        )
    )
