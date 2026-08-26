# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Transpose an [M, K] matrix through a channel, on air.api.

    chan_in.put(A)                        # L3 -> channel, at launch scope
    ...
    chan_in.get(t)                        # channel -> L1, in the herd
    chan_out.put(t.transpose(1, 0))       # L1 -> channel, axes swapped
    ...
    chan_out.get(B)                       # channel -> L3, at launch scope

The third of the three transposes in this directory and the only one that uses
channels. ``dma/transpose.py`` moves the tile with two ``ops.load``/``ops.store``
transfers; this one hands it to a named channel instead, so the herd never names
the L3 tensors at all -- a channel is a module-level symbol, not an operand, and
that is the whole point of it.

The transpose itself is identical to the DMA variant: a property of the access
pattern the *put* walks, not a computation. ``t.transpose(1, 0)`` derives sizes
[k, m] and strides [1, k] from the permutation, where the predecessor passed
``sizes=[1, k, m], strides=[1, 1, k]`` to ``ChannelPut`` as literal lists. Same
descriptor, minus the leading unit axis the DSL has no reason to write.

**The puts and gets sit at launch scope, around the segment rather than inside
it**, which is how the predecessor is written and is load-bearing: the put has
to happen before the segment runs and the get after it, because they are the
two ends of the stream the segment consumes and produces.

The L1 buffer is ``[m, k]`` rather than the predecessor's flat ``[k * m]``. It
holds the same bytes; the shape is now the shape of the thing in it, which is
what lets the permutation be written at all.

Both element types the Makefile exercises still work -- ``uint32`` and
``float32``. Nothing here is arithmetic, so the unsigned type is carried by the
transfers without being computed on, which is the one place air.api accepts one.

``--target`` is new and defaults to detecting the installed part.
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

    chan_in = air.channel("ChanIn")
    chan_out = air.channel("ChanOut")

    A = air.tensor([m, k], dt)
    B = air.tensor([k, m], dt)

    with air.launch(name="transpose") as launch:

        @launch.body
        def _():
            # Into the stream, before the segment that consumes it.
            chan_in.put(A)

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="herd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            t = air.alloc([m, k], dt, scope=h.private())
                            chan_in.get(t)
                            # The whole example: the same tile, walked with its
                            # axes swapped on the way out. A view, not a copy.
                            chan_out.put(t.transpose(1, 0))

            # Out of the stream, after the segment that filled it.
            chan_out.get(B)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the matrix_scalar_add/single_core_channel example",
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
