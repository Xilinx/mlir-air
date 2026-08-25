# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""bf16 matrix transpose using an external kernel, on air.api.

    transpose_bf16 = air.extern("transpose_bf16", link_with="transpose.o")
    ...
    air.ops.load(l1_in, A)
    transpose_bf16(l1_in, l1_out)
    air.ops.store(l1_out, B)

**Why this one has a kernel and its sibling does not.** ``dma/transpose.py``
transposes by walking the tile with its axes swapped -- a DMA descriptor with
sizes [k, m] and strides [1, k] -- and needs no compute at all. That route is
closed here: on AIE the innermost DMA stride must be 1 for data narrower than
32 bits, so a bf16 transpose cannot be expressed as an access pattern. The
matrix is DMAed into L1 contiguously instead and a scalar C++ kernel does the
transpose in place of the descriptor.

So the two variants are not duplicates. They are the two halves of one fact
about the hardware, and the element type is what selects between them.

Everything here is flat: the L3 tensors are ``[m * k]`` and ``[k * m]`` and the
L1 tiles are ``[m * k]``, because the kernel takes raw buffers and computes the
indices itself from its ``DIM_M``/``DIM_N`` defines. That is the predecessor's
shape too, kept rather than tidied -- the flatness is part of the contract with
transpose.cc, not an artifact.

``air.extern`` carries ``link_with="transpose.o"``, which stamps the attribute
on both the declaration and the herd; the Makefile compiles transpose.cc to
exactly that name in the build directory.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

INOUT_DATATYPE = bfloat16

# The object the Makefile compiles transpose.cc into, in the build directory.
EXTERN_OBJECT = "transpose.o"


def build_module(m, k):
    # The kernel is compiled with -DDIM_M/-DDIM_N, so it knows the shape; the
    # buffers it is handed are flat.
    transpose_bf16 = air.extern("transpose_bf16", link_with=EXTERN_OBJECT)

    A = air.tensor([m * k], bf16)
    B = air.tensor([k * m], bf16)

    with air.launch(name="transpose") as launch:

        @launch.body
        def _():
            with air.segment(name="seg"):
                with air.herd(
                    [range(1)], name="herd", shape=(1,), link_with=EXTERN_OBJECT
                ) as h:

                    @h.body
                    def _(tx):
                        l1_in = air.alloc([m * k], bf16, scope=h.private())
                        l1_out = air.alloc([m * k], bf16, scope=h.private())

                        air.ops.load(l1_in, A)
                        transpose_bf16(l1_in, l1_out)
                        air.ops.store(l1_out, B)

    return launch


if __name__ == "__main__":
    M = 64
    K = 32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the bf16 transpose example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-m", type=int, default=M, help="Matrix rows")
    parser.add_argument("-k", type=int, default=K, help="Matrix columns")
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

    launch = build_module(args.m, args.k)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_matrix = np.random.uniform(-1.0, 1.0, (args.m, args.k)).astype(INOUT_DATATYPE)
    expected_output = np.transpose(input_matrix)

    if args.compile_mode == "compile-and-run":
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
                inputs=[input_matrix.reshape(-1)],
                expected_outputs=[expected_output.reshape(-1)],
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            output_format=args.output_format,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
