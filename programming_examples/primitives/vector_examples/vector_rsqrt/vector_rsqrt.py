# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reciprocal square root, on air.api, in the three spellings this example ships.

    c[:] = air.ops.rsqrt(a[:])                              # --version 1
    c[:] = air.ops.rsqrt(a[:])   with vector=0              # --version 2
    c[:] = ops.cast(ops.rsqrt(ops.cast(a[:], f32)), bf16)   # --version 3

The predecessor was three near-identical files (``vector_rsqrt_v{1,2,3}.py``)
differing in one line of compute and in which element type they picked. They are
one file here with a ``--version`` flag, because on air.api that one line *is*
the difference: everything around it -- the tiling, the herd, the DMA, the
verification -- was already identical, and was duplicated three times.

The three still mean different things, and each keeps its own lit:

1. **vector rsqrt in the natural type for the generation.** bf16 on npu1,
   f32 on npu2, as the predecessor chose.
2. **scalar rsqrt, f32, npu2 only.** ``vector=0`` sends the emitter down its
   scalar path, which is the point of this variant: it is the one that checks
   ``math.rsqrt`` legalizes on a plain f32 rather than on a vector. The
   predecessor refused npu1 outright and so does this.
3. **bf16 in and out, computed in f32.** Two ``ops.cast`` nodes around the
   rsqrt. Everything below a cast is read and computed in the source type, so
   this reads bf16, widens, computes f32, narrows, and stores bf16 -- what the
   predecessor spelled with explicit ``arith.extf``/``truncf`` on the vectors.

**The object file.** On npu1 ``math.rsqrt`` is not an instruction: the AIE
lowering turns it into a call to ``getRsqrtBf16`` from ``extern_func.o``, built
from the ``extern_func.cc`` beside this file. The herd therefore has to carry
``link_with``, and nothing at trace time emits a call to hang it off, which is
what ``air.herd(link_with=...)`` is for. On npu2 rsqrt is native and no object is
linked -- linking an aie2 object into an aie2p build is not a no-op.

That branch keys off the *resolved* target rather than ``--arch``, so
``--target auto`` is the single source of truth for the generation. The
predecessor keyed both the object and the element type off ``--arch`` and had no
``--target`` at all, which left the two able to disagree.

The herd is [NUM_TILES, 1] rather than [1, NUM_TILES] -- a 1-D air.api herd is
laid out along x, the orientation that places on both generations -- and the
strip-mine is the DSL's rather than a hand-built AffineMap.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16, f32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

NUM_TILES = 2
VECTOR_SIZE = 16

# The object the npu1 lowering of math.rsqrt calls into. The Makefile compiles
# extern_func.cc to exactly this name in the build directory.
EXTERN_OBJECT = "extern_func.o"


def needs_extern_object(target):
    """npu1 (aie2) has no rsqrt instruction and calls getRsqrtBf16 for it."""
    return target == "npu1"


def element_type(version, target):
    """The element type of the L3 tensors, per version and generation.

    Version 3 is defined by its buffers being bf16 while the arithmetic is f32,
    so it does not vary with the generation. Versions 1 and 2 compute in the
    buffer type, and version 1 follows the predecessor in taking whichever type
    the generation does rsqrt in natively.
    """
    if version == 3:
        return bf16
    if version == 2:
        return f32
    return bf16 if target == "npu1" else f32


def build_module(n, tile_n, version=1, target="auto"):
    assert n % (tile_n * NUM_TILES) == 0
    resolved = air.resolve_target(target)
    if version == 2 and resolved == "npu1":
        raise ValueError(
            "version 2 (scalar rsqrt) is npu2-only: it computes in f32, and "
            "npu1 does rsqrt through a bf16 call into extern_func.o. Use "
            "--version 1 there, as the predecessor did."
        )
    dt = element_type(version, resolved)
    # Version 2 is the scalar variant; vector=0 is how that is asked for.
    width = 0 if version == 2 else VECTOR_SIZE
    assert width == 0 or tile_n % width == 0

    A = air.tensor([n], dt)
    C = air.tensor([n], dt)

    with air.launch(name="vector_rsqrt") as launch:

        @launch.body
        def _():
            with air.herd(
                [range(0, n, tile_n)],
                name="herd_0",
                shape=(NUM_TILES,),
                link_with=EXTERN_OBJECT if needs_extern_object(resolved) else None,
            ) as h:

                @h.body
                def _(tx):
                    i0 = tx * tile_n
                    a = air.alloc([tile_n], dt, scope=h.private(), vector=width)
                    c = air.alloc([tile_n], dt, scope=h.private(), vector=width)

                    air.ops.load(a, A[i0 : i0 + tile_n])

                    if version == 3:
                        c[:] = air.ops.cast(
                            air.ops.rsqrt(air.ops.cast(a[:], f32)), bf16
                        )
                    else:
                        c[:] = air.ops.rsqrt(a[:])

                    air.ops.store(c, C[i0 : i0 + tile_n])

    return launch


if __name__ == "__main__":
    N = 512
    TILE_N = 64

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the vector_rsqrt example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--version",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="1: vector rsqrt in the generation's native type. 2: scalar f32 "
        "rsqrt (npu2 only). 3: bf16 buffers, f32 arithmetic",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["aie2", "aie2p"],
        default="aie2",
        help="Accepted for Makefile compatibility and otherwise unused: the "
        "generation comes from --target, and both the element type and whether "
        "extern_func.o is linked follow from that. Inherited from the "
        "predecessor, where this flag chose them",
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

    launch = build_module(args.n, args.tile_n, args.version, args.target)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np_dtype = {bf16: bfloat16, f32: np.float32}[
        element_type(args.version, launch.target)
    ]
    # Positive inputs only -- rsqrt of a non-positive value is not a number --
    # and [0.1, 3.0] stays well inside the bfloat16 range.
    np.random.seed(10)
    input_a = np.abs(np.random.uniform(0.1, 3.0, args.n)).astype(np_dtype)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])
        sampled_values = np.array(
            [1.0 / np.sqrt(input_a[i]) for i in sampled_indices[0]],
            dtype=np_dtype,
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
            instance_name="vector_rsqrt",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
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
        module_function = backend.compile(mlir_module)
        backend.unload()
