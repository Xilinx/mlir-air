# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Passthrough a vector one chunk at a time, over plain DMA, on air.api.

The sibling ``passthrough_channel`` streams the same vector through a channel
pair. This one does it with transfers instead:

    for i in air.sequential(0, n, chunk)
        air.ops.load(tile_in, A[i : i + chunk])     L3 -> L1
        tile_out[:] = tile_in[:]
        air.ops.store(tile_out, B[i : i + chunk])   L1 -> L3

which is the difference worth having both for. A channel is a stream, so one put
of the whole vector can feed many gets of a chunk each and neither end names an
offset. A transfer is addressed, so the offset is spelled out per trip and the
shapes must agree -- and the loop variable is what supplies it.

The six element types the predecessor accepted are all still here, including the
two unsigned ones. ``uint8`` is the default and the one three of the four lits
run, so it is the case that has to keep working: an unsigned tile is *movable
but not computable* in air.api, and this kernel only ever moves and copies it,
which is exactly the line the DSL draws. Declaring ``i8`` for ``uint8`` data
instead -- as ``passthrough_channel`` still does -- would make the emitted
memref element type disagree with the array the runner is handed.

Unchanged from the raw-bindings version this replaces, except for two things:

* The chunk buffers are hoisted above the loop and reused across trips, which is
  what the loop is for; the predecessor allocated a fresh pair per trip.
* The copy is written ``tile_out[:] = tile_in[:]`` rather than a scalar loop over
  every element. For the four signless types that vectorises; for ``uint8`` and
  ``uint16`` the emitter stays scalar, because a vector copy needs a padding
  value and that padding value is an ``arith.constant``, which will not take an
  unsigned type. So the unsigned configs emit the same scalar loop the
  predecessor did, and the signless ones are strictly faster.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.backend.xrt_runner import XRTRunner

from air import api as air
from air.api import bf16, f32, i8, i16, ui8, ui16

# numpy dtype -> air.api element type. The unsigned entries are the point of the
# table: `type_mapper` maps np.uint8 onto MLIR's signful `ui8`, so declaring i8
# here would emit a kernel whose signature disagrees with its inputs.
DTYPE = {
    "uint8": (np.uint8, ui8),
    "int8": (np.int8, i8),
    "int16": (np.int16, i16),
    "uint16": (np.uint16, ui16),
    "float32": (np.float32, f32),
    "bfloat16": (bfloat16, bf16),
}
DEFAULT_DTYPE = "uint8"


def build_module(vector_size, num_subvectors, dt):
    assert vector_size % num_subvectors == 0
    chunk = vector_size // num_subvectors

    A = air.tensor([vector_size], dt)
    B = air.tensor([vector_size], dt)

    with air.launch(name="copy") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="copyherd", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            tile_in = air.alloc([chunk], dt, scope=h.private())
                            tile_out = air.alloc([chunk], dt, scope=h.private())

                            for i in air.sequential(0, num_subvectors * chunk, chunk):
                                air.ops.load(tile_in, A[i : i + chunk])
                                tile_out[:] = tile_in[:]
                                air.ops.store(tile_out, B[i : i + chunk])

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
    )
    parser.add_argument(
        "-s",
        "--vector_size",
        type=int,
        default=4096,
        help="The size (in bytes) of the data vector to passthrough",
    )
    parser.add_argument(
        "--subvector_size",
        type=int,
        default=4,
        help="The number of sub-vectors to break the vector into",
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
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "-t",
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=DTYPE.keys(),
        help="The data type to use (default: uint8)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    args = parser.parse_args()

    np_dtype, dt = DTYPE[args.dtype]
    launch = build_module(args.vector_size, args.subvector_size, dt)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(args.vector_size, dtype=np_dtype)
    output_b = np.arange(args.vector_size, dtype=np_dtype)

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="copy",
        runtime_loop_tiling_sizes=[4, 4],
    )
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
