# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.backend.xrt_runner import XRTRunner
from shim_dma_2d import *

INOUT_DATATYPE = np.int32


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the shim dma 2d example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--peano_path",
        default=None,
        type=str,
        help="Use peano to compile",
    )
    args = parser.parse_args()

    mlir_module = build_module()

    input_a = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
    output_b = np.zeros(shape=IMAGE_SIZE, dtype=INOUT_DATATYPE)
    for h in range(TILE_HEIGHT):
        for w in range(TILE_WIDTH):
            output_b[h, w] = input_a[h, w]

    runner = XRTRunner(verbose=args.verbose, peano_path=args.peano_path)
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))
