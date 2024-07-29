# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import numpy as np
from air.backend.xrt_runner import XRTRunner

from segment_alloc import *

INOUT_DATATYPE = np.uint32


def main(verbose=False, experimental_passes=False):
    mlir_module = build_module()

    input_a = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
    output_b = np.zeros(shape=IMAGE_SIZE, dtype=INOUT_DATATYPE)
    for h in range(TILE_HEIGHT):
        for w in range(TILE_WIDTH):
            output_b[h, w] = input_a[h, w]

    runner = XRTRunner(verbose=verbose, experimental_passes=experimental_passes)
    exit(runner.run_test(mlir_module, inputs=[input_a], expected_outputs=[output_b]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the segment_alloc example",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()
    main(experimental_passes=True, verbose=args.verbose)
