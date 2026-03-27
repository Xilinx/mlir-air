# run.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from air.backend.xrt_runner import XRTRunner, XRTBackend, type_mapper, make_air_parser, run_on_npu
from shim_dma_2d import *

INOUT_DATATYPE = np.int32


if __name__ == "__main__":
    parser = make_air_parser("Builds, runs, and tests the shim_dma_2d example")
    args = parser.parse_args()

    mlir_module = build_module()

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    input_a = np.arange(np.prod(IMAGE_SIZE), dtype=INOUT_DATATYPE).reshape(IMAGE_SIZE)
    output_b = np.zeros(shape=IMAGE_SIZE, dtype=INOUT_DATATYPE)
    for h in range(TILE_HEIGHT):
        for w in range(TILE_WIDTH):
            output_b[h, w] = input_a[h, w]

    exit(
        run_on_npu(
            args,
            mlir_module,
            inputs=[input_a],
            instance_name="copy",
            expected_outputs=[output_b],
        )
    )
