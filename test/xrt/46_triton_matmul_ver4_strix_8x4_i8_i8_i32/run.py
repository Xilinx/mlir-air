# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import numpy as np
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
import filelock
from ml_dtypes import bfloat16 as bfloat16_t

parser = argparse.ArgumentParser(
    prog="run.py",
    description="Builds, runs, and tests the matmul example",
)
parser.add_argument(
    "--input-ir",
    type=str,
    dest="input_ir",
    default="asm_src.mlir",
    help="Input IR file path",
)

parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform.mlir",
    help="Transform script path",
)
parser.add_argument(
    "--compile-only",
    action="store_true",
    help="Only compile without running validation (for profiling)",
)
parser.add_argument(
    "--output-format",
    type=str,
    dest="output_format",
    default="xclbin",
    choices=["elf", "xclbin"],
    help="Output format: 'xclbin' (default) or 'elf'",
)
parser.add_argument(
    "--debug-ir",
    type=str,
    dest="debug_ir",
    default=None,
    metavar="OUTPUT_FILE",
    help="Print the transformed IR to the specified file and exit (for debugging)",
)
args = parser.parse_args()

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    # read the input IR (Triton Shared MLIR) from a file
    with open(args.input_ir, "r") as f:
        air_tiled_ir_string = f.read()

    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Tiling
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-override-memref-memory-space{scope=func memory-space=1}",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    # Load the MLIR transform IR from an external file
    with open(args.transform_script, "r") as f:
        transform_ir_string = f.read()
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, air_module)

    # Print the IR for debugging and exit if --debug-ir is specified
    if args.debug_ir:
        import os

        output_file = args.debug_ir
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            f.write(str(air_module))
        print(f"Transformed IR written to {output_file}")
        exit(0)

    ################################################
    ## Binding scf.parallel to air hierarchies
    ################################################
    M, N, K = 1024, 1024, 1024
    input_size = (M, N, K)
    tile_size = (512, 256, K)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
                "canonicalize",
                "cse",
                "air-copy-to-dma",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ###############################################
    # Run compile and load
    ###############################################

    # Determine output file extension based on format
    output_ext = "elf" if args.output_format == "elf" else "xclbin"

    if args.compile_only:
        # Compile-only mode: generate binary without validation
        print(f"Compile-only mode: generating {output_ext} binary...")
        backend = XRTBackend(
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="bare_matmul",
        )
        module_function = backend.compile(air_module)
        backend.unload()
        print("Compilation complete. Generated files:")
        print(f"  - air.{output_ext}")
        if args.output_format == "xclbin":
            print("  - air.insts.bin")
        print("Run profiling with: ./test.exe")
        exit(0)
    else:
        # Normal mode: compile and run validation

        input_type = np.int8
        output_type = np.int32
        A = np.random.randint(
            low=0, high=8, size=(M, K), dtype=input_type
        )  # Shape [M, K]
        B = np.random.randint(
            low=0, high=8, size=(K, N), dtype=input_type
        )  # Shape [K, N]

        C = np.matmul(A.astype(output_type), B.astype(output_type)).astype(
            output_type
        )  # Shape [M, N]

        runner = XRTRunner(
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="bare_matmul",
            # verbose=True,
        )
        exit(
            runner.run_test(
                air_module,
                inputs=[A, B],
                expected_outputs=[C],
                # rtol=1e-1,
            )
        )
