# run.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import os
import numpy as np
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
import filelock
from ml_dtypes import bfloat16

# Get the directory containing this script
script_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    prog="run.py",
    description="Builds, runs, and tests the softmax example",
)
parser.add_argument(
    "--input-mlir",
    type=str,
    dest="input_mlir",
    default=os.path.join(script_dir, "input_ir/original.mlir"),
    help="Input MLIR file path (default: input_ir/original.mlir)",
)
parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform.mlir",
    help="Transform script path",
)
parser.add_argument(
    "--M",
    type=int,
    dest="M",
    default=256,
    help="M (parallel) dimension size",
)
parser.add_argument(
    "--N",
    type=int,
    dest="N",
    default=256,
    help="N (reduction) dimension size",
)
parser.add_argument(
    "--compile-only",
    action="store_true",
    help="Only compile to xclbin without running validation (for profiling)",
)
parser.add_argument(
    "--debug-ir",
    type=str,
    dest="debug_ir",
    default=None,
    metavar="OUTPUT_FILE",
    help="Print the transformed IR to the specified file and exit (for debugging)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose mode to show all passes during compilation",
)
parser.add_argument(
    "--debug-aircc",
    action="store_true",
    dest="debug_aircc",
    help="Enable debug mode in aircc to emit IR after each individual pass for fine-grained inspection",
)
args = parser.parse_args()


def softmax(x, axis=-1):
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    # Resolve input MLIR path - if not absolute and not found, try script directory
    input_mlir_path = args.input_mlir
    if not os.path.isabs(input_mlir_path) and not os.path.exists(input_mlir_path):
        input_mlir_path = os.path.join(script_dir, input_mlir_path)

    # Load the input MLIR from file
    print(f"Loading input MLIR from: {input_mlir_path}")
    with open(input_mlir_path, "r") as f:
        air_tiled_ir_string = f.read()
    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Tiling
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-resolve-tensor-opoperand-conflicts",
                "air-override-memref-memory-space{scope=func memory-space=1}",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## Tiling
    ################################################

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

    ###############################################
    # Binding scf.paralell to air hierarchies
    ###############################################
    M, N = args.M, args.N
    input_size = (M, N)
    tile_size = (4, N)  # herd size = 4 (4 AIE cores)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},1}})",
                "air-par-to-launch{depth=-1 has-air-segment=true}",
                "air-copy-to-dma",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ###############################################
    # Run compile and load
    ###############################################

    if args.compile_only:
        # Compile-only mode: generate xclbin and instruction binary without validation
        print("Compile-only mode: generating xclbin and instruction binary...")
        backend = XRTBackend(
            omit_while_true_loop=False,
            verbose=args.verbose,
            debug_ir=args.debug_aircc,
            runtime_loop_tiling_sizes=[],
        )
        module_function = backend.compile(air_module)
        backend.unload()
        print("Compilation complete. Generated files:")
        print("  - air.xclbin")
        print("  - air.insts.bin")
        print("Run profiling with: ./test.exe")
        exit(0)
    else:
        # Normal mode: compile and run validation
        input_type = bfloat16
        # Generate random input in range [-512, 512]
        A = (np.random.rand(M, N) * 1024 - 512).astype(
            input_type
        )  # Shape [M, N], range [-512, 512]
        C = softmax(A).astype(input_type)

        ###### Compile and test
        runner = XRTRunner(
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[],  # No tiling = single large DMA transfer
            verbose=args.verbose,
            debug_ir=args.debug_aircc,
        )
        exit(
            runner.run_test(
                air_module,
                inputs=[A],
                expected_outputs=[C],
                rtol=0.04,  # 4% relative tolerance (matches mlir-aie reference)
                atol=0.001,  # Absolute tolerance (matches mlir-aie reference)
            )
        )
