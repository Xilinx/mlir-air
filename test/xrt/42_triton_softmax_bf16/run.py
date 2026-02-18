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
    default=os.path.join(script_dir, "input_ir/initial/4x1024_1d.mlir"),
    help="Input MLIR file path (default: input_ir/initial/4x1024_1d.mlir)",
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
    # default=256,
    default=1024,
    help="N (reduction) dimension size",
)
parser.add_argument(
    "--tile-rows",
    type=int,
    dest="tile_rows",
    default=4,
    help="Number of rows per herd tile (default: 4, use 16 for 4x4 herd)",
)
parser.add_argument(
    "--herd-shape",
    type=str,
    dest="herd_shape",
    default="1x4",
    choices=["1x4", "4x4"],
    help="Herd shape: '1x4' for 1D (4 cores), '4x4' for 2D (16 cores)",
)
parser.add_argument(
    "--compile-only",
    action="store_true",
    help="Only compile to xclbin without running validation (for profiling)",
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
parser.add_argument(
    "--pre-transformed-ir",
    type=str,
    dest="pre_transformed_ir",
    default=None,
    metavar="MLIR_FILE",
    help="Load pre-transformed IR directly, skipping the transform script (for testing optimized IR)",
)
parser.add_argument(
    "--output-format",
    type=str,
    dest="output_format",
    default="xclbin",
    choices=["elf", "xclbin"],
    help="Output format: 'xclbin' (default) or 'elf'",
)
args = parser.parse_args()


def softmax(x, axis=-1):
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


with air.ir.Context() as ctx, Location.unknown():

    # Get M, N dimensions (used for validation)
    M, N = args.M, args.N

    # Check if using pre-transformed IR (skip initial transform steps)
    if args.pre_transformed_ir:
        ################################################
        ## Load Pre-Transformed IR Directly
        ################################################
        pre_transformed_path = args.pre_transformed_ir
        if not os.path.isabs(pre_transformed_path) and not os.path.exists(
            pre_transformed_path
        ):
            pre_transformed_path = os.path.join(script_dir, pre_transformed_path)

        print(f"Loading pre-transformed IR from: {pre_transformed_path}")
        with open(pre_transformed_path, "r") as f:
            pre_transformed_ir_string = f.read()
        air_module = Module.parse(pre_transformed_ir_string)
        print("Skipping transform script - using pre-transformed IR")

        ###############################################
        # Binding scf.parallel to air hierarchies
        ###############################################
        # Parse herd shape (e.g., "1x4" -> (1, 4), "4x4" -> (4, 4))
        herd_cols, herd_rows = map(int, args.herd_shape.split("x"))
        total_cores = herd_cols * herd_rows

        if args.herd_shape == "4x4":
            # 2D mode: 4×4 herd (16 cores)
            # Input is 256×1024, each of 16 cores handles 1 row (256/16=16 rows total per launch)
            # loop-bounds = (M/16, 1, 1) for outer loop
            # But the transform creates scf.forall with 4×4 iteration, so we just need outer wrapping
            launch_rows = M // total_cores
            launch_size = (launch_rows, 1, 1)
            print(f"Herd configuration: 4×4 (16 cores), launch size: {launch_size}")
        else:
            # 1D mode: 1×4 herd (4 cores)
            input_size = (M, N)
            tile_size = (args.tile_rows, N)
            launch_size = (M // args.tile_rows, 1, 1)
            print(
                f"Herd configuration: 1×{args.tile_rows} ({args.tile_rows} cores), launch size: {launch_size}"
            )

        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
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
    else:
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

        ###############################################
        # Binding scf.parallel to air hierarchies
        ###############################################
        # Parse herd shape (e.g., "1x4" -> (1, 4), "4x4" -> (4, 4))
        herd_cols, herd_rows = map(int, args.herd_shape.split("x"))
        total_cores = herd_cols * herd_rows

        if args.herd_shape == "4x4":
            # 2D mode: 4×4 herd (16 cores)
            # Input is 256×1024, each of 16 cores handles 1 row (256/16=16 rows total per launch)
            # loop-bounds = (M/16, 1, 1) for outer loop
            # But the transform creates scf.forall with 4×4 iteration, so we just need outer wrapping
            launch_rows = M // total_cores
            launch_size = (launch_rows, 1, 1)
            print(f"Herd configuration: 4×4 (16 cores), launch size: {launch_size}")
        else:
            # 1D mode: 1×4 herd (4 cores)
            launch_size = (M // args.tile_rows, 1, 1)
            print(
                f"Herd configuration: 1×{args.tile_rows} ({args.tile_rows} cores), launch size: {launch_size}"
            )

        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
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
            output_format=args.output_format,
            instance_name="softmax_kernel",
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
            output_format=args.output_format,
            instance_name="softmax_kernel",
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
