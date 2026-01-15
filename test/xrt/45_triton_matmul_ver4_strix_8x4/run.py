# run.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
from ml_dtypes import bfloat16
import filelock

parser = argparse.ArgumentParser(
    prog="run.py",
    description="Builds, runs, and tests the matmul example",
)
parser.add_argument(
    "--transform-script",
    type=str,
    dest="transform_script",
    default="transform.mlir",
    help="Transform script path",
)
args = parser.parse_args()

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    module {
      func.func @bare_matmul(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
        %cst = arith.constant 0.000000e+00 : f32
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        %c256_i32 = arith.constant 256 : i32
        %c512_i32 = arith.constant 512 : i32
        %0 = arith.muli %arg6, %c512_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg7, %c256_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %1, %c512 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [512, 512], strides: [512, 1] : memref<*xbf16> to memref<512x512xbf16, strided<[512, 1], offset: ?>>
        %alloc = memref.alloc() : memref<512x512xbf16>
        memref.copy %reinterpret_cast, %alloc : memref<512x512xbf16, strided<[512, 1], offset: ?>> to memref<512x512xbf16>
        %5 = bufferization.to_tensor %alloc restrict writable : memref<512x512xbf16> to tensor<512x512xbf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [512, 256], strides: [1024, 1] : memref<*xbf16> to memref<512x256xbf16, strided<[1024, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<512x256xbf16>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<512x256xbf16, strided<[1024, 1], offset: ?>> to memref<512x256xbf16>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<512x256xbf16> to tensor<512x256xbf16>
        %7 = tensor.empty() : tensor<512x256xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<512x256xf32>) -> tensor<512x256xf32>
        %9 = linalg.matmul ins(%5, %6 : tensor<512x512xbf16>, tensor<512x256xbf16>) outs(%8 : tensor<512x256xf32>) -> tensor<512x256xf32>
        %10 = arith.muli %1, %c1024 : index
        %11 = arith.addi %10, %3 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [512, 256], strides: [1024, 1] : memref<*xf32> to memref<512x256xf32, strided<[1024, 1], offset: ?>>
        bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<512x256xf32>, memref<512x256xf32, strided<[1024, 1], offset: ?>>) -> ()
        return
      }
    }
    """
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

    ################################################
    ## Binding scf.parallel to air hierarchies
    ################################################
    M, N, K = 2048, 1024, 512
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

    input_type = bfloat16
    output_type = np.float32
    A = np.random.rand(M, K).astype(input_type)  # Shape [M, K]
    B = np.random.rand(K, N).astype(input_type)  # Shape [K, N]
    C = np.matmul(A, B).astype(output_type)  # Shape [M, N]

    ###### Compile and test
    runner = XRTRunner(
        omit_while_true_loop=False,
    )
    exit(
        runner.run_test(
            air_module,
            inputs=[A, B],
            expected_outputs=[C],
            rtol=1e-1,
        )
    )

    # with open("air_tiled.mlir", "w") as f:
    #     f.write(str(air_module))
