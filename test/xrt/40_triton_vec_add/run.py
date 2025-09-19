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
    #map = affine_map<(d0, d1) -> (d0, d1)>
    module {
      func.func @vecadd(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = arith.muli %arg6, %c256_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [256, 1], strides: [1, 1] : memref<*xbf16> to memref<256x1xbf16, strided<[1, 1], offset: ?>>
        %alloc = memref.alloc() : memref<256x1xbf16>
        memref.copy %reinterpret_cast, %alloc : memref<256x1xbf16, strided<[1, 1], offset: ?>> to memref<256x1xbf16>
        %2 = bufferization.to_tensor %alloc restrict writable : memref<256x1xbf16> to tensor<256x1xbf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [256, 1], strides: [1, 1] : memref<*xbf16> to memref<256x1xbf16, strided<[1, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<256x1xbf16>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<256x1xbf16, strided<[1, 1], offset: ?>> to memref<256x1xbf16>
        %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<256x1xbf16> to tensor<256x1xbf16>
        %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<256x1xbf16>, tensor<256x1xbf16>) outs(%2 : tensor<256x1xbf16>) {
        ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
          %5 = arith.addf %in, %in_3 : bf16
          linalg.yield %5 : bf16
        } -> tensor<256x1xbf16>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [256, 1], strides: [1, 1] : memref<*xbf16> to memref<256x1xbf16, strided<[1, 1], offset: ?>>
        bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<256x1xbf16>, memref<256x1xbf16, strided<[1, 1], offset: ?>>) -> ()
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
                "air-resolve-tensor-opoperand-conflicts",
                "air-override-memref-memory-space{scope=func memory-space=1}",
                "linalg-fuse-elementwise-ops",
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
    ## Binding scf.paralell to air hierarchies
    ################################################
    M, N, K = 1024, 1, 1
    input_size = (M, N, K)
    tile_size = (256, 1, K)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
                "air-par-to-launch{depth=0 has-air-segment=true}",
                "canonicalize",
                "cse",
                "air-par-to-herd{depth=-1}",
                "air-copy-to-dma",
                "func.func(air-herd-vectorize)",
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

    input_type = bfloat16
    output_type = bfloat16
    A = np.random.rand(
        M,
    ).astype(
        input_type
    )  # Shape [M, K]
    B = np.random.rand(
        M,
    ).astype(
        input_type
    )  # Shape [K, N]
    C = np.add(A, B).astype(output_type)  # Shape [M, N]

    ###### Compile and test
    runner = XRTRunner(
        omit_while_true_loop=False,
        air_loop_fusion=True,
        verbose=True,
    )
    exit(
        runner.run_test(
            air_module,
            inputs=[A, B],
            expected_outputs=[C],
            rtol=1e-2,
        )
    )
