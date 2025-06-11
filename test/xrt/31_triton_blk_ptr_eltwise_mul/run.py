# run.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.ir import *
import air.passmanager
import filelock

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    module {
      func.func @kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
        %c64_i32 = arith.constant 64 : i32
        %c32_i32 = arith.constant 32 : i32
        %c64 = arith.constant 64 : index
        %0 = arith.muli %arg6, %c64_i32 : i32
        %1 = arith.muli %arg7, %c32_i32 : i32
        %2 = arith.index_cast %0 : i32 to index
        %3 = arith.muli %2, %c64 : index
        %4 = arith.index_cast %1 : i32 to index
        %5 = arith.addi %3, %4 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [16, 16, 4, 2], strides: [256, 2, 64, 1] : memref<*xf32> to memref<16x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x16x4x2xf32>
        memref.copy %reinterpret_cast, %alloc : memref<16x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>> to memref<16x16x4x2xf32>
        %6 = bufferization.to_tensor %alloc restrict writable : memref<16x16x4x2xf32> to tensor<16x16x4x2xf32>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [16, 16, 4, 2], strides: [256, 2, 64, 1] : memref<*xf32> to memref<16x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<16x16x4x2xf32>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<16x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>> to memref<16x16x4x2xf32>
        %7 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x16x4x2xf32> to tensor<16x16x4x2xf32>
        %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6, %7 : tensor<16x16x4x2xf32>, tensor<16x16x4x2xf32>) outs(%6 : tensor<16x16x4x2xf32>) {
        ^bb0(%in: f32, %in_3: f32, %out: f32):
          %9 = arith.mulf %in, %in_3 : f32
          linalg.yield %9 : f32
        } -> tensor<16x16x4x2xf32>
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [16, 16, 4, 2], strides: [256, 2, 64, 1] : memref<*xf32> to memref<16x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>>
        bufferization.materialize_in_destination %8 in writable %reinterpret_cast_2 : (tensor<16x16x4x2xf32>, memref<16x16x4x2xf32, strided<[256, 2, 64, 1], offset: ?>>) -> ()
        return
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "func.func(air-wrap-func-with-parallel{loop-bounds=2,2,1})",
                "canonicalize",
                "cse",
                "one-shot-bufferize{copy-before-write}",
                "buffer-results-to-out-params",
                "air-par-to-herd{depth=-1}",
                "air-insert-launch-around-herd{insert-segment=false}",
                "func.func(air-force-l1-memref-in-herd)",
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

    input_size = (128, 64)
    input_type = np.float32
    inputs_a = (np.random.rand(*input_size)).reshape(input_size).astype(input_type)
    inputs_b = (np.random.rand(*input_size)).reshape(input_size).astype(input_type)
    ref = (inputs_a * inputs_b).astype(input_type)

    ###### Compile and test
    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        omit_pingpong=True,
    )
    exit(
        runner.run_test(
            air_module,
            inputs=[inputs_a, inputs_b],
            expected_outputs=[ref],
            rtol=1e-3,
        )
    )
