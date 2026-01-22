# run.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
import filelock

with air.ir.Context() as ctx, Location.unknown():

    ################################################
    ## Input SCF and Linalg IR
    ################################################

    air_tiled_ir_string = """
    module {
      func.func @bare_matmul(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
        %cst = arith.constant 0.000000e+00 : f32
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c32_i32 = arith.constant 32 : i32
        %0 = arith.muli %arg6, %c32_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg7, %c32_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %1, %c32 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [32, 32], strides: [32, 1] : memref<*xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
        %alloc = memref.alloc() : memref<32x32xf32>
        memref.copy %reinterpret_cast, %alloc : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<32x32xf32>
        %5 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf32> to tensor<32x32xf32>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [32, 32], strides: [64, 1] : memref<*xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<32x32xf32>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<32x32xf32, strided<[64, 1], offset: ?>> to memref<32x32xf32>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
        %7 = tensor.empty() : tensor<32x32xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %9 = linalg.matmul ins(%5, %6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %10 = arith.muli %1, %c64 : index
        %11 = arith.addi %10, %3 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [64, 1] : memref<*xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
        bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<32x32xf32>, memref<32x32xf32, strided<[64, 1], offset: ?>>) -> ()
        return
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################
    M, N, K = 64, 64, 32
    input_size = (M, N, K)
    tile_size = (32, 32, 32)
    herd_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "one-shot-bufferize",
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={herd_size[0]},{herd_size[1]},{herd_size[2]}}})",
                "canonicalize",
                "cse",
                "air-override-memref-memory-space{scope=func memory-space=2}",
                "air-par-to-herd{depth=-1}",
                "air-insert-launch-around-herd{insert-segment=false}",
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

    input_type = np.float32
    A = np.random.rand(M, K).astype(input_type)  # Shape [M, K]
    B = np.random.rand(K, N).astype(input_type)  # Shape [K, N]
    C = np.matmul(A, B).astype(input_type)  # Shape [M, N]

    ###### Compile and test
    runner = XRTRunner(
        omit_while_true_loop=False,
        use_lock_race_condition_fix=True,
    )
    exit(
        runner.run_test(
            air_module,
            inputs=[A, B],
            expected_outputs=[C],
            rtol=1e-3,
        )
    )
