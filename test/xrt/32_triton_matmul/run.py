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
        %c128 = arith.constant 128 : index
        %c64_i32 = arith.constant 64 : i32
        %0 = arith.muli %arg6, %c64_i32 : i32
        %1 = arith.index_cast %0 : i32 to index
        %2 = arith.muli %arg7, %c64_i32 : i32
        %3 = arith.index_cast %2 : i32 to index
        %4 = arith.muli %1, %c128 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [64, 128], strides: [128, 1] : memref<*xf32> to memref<64x128xf32, strided<[128, 1], offset: ?>>
        %alloc = memref.alloc() : memref<64x128xf32, 1>
        memref.copy %reinterpret_cast, %alloc : memref<64x128xf32, strided<[128, 1], offset: ?>> to memref<64x128xf32, 1>
        %5 = bufferization.to_tensor %alloc restrict writable : memref<64x128xf32, 1> to tensor<64x128xf32>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [128, 64], strides: [128, 1] : memref<*xf32> to memref<128x64xf32, strided<[128, 1], offset: ?>>
        %alloc_1 = memref.alloc() : memref<128x64xf32, 1>
        memref.copy %reinterpret_cast_0, %alloc_1 : memref<128x64xf32, strided<[128, 1], offset: ?>> to memref<128x64xf32, 1>
        %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<128x64xf32, 1> to tensor<128x64xf32>
        %7 = tensor.empty() : tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %9 = linalg.matmul ins(%5, %6 : tensor<64x128xf32>, tensor<128x64xf32>) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %10 = arith.addi %4, %3 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [64, 64], strides: [128, 1] : memref<*xf32> to memref<64x64xf32, strided<[128, 1], offset: ?>>
        bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<64x64xf32>, memref<64x64xf32, strided<[128, 1], offset: ?>>) -> ()
        return
      }
    }
    """
    air_module = Module.parse(air_tiled_ir_string)

    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################
    M, N, K = 128, 128, 128
    input_size = (M, N, K)
    tile_size = (64, 64, 128)
    launch_size = tuple(i // t for i, t in zip(input_size, tile_size))

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "one-shot-bufferize",
                f"func.func(air-wrap-func-with-parallel{{loop-bounds={launch_size[0]},{launch_size[1]},{launch_size[2]}}})",
                "air-override-memref-memory-space{scope=func memory-space=1}",
                "air-par-to-launch{depth=0 has-air-segment=true}",
            ]
        )
        + ")"
    )
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    ################################################
    ## Tiling
    ################################################

    transform_ir_string = """
    transform.with_pdl_patterns {
    ^bb0(%arg0: !pdl.operation):
        transform.sequence %arg0 : !pdl.operation failures(propagate) {
        ^bb1(%arg1: !pdl.operation):
            %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
            %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation
            %matmul_1, %loop = transform.air.linalg_tile %matmul [32, 32, 0]
            %fill_1 = transform.air.fuse_into_containing_op %fill into %loop
            transform.air.linalg_promote %fill_1 {"operands_to_promote"=[1], "memory_space"="L1"}
            transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[2], "memory_space"="L1"}
            %matmul_2, %reduction_loop = transform.air.linalg_tile %matmul_1 [0, 0, 32]
            transform.air.linalg_promote %matmul_2 {"operands_to_promote"=[0,1], "memory_space"="L1"}
            %scffor = transform.loop.forall_to_for %reduction_loop  : (!pdl.operation) -> !pdl.operation
        }
    }
    """
    transform_ir = Module.parse(transform_ir_string)
    run_transform(transform_ir, air_module)

    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "air-par-to-herd{depth=-1}",
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
