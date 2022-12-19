//===- canonicalize_hierarchy_args.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize | FileCheck %s

// Prune block arguments for air.herd and air.partition
// CHECK: %[[EVENT0:.*]] = air.launch async{{.*}}args(%[[VALUE1:.*]]=%[[VALUE0:.*]])
// CHECK: %[[EVENT1:.*]] = air.partition async{{.*}}args(%[[VALUE2:.*]]=%[[VALUE1]])
// CHECK: %[[EVENT2:.*]] = air.herd async{{.*}}args(%[[VALUE3:.*]]=%[[VALUE2]])

module {
  func.func @foo(%arg0: memref<256xi32, 2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<256xi32, 2> attributes {id = 13 : i32} {
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %asyncToken, %valOut = air.execute -> (memref<512xi32, 3>) {
        %3 = memref.alloc() : memref<512xi32, 3>
        air.execute_terminator %3 : memref<512xi32, 3>
      } {id = 1 : i32}
      %2 = air.partition async  unroll(%arg6, %arg7) in (%arg8=%c1_1, %arg9=%c1_1) args(%arg10=%arg5, %newarg1=%valOut) : memref<256xi32, 2>, memref<512xi32, 3> attributes {id = 82 : i32} {
        %c0_3 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        %4 = air.herd async  tile (%arg11, %arg12) in (%arg13=%c1_4, %arg14=%c1_4) args(%arg15=%arg10, %newarg2=%newarg1) : memref<256xi32, 2>, memref<512xi32, 3> attributes {id = 15 : i32} {
          %c0_8 = arith.constant 0 : index
          %asyncToken_9, %valOut_10 = air.execute -> (memref<128xi32, 1>) {
            %6 = memref.alloc() : memref<128xi32, 1>
            air.execute_terminator %6 : memref<128xi32, 1>
          } {id = 3 : i32}
          %5 = air.dma_memcpy_nd async [%asyncToken_9] (%valOut_10[] [] [], %arg15[%c0_8] [%c0_8] [%c0_8]) {id = 43 : i32} : (memref<128xi32, 1>, memref<256xi32, 2>)
          air.herd_terminator
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return
  }
}