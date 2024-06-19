//===- prune_redundant_edges.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// Prune redundant dependency edges
// CHECK: %[[EVENT0:.*]] = air.segment async
// CHECK: %[[EVENT1:.*]] = air.herd async
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd async{{.*}}id = 3
// CHECK-NEXT: %[[EVENT3:.*]] = air.execute [%[[EVENT2]]]
// CHECK: %[[EVENT4:.*]] = air.execute [%[[EVENT1]]]
// CHECK: %[[EVENT5:.*]] = air.execute [%[[EVENT0]]]

module {
  func.func @foo(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<1024xi32> attributes {id = 3 : i32} {
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %asyncToken, %valOut = air.execute -> (memref<512xi32>){
        %3 = memref.alloc() : memref<512xi32>
        air.execute_terminator %3 : memref<512xi32>
      } {id = 1 : i32}
      %1 = air.dma_memcpy_nd async [%asyncToken] (%valOut[] [] [], %arg5[%c0_0] [%c0_0] [%c0_0]) {id = 1 : i32} : (memref<512xi32>, memref<1024xi32>)
      %2 = air.segment async [%1]  unroll(%arg6, %arg7) in (%arg8=%c1_1, %arg9=%c1_1) args(%arg10=%valOut) : memref<512xi32> attributes {id = 2 : i32} {
        %c0_3 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        %asyncToken_5, %valOut_6 = air.execute -> (memref<256xi32, 1>) {
          %5 = memref.alloc() : memref<256xi32, 1>
          air.execute_terminator %5 : memref<256xi32, 1>
        } {id = 2 : i32}
        %3 = air.dma_memcpy_nd async [%asyncToken_5] (%valOut_6[] [] [], %arg10[%c0_3] [%c0_3] [%c0_3]) {id = 2 : i32} : (memref<256xi32, 1>, memref<512xi32>)
        %4 = air.herd async [%3]  tile (%arg11, %arg12) in (%arg13=%c1_4, %arg14=%c1_4) args(%arg15=%valOut_6) : memref<256xi32, 1> attributes {id = 1 : i32} {
          %c0_8 = arith.constant 0 : index
          %asyncToken_9, %valOut_10 = air.execute -> (memref<128xi32, 2>){
            %6 = memref.alloc() : memref<128xi32, 2>
            air.execute_terminator %6 : memref<128xi32, 2>
          } {id = 3 : i32}
          %5 = air.dma_memcpy_nd async [%asyncToken_9] (%valOut_10[] [] [], %arg15[%c0_8] [%c0_8] [%c0_8]) {id = 3 : i32} : (memref<128xi32, 2>, memref<256xi32, 1>)
          %asyncToken_11 = air.execute [%5, %asyncToken_9] {
            memref.dealloc %valOut_10 : memref<128xi32, 2>
            air.execute_terminator
          } {id = 4 : i32}
        }
        %asyncToken_7 = air.execute [%4, %asyncToken_5, %3] {
          memref.dealloc %valOut_6 : memref<256xi32, 1>
          air.execute_terminator
        } {id = 5 : i32}
      }
      %asyncToken_2 = air.execute [%2, %1] {
        memref.dealloc %valOut : memref<512xi32>
        air.execute_terminator
      } {id = 6 : i32}
    }
    return
  }
}