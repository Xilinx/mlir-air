//===- air_cascade_region.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=npu2" | FileCheck %s

// Each core sends or receives its own [32, 64] slab of a [2, 32, 64] buffer
// over the cascade. The slab of core 1 starts at element 2048, and all 2048 of
// its elements go, 16 at a time.

// CHECK-DAG:     %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:     %[[tile_2_4:.*]] = aie.tile(2, 4)
// CHECK:         aie.core(%[[tile_2_4]])
// CHECK:           %[[FLAT:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2]]
// CHECK:           %[[SLAB:.*]] = memref.subview %[[FLAT]][%{{.*}}] [2048] [1]
// CHECK:           scf.for %[[I:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK:             memref.subview %[[SLAB]][%[[I]]] [16] [1]
// CHECK:             aie.put_cascade
// CHECK:         aie.core(%[[tile_2_3]])
// CHECK:           %[[FLAT:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2]]
// CHECK:           %[[SLAB:.*]] = memref.subview %[[FLAT]][%{{.*}}] [2048] [1]
// CHECK:           scf.for %[[I:.*]] = %c0{{.*}} to %c2048{{.*}} step %c16{{.*}} {
// CHECK:             memref.subview %[[SLAB]][%[[I]]] [16] [1]
// CHECK:             aie.get_cascade
// CHECK:         aie.cascade_flow(%[[tile_2_4]], %[[tile_2_3]])

#set = affine_set<()[s0] : (s0 - 1 == 0)>
module {
  air.channel @cascade [1] {channel_type = "npu_cascade"}
  func.func @cascade_region(%arg0: memref<32x64xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x64xi32> {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %1 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2) {
        %c1_i32 = arith.constant 1 : i32
        %c1_1 = arith.constant 1 : index
        %async_token, %buf = air.execute -> (memref<2x32x64xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<2x32x64xi32, 2 : i32>
          air.execute_terminator %alloc : memref<2x32x64xi32, 2 : i32>
        }
        %async_token_2 = air.execute [%async_token] {
          linalg.fill ins(%c1_i32 : i32) outs(%buf : memref<2x32x64xi32, 2 : i32>)
        }
        %2 = affine.if #set()[%ty] -> !air.async.token {
          %3 = arith.subi %ty, %c1_1 : index
          %4 = air.channel.put async [%async_token_2] @cascade[%3] (%buf[%ty, 0, 0] [1, 32, 64] [2048, 64, 1]) : (memref<2x32x64xi32, 2 : i32>)
          affine.yield %4 : !air.async.token
        } else {
          %3 = air.channel.get async [%async_token_2] @cascade[%ty] (%buf[%ty, 0, 0] [1, 32, 64] [2048, 64, 1]) : (memref<2x32x64xi32, 2 : i32>)
          affine.yield %3 : !air.async.token
        }
      }
    }
    return
  }
}
