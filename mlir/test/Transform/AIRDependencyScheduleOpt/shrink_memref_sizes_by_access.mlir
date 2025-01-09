//===- shrink_memref_sizes_by_access.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access | FileCheck %s

// Shrink the sizes of memrefs based on the actual access patterns.

// CHECK: %[[TOK0:.*]], %[[RES0:.*]] = air.execute -> (memref<1x1x16x16x4x4xbf16, 2 : i32>) {
// CHECK-NEXT: memref.alloc() {{.*}} : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK-NEXT: air.execute_terminator
// CHECK-NEXT: }
// CHECK: linalg.fill {{.*}} outs(%[[RES0]] : memref<1x1x16x16x4x4xbf16, 2 : i32>)
// CHECK: air.channel.put async [%{{.*}}]  @channel_0[%{{.*}}, %{{.*}}] (%[[RES0]]{{.*}} : (memref<1x1x16x16x4x4xbf16, 2 : i32>)
// CHECK: memref.dealloc %[[RES0]] : memref<1x1x16x16x4x4xbf16, 2 : i32>

module {
  air.channel @channel_0 [4, 4]
  func.func @func0() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0) in (%arg1=%c1) {
      %1 = air.segment @segment_0 async  {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async  tile (%arg2, %arg3) in (%arg4=%c4, %arg5=%c4) {
          %c0 = arith.constant 0 : index
          %c256 = arith.constant 256 : index
          %c4_0 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c4096 = arith.constant 4096 : index
          %c16384 = arith.constant 16384 : index
          %c1_1 = arith.constant 1 : index
          %cst = arith.constant 0.000000e+00 : bf16
          %async_token, %results = air.execute -> (memref<4x4x16x16x4x4xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<4x4x16x16x4x4xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<4x4x16x16x4x4xbf16, 2 : i32>
          }
          %subview = memref.subview %results[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x4x16x16x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
          %async_token_2 = air.execute [%async_token] {
            linalg.fill ins(%cst : bf16) outs(%subview : memref<1x1x16x16x4x4xbf16, strided<[16384, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
          }
          %3 = air.channel.put async [%async_token_2] @channel_0[%arg2, %arg3] (%results[%arg2, %arg3, %c0, %c0, %c0, %c0] [%c1_1, %c1_1, %c16, %c4_0, %c16, %c4_0] [%c16384, %c4096, %c16, %c4_0, %c256, %c1_1]) {id = 37 : i32} : (memref<4x4x16x16x4x4xbf16, 2 : i32>)
          %async_token_3 = air.execute [%3] {
            memref.dealloc %results : memref<4x4x16x16x4x4xbf16, 2 : i32>
          }
        }
      }
    }
    return
  }
}
