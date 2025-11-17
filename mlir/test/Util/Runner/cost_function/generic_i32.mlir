//===- generic_i32.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Test trace latency modelling of linalg.generic ops.

// CHECK: "name": "LinalgOp(linalg.generic)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.00[[#%d,TIME0:]],
// CHECK: "name": "LinalgOp(linalg.generic)",
// CHECK: "ph": "E",
// CHECK: "ts": 4.[[#TIME0 + 4096 - 4000 + 100]],

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test(%arg0: memref<256x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>, %arg3: memref<1024x1024xi32>) -> memref<256x1024xi32> {
    %c1 = arith.constant 1 : index
    %async_token_1, %results_2 = air.execute -> (memref<256x1024xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xi32>
      air.execute_terminator %alloc : memref<256x1024xi32>
    }
    %0 = air.launch async [%async_token_1] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xi32>, memref<1024x1024xi32> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xi32>, memref<1024x1024xi32> attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %cst_8 = arith.constant 2 : i32
          %cst_9 = arith.constant 1 : i32
          %cst_10 = arith.constant 1 : i32
          %async_token_3, %results_4 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %async_token_5, %results_6 = air.execute -> (memref<32x32xi32, 2>) {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.execute_terminator %alloc : memref<32x32xi32, 2>
          }
          %async_token_7 = air.execute [%async_token_3, %async_token_5] {
            linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%results_4 : memref<32x32xi32, 2>) outs(%results_6 : memref<32x32xi32, 2>) {
            ^bb0(%in: i32, %out: i32):
              %9 = arith.divsi %in, %cst_8 : i32
              %11 = arith.addi %9, %cst_9 : i32
              %12 = arith.muli %11, %cst_10 : i32
              %13 = arith.muli %in, %12 : i32
              linalg.yield %13 : i32
            }
          }
          %async_token_10 = air.execute [%async_token_7] {
            memref.dealloc %results_4 : memref<32x32xi32, 2>
          }
          %async_token_11 = air.execute [%async_token_7] {
            memref.dealloc %results_6 : memref<32x32xi32, 2>
          }
        }
      }
    }
    return %results_2 : memref<256x1024xi32>
  }
}
