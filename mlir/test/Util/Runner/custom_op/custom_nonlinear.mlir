//===- custom_nonlinear.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// AIR Runner simulation of a user hand-written kernel that performs nonlinear 
// function after matmul.

// RUN: air-runner %s -f nonlinear -m %S/arch.json | FileCheck %s

// CHECK: "name": "air.custom",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK: "ts": 0.[[#%d,TIME0:]],
// CHECK: "name": "air.custom",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK: "ts": 0.[[#TIME0 + 200]],

module {
  func.func @nonlinear(%arg0: memref<32x32xi8>, %arg1: memref<32x32xi8>, %arg2: memref<32x32xi8>, %arg3: memref<32x32xi8>) -> memref<32x32xi8> {
    %c1 = arith.constant 1 : index
    %async_token_1, %results_2 = air.execute -> (memref<32x32xi8>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32xi8>
      air.execute_terminator %alloc : memref<32x32xi8>
    }
    %0 = air.launch async [%async_token_1] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<32x32xi8>, memref<32x32xi8> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<32x32xi8>, memref<32x32xi8> attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) args(%arg25=%arg19, %arg26=%arg20) : memref<32x32xi8>, memref<32x32xi8> {
          %async_token_3, %results_4 = air.execute -> (memref<32x32xi8, 2>) {
            %alloc = memref.alloc() : memref<32x32xi8, 2>
            air.execute_terminator %alloc : memref<32x32xi8, 2>
          }
          %async_token_5, %results_6 = air.execute -> (memref<32x32xi8, 2>) {
            %alloc = memref.alloc() : memref<32x32xi8, 2>
            air.execute_terminator %alloc : memref<32x32xi8, 2>
          }
          %async_token_7, %results_8 = air.execute -> (memref<32x32xi8, 2>) {
            %alloc = memref.alloc() : memref<32x32xi8, 2>
            air.execute_terminator %alloc : memref<32x32xi8, 2>
          }
          
          %3 = air.dma_memcpy_nd async [%async_token_3] (%results_4[] [] [], %arg25[] [] []) : (memref<32x32xi8, 2>, memref<32x32xi8>)
          %4 = air.dma_memcpy_nd async [%async_token_5] (%results_6[] [] [], %arg26[] [] []) : (memref<32x32xi8, 2>, memref<32x32xi8>)
          %async_token_9 = air.execute [%3, %4] {
            linalg.matmul ins(%results_4, %results_6 : memref<32x32xi8, 2>, memref<32x32xi8, 2>) outs(%results_8 : memref<32x32xi8, 2>)
          }
          %5 = air.execute [%async_token_9] {
            air.custom @nonlin  operands (%results_8) : memref<32x32xi8, 2>
          }
          %async_token_10 = air.execute [%5] {
            memref.dealloc %results_4 : memref<32x32xi8, 2>
          }
          %async_token_11 = air.execute [%5] {
            memref.dealloc %results_6 : memref<32x32xi8, 2>
          }
          %async_token_12 = air.execute [%5] {
            memref.dealloc %results_8 : memref<32x32xi8, 2>
          }
        }
      }
    }
    return %results_2 : memref<32x32xi8>
  }
}
