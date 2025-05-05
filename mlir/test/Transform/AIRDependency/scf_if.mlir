//===- scf_if.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// CHECK: %[[EVENT0:.*]] = air.wait_all async
// CHECK-NEXT: %[[EVENT1:.*]] = scf.if
// CHECK: %[[EVENT2:.*]] = scf.for
// CHECK: %[[EVENT3:.*]] = air.wait_all async
// CHECK-NEXT: scf.yield %[[EVENT3]]
// CHECK: %[[EVENT4:.*]] = air.wait_all async [%[[EVENT2]]]
// CHECK-NEXT: scf.yield %[[EVENT4]]
// CHECK-NEXT: } else {
// CHECK: %[[EVENT5:.*]] = scf.for
// CHECK: %[[EVENT6:.*]] = air.wait_all async
// CHECK-NEXT: scf.yield %[[EVENT6]]
// CHECK: %[[EVENT7:.*]] = air.wait_all async [%[[EVENT5]]]
// CHECK-NEXT: scf.yield %[[EVENT7]]

module {
  func.func @scf_if() {
    %c1_0 = arith.constant 1 : index
    %true = arith.constant true
    %alloc = memref.alloc() : memref<48xi32, 1 : i32>
    air.herd @herd_0  tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) args(%arg10=%alloc, %arg11=%true) : memref<48xi32, 1 : i32>, i1 attributes {link_with = "vm.o"} {
      %c100_i32 = arith.constant 100 : i32
      %c1_1 = arith.constant 1 : index
      %c48 = arith.constant 48 : index
      %c0 = arith.constant 0 : index
      %alloc_2 = memref.alloc() : memref<48xi32, 2 : i32>
      %alloc_3 = memref.alloc() : memref<48xi32, 2 : i32>
      scf.if %arg11 {
        scf.for %arg12 = %c0 to %c48 step %c1_1 {
          %0 = memref.load %alloc_2[%arg12] : memref<48xi32, 2 : i32>
          %1 = arith.addi %0, %c100_i32 : i32
          memref.store %1, %alloc_3[%arg12] : memref<48xi32, 2 : i32>
        }
      } else {
        scf.for %arg12 = %c0 to %c48 step %c1_1 {
          %0 = memref.load %alloc_2[%arg12] : memref<48xi32, 2 : i32>
          %1 = arith.muli %0, %c100_i32 : i32
          memref.store %1, %alloc_3[%arg12] : memref<48xi32, 2 : i32>
        }
      }
      air.dma_memcpy_nd (%arg10[] [] [], %alloc_3[] [] []) : (memref<48xi32, 1 : i32>, memref<48xi32, 2 : i32>)
    }
    return
  }
}

