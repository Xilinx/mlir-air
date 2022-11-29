//===- airrt_alloc_dealloc.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: call @__airrt_alloc_L2_1d1i32(%[[C64]]) : (index) -> memref<?xi32, 1>
// CHECK: %[[C25:.*]] = arith.constant 25 : index
// CHECK: call @__airrt_alloc_L2_2d1i32(%[[C25]]) : (index) -> memref<?x?xi32, 1>
// CHECK: %[[C6:.*]] = arith.constant 6 : index
// CHECK: call @__airrt_alloc_L2_3d1i32(%[[C6]]) : (index) -> memref<?x?x?xi32, 1>
// CHECK: %[[C24:.*]] = arith.constant 24 : index
// CHECK: call @__airrt_alloc_L2_4d1i32(%[[C24]]) : (index) -> memref<?x?x?x?xi32, 1>
// CHECK: call @__airrt_dealloc_L2_1d1i32({{.*}}) : (memref<?xi32, 1>) -> ()
// CHECK: call @__airrt_dealloc_L2_2d1i32({{.*}}) : (memref<?x?xi32, 1>) -> ()
// CHECK: call @__airrt_dealloc_L2_3d1i32({{.*}}) : (memref<?x?x?xi32, 1>) -> ()
// CHECK: call @__airrt_dealloc_L2_4d1i32({{.*}}) : (memref<?x?x?x?xi32, 1>) -> ()
module {
  func.func @f() {
    %1 = airrt.alloc : memref<64xi32, 1>
    %2 = airrt.alloc : memref<5x5xi32, 1>
    %3 = airrt.alloc : memref<1x2x3xi32, 1>
    %4 = airrt.alloc : memref<1x2x3x4xi32, 1>
    airrt.dealloc %1 : memref<64xi32, 1>
    airrt.dealloc %2 : memref<5x5xi32, 1>
    airrt.dealloc %3 : memref<1x2x3xi32, 1>
    airrt.dealloc %4 : memref<1x2x3x4xi32, 1>
    return
  }
}

