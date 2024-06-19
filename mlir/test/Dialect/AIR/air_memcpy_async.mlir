//===- air_memcpy_async.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s
module {

// CHECK-LABEL: module

// CHECK: func.func @memcpy_nd
func.func @memcpy_nd(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  air.herd tile (%arg1, %arg2) in (%arg3=%c4, %arg4=%c1) args(%arg5=%arg0) : memref<4096xi32>attributes {sym_name = "memcpy_nd"} {
    %c32 = arith.constant 32 : index
    %0 = arith.muli %arg1, %c32 : index
    %1 = memref.alloc() : memref<32xi32, 2>
    %c1_0 = arith.constant 1 : index
    // CHECK: air.dma_memcpy_nd
    air.dma_memcpy_nd (%1[] [] [], %arg5[%0] [%c32] [%c1_0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    air.dma_memcpy_nd (%arg5[%0] [%c32] [%c1_0], %1[] [] []) {id = 2 : i32} : (memref<4096xi32>, memref<32xi32, 2>)
    memref.dealloc %1 : memref<32xi32, 2>
  }
  return
}

}
