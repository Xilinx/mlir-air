//===- matmul_nd.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

module {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<16x4x64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %1 = memref.alloc() : memref<16x4x64xi32, 1>
    // CHECK: = air.execute
    // CHECK: air.execute_terminator
    air.dma_memcpy_nd (%1[] [] [], %arg1[%c0, %c0, %c0] [%c16, %c4, %c64] [%c256, %c64, %c1]) {id = 1 : i32} : (memref<16x4x64xi32, 1>, memref<16x4x64xi32>)
    // CHECK: = air.dma_memcpy_nd async
    %2 = memref.alloc() : memref<64x64xi32, 1>
    // CHECK: = air.execute
    // CHECK: air.execute_terminator
    air.herd tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%1, %arg7=%2) : memref<16x4x64xi32, 1>,memref<64x64xi32, 1> attributes {sym_name = "herd_0"} {
    // CHECK: = air.herd @herd_0 async
      %c32_0 = arith.constant 32 : index
      %c0_0 = arith.constant 0 : index
      %c64_0 = arith.constant 64 : index
      %c256_0 = arith.constant 256 : index
      %c1_0 = arith.constant 1 : index
      %c4_0 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %shape = memref.alloc() : memref<2xindex>
      // CHECK-NOT: = air.execute
      // CHECK-NOT: air.execute_terminator
      memref.store %c32_0, %shape[%c0_0] : memref<2xindex>
      // CHECK-NOT: = air.execute
      // CHECK-NOT: air.execute_terminator
      memref.store %c32_0, %shape[%c1_0] : memref<2xindex>
      // CHECK-NOT: = air.execute
      // CHECK-NOT: air.execute_terminator
      %3 = arith.muli %arg2, %c8 : index
      // CHECK: = air.execute
      // CHECK: air.execute_terminator
      %4 = arith.muli %arg2, %c32_0 : index
      // CHECK: = air.execute
      // CHECK: air.execute_terminator
      // CHECK: = air.wait_all async
      scf.for %arg8 = %c0_0 to %c64_0 step %c32_0 {
        %5 = memref.alloc() : memref<8x4x32xi32, 2>
        air.dma_memcpy_nd (%5[] [] [], %arg6[%3, %c0_0, %arg8] [%c8, %c4_0, %c32_0] [%c256_0, %c64_0, %c1_0]) {id = 2 : i32} : (memref<8x4x32xi32, 2>, memref<16x4x64xi32, 1>)
        // CHECK: = air.dma_memcpy_nd async
        %6 = memref.reshape %5(%shape) : (memref<8x4x32xi32, 2>, memref<2xindex>) -> memref<32x32xi32, 2>
        // CHECK: memref.reshape
        air.dma_memcpy_nd (%arg7[%4, %arg8] [%c32_0, %c32_0] [%c64_0, %c1_0], %6[] [] []) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
        // CHECK: = air.dma_memcpy_nd async
        memref.dealloc %5 : memref<8x4x32xi32, 2>
        // CHECK: = air.execute
      }
    }
    air.dma_memcpy_nd (%arg0[%c0, %c0] [%c64, %c64] [%c64, %c1], %2[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<64x64xi32, 1>)
    // CHECK: = air.dma_memcpy_nd async
    memref.dealloc %1 : memref<16x4x64xi32, 1>
    // CHECK: = air.execute
    memref.dealloc %2 : memref<64x64xi32, 1>
    // CHECK: = air.execute
    return
  }
}
