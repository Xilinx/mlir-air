//===- matmul_nd.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

module attributes {torch.debug_module_name = "mmult"}  {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x16x4xi32>, %arg2: memref<?x?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64xi32>
    // CHECK: = air.execute
    // CHECK: air.execute_terminator
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<64x64xi32>)
    // CHECK: = air.execute
    %1 = memref.cast %arg2 : memref<?x?xi32> to memref<64x64xi32>
    // CHECK: = air.execute
    // CHECK: air.execute_terminator
    linalg.copy ins(%0 : memref<64x64xi32>) outs(%1 : memref<64x64xi32>)
    // CHECK: = air.execute
    %2 = memref.alloc() : memref<64x64xi32, 1>
    // CHECK: = air.execute
    // CHECK: air.execute_terminator
    %3 = memref.alloc() : memref<64x16x4xi32, 1>
    // CHECK: = air.execute
    // CHECK: air.execute_terminator
    %4 = memref.alloc() : memref<64x64xi32, 1>
    // CHECK: = air.execute
    // CHECK: air.execute_terminator
    air.dma_memcpy_nd (%2[] [] [], %arg0[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    // CHECK: = air.dma_memcpy_nd async
    air.dma_memcpy_nd (%3[] [] [], %arg1[%c0, %c0, %c0] [%c64, %c16, %c4] [%c64, %c4, %c1]) {id = 2 : i32} : (memref<64x16x4xi32, 1>, memref<64x16x4xi32>)
    // CHECK: = air.dma_memcpy_nd async
    air.dma_memcpy_nd (%4[] [] [], %1[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    // CHECK: = air.dma_memcpy_nd async
    air.herd tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%2, %arg8=%3, %arg9=%4) : memref<64x64xi32, 1>,memref<64x16x4xi32, 1>,memref<64x64xi32, 1>attributes {sym_name = "herd_0"} {
    // CHECK: = air.herd @herd_0 async
      %c8 = arith.constant 8 : index
      %c32_0 = arith.constant 32 : index
      %c0_0 = arith.constant 0 : index
      %c4_0 = arith.constant 4 : index
      %c64_1 = arith.constant 64 : index
      %c1_2 = arith.constant 1 : index
      %shape = memref.alloc() : memref<2xindex>
      // CHECK-NOT: = air.execute
      // CHECK-NOT: air.execute_terminator
      memref.store %c32_0, %shape[%c0_0] : memref<2xindex>
      // CHECK-NOT: = air.execute
      // CHECK-NOT: air.execute_terminator
      memref.store %c32_0, %shape[%c1_2] : memref<2xindex>
      // CHECK-NOT: = air.execute
      // CHECK-NOT: air.execute_terminator
      %5 = arith.muli %arg3, %c32_0 : index
      // CHECK: = air.execute
      // CHECK: air.execute_terminator
      %6 = arith.muli %arg4, %c8 : index
      // CHECK: = air.execute
      // CHECK: air.execute_terminator
      %7 = arith.muli %arg4, %c32_0 : index
      // CHECK: = air.execute
      // CHECK: air.execute_terminator
      // CHECK: = air.wait_all async
      scf.for %arg10 = %c0_0 to %c64_1 step %c32_0 {
        %8 = memref.alloc() : memref<32x32xi32, 2>
        %9 = memref.alloc() : memref<32x8x4xi32, 2>
        %10 = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%8[] [] [], %arg7[%5, %arg10] [%c32_0, %c32_0] [%c64_1, %c1_2]) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        // CHECK: = air.dma_memcpy_nd async
        air.dma_memcpy_nd (%9[] [] [], %arg8[%arg10, %6, %c0_0] [%c32_0, %c8, %c4_0] [%c64_1, %c4_0, %c1_2]) {id = 5 : i32} : (memref<32x8x4xi32, 2>, memref<64x16x4xi32, 1>)
        // CHECK: = air.dma_memcpy_nd async
        air.dma_memcpy_nd (%10[] [] [], %arg9[%5, %7] [%c32_0, %c32_0] [%c64_1, %c1_2]) {id = 6 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        // CHECK: = air.dma_memcpy_nd async
        %reshape = memref.reshape %9(%shape) : (memref<32x8x4xi32, 2>, memref<2xindex>) -> memref<32x32xi32, 2>
        linalg.matmul ins(%8, %reshape : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%10 : memref<32x32xi32, 2>)
        // CHECK: = air.execute
        air.dma_memcpy_nd (%arg9[%5, %7] [%c32_0, %c32_0] [%c64_1, %c1_2], %10[] [] []) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
        // CHECK: = air.dma_memcpy_nd async
        memref.dealloc %8 : memref<32x32xi32, 2>
        // CHECK: = air.execute
        memref.dealloc %9 : memref<32x8x4xi32, 2>
        // CHECK: = air.execute
        memref.dealloc %10 : memref<32x32xi32, 2>
        // CHECK: = air.execute
      }
    }
    air.dma_memcpy_nd (%1[%c0, %c0] [%c64, %c64] [%c64, %c1], %4[] [] []) {id = 8 : i32} : (memref<64x64xi32>, memref<64x64xi32, 1>)
    // CHECK: = air.dma_memcpy_nd async
    memref.dealloc %2 : memref<64x64xi32, 1>
    // CHECK: = air.execute
    memref.dealloc %3 : memref<64x16x4xi32, 1>
    // CHECK: = air.execute
    memref.dealloc %4 : memref<64x64xi32, 1>
    // CHECK: = air.execute
    return
  }
}
