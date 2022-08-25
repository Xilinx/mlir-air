// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-dependency | FileCheck %s

module attributes {torch.debug_module_name = "mmult"}  {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<?x?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<64x64xi32>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<64x64xi32>)
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    %1 = memref.cast %arg2 : memref<?x?xi32> to memref<64x64xi32>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    linalg.copy ins(%0 : memref<64x64xi32>) outs(%1 : memref<64x64xi32>)
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    %2 = memref.alloc() : memref<64x64xi32, 1>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    %3 = memref.alloc() : memref<64x64xi32, 1>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    %4 = memref.alloc() : memref<64x64xi32, 1>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    air.dma_memcpy_nd (%2[] [] [], %arg0[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 1 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    // CHECK: = air.dma_memcpy_nd async
    air.dma_memcpy_nd (%3[] [] [], %arg1[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 2 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    // CHECK: = air.dma_memcpy_nd async
    air.dma_memcpy_nd (%4[] [] [], %1[%c0, %c0] [%c64, %c64] [%c64, %c1]) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<64x64xi32>)
    // CHECK: = air.dma_memcpy_nd async
    air.herd tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%2, %arg8=%3, %arg9=%4) : memref<64x64xi32, 1>,memref<64x64xi32, 1>,memref<64x64xi32, 1>attributes {sym_name = "herd_0"} {
    // CHECK: = air.herd @herd_0 async
      %c32 = arith.constant 32 : index
      %c0_0 = arith.constant 0 : index
      %c64_1 = arith.constant 64 : index
      %c1_2 = arith.constant 1 : index
      %5 = arith.muli %arg3, %c32 : index
      // CHECK: = air.execute async
      // CHECK: air.execute_terminator
      %6 = arith.muli %arg4, %c32 : index
      // CHECK: = air.execute async
      // CHECK: air.execute_terminator
      // CHECK: = air.wait_all async
      scf.for %arg10 = %c0_0 to %c64_1 step %c32 {
        %7 = memref.alloc() : memref<32x32xi32, 2>
        %8 = memref.alloc() : memref<32x32xi32, 2>
        %9 = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%7[] [] [], %arg7[%5, %arg10] [%c32, %c32] [%c64_1, %c1_2]) {id = 4 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        // CHECK: = air.dma_memcpy_nd async
        air.dma_memcpy_nd (%8[] [] [], %arg8[%arg10, %6] [%c32, %c32] [%c64_1, %c1_2]) {id = 5 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        // CHECK: = air.dma_memcpy_nd async
        air.dma_memcpy_nd (%9[] [] [], %arg9[%5, %6] [%c32, %c32] [%c64_1, %c1_2]) {id = 6 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        // CHECK: = air.dma_memcpy_nd async
        linalg.matmul ins(%7, %8 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%9 : memref<32x32xi32, 2>)
        // CHECK: = air.execute async
        // CHECK: air.execute_terminator
        air.dma_memcpy_nd (%arg9[%5, %6] [%c32, %c32] [%c64_1, %c1_2], %9[] [] []) {id = 7 : i32} : (memref<64x64xi32, 1>, memref<32x32xi32, 2>)
        // CHECK: = air.dma_memcpy_nd async
        memref.dealloc %7 : memref<32x32xi32, 2>
        // CHECK: = air.execute async
        // CHECK: air.execute_terminator
        memref.dealloc %8 : memref<32x32xi32, 2>
        // CHECK: = air.execute async
        // CHECK: air.execute_terminator
        memref.dealloc %9 : memref<32x32xi32, 2>
        // CHECK: = air.execute async
        // CHECK: air.execute_terminator
      }
      air.herd_terminator
      // CHECK: air.herd_terminator
    }
    air.dma_memcpy_nd (%1[%c0, %c0] [%c64, %c64] [%c64, %c1], %4[] [] []) {id = 8 : i32} : (memref<64x64xi32>, memref<64x64xi32, 1>)
    // CHECK: = air.dma_memcpy_nd async
    memref.dealloc %2 : memref<64x64xi32, 1>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    memref.dealloc %3 : memref<64x64xi32, 1>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    memref.dealloc %4 : memref<64x64xi32, 1>
    // CHECK: = air.execute async
    // CHECK: air.execute_terminator
    return
  }
}