//===- scf_for.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

module {

  // Scf.for on IndexType.

  // CHECK-LABEL: func0
  // CHECK: scf.for{{.*}}iter_args(%[[ITERARG0:.*]] = %{{.*}})
  // CHECK: arith.muli
  // CHECK: arith.addi

  // CHECK: scf.for{{.*}}iter_args(%[[ITERARG1:.*]] = %{{.*}})
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: %[[EXECTOK8:.*]], %[[EXECVAL8:.*]] = air.execute
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: air.execute_terminator
  // CHECK: %[[DMA0:.*]] = air.dma_memcpy_nd async [%[[ITERARG1]], %[[EXECTOK8]]]
  // CHECK: air.dma_memcpy_nd async [%[[ITERARG1]], %[[DMA0]]]
  // CHECK: scf.yield
  // CHECK: scf.yield

  func.func @func0(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: index, %arg3: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.muli %arg2, %c128 : index
    %2 = arith.muli %arg3, %c64 : index
    scf.for %arg4 = %c0 to %c32 step %c1 {
      %4 = arith.muli %arg4, %c4 : index
      %5 = arith.addi %0, %4 : index
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %6 = arith.muli %arg5, %c2 : index
        %7 = arith.addi %2, %6 : index
        %9 = arith.muli %5, %c128 : index
        %11 = arith.addi %9, %7 : index
        %alloc = memref.alloc() : memref<4x2xf32, 2 : i32>
        air.dma_memcpy_nd (%alloc[] [] [], %arg0[%c0, %11] [%c4, %c2] [%c128, %c1]) {id = 1 : i32} : (memref<4x2xf32, 2 : i32>, memref<*xf32>)
        air.dma_memcpy_nd (%arg1[%c0, %11] [%c4, %c2] [%c128, %c1], %alloc[] [] []) {id = 2 : i32} : (memref<*xf32>, memref<4x2xf32, 2 : i32>)
      }
    }
    return
  }

  // Scf.for on IntegerType.

  // CHECK-LABEL: func1
  // CHECK: scf.for{{.*}}iter_args(%[[ITERARG0:.*]] = %{{.*}})
  // CHECK: arith.muli
  // CHECK: arith.addi

  // CHECK: scf.for{{.*}}iter_args(%[[ITERARG1:.*]] = %{{.*}})
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: arith.index_cast
  // CHECK: arith.muli
  // CHECK: arith.index_cast
  // CHECK: arith.addi
  // CHECK: %[[EXECTOK8:.*]], %[[EXECVAL8:.*]] = air.execute
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: air.execute_terminator
  // CHECK: %[[DMA0:.*]] = air.dma_memcpy_nd async [%[[ITERARG1]], %[[EXECTOK8]]]
  // CHECK: air.dma_memcpy_nd async [%[[ITERARG1]], %[[DMA0]]]
  // CHECK: scf.yield
  // CHECK: scf.yield

  func.func @func1(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: index, %arg3: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.muli %arg2, %c128 : index
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.muli %arg3, %c64 : index
    %3 = arith.index_cast %2 : index to i32
    scf.for %arg4 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %4 = arith.muli %arg4, %c4_i32 : i32
      %5 = arith.addi %1, %4 : i32
      scf.for %arg5 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %6 = arith.muli %arg5, %c2_i32 : i32
        %7 = arith.addi %3, %6 : i32
        %8 = arith.index_cast %5 : i32 to index
        %9 = arith.muli %8, %c128 : index
        %10 = arith.index_cast %7 : i32 to index
        %11 = arith.addi %9, %10 : index
        %alloc = memref.alloc() : memref<4x2xf32, 2 : i32>
        air.dma_memcpy_nd (%alloc[] [] [], %arg0[%c0, %11] [%c4, %c2] [%c128, %c1]) {id = 1 : i32} : (memref<4x2xf32, 2 : i32>, memref<*xf32>)
        air.dma_memcpy_nd (%arg1[%c0, %11] [%c4, %c2] [%c128, %c1], %alloc[] [] []) {id = 2 : i32} : (memref<*xf32>, memref<4x2xf32, 2 : i32>)
      }
    }
    return
  }

  // Test nested loops with multiple existing iter_args to ensure they are preserved
  // when async token is added as an additional iter_arg.

  // CHECK-LABEL: func2
  // CHECK: %[[INIT0:.*]] = arith.constant dense<0> : vector<1x1x8x8xi16>
  // CHECK: %[[INIT1:.*]] = arith.constant dense<0> : vector<1x1x8x8xi16>
  // CHECK: %[[INIT2:.*]] = arith.constant dense<0> : vector<1x1x8x8xi16>
  // CHECK: %[[INIT3:.*]] = arith.constant dense<0> : vector<1x1x8x8xi16>
  // CHECK: scf.for{{.*}}iter_args(%[[V0:.*]] = %[[INIT0]], %[[V1:.*]] = %[[INIT1]], %[[V2:.*]] = %[[INIT2]], %[[V3:.*]] = %[[INIT3]], %[[TOKEN0:.*]] = %{{.*}})
  // CHECK: %[[EXECTOK0:.*]], %[[EXECVAL0:.*]] = air.execute
  // CHECK: %[[DMA0:.*]] = air.dma_memcpy_nd async [%[[TOKEN0]], %[[EXECTOK0]]]
  // CHECK: %[[INNERFORRES:.*]]:5 = scf.for{{.*}}iter_args(%[[V0_INNER:.*]] = %[[V0]], %[[V1_INNER:.*]] = %[[V1]], %[[V2_INNER:.*]] = %[[V2]], %[[V3_INNER:.*]] = %[[V3]], %[[TOKEN1:.*]] = %{{.*}})
  // CHECK: %[[EXECTOK1:.*]], %[[EXECVAL1:.*]] = air.execute
  // CHECK: %[[DMA1:.*]] = air.dma_memcpy_nd async [%[[TOKEN1]], %[[EXECTOK1]]]
  // CHECK: %[[WAITYIELD:.*]] = air.wait_all async [%[[DMA1]]]
  // CHECK: scf.yield %[[V0_INNER]], %[[V1_INNER]], %[[V2_INNER]], %[[V3_INNER]], %[[WAITYIELD]]
  // CHECK: %[[WAITYIELD_OUTER:.*]] = air.wait_all async [%[[DMA0]], %[[INNERFORRES]]#4]
  // CHECK: scf.yield %[[INNERFORRES]]#0, %[[INNERFORRES]]#1, %[[INNERFORRES]]#2, %[[INNERFORRES]]#3, %[[WAITYIELD_OUTER]]

  func.func @func2(%arg0: memref<256x1024xi16>, %arg1: memref<1024x1024xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    
    // Initialize 4 vector iter_args
    %cst = arith.constant 0 : i16
    %init0 = arith.constant dense<0> : vector<1x1x8x8xi16>
    %init1 = arith.constant dense<0> : vector<1x1x8x8xi16>
    %init2 = arith.constant dense<0> : vector<1x1x8x8xi16>
    %init3 = arith.constant dense<0> : vector<1x1x8x8xi16>
    
    // Outer loop with 4 vector iter_args
    %res:4 = scf.for %i = %c0 to %c256 step %c8 iter_args(%v0 = %init0, %v1 = %init1, %v2 = %init2, %v3 = %init3) 
        -> (vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>) {
      
      %alloc0 = memref.alloc() : memref<1x1x8x8xi16, 2 : i32>
      air.dma_memcpy_nd (%alloc0[] [] [], %arg0[%i, %c0] [%c8, %c8] [%c1024, %c1]) {id = 1 : i32} 
        : (memref<1x1x8x8xi16, 2 : i32>, memref<256x1024xi16>)
      
      // Inner nested loop also with 4 vector iter_args
      %inner_res:4 = scf.for %j = %c0 to %c1024 step %c8 iter_args(%v0_inner = %v0, %v1_inner = %v1, %v2_inner = %v2, %v3_inner = %v3) 
          -> (vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>) {
        
        %alloc1 = memref.alloc() : memref<1x1x8x8xi16, 2 : i32>
        air.dma_memcpy_nd (%alloc1[] [] [], %arg1[%c0, %j] [%c8, %c8] [%c1024, %c1]) {id = 2 : i32} 
          : (memref<1x1x8x8xi16, 2 : i32>, memref<1024x1024xi16>)
        
        // Yield the 4 vectors unchanged (just pass them through)
        scf.yield %v0_inner, %v1_inner, %v2_inner, %v3_inner 
            : vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>
      }
      
      // Yield the 4 vectors from inner loop
      scf.yield %inner_res#0, %inner_res#1, %inner_res#2, %inner_res#3 
          : vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>, vector<1x1x8x8xi16>
    }
    
    return
  }
}
