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
}

