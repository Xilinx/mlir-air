//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test basic cascade memcpy elimination - the exact pattern from the original example
// CHECK-LABEL: @test_basic_cascade_elimination
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1024xi32>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC]][] [] [], %[[ALLOC_3]][] [] []) : (memref<1024xi32>, memref<2x16x8xi32, 1>)
func.func @test_basic_cascade_elimination(%arg0: memref<2048xi32>, %arg1: memref<2048x1024xi32>) -> memref<1024xi32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xi32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x128xi32>
  %alloc_3 = memref.alloc() : memref<2x16x8xi32, 1>
  
  // This is the cascade pattern we want to eliminate:
  // First memcpy: %alloc_3 -> %alloc_2 (intermediate buffer)
  air.dma_memcpy_nd (%alloc_2[] [] [], %alloc_3[] [] []) : (memref<2x128xi32>, memref<2x16x8xi32, 1>)
  // Second memcpy: %alloc_2 -> %alloc (final destination)
  air.dma_memcpy_nd (%alloc[] [] [], %alloc_2[] [] []) : (memref<1024xi32>, memref<2x128xi32>)
  
  return %alloc : memref<1024xi32>
}

// -----

// Test that non-default access patterns are NOT eliminated
// CHECK-LABEL: @test_non_default_access_patterns_not_eliminated
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xi32>
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<2x128xi32>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC_1]]{{.*}}%[[ALLOC_3]][] [] [])
// CHECK: air.dma_memcpy_nd (%[[ALLOC]][] [] [], %[[ALLOC_1]][] [] [])
func.func @test_non_default_access_patterns_not_eliminated() -> memref<1024xi32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<1024xi32>
  %alloc_2 = memref.alloc() : memref<2x128xi32>
  %alloc_3 = memref.alloc() : memref<2x16x8xi32, 1>
  
  // Non-default access pattern (has offsets, sizes, strides)
  air.dma_memcpy_nd (%alloc_2[%c0] [%c128] [%c1], %alloc_3[] [] []) : (memref<2x128xi32>, memref<2x16x8xi32, 1>)
  air.dma_memcpy_nd (%alloc[] [] [], %alloc_2[] [] []) : (memref<1024xi32>, memref<2x128xi32>)
  
  return %alloc : memref<1024xi32>
}

// -----

// Test that intermediate buffer with more than 2 uses is NOT eliminated
// CHECK-LABEL: @test_intermediate_buffer_multiple_uses_not_eliminated
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xi32>
// CHECK: %[[ALLOC_2:.*]] = memref.alloc() : memref<2x128xi32>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: %[[ALLOC_4:.*]] = memref.alloc() : memref<2x128xi32>
// CHECK: air.dma_memcpy_nd (%[[ALLOC_2]][] [] [], %[[ALLOC_3]][] [] [])
// CHECK: air.dma_memcpy_nd (%[[ALLOC]][] [] [], %[[ALLOC_2]][] [] [])
// CHECK: air.dma_memcpy_nd (%[[ALLOC_4]][] [] [], %[[ALLOC_2]][] [] [])
func.func @test_intermediate_buffer_multiple_uses_not_eliminated() -> memref<1024xi32> {
  %alloc = memref.alloc() : memref<1024xi32>
  %alloc_2 = memref.alloc() : memref<2x128xi32>
  %alloc_3 = memref.alloc() : memref<2x16x8xi32, 1>
  %alloc_4 = memref.alloc() : memref<2x128xi32>
  
  // Intermediate buffer used 3 times (should not be eliminated)
  air.dma_memcpy_nd (%alloc_2[] [] [], %alloc_3[] [] []) : (memref<2x128xi32>, memref<2x16x8xi32, 1>)
  air.dma_memcpy_nd (%alloc[] [] [], %alloc_2[] [] []) : (memref<1024xi32>, memref<2x128xi32>)
  air.dma_memcpy_nd (%alloc_4[] [] [], %alloc_2[] [] []) : (memref<2x128xi32>, memref<2x128xi32>)
  
  return %alloc : memref<1024xi32>
}

// -----

// Test cascade elimination with async dependencies
// CHECK-LABEL: @test_cascade_with_async_dependencies
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xi32>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC]][] [] [], %[[ALLOC_3]][] [] []) : (memref<1024xi32>, memref<2x16x8xi32, 1>)
func.func @test_cascade_with_async_dependencies() -> memref<1024xi32> {
  %alloc = memref.alloc() : memref<1024xi32>
  %alloc_2 = memref.alloc() : memref<2x128xi32>
  %alloc_3 = memref.alloc() : memref<2x16x8xi32, 1>
  
  // Cascade with async tokens
  %token1 = air.dma_memcpy_nd async (%alloc_2[] [] [], %alloc_3[] [] []) : (memref<2x128xi32>, memref<2x16x8xi32, 1>)
  %token2 = air.dma_memcpy_nd async [%token1] (%alloc[] [] [], %alloc_2[] [] []) : (memref<1024xi32>, memref<2x128xi32>)
  
  return %alloc : memref<1024xi32>
}

// -----

// Test multiple cascade patterns in the same function
// CHECK-LABEL: @test_multiple_cascade_patterns
// CHECK: %[[ALLOC_A:.*]] = memref.alloc() : memref<1024xi32>
// CHECK: %[[ALLOC_B:.*]] = memref.alloc() : memref<512xi32>
// CHECK: %[[ALLOC_SRC1:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: %[[ALLOC_SRC2:.*]] = memref.alloc() : memref<1x16x8xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC_A]][] [] [], %[[ALLOC_SRC1]][] [] [])
// CHECK: air.dma_memcpy_nd (%[[ALLOC_B]][] [] [], %[[ALLOC_SRC2]][] [] [])
func.func @test_multiple_cascade_patterns() -> (memref<1024xi32>, memref<512xi32>) {
  %alloc_a = memref.alloc() : memref<1024xi32>
  %alloc_b = memref.alloc() : memref<512xi32>
  %alloc_int1 = memref.alloc() : memref<2x128xi32>
  %alloc_int2 = memref.alloc() : memref<1x64xi32>
  %alloc_src1 = memref.alloc() : memref<2x16x8xi32, 1>
  %alloc_src2 = memref.alloc() : memref<1x16x8xi32, 1>
  
  // First cascade pattern
  air.dma_memcpy_nd (%alloc_int1[] [] [], %alloc_src1[] [] []) : (memref<2x128xi32>, memref<2x16x8xi32, 1>)
  air.dma_memcpy_nd (%alloc_a[] [] [], %alloc_int1[] [] []) : (memref<1024xi32>, memref<2x128xi32>)
  
  // Second cascade pattern
  air.dma_memcpy_nd (%alloc_int2[] [] [], %alloc_src2[] [] []) : (memref<1x64xi32>, memref<1x16x8xi32, 1>)
  air.dma_memcpy_nd (%alloc_b[] [] [], %alloc_int2[] [] []) : (memref<512xi32>, memref<1x64xi32>)
  
  return %alloc_a, %alloc_b : memref<1024xi32>, memref<512xi32>
}

// -----

// Test cascade pattern within scf.forall
// CHECK-LABEL: @test_cascade_in_scf_forall
// CHECK: scf.forall (%[[ARG3:.*]]) in (2) {
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xi32>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK-NOT: %[[ALLOC_2:.*]] = memref.alloc() : memref<2x128xi32>
// CHECK: air.dma_memcpy_nd (%[[ALLOC]][] [] [], %[[ALLOC_3]][] [] [])
func.func @test_cascade_in_scf_forall() {
  scf.forall (%arg3) in (2) {
    %alloc = memref.alloc() : memref<1024xi32>
    %alloc_2 = memref.alloc() : memref<2x128xi32>
    %alloc_3 = memref.alloc() : memref<2x16x8xi32, 1>
    
    air.dma_memcpy_nd (%alloc_2[] [] [], %alloc_3[] [] []) : (memref<2x128xi32>, memref<2x16x8xi32, 1>)
    air.dma_memcpy_nd (%alloc[] [] [], %alloc_2[] [] []) : (memref<1024xi32>, memref<2x128xi32>)
  }
  return
}

// -----

// Test that non-cascade patterns are not affected
// CHECK-LABEL: @test_non_cascade_patterns_unaffected
// CHECK: %[[ALLOC_A:.*]] = memref.alloc() : memref<1024xi32>
// CHECK: %[[ALLOC_B:.*]] = memref.alloc() : memref<512xi32>
// CHECK: %[[ALLOC_C:.*]] = memref.alloc() : memref<256xi32>
// CHECK: air.dma_memcpy_nd (%[[ALLOC_A]][] [] [], %[[ALLOC_B]][] [] [])
// CHECK: air.dma_memcpy_nd (%[[ALLOC_B]][] [] [], %[[ALLOC_C]][] [] [])
func.func @test_non_cascade_patterns_unaffected() -> memref<1024xi32> {
  %alloc_a = memref.alloc() : memref<1024xi32>
  %alloc_b = memref.alloc() : memref<512xi32>
  %alloc_c = memref.alloc() : memref<256xi32>
  
  // These are separate operations, not a cascade pattern
  air.dma_memcpy_nd (%alloc_a[] [] [], %alloc_b[] [] []) : (memref<1024xi32>, memref<512xi32>)
  air.dma_memcpy_nd (%alloc_b[] [] [], %alloc_c[] [] []) : (memref<512xi32>, memref<256xi32>)
  
  return %alloc_a : memref<1024xi32>
}

// -----

// Test complex pattern from the original example with more context
// CHECK-LABEL: @test_complex_original_pattern
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1024xi32>
// CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<16x128xi32, 1>
// CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<2x16x128x128xi32, 1>
// CHECK: %[[ALLOC_3:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: scf.forall (%[[ARG3:.*]]) in (2) {
// CHECK-NOT: %[[ALLOC_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2x128xi32>
// CHECK: air.dma_memcpy_nd (%[[ALLOC]][] [] [], %[[ALLOC_3]][] [] [])
func.func @test_complex_original_pattern(%arg0: memref<2048xi32>, %arg1: memref<2048x1024xi32>) -> memref<1024xi32> {
  %c2048 = arith.constant 2048 : index
  %c8 = arith.constant 8 : index
  %c16384 = arith.constant 16384 : index
  %c262144 = arith.constant 262144 : index
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index
  %c1024 = arith.constant 1024 : index
  %c131072 = arith.constant 131072 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xi32>
  scf.forall (%arg2) in (4) {
    %alloc_0 = memref.alloc() : memref<16x128xi32, 1>
    air.dma_memcpy_nd (%alloc_0[] [] [], %arg0[] [] []) : (memref<16x128xi32, 1>, memref<2048xi32>)
    %alloc_1 = memref.alloc() : memref<2x16x128x128xi32, 1>
    air.dma_memcpy_nd (%alloc_1[] [] [], %arg1[%c0, %c0, %c0, %c0] [%c2, %c16, %c128, %c128] [%c128, %c131072, %c1024, %c1]) : (memref<2x16x128x128xi32, 1>, memref<2048x1024xi32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x128xi32>
    %alloc_3 = memref.alloc() : memref<2x16x8xi32, 1>
    scf.forall (%arg3) in (2) {
      %subview = memref.subview %alloc_3[%arg3, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: ?>, 1>
      %alloc_4 = memref.alloc() : memref<1x16x8xi32, 2>
      linalg.fill ins(%c0_i32 : i32) outs(%alloc_4 : memref<1x16x8xi32, 2>)
      memref.copy %alloc_4, %subview : memref<1x16x8xi32, 2> to memref<1x16x8xi32, strided<[128, 8, 1], offset: ?>, 1>
    }
    // This is the cascade pattern that should be eliminated:
    air.dma_memcpy_nd (%alloc_2[] [] [], %alloc_3[] [] []) : (memref<2x128xi32>, memref<2x16x8xi32, 1>)
    air.dma_memcpy_nd (%alloc[] [] [], %alloc_2[] [] []) : (memref<1024xi32>, memref<2x128xi32>)
    memref.dealloc %alloc_0 : memref<16x128xi32, 1>
    memref.dealloc %alloc_1 : memref<2x16x128x128xi32, 1>
    memref.dealloc %alloc_3 : memref<2x16x8xi32, 1>
  }
  return %alloc : memref<1024xi32>
}
