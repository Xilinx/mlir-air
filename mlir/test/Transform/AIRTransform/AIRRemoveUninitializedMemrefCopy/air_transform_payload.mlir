//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test basic uninitialized memref copy removal
// CHECK-LABEL: @test_basic_uninitialized_copy
// CHECK: %[[ALLOC_12:.*]] = memref.alloc() : memref<1x16x8xi32, 2>
// CHECK-NOT: memref.copy %{{.*}}, %[[ALLOC_12]]
// CHECK: linalg.fill ins(%{{.*}} : i32) outs(%[[ALLOC_12]] : memref<1x16x8xi32, 2>)
func.func @test_basic_uninitialized_copy() {
  %c0_i32 = arith.constant 0 : i32
  %alloc_6 = memref.alloc() : memref<2x16x8xi32, 1>
  %subview_11 = memref.subview %alloc_6[0, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1>
  %alloc_12 = memref.alloc() : memref<1x16x8xi32, 2>
  memref.copy %subview_11, %alloc_12 : memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1> to memref<1x16x8xi32, 2>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_12 : memref<1x16x8xi32, 2>)
  return
}

// -----

// Test that copy from single-fill source is replaced with fill
// CHECK-LABEL: @test_copy_replaced_with_fill
// CHECK: %[[ALLOC_6:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: linalg.fill ins(%[[C0:.*]] : i32) outs(%[[ALLOC_6]] : memref<2x16x8xi32, 1>)
// CHECK: %[[ALLOC_12:.*]] = memref.alloc() : memref<1x16x8xi32, 2>
// CHECK-NOT: memref.copy
// CHECK: linalg.fill ins(%[[C0]] : i32) outs(%[[ALLOC_12]] : memref<1x16x8xi32, 2>)
func.func @test_copy_replaced_with_fill() {
  %c0_i32 = arith.constant 0 : i32
  %alloc_6 = memref.alloc() : memref<2x16x8xi32, 1>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<2x16x8xi32, 1>)
  %subview_11 = memref.subview %alloc_6[0, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1>
  %alloc_12 = memref.alloc() : memref<1x16x8xi32, 2>
  memref.copy %subview_11, %alloc_12 : memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1> to memref<1x16x8xi32, 2>
  return
}

// -----

// Test copy with write between alloc and copy (should NOT be removed)
// CHECK-LABEL: @test_copy_with_intermediate_write
// CHECK: %[[ALLOC_6:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: memref.store %{{.*}}, %[[ALLOC_6]]
// CHECK: %[[SUBVIEW_11:.*]] = memref.subview %[[ALLOC_6]][0, 0, 0] [1, 16, 8] [1, 1, 1]
// CHECK: %[[ALLOC_12:.*]] = memref.alloc() : memref<1x16x8xi32, 2>
// CHECK: memref.copy %[[SUBVIEW_11]], %[[ALLOC_12]]
func.func @test_copy_with_intermediate_write() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %alloc_6 = memref.alloc() : memref<2x16x8xi32, 1>
  memref.store %c0_i32, %alloc_6[%c0, %c0, %c0] : memref<2x16x8xi32, 1>
  %subview_11 = memref.subview %alloc_6[0, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1>
  %alloc_12 = memref.alloc() : memref<1x16x8xi32, 2>
  memref.copy %subview_11, %alloc_12 : memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1> to memref<1x16x8xi32, 2>
  return
}

// -----

// Test nested scf.forall with uninitialized copy
// CHECK-LABEL: @test_nested_scf_forall
// CHECK: scf.forall (%[[ARG3:.*]]) in (2) {
// CHECK: %[[ALLOC_12:.*]] = memref.alloc() : memref<1x16x8xi32, 2>
// CHECK-NOT: memref.copy %{{.*}}, %[[ALLOC_12]]
// CHECK: linalg.fill ins(%{{.*}} : i32) outs(%[[ALLOC_12]] : memref<1x16x8xi32, 2>)
func.func @test_nested_scf_forall() {
  %c0_i32 = arith.constant 0 : i32
  scf.forall (%arg3) in (2) {
    %alloc_6 = memref.alloc() : memref<2x16x8xi32, 1>
    %subview_11 = memref.subview %alloc_6[%arg3, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: ?>, 1>
    %alloc_12 = memref.alloc() : memref<1x16x8xi32, 2>
    memref.copy %subview_11, %alloc_12 : memref<1x16x8xi32, strided<[128, 8, 1], offset: ?>, 1> to memref<1x16x8xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_12 : memref<1x16x8xi32, 2>)
  }
  return
}

// -----

// Test copy from function argument (should NOT be removed)
// CHECK-LABEL: @test_copy_from_function_arg
// CHECK: memref.copy %[[ARG0:.*]], %[[ALLOC:.*]]
func.func @test_copy_from_function_arg(%arg0: memref<16x8xi32>) {
  %alloc = memref.alloc() : memref<16x8xi32, 2>
  memref.copy %arg0, %alloc : memref<16x8xi32> to memref<16x8xi32, 2>
  return
}

// -----

// Test complex pattern from the original example
// CHECK-LABEL: @test_complex_pattern
// CHECK: scf.forall (%[[ARG3:.*]]) in (2) {
// CHECK: %[[ALLOC_12:.*]] = memref.alloc() : memref<1x16x8xi32, 2>
// CHECK-NOT: memref.copy %{{.*}}, %[[ALLOC_12]]
// CHECK: linalg.fill ins(%{{.*}} : i32) outs(%[[ALLOC_12]] : memref<1x16x8xi32, 2>)
func.func @test_complex_pattern() {
  %c0_i32 = arith.constant 0 : i32
  %alloc_2 = memref.alloc() : memref<2x16x128x128xi32, 1>
  %alloc_6 = memref.alloc() : memref<2x16x8xi32, 1>
  scf.forall (%arg3) in (2) {
    %subview_10 = memref.subview %alloc_2[%arg3, 0, 0, 0] [1, 16, 128, 128] [1, 1, 1, 1] : memref<2x16x128x128xi32, 1> to memref<1x16x128x128xi32, strided<[262144, 16384, 128, 1], offset: ?>, 1>
    %subview_11 = memref.subview %alloc_6[%arg3, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: ?>, 1>
    %alloc_12 = memref.alloc() : memref<1x16x8xi32, 2>
    memref.copy %subview_11, %alloc_12 : memref<1x16x8xi32, strided<[128, 8, 1], offset: ?>, 1> to memref<1x16x8xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_12 : memref<1x16x8xi32, 2>)
  }
  return
}

// -----

// Test multiple levels of subviews
// CHECK-LABEL: @test_multiple_subview_levels
// CHECK-NOT: memref.copy %{{.*}}, %{{.*}}
func.func @test_multiple_subview_levels() {
  %alloc_base = memref.alloc() : memref<4x32x16xi32, 1>
  %subview_1 = memref.subview %alloc_base[0, 0, 0] [2, 16, 8] [1, 1, 1] : memref<4x32x16xi32, 1> to memref<2x16x8xi32, strided<[512, 16, 1]>, 1>
  %subview_2 = memref.subview %subview_1[0, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, strided<[512, 16, 1]>, 1> to memref<1x16x8xi32, strided<[512, 16, 1]>, 1>
  %alloc_dst = memref.alloc() : memref<1x16x8xi32, 2>
  memref.copy %subview_2, %alloc_dst : memref<1x16x8xi32, strided<[512, 16, 1]>, 1> to memref<1x16x8xi32, 2>
  return
}

// -----

// Test multiple uninitialized copies removal
// CHECK-LABEL: @test_multiple_uninitialized_copies
// CHECK: %[[ALLOC_DST_1:.*]] = memref.alloc() : memref<4x128xi32, 2>
// CHECK: %[[ALLOC_DST_2:.*]] = memref.alloc() : memref<1x4x128x128xi32, 2>
// CHECK-NOT: memref.copy %{{.*}}, %[[ALLOC_DST_1]]
// CHECK-NOT: memref.copy %{{.*}}, %[[ALLOC_DST_2]]
// CHECK: linalg.generic
func.func @test_multiple_uninitialized_copies() {
  %alloc_1 = memref.alloc() : memref<16x128xi32, 1>
  %alloc_2 = memref.alloc() : memref<2x16x128x128xi32, 1>
  
  %subview_1 = memref.subview %alloc_1[0, 0] [4, 128] [1, 1] : memref<16x128xi32, 1> to memref<4x128xi32, strided<[128, 1]>, 1>
  %subview_2 = memref.subview %alloc_2[0, 0, 0, 0] [1, 4, 128, 128] [1, 1, 1, 1] : memref<2x16x128x128xi32, 1> to memref<1x4x128x128xi32, strided<[262144, 16384, 128, 1]>, 1>
  
  %alloc_dst_1 = memref.alloc() : memref<4x128xi32, 2>
  %alloc_dst_2 = memref.alloc() : memref<1x4x128x128xi32, 2>
  
  memref.copy %subview_1, %alloc_dst_1 : memref<4x128xi32, strided<[128, 1]>, 1> to memref<4x128xi32, 2>
  memref.copy %subview_2, %alloc_dst_2 : memref<1x4x128x128xi32, strided<[262144, 16384, 128, 1]>, 1> to memref<1x4x128x128xi32, 2>
  
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, d0, d1, 0)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_dst_1 : memref<4x128xi32, 2>) outs(%alloc_dst_2 : memref<1x4x128x128xi32, 2>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}

// -----

// Test basic uninitialized linalg.copy removal
// CHECK-LABEL: @test_basic_uninitialized_linalg_copy
// CHECK: %[[ALLOC_12:.*]] = memref.alloc() : memref<1x16x8xi32, 2>
// CHECK-NOT: linalg.copy ins(%{{.*}} : {{.*}}) outs(%[[ALLOC_12]] : {{.*}})
// CHECK: linalg.fill ins(%{{.*}} : i32) outs(%[[ALLOC_12]] : memref<1x16x8xi32, 2>)
func.func @test_basic_uninitialized_linalg_copy() {
  %c0_i32 = arith.constant 0 : i32
  %alloc_6 = memref.alloc() : memref<2x16x8xi32, 1>
  %subview_11 = memref.subview %alloc_6[0, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1>
  %alloc_12 = memref.alloc() : memref<1x16x8xi32, 2>
  linalg.copy ins(%subview_11 : memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1>) outs(%alloc_12 : memref<1x16x8xi32, 2>)
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_12 : memref<1x16x8xi32, 2>)
  return
}

// -----

// Test that linalg.copy from single-fill source is replaced with fill
// CHECK-LABEL: @test_linalg_copy_replaced_with_fill
// CHECK: %[[ALLOC_6:.*]] = memref.alloc() : memref<2x16x8xi32, 1>
// CHECK: linalg.fill ins(%[[C0:.*]] : i32) outs(%[[ALLOC_6]] : memref<2x16x8xi32, 1>)
// CHECK: %[[ALLOC_12:.*]] = memref.alloc() : memref<1x16x8xi32, 2>
// CHECK-NOT: linalg.copy
// CHECK: linalg.fill ins(%[[C0]] : i32) outs(%[[ALLOC_12]] : memref<1x16x8xi32, 2>)
func.func @test_linalg_copy_replaced_with_fill() {
  %c0_i32 = arith.constant 0 : i32
  %alloc_6 = memref.alloc() : memref<2x16x8xi32, 1>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<2x16x8xi32, 1>)
  %subview_11 = memref.subview %alloc_6[0, 0, 0] [1, 16, 8] [1, 1, 1] : memref<2x16x8xi32, 1> to memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1>
  %alloc_12 = memref.alloc() : memref<1x16x8xi32, 2>
  linalg.copy ins(%subview_11 : memref<1x16x8xi32, strided<[128, 8, 1], offset: 0>, 1>) outs(%alloc_12 : memref<1x16x8xi32, 2>)
  return
}

// -----

// Test mixed memref.copy and linalg.copy removal
// CHECK-LABEL: @test_mixed_copy_types
// CHECK: %[[ALLOC_DST_1:.*]] = memref.alloc() : memref<16x8xi32, 2>
// CHECK: %[[ALLOC_DST_2:.*]] = memref.alloc() : memref<16x8xi32, 2>
// CHECK-NOT: memref.copy %{{.*}}, %[[ALLOC_DST_1]]
// CHECK-NOT: linalg.copy ins(%{{.*}} : {{.*}}) outs(%[[ALLOC_DST_2]] : {{.*}})
// CHECK: linalg.generic
func.func @test_mixed_copy_types() {
  %alloc_1 = memref.alloc() : memref<16x8xi32, 1>
  %alloc_2 = memref.alloc() : memref<16x8xi32, 1>
  %alloc_dst_1 = memref.alloc() : memref<16x8xi32, 2>
  %alloc_dst_2 = memref.alloc() : memref<16x8xi32, 2>
  
  memref.copy %alloc_1, %alloc_dst_1 : memref<16x8xi32, 1> to memref<16x8xi32, 2>
  linalg.copy ins(%alloc_2 : memref<16x8xi32, 1>) outs(%alloc_dst_2 : memref<16x8xi32, 2>)
  
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_dst_1 : memref<16x8xi32, 2>) outs(%alloc_dst_2 : memref<16x8xi32, 2>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %in : i32
  }
  return
}
