//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test basic memref.copy to linalg.copy conversion
// CHECK-LABEL: @test_basic_conversion
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16x32xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16x32xf32>
// CHECK: linalg.copy ins(%[[SRC]] : memref<16x32xf32>) outs(%[[DST]] : memref<16x32xf32>)
// CHECK-NOT: memref.copy
func.func @test_basic_conversion() {
  %src = memref.alloc() : memref<16x32xf32>
  %dst = memref.alloc() : memref<16x32xf32>
  memref.copy %src, %dst : memref<16x32xf32> to memref<16x32xf32>
  return
}

// -----

// Test conversion with different element types
// CHECK-LABEL: @test_different_element_types
// CHECK: %[[SRC_I32:.*]] = memref.alloc() : memref<8x8xi32>
// CHECK: %[[DST_I32:.*]] = memref.alloc() : memref<8x8xi32>
// CHECK: linalg.copy ins(%[[SRC_I32]] : memref<8x8xi32>) outs(%[[DST_I32]] : memref<8x8xi32>)
// CHECK: %[[SRC_F16:.*]] = memref.alloc() : memref<4x4xf16>
// CHECK: %[[DST_F16:.*]] = memref.alloc() : memref<4x4xf16>
// CHECK: linalg.copy ins(%[[SRC_F16]] : memref<4x4xf16>) outs(%[[DST_F16]] : memref<4x4xf16>)
// CHECK-NOT: memref.copy
func.func @test_different_element_types() {
  %src_i32 = memref.alloc() : memref<8x8xi32>
  %dst_i32 = memref.alloc() : memref<8x8xi32>
  memref.copy %src_i32, %dst_i32 : memref<8x8xi32> to memref<8x8xi32>
  
  %src_f16 = memref.alloc() : memref<4x4xf16>
  %dst_f16 = memref.alloc() : memref<4x4xf16>
  memref.copy %src_f16, %dst_f16 : memref<4x4xf16> to memref<4x4xf16>
  return
}

// -----

// Test conversion with different memory spaces
// CHECK-LABEL: @test_memory_spaces
// CHECK: %[[SRC_L1:.*]] = memref.alloc() : memref<16x16xf32, 1>
// CHECK: %[[DST_L2:.*]] = memref.alloc() : memref<16x16xf32, 2>
// CHECK: linalg.copy ins(%[[SRC_L1]] : memref<16x16xf32, 1>) outs(%[[DST_L2]] : memref<16x16xf32, 2>)
// CHECK-NOT: memref.copy
func.func @test_memory_spaces() {
  %src_l1 = memref.alloc() : memref<16x16xf32, 1>
  %dst_l2 = memref.alloc() : memref<16x16xf32, 2>
  memref.copy %src_l1, %dst_l2 : memref<16x16xf32, 1> to memref<16x16xf32, 2>
  return
}

// -----

// Test conversion with subviews
// CHECK-LABEL: @test_subview_conversion
// CHECK: %[[BASE:.*]] = memref.alloc() : memref<32x32xf32>
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][0, 0] [16, 16] [1, 1]
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK: linalg.copy ins(%[[SUBVIEW]] : memref<16x16xf32, strided<[32, 1]>>) outs(%[[DST]] : memref<16x16xf32>)
// CHECK-NOT: memref.copy
func.func @test_subview_conversion() {
  %base = memref.alloc() : memref<32x32xf32>
  %subview = memref.subview %base[0, 0] [16, 16] [1, 1] : memref<32x32xf32> to memref<16x16xf32, strided<[32, 1]>>
  %dst = memref.alloc() : memref<16x16xf32>
  memref.copy %subview, %dst : memref<16x16xf32, strided<[32, 1]>> to memref<16x16xf32>
  return
}

// -----

// Test multiple copies in the same function
// CHECK-LABEL: @test_multiple_copies
// CHECK: %[[SRC1:.*]] = memref.alloc() : memref<8x8xf32>
// CHECK: %[[DST1:.*]] = memref.alloc() : memref<8x8xf32>
// CHECK: %[[SRC2:.*]] = memref.alloc() : memref<4x4xi32>
// CHECK: %[[DST2:.*]] = memref.alloc() : memref<4x4xi32>
// CHECK: linalg.copy ins(%[[SRC1]] : memref<8x8xf32>) outs(%[[DST1]] : memref<8x8xf32>)
// CHECK: linalg.copy ins(%[[SRC2]] : memref<4x4xi32>) outs(%[[DST2]] : memref<4x4xi32>)
// CHECK-NOT: memref.copy
func.func @test_multiple_copies() {
  %src1 = memref.alloc() : memref<8x8xf32>
  %dst1 = memref.alloc() : memref<8x8xf32>
  %src2 = memref.alloc() : memref<4x4xi32>
  %dst2 = memref.alloc() : memref<4x4xi32>
  
  memref.copy %src1, %dst1 : memref<8x8xf32> to memref<8x8xf32>
  memref.copy %src2, %dst2 : memref<4x4xi32> to memref<4x4xi32>
  return
}

// -----

// Test conversion within scf.for loop
// CHECK-LABEL: @test_scf_for_conversion
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[SRC:.*]] = memref.alloc() : memref<16xf32>
// CHECK:   %[[DST:.*]] = memref.alloc() : memref<16xf32>
// CHECK:   linalg.copy ins(%[[SRC]] : memref<16xf32>) outs(%[[DST]] : memref<16xf32>)
// CHECK-NOT:   memref.copy
// CHECK: }
func.func @test_scf_for_conversion() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %c4 step %c1 {
    %src = memref.alloc() : memref<16xf32>
    %dst = memref.alloc() : memref<16xf32>
    memref.copy %src, %dst : memref<16xf32> to memref<16xf32>
  }
  return
}

// -----

// Test conversion within scf.forall
// CHECK-LABEL: @test_scf_forall_conversion
// CHECK: scf.forall (%[[ARG:.*]]) in (4) {
// CHECK:   %[[SRC:.*]] = memref.alloc() : memref<8x8xf32, 2>
// CHECK:   %[[DST:.*]] = memref.alloc() : memref<8x8xf32, 1>
// CHECK:   linalg.copy ins(%[[SRC]] : memref<8x8xf32, 2>) outs(%[[DST]] : memref<8x8xf32, 1>)
// CHECK-NOT:   memref.copy
// CHECK: }
func.func @test_scf_forall_conversion() {
  scf.forall (%i) in (4) {
    %src = memref.alloc() : memref<8x8xf32, 2>
    %dst = memref.alloc() : memref<8x8xf32, 1>
    memref.copy %src, %dst : memref<8x8xf32, 2> to memref<8x8xf32, 1>
  }
  return
}

// -----

// Test conversion mixed with linalg operations
// CHECK-LABEL: @test_mixed_with_linalg
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK: %[[TEMP:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[SRC]] : memref<16x16xf32>)
// CHECK: linalg.copy ins(%[[SRC]] : memref<16x16xf32>) outs(%[[TEMP]] : memref<16x16xf32>)
// CHECK: linalg.copy ins(%[[TEMP]] : memref<16x16xf32>) outs(%[[DST]] : memref<16x16xf32>)
// CHECK-NOT: memref.copy
func.func @test_mixed_with_linalg() {
  %cst = arith.constant 0.0 : f32
  %src = memref.alloc() : memref<16x16xf32>
  %dst = memref.alloc() : memref<16x16xf32>
  %temp = memref.alloc() : memref<16x16xf32>
  
  linalg.fill ins(%cst : f32) outs(%src : memref<16x16xf32>)
  memref.copy %src, %temp : memref<16x16xf32> to memref<16x16xf32>
  memref.copy %temp, %dst : memref<16x16xf32> to memref<16x16xf32>
  return
}

// -----

// Test 1D memref conversion
// CHECK-LABEL: @test_1d_memref
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<128xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<128xf32>
// CHECK: linalg.copy ins(%[[SRC]] : memref<128xf32>) outs(%[[DST]] : memref<128xf32>)
// CHECK-NOT: memref.copy
func.func @test_1d_memref() {
  %src = memref.alloc() : memref<128xf32>
  %dst = memref.alloc() : memref<128xf32>
  memref.copy %src, %dst : memref<128xf32> to memref<128xf32>
  return
}

// -----

// Test 3D memref conversion
// CHECK-LABEL: @test_3d_memref
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<4x8x16xf32>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<4x8x16xf32>
// CHECK: linalg.copy ins(%[[SRC]] : memref<4x8x16xf32>) outs(%[[DST]] : memref<4x8x16xf32>)
// CHECK-NOT: memref.copy
func.func @test_3d_memref() {
  %src = memref.alloc() : memref<4x8x16xf32>
  %dst = memref.alloc() : memref<4x8x16xf32>
  memref.copy %src, %dst : memref<4x8x16xf32> to memref<4x8x16xf32>
  return
}

// -----

// Test conversion with function arguments
// CHECK-LABEL: @test_function_args
// CHECK-SAME: (%[[ARG0:.*]]: memref<16x16xf32>, %[[ARG1:.*]]: memref<16x16xf32>)
// CHECK: linalg.copy ins(%[[ARG0]] : memref<16x16xf32>) outs(%[[ARG1]] : memref<16x16xf32>)
// CHECK-NOT: memref.copy
func.func @test_function_args(%src: memref<16x16xf32>, %dst: memref<16x16xf32>) {
  memref.copy %src, %dst : memref<16x16xf32> to memref<16x16xf32>
  return
}

// -----

// Test complex pattern with multiple subviews and copies
// CHECK-LABEL: @test_complex_pattern
// CHECK: %[[BASE2:.*]] = memref.alloc() : memref<32x32xf32, 2>
// CHECK: scf.forall (%[[I:.*]], %[[J:.*]]) in (2, 2) {
// CHECK:   %[[SUBVIEW2:.*]] = memref.subview %[[BASE2]][%{{.*}}, %{{.*}}] [16, 16] [1, 1]
// CHECK:   %[[TEMP:.*]] = memref.alloc() : memref<16x16xf32, 2>
// CHECK:   linalg.copy ins(%[[SUBVIEW2]] : memref<16x16xf32, strided<[32, 1], offset: ?>, 2>) outs(%[[TEMP]] : memref<16x16xf32, 2>)
// CHECK-NOT:   memref.copy
// CHECK: }
func.func @test_complex_pattern() {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %base1 = memref.alloc() : memref<64x64xf32, 1>
  %base2 = memref.alloc() : memref<32x32xf32, 2>
  
  scf.forall (%i, %j) in (2, 2) {
    %offset_i = arith.muli %i, %c32 : index
    %offset_j = arith.muli %j, %c32 : index
    %subview1 = memref.subview %base1[%offset_i, %offset_j] [32, 32] [1, 1] : memref<64x64xf32, 1> to memref<32x32xf32, strided<[64, 1], offset: ?>, 1>
    
    %sub_offset_i = arith.muli %i, %c16 : index
    %sub_offset_j = arith.muli %j, %c16 : index
    %subview2 = memref.subview %base2[%sub_offset_i, %sub_offset_j] [16, 16] [1, 1] : memref<32x32xf32, 2> to memref<16x16xf32, strided<[32, 1], offset: ?>, 2>
    
    %temp = memref.alloc() : memref<16x16xf32, 2>
    memref.copy %subview2, %temp : memref<16x16xf32, strided<[32, 1], offset: ?>, 2> to memref<16x16xf32, 2>
  }
  return
}

// -----

// Test empty function (no copies to convert)
// CHECK-LABEL: @test_empty_function
// CHECK: return
func.func @test_empty_function() {
  return
}

// -----

// Test function with only other operations (no copies)
// CHECK-LABEL: @test_no_copies
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[ALLOC]] : memref<16x16xf32>)
// CHECK-NOT: memref.copy
// CHECK-NOT: linalg.copy
func.func @test_no_copies() {
  %cst = arith.constant 1.0 : f32
  %alloc = memref.alloc() : memref<16x16xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
  return
}
