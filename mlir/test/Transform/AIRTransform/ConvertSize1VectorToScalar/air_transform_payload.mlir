//===- air_transform_payload.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// Test case 1: vector.transfer_read/write with vector<1xT> → memref.load/store
// CHECK-LABEL: @test_transfer_read_write
func.func @test_transfer_read_write(%mem: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.0 : f32
  
  // CHECK: %[[LOAD:.*]] = memref.load %arg0[%c0, %c1] : memref<8x8xf32>
  // CHECK-NOT: vector.transfer_read
  %v = vector.transfer_read %mem[%c0, %c1], %pad {in_bounds = [true]} : memref<8x8xf32>, vector<1xf32>
  
  // CHECK: %[[MUL:.*]] = arith.mulf %[[LOAD]], %[[LOAD]] : f32
  // CHECK-NOT: vector<1xf32>
  %result = arith.mulf %v, %v : vector<1xf32>
  
  // CHECK: memref.store %[[MUL]], %arg0[%c0, %c1] : memref<8x8xf32>
  // CHECK-NOT: vector.transfer_write
  vector.transfer_write %result, %mem[%c0, %c1] {in_bounds = [true]} : vector<1xf32>, memref<8x8xf32>
  return
}

// Test case 2: vector.load/store with vector<1xT> → memref.load/store
// CHECK-LABEL: @test_vector_load_store
func.func @test_vector_load_store(%mem: memref<16xi32>) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  
  // CHECK: %[[LOAD:.*]] = memref.load %arg0[%c5] : memref<16xi32>
  // CHECK-NOT: vector.load
  %v = vector.load %mem[%c5] : memref<16xi32>, vector<1xi32>
  
  // CHECK: %[[ADD:.*]] = arith.addi %[[LOAD]], %[[LOAD]] : i32
  %result = arith.addi %v, %v : vector<1xi32>
  
  // CHECK: memref.store %[[ADD]], %arg0[%c5] : memref<16xi32>
  // CHECK-NOT: vector.store
  vector.store %result, %mem[%c5] : memref<16xi32>, vector<1xi32>
  return
}

// Test case 3: Multi-dimensional size-1 vectors (vector<1x1xT>)
// CHECK-LABEL: @test_multidim_size1
func.func @test_multidim_size1(%mem: memref<4x4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.0 : f16
  
  // vector<1x1xf16> should become scalar f16
  // CHECK: %[[LOAD:.*]] = memref.load %arg0[%c1, %c1] : memref<4x4xf16>
  %v = vector.transfer_read %mem[%c1, %c1], %pad {in_bounds = [true, true]} : memref<4x4xf16>, vector<1x1xf16>
  
  // CHECK: %[[SQRT:.*]] = math.sqrt %[[LOAD]] : f16
  %result = math.sqrt %v : vector<1x1xf16>
  
  // CHECK: memref.store %[[SQRT]], %arg0[%c1, %c1] : memref<4x4xf16>
  vector.transfer_write %result, %mem[%c1, %c1] {in_bounds = [true, true]} : vector<1x1xf16>, memref<4x4xf16>
  return
}

// Test case 4: Arithmetic operations on vector<1xT>
// CHECK-LABEL: @test_arith_ops
func.func @test_arith_ops(%a: vector<1xf32>, %b: vector<1xf32>) -> vector<1xf32> {
  // After conversion, these should all be scalar operations
  // CHECK: %[[V0:.*]] = vector.extract %arg0[0] : f32 from vector<1xf32>
  // CHECK: %[[V1:.*]] = vector.extract %arg1[0] : f32 from vector<1xf32>
  // CHECK: %[[V2:.*]] = arith.addf %[[V0]], %[[V1]] : f32
  %add = arith.addf %a, %b : vector<1xf32>
  
  // CHECK: %[[V3:.*]] = vector.extract %arg0[0] : f32 from vector<1xf32>
  // CHECK: %[[V4:.*]] = arith.mulf %[[V2]], %[[V3]] : f32
  %mul = arith.mulf %add, %a : vector<1xf32>
  
  // CHECK: %[[V5:.*]] = vector.extract %arg1[0] : f32 from vector<1xf32>
  // CHECK: %[[V6:.*]] = arith.divf %[[V4]], %[[V5]] : f32
  %div = arith.divf %mul, %b : vector<1xf32>
  
  // CHECK: %[[V7:.*]] = vector.broadcast %[[V6]] : f32 to vector<1xf32>
  // CHECK: return %[[V7]] : vector<1xf32>
  return %div : vector<1xf32>
}

// Test case 5: Math operations on vector<1xT>
// CHECK-LABEL: @test_math_ops
func.func @test_math_ops(%x: vector<1xf32>) -> vector<1xf32> {
  // CHECK: %[[EXT:.*]] = vector.extract %arg0[0] : f32 from vector<1xf32>
  
  // CHECK: %[[EXP:.*]] = math.exp %[[EXT]] : f32
  %exp = math.exp %x : vector<1xf32>
  
  // CHECK: %[[LOG:.*]] = math.log %[[EXP]] : f32
  %log = math.log %exp : vector<1xf32>
  
  // CHECK: %[[RSQRT:.*]] = math.rsqrt %[[LOG]] : f32
  %rsqrt = math.rsqrt %log : vector<1xf32>
  
  // CHECK: %[[BCAST:.*]] = vector.broadcast %[[RSQRT]] : f32 to vector<1xf32>
  // CHECK: return %[[BCAST]]
  return %rsqrt : vector<1xf32>
}

// Test case 6: Loop with vector<1xT> iter_args
// CHECK-LABEL: @test_loop_iter_args
func.func @test_loop_iter_args(%init: vector<1xf32>) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<1.0> : vector<1xf32>
  
  // The loop keeps vector<1xf32> signature but uses scalar ops inside
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[V0:.*]] = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %arg0) -> (vector<1xf32>) {
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg = %init) -> (vector<1xf32>) {
    // CHECK: %[[V1:.*]] = vector.extract %arg2[0] : f32 from vector<1xf32>
    // CHECK: %[[V2:.*]] = arith.addf %[[V1]], %[[CST]] : f32
    %add = arith.addf %arg, %cst : vector<1xf32>
    // CHECK: %[[V3:.*]] = vector.broadcast %[[V2]] : f32 to vector<1xf32>
    // CHECK: scf.yield %[[V3]] : vector<1xf32>
    scf.yield %add : vector<1xf32>
  }
  
  // CHECK: return %[[V0]] : vector<1xf32>
  return %result : vector<1xf32>
}

// Test case 7: Mixed operations - some vector<1xT>, some larger vectors
// CHECK-LABEL: @test_mixed_vectors
func.func @test_mixed_vectors(%mem: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  
  // vector<1xf32> should become scalar
  // CHECK: %[[SCALAR_LOAD:.*]] = memref.load %arg0[%c0] : memref<16xf32>
  %scalar_vec = vector.transfer_read %mem[%c0], %pad {in_bounds = [true]} : memref<16xf32>, vector<1xf32>
  
  // vector<8xf32> should remain a vector
  // CHECK: %[[VEC_LOAD:.*]] = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<16xf32>, vector<8xf32>
  %large_vec = vector.transfer_read %mem[%c0], %pad {in_bounds = [true]} : memref<16xf32>, vector<8xf32>
  
  // Broadcast scalar to vector<8xf32>
  // CHECK: %[[BCAST:.*]] = vector.broadcast %[[SCALAR_LOAD]] : f32 to vector<8xf32>
  %broadcast = vector.broadcast %scalar_vec : vector<1xf32> to vector<8xf32>
  
  // Operate on the larger vector (should not be converted)
  // CHECK: %[[MUL:.*]] = arith.mulf %[[VEC_LOAD]], %[[BCAST]] : vector<8xf32>
  %mul = arith.mulf %large_vec, %broadcast : vector<8xf32>
  
  // CHECK: vector.transfer_write %[[MUL]]
  vector.transfer_write %mul, %mem[%c0] {in_bounds = [true]} : vector<8xf32>, memref<16xf32>
  return
}

// Test case 8: Constants with vector<1xT>
// CHECK-LABEL: @test_constants
func.func @test_constants() -> vector<1xf32> {
  // Constants get constant-folded: 2.5 * 2.5 = 6.25
  // CHECK: %[[CST:.*]] = arith.constant dense<6.250000e+00> : vector<1xf32>
  %cst = arith.constant dense<2.5> : vector<1xf32>
  
  %result = arith.mulf %cst, %cst : vector<1xf32>
  
  // CHECK: return %[[CST]] : vector<1xf32>
  return %result : vector<1xf32>
}

// Test case 9: vector.extract and vector.broadcast patterns
// These should be eliminated when connecting to size-1 vectors
// CHECK-LABEL: @test_extract_broadcast_elim
func.func @test_extract_broadcast_elim(%mem: memref<10xbf16>) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %pad = arith.constant 0.0 : bf16
  
  // CHECK: %[[LOAD:.*]] = memref.load %arg0[%c5] : memref<10xbf16>
  %v1 = vector.transfer_read %mem[%c5], %pad {in_bounds = [true]} : memref<10xbf16>, vector<1xbf16>
  
  // This extract should be eliminated after conversion
  // CHECK-NOT: vector.extract
  %scalar = vector.extract %v1[0] : bf16 from vector<1xbf16>
  
  // Operate on scalar
  // CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %[[LOAD]] : bf16
  %add = arith.addf %scalar, %scalar : bf16
  
  // This broadcast should be eliminated
  // CHECK-NOT: vector.broadcast {{.*}} : bf16 to vector<1xbf16>
  %v2 = vector.broadcast %add : bf16 to vector<1xbf16>
  
  // CHECK: memref.store %[[ADD]], %arg0[%c0] : memref<10xbf16>
  vector.transfer_write %v2, %mem[%c0] {in_bounds = [true]} : vector<1xbf16>, memref<10xbf16>
  return
}

// Test case 10: Integer types
// CHECK-LABEL: @test_integer_types
func.func @test_integer_types(%a: vector<1xi32>, %b: vector<1xi32>) -> vector<1xi32> {
  // CHECK: %[[V0:.*]] = vector.extract %arg0[0] : i32 from vector<1xi32>
  // CHECK: %[[V1:.*]] = vector.extract %arg1[0] : i32 from vector<1xi32>
  // CHECK: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : i32
  %add = arith.addi %a, %b : vector<1xi32>
  
  // CHECK: %[[V3:.*]] = vector.extract %arg1[0] : i32 from vector<1xi32>
  // CHECK: %[[V4:.*]] = arith.muli %[[V2]], %[[V3]] : i32
  %mul = arith.muli %add, %b : vector<1xi32>
  
  // CHECK: %[[V5:.*]] = vector.broadcast %[[V4]] : i32 to vector<1xi32>
  // CHECK: return %[[V5]] : vector<1xi32>
  return %mul : vector<1xi32>
}

// Test case 11: Negative test - larger vectors should NOT be converted
// CHECK-LABEL: @test_no_convert_larger_vectors
func.func @test_no_convert_larger_vectors(%a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  // These should remain as vector operations
  // CHECK: arith.addf %arg0, %arg1 : vector<4xf32>
  %result = arith.addf %a, %b : vector<4xf32>
  // CHECK: return {{.*}} : vector<4xf32>
  return %result : vector<4xf32>
}

// Test case 12: Block arguments in nested regions
// CHECK-LABEL: @test_block_arguments
func.func @test_block_arguments(%cond: i1, %true_val: vector<1xf32>, %false_val: vector<1xf32>) -> vector<1xf32> {
  // The scf.if keeps vector<1xf32> signature but uses scalar ops inside
  // CHECK: %[[V0:.*]] = scf.if %arg0 -> (vector<1xf32>) {
  %result = scf.if %cond -> (vector<1xf32>) {
    // CHECK: %[[V1:.*]] = vector.extract %arg1[0] : f32 from vector<1xf32>
    // CHECK: %[[V2:.*]] = vector.extract %arg1[0] : f32 from vector<1xf32>
    // CHECK: %[[V3:.*]] = arith.addf %[[V1]], %[[V2]] : f32
    %add = arith.addf %true_val, %true_val : vector<1xf32>
    // CHECK: %[[V4:.*]] = vector.broadcast %[[V3]] : f32 to vector<1xf32>
    // CHECK: scf.yield %[[V4]] : vector<1xf32>
    scf.yield %add : vector<1xf32>
  } else {
    // CHECK: %[[V1_:.*]] = vector.extract %arg2[0] : f32 from vector<1xf32>
    // CHECK: %[[V2_:.*]] = vector.extract %arg2[0] : f32 from vector<1xf32>
    // CHECK: %[[V3_:.*]] = arith.mulf %[[V1_]], %[[V2_]] : f32
    %mul = arith.mulf %false_val, %false_val : vector<1xf32>
    // CHECK: %[[V4_:.*]] = vector.broadcast %[[V3_]] : f32 to vector<1xf32>
    // CHECK: scf.yield %[[V4_]] : vector<1xf32>
    scf.yield %mul : vector<1xf32>
  }
  
  // CHECK: return %[[V0]] : vector<1xf32>
  return %result : vector<1xf32>
}

// Test case 13: Complex computation chain
// CHECK-LABEL: @test_computation_chain
func.func @test_computation_chain(%mem: memref<100xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %pad = arith.constant 0.0 : f32
  
  // Load three scalar values (originally vector<1xf32>)
  // CHECK: %[[V0:.*]] = memref.load %arg0[%c0] : memref<100xf32>
  %v0 = vector.transfer_read %mem[%c0], %pad {in_bounds = [true]} : memref<100xf32>, vector<1xf32>
  
  // CHECK: %[[V1:.*]] = memref.load %arg0[%c1] : memref<100xf32>
  %v1 = vector.transfer_read %mem[%c1], %pad {in_bounds = [true]} : memref<100xf32>, vector<1xf32>
  
  // CHECK: %[[V2:.*]] = memref.load %arg0[%c2] : memref<100xf32>
  %v2 = vector.transfer_read %mem[%c2], %pad {in_bounds = [true]} : memref<100xf32>, vector<1xf32>
  
  // Compute: (v0 + v1) * v2 - all as scalars
  // CHECK: %[[ADD:.*]] = arith.addf %[[V0]], %[[V1]] : f32
  %add = arith.addf %v0, %v1 : vector<1xf32>
  
  // CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[V2]] : f32
  %mul = arith.mulf %add, %v2 : vector<1xf32>
  
  // Extract to get final scalar result
  // CHECK-NOT: vector.extract
  // CHECK: return %[[MUL]] : f32
  %result = vector.extract %mul[0] : f32 from vector<1xf32>
  return %result : f32
}

// Test case 14: Type conversions (extf, truncf)
// CHECK-LABEL: @test_type_conversions
func.func @test_type_conversions(%mem_bf16: memref<8xbf16>, %mem_f32: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %pad_bf16 = arith.constant 0.0 : bf16
  %pad_f32 = arith.constant 0.0 : f32
  
  // Load bf16 scalar
  // CHECK: %[[BF16_LOAD:.*]] = memref.load %arg0[%c0] : memref<8xbf16>
  %bf16_v = vector.transfer_read %mem_bf16[%c0], %pad_bf16 {in_bounds = [true]} : memref<8xbf16>, vector<1xbf16>
  
  // Extend to f32
  // CHECK: %[[EXT:.*]] = arith.extf %[[BF16_LOAD]] : bf16 to f32
  %f32_v = arith.extf %bf16_v : vector<1xbf16> to vector<1xf32>
  
  // Operate on f32
  // CHECK: %[[ADD:.*]] = arith.addf %[[EXT]], %[[EXT]] : f32
  %add = arith.addf %f32_v, %f32_v : vector<1xf32>
  
  // Truncate back to bf16
  // CHECK: %[[TRUNC:.*]] = arith.truncf %[[ADD]] : f32 to bf16
  %bf16_result = arith.truncf %add : vector<1xf32> to vector<1xbf16>
  
  // Store bf16 scalar
  // CHECK: memref.store %[[TRUNC]], %arg0[%c0] : memref<8xbf16>
  vector.transfer_write %bf16_result, %mem_bf16[%c0] {in_bounds = [true]} : vector<1xbf16>, memref<8xbf16>
  return
}
