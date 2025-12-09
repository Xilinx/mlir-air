//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

// Test case 1: Basic elementwise fusion - multiply followed by add (from op definition example)
// CHECK-LABEL: @basic_elementwise_fusion
func.func @basic_elementwise_fusion(%input: tensor<16xf32>, %input2: tensor<16xf32>) -> tensor<16xf32> {
  %cst = arith.constant 2.0 : f32
  %temp1 = tensor.empty() : tensor<16xf32>
  %temp2 = tensor.empty() : tensor<16xf32>
  
  // Producer: multiply input by constant
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<16xf32>) outs(%temp1 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = arith.mulf %arg0, %cst : f32
    linalg.yield %1 : f32
  } -> tensor<16xf32>
  
  // Consumer: add producer result with another input
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0, %input2 : tensor<16xf32>, tensor<16xf32>) outs(%temp2 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %2 = arith.addf %arg0, %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<16xf32>
  
  // After fusion, should have single linalg.generic with both operations
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<16xf32>, tensor<16xf32>)
  // CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MUL:.*]] = arith.mulf %[[ARG0]]
  // CHECK:   %[[ADD:.*]] = arith.addf %[[MUL]], %[[ARG1]]
  // CHECK:   linalg.yield %[[ADD]]
  // CHECK-NOT: linalg.generic
  
  return %result : tensor<16xf32>
}

// Test case 2: Chain of multiple elementwise operations
// CHECK-LABEL: @chain_of_elementwise_ops
func.func @chain_of_elementwise_ops(%input: tensor<8xf32>) -> tensor<8xf32> {
  %cst1 = arith.constant 2.0 : f32
  %cst2 = arith.constant 3.0 : f32
  %cst3 = arith.constant 1.0 : f32
  %temp1 = tensor.empty() : tensor<8xf32>
  %temp2 = tensor.empty() : tensor<8xf32>
  %temp3 = tensor.empty() : tensor<8xf32>
  
  // First op: multiply by 2
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<8xf32>) outs(%temp1 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = arith.mulf %arg0, %cst1 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  
  // Second op: add 3
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<8xf32>) outs(%temp2 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %3 = arith.addf %arg0, %cst2 : f32
    linalg.yield %3 : f32
  } -> tensor<8xf32>
  
  // Third op: subtract 1
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%2 : tensor<8xf32>) outs(%temp3 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %4 = arith.subf %arg0, %cst3 : f32
    linalg.yield %4 : f32
  } -> tensor<8xf32>
  
  // All three operations should be fused into one
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}} : tensor<8xf32>)
  // CHECK: ^bb0(%[[ARG:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MUL:.*]] = arith.mulf %[[ARG]]
  // CHECK:   %[[ADD:.*]] = arith.addf %[[MUL]]
  // CHECK:   %[[SUB:.*]] = arith.subf %[[ADD]]
  // CHECK:   linalg.yield %[[SUB]]
  // CHECK-NOT: linalg.generic
  
  return %result : tensor<8xf32>
}

// Test case 3: 2D tensor elementwise fusion
// CHECK-LABEL: @elementwise_fusion_2d
func.func @elementwise_fusion_2d(%input1: tensor<4x8xf32>, %input2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %temp1 = tensor.empty() : tensor<4x8xf32>
  %temp2 = tensor.empty() : tensor<4x8xf32>
  
  // Producer: element-wise absolute value
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input1 : tensor<4x8xf32>) outs(%temp1 : tensor<4x8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = math.absf %arg0 : f32
    linalg.yield %1 : f32
  } -> tensor<4x8xf32>
  
  // Consumer: element-wise maximum
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %input2 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%temp2 : tensor<4x8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %2 = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<4x8xf32>
  
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<4x8xf32>)
  // CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[ABS:.*]] = math.absf %[[ARG0]]
  // CHECK:   %[[MAX:.*]] = arith.maximumf %[[ABS]], %[[ARG1]]
  // CHECK:   linalg.yield %[[MAX]]
  
  return %result : tensor<4x8xf32>
}

// Test case 4: Fill operation folding into generic
// CHECK-LABEL: @fold_fill_into_generic
func.func @fold_fill_into_generic(%input: tensor<16xf32>) -> tensor<16xf32> {
  %cst = arith.constant 0.0 : f32
  %temp1 = tensor.empty() : tensor<16xf32>
  %temp2 = tensor.empty() : tensor<16xf32>
  
  // Fill operation
  %0 = linalg.fill ins(%cst : f32) outs(%temp1 : tensor<16xf32>) -> tensor<16xf32>
  
  // Generic operation that uses the filled tensor
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input, %0 : tensor<16xf32>, tensor<16xf32>) outs(%temp2 : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %1 = arith.addf %arg0, %arg1 : f32
    linalg.yield %1 : f32
  } -> tensor<16xf32>
  
  // After fusion, fill should be folded and constant used directly
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}} : tensor<16xf32>)
  // CHECK: ^bb0(%[[ARG:.*]]: f32, %{{.*}}: f32):
  // CHECK:   arith.addf %[[ARG]]
  
  return %result : tensor<16xf32>
}

// Test case 5: Multiple producers feeding into one consumer
// CHECK-LABEL: @multiple_producers_one_consumer
func.func @multiple_producers_one_consumer(%input1: tensor<8xf32>, %input2: tensor<8xf32>) -> tensor<8xf32> {
  %cst1 = arith.constant 2.0 : f32
  %cst2 = arith.constant 3.0 : f32
  %temp1 = tensor.empty() : tensor<8xf32>
  %temp2 = tensor.empty() : tensor<8xf32>
  %temp3 = tensor.empty() : tensor<8xf32>
  
  // Producer 1: multiply input1 by 2
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input1 : tensor<8xf32>) outs(%temp1 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = arith.mulf %arg0, %cst1 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  
  // Producer 2: multiply input2 by 3
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input2 : tensor<8xf32>) outs(%temp2 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %3 = arith.mulf %arg0, %cst2 : f32
    linalg.yield %3 : f32
  } -> tensor<8xf32>
  
  // Consumer: add both producer results
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0, %2 : tensor<8xf32>, tensor<8xf32>) outs(%temp3 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %4 = arith.addf %arg0, %arg1 : f32
    linalg.yield %4 : f32
  } -> tensor<8xf32>
  
  // Both producers should be fused into consumer
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>)
  // CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MUL1:.*]] = arith.mulf %[[ARG1]]
  // CHECK:   %[[MUL2:.*]] = arith.mulf %[[ARG0]]
  // CHECK:   %[[ADD:.*]] = arith.addf %[[MUL2]], %[[MUL1]]
  // CHECK:   linalg.yield %[[ADD]]
  // CHECK-NOT: linalg.generic
  
  return %result : tensor<8xf32>
}

// Test case 6: Complex elementwise operations (exp, log, etc.)
// CHECK-LABEL: @complex_elementwise_ops
func.func @complex_elementwise_ops(%input: tensor<8xf32>) -> tensor<8xf32> {
  %temp1 = tensor.empty() : tensor<8xf32>
  %temp2 = tensor.empty() : tensor<8xf32>
  
  // Producer: exp operation
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<8xf32>) outs(%temp1 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = math.exp %arg0 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  
  // Consumer: log operation
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<8xf32>) outs(%temp2 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = math.log %arg0 : f32
    linalg.yield %2 : f32
  } -> tensor<8xf32>
  
  // Should fuse exp and log
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}} : tensor<8xf32>)
  // CHECK: ^bb0(%[[ARG:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[EXP:.*]] = math.exp %[[ARG]]
  // CHECK:   %[[LOG:.*]] = math.log %[[EXP]]
  // CHECK:   linalg.yield %[[LOG]]
  
  return %result : tensor<8xf32>
}

// Test case 7: Negative test - non-elementwise operation (should not fuse)
// CHECK-LABEL: @no_fusion_non_elementwise
func.func @no_fusion_non_elementwise(%input: tensor<8x8xf32>) -> tensor<8xf32> {
  %cst = arith.constant 2.0 : f32
  %temp1 = tensor.empty() : tensor<8xf32>
  %temp2 = tensor.empty() : tensor<8xf32>
  
  // Reduction operation (not elementwise)
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<8x8xf32>) outs(%temp1 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = arith.addf %arg0, %arg1 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  
  // Elementwise operation
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<8xf32>) outs(%temp2 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = arith.mulf %arg0, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<8xf32>
  
  // Should NOT fuse because first operation is a reduction
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel"]
  
  return %result : tensor<8xf32>
}

// Test case 8: Negative test - operation with multiple uses (should not always fuse)
// CHECK-LABEL: @no_fusion_multiple_uses
func.func @no_fusion_multiple_uses(%input: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %cst1 = arith.constant 2.0 : f32
  %cst2 = arith.constant 3.0 : f32
  %temp1 = tensor.empty() : tensor<8xf32>
  %temp2 = tensor.empty() : tensor<8xf32>
  %temp3 = tensor.empty() : tensor<8xf32>
  
  // Producer with multiple uses
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<8xf32>) outs(%temp1 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %1 = arith.mulf %arg0, %cst1 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  
  // First consumer
  %result1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<8xf32>) outs(%temp2 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = arith.addf %arg0, %cst2 : f32
    linalg.yield %2 : f32
  } -> tensor<8xf32>
  
  // Second consumer (uses same producer result)
  %result2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<8xf32>) outs(%temp3 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %3 = arith.subf %arg0, %cst2 : f32
    linalg.yield %3 : f32
  } -> tensor<8xf32>
  
  // When producer has multiple uses, the fusion creates duplicates
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.subf
  
  return %result1, %result2 : tensor<8xf32>, tensor<8xf32>
}

// Test case 9: Broadcast semantics
// CHECK-LABEL: @elementwise_with_broadcast
func.func @elementwise_with_broadcast(%input: tensor<8xf32>, %scalar: tensor<f32>) -> tensor<8xf32> {
  %temp1 = tensor.empty() : tensor<8xf32>
  %temp2 = tensor.empty() : tensor<8xf32>
  
  // Producer: broadcast scalar and add to input
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input, %scalar : tensor<8xf32>, tensor<f32>) outs(%temp1 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %1 = arith.addf %arg0, %arg1 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  
  // Consumer: negate
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%0 : tensor<8xf32>) outs(%temp2 : tensor<8xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %2 = arith.negf %arg0 : f32
    linalg.yield %2 : f32
  } -> tensor<8xf32>
  
  // Should fuse broadcast and negation
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<f32>)
  // CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[ADD:.*]] = arith.addf %[[ARG0]], %[[ARG1]]
  // CHECK:   %[[NEG:.*]] = arith.negf %[[ADD]]
  // CHECK:   linalg.yield %[[NEG]]
  
  return %result : tensor<8xf32>
}
