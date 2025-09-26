//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// CHECK: %[[transposed:.*]] = linalg.transpose ins(%{{.*}} : tensor<8x16x32xi32>) outs(%{{.*}} : tensor<8x32x16xi32>) permutation = [0, 2, 1] 
// CHECK: %[[reduced:.*]] = linalg.reduce ins(%[[transposed]] : tensor<8x32x16xi32>) outs(%{{.*}} : tensor<8x32xi32>) dimensions = [2] 

module {
  func.func @transpose_reduce_test(%arg0: tensor<8x16x32xi32>) -> tensor<8x32xi32> {
    %c0_i32 = arith.constant 0 : i32
    %init = tensor.empty() : tensor<8x32xi32>
    %filled = linalg.fill ins(%c0_i32 : i32) outs(%init : tensor<8x32xi32>) -> tensor<8x32xi32>
    
    // This linalg.reduce operation reduces along dimension 1 (middle dimension)
    // The TransposeReduceOp should transpose the input to make dimension 1 innermost
    // Input shape: [8, 16, 32] -> should transpose to [8, 32, 16] to make reduction dim innermost
    %reduced = linalg.reduce ins(%arg0 : tensor<8x16x32xi32>) outs(%filled : tensor<8x32xi32>) dimensions = [1] 
      (%in: i32, %1: i32) {
        %sum = arith.addi %in, %1 : i32
        linalg.yield %sum : i32
      }
    
    return %reduced : tensor<8x32xi32>
  }
}
