//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// CHECK: %[[init_buf:.*]] = bufferization.to_buffer
// CHECK: scf.parallel (%{{.*}}) = (%c0{{.*}}) to (%c4{{.*}}) step (%c1{{.*}}) init (%[[init_buf]]) -> memref<32xi32, 2>
// CHECK: scf.reduce
// CHECK: linalg.add
// CHECK: scf.reduce.return
#map = affine_map<()[s0] -> (s0 * 32)>
module {
  func.func @forward(%arg0: memref<512x64xi32>, %arg1: index) -> memref<32xi32> {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<64xi32, 1>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<64xi32, 1> to tensor<64xi32>
    %1 = affine.apply #map()[%arg1]
    %extracted_slice = tensor.extract_slice %0[%1] [32] [1] : tensor<64xi32> to tensor<32xi32>
    %2 = bufferization.alloc_tensor() : tensor<32xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<32xi32>) -> tensor<32xi32>
    %4 = tensor.empty() : tensor<32x4xi32>
    %5 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %4) -> (tensor<32x4xi32>) {
      %extracted_slice_0 = tensor.extract_slice %arg3[0, %arg2] [32, 1] [1, 1] : tensor<32x4xi32> to tensor<32x1xi32>
      %8 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_0 : tensor<32x1xi32>) -> tensor<32x1xi32>
      %extracted_slice_1 = tensor.extract_slice %8[0, 0] [32, 1] [1, 1] : tensor<32x1xi32> to tensor<32xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %extracted_slice_1 into %arg3[0, %arg2] [32, 1] [1, 1] : tensor<32xi32> into tensor<32x4xi32>
      }
    }
    %reduced = linalg.reduce ins(%5 : tensor<32x4xi32>) outs(%3 : tensor<32xi32>) dimensions = [1] 
      (%in: i32, %init: i32) {
        %8 = arith.addi %in, %init : i32
        linalg.yield %8 : i32
      }
    %6 = linalg.copy ins(%reduced : tensor<32xi32>) outs(%extracted_slice : tensor<32xi32>) -> tensor<32xi32>
    %7 = bufferization.to_buffer %6 : tensor<32xi32> to memref<32xi32>
    return %7 : memref<32xi32>
  }
}
