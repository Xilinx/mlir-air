//===- transform-par-to-launch.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-transform='filename=%s' | FileCheck %s

// CHECK-LABEL @air_par_to_launch
// CHECK: air.launch
func.func @air_par_to_launch() {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c128, %c128) step (%c32, %c32) {
    %alloc = memref.alloc() : memref<1xi32>
    %c = arith.constant 0 : i32
    linalg.fill ins(%c : i32) outs(%alloc : memref<1xi32>)
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["scf.parallel"]} in %arg1: (!transform.any_op) -> !transform.any_op
      %1 = transform.air.par_to_launch %0 : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
}
