//===- transform-ops2.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-transform='filename=%s' | FileCheck %s

// CHECK-LABEL @air_par_to_herd_vert
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: air.herd @herd_0  tile ({{.*}}) in (%{{.*}}=%[[C1]], %{{.*}}=%[[C4]])
func.func @air_par_to_herd_vert() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%a) = (%c0) to (%c4) step (%c1) {
    %alloc = memref.alloc() : memref<1xi32, 2>
    %c = arith.constant 0 : i32
    linalg.fill ins(%c : i32) outs(%alloc : memref<1xi32, 2>)
  }
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["scf.parallel"]} in %arg1: (!transform.any_op) -> !transform.any_op
      %1 = transform.air.par_to_herd %0 {"first_dim"=1} : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
}
