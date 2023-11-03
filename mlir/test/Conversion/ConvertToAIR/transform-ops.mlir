//===- transform-ops.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -test-transform-dialect-interpreter -split-input-file -verify-diagnostics | FileCheck %s

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
    scf.yield
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_par : benefit(1) {
    %args = pdl.operands
    %results = pdl.types
    %op = pdl.operation "scf.parallel"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    pdl.rewrite %op with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1 : !pdl.operation):
      %0 = pdl_match @match_par in %arg1 : (!pdl.operation) -> !pdl.operation
      %1 = transform.air.par_to_launch %0
  }
}

// -----

// CHECK-LABEL @air_par_to_herd
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0  tile ({{.*}}) in (%{{.*}}=%[[C4]], %{{.*}}=%[[C1]])
func.func @air_par_to_herd() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%a) = (%c0) to (%c4) step (%c1) {
    %alloc = memref.alloc() : memref<1xi32>
    %c = arith.constant 0 : i32
    linalg.fill ins(%c : i32) outs(%alloc : memref<1xi32>)
    scf.yield
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_par : benefit(1) {
    %args = pdl.operands
    %results = pdl.types
    %op = pdl.operation "scf.parallel"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    pdl.rewrite %op with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1 : !pdl.operation):
      %0 = pdl_match @match_par in %arg1 : (!pdl.operation) -> !pdl.operation
      %1 = transform.air.par_to_herd %0
  }
}

// -----

// CHECK-LABEL @air_par_to_herd_vert
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: air.herd @herd_0  tile ({{.*}}) in (%{{.*}}=%[[C1]], %{{.*}}=%[[C4]])
func.func @air_par_to_herd_vert() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%a) = (%c0) to (%c4) step (%c1) {
    %alloc = memref.alloc() : memref<1xi32>
    %c = arith.constant 0 : i32
    linalg.fill ins(%c : i32) outs(%alloc : memref<1xi32>)
    scf.yield
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_par : benefit(1) {
    %args = pdl.operands
    %results = pdl.types
    %op = pdl.operation "scf.parallel"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    pdl.rewrite %op with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1 : !pdl.operation):
      %0 = pdl_match @match_par in %arg1 : (!pdl.operation) -> !pdl.operation
      %1 = transform.air.par_to_herd %0 {"first_dim"=1}
  }
}
