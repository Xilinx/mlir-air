//===- preconditions.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -transform-interpreter -verify-diagnostics

// A read as a target: a merge could erase it.
func.func @read_target(%m: memref<2x16xbf16, 2>) -> (vector<1x16xbf16>, vector<1x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : bf16
  %a = vector.transfer_read %m[%c0, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  %b = vector.transfer_read %m[%c1, %c0], %p {in_bounds = [true, true]} : memref<2x16xbf16, 2>, vector<1x16xbf16>
  return %a, %b : vector<1x16xbf16>, vector<1x16xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %r = transform.structured.match ops{["vector.transfer_read"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected targets that contain reads, not a read}}
    %m = transform.air.merge_adjacent_reads %r max_bits = 512 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
