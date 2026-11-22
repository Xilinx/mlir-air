//===- scf_if_results.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dependency-canonicalize | FileCheck %s

// An scf.if or affine.if that yields a value: air-dependency appends its token
// after the value. The write after the if depends on the read inside it,
// elevated to the if, so it must take the if's token, not its first result.

// CHECK-LABEL: func.func @scf_if_with_results
// CHECK: %[[IF:.*]]:2 = scf.if %{{.*}} -> (vector<64xbf16>, !air.async.token)
// CHECK: air.execute [%[[IF]]#1
// CHECK-NEXT: vector.transfer_write %[[IF]]#0
func.func @scf_if_with_results(%a: memref<64xbf16>, %c: i1) {
  %i = arith.constant 0 : index
  %z = arith.constant 0.0 : bf16
  %v0 = vector.transfer_read %a[%i], %z {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %r = scf.if %c -> (vector<64xbf16>) {
    %w = vector.transfer_read %a[%i], %z {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
    %s = arith.addf %w, %v0 : vector<64xbf16>
    scf.yield %s : vector<64xbf16>
  } else {
    scf.yield %v0 : vector<64xbf16>
  }
  vector.transfer_write %r, %a[%i] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  return
}

// CHECK-LABEL: func.func @affine_if_with_results
// CHECK: %[[AIF:.*]]:2 = affine.if #{{.*}}(%{{.*}}) -> (vector<64xbf16>, !air.async.token)
// CHECK: air.execute [%[[AIF]]#1
// CHECK-NEXT: vector.transfer_write %[[AIF]]#0
#set = affine_set<(d0) : (d0 == 0)>
func.func @affine_if_with_results(%a: memref<64xbf16>, %n: index) {
  %i = arith.constant 0 : index
  %z = arith.constant 0.0 : bf16
  %v0 = vector.transfer_read %a[%i], %z {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %r = affine.if #set(%n) -> (vector<64xbf16>) {
    %w = vector.transfer_read %a[%i], %z {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
    %s = arith.addf %w, %v0 : vector<64xbf16>
    affine.yield %s : vector<64xbf16>
  } else {
    affine.yield %v0 : vector<64xbf16>
  }
  vector.transfer_write %r, %a[%i] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  return
}
