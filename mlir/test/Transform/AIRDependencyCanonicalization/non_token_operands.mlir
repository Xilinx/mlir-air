//===- non_token_operands.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize --split-input-file --verify-diagnostics | FileCheck %s

// A branch op's joint token is joined from the token each branch yields. A
// branch may also yield ordinary data, whose defining op is not a graph vertex.

// CHECK-LABEL: func.func @index_switch_non_token_result
// CHECK: scf.index_switch
// CHECK: air.channel.put async
air.channel @channel_0 [1]
func.func @index_switch_non_token_result(%arg0: memref<32xi32>, %arg1: memref<32xi32>, %sel: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7_i32 = arith.constant 7 : i32
  %c8_i32 = arith.constant 8 : i32
  %r:2 = scf.index_switch %sel -> !air.async.token, i32
  case 0 {
    %t = air.channel.put async @channel_0[%c0] (%arg0[%c0] [%c4] [%c1]) : (memref<32xi32>)
    scf.yield %t, %c7_i32 : !air.async.token, i32
  }
  default {
    %t = air.wait_all async
    scf.yield %t, %c8_i32 : !air.async.token, i32
  }
  %f:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %r#0, %b = %r#1) -> (!air.async.token, i32) {
    %g = air.channel.get async [%a] @channel_0[%c0] (%arg1[%c0] [%c4] [%c1]) : (memref<32xi32>)
    %n = arith.addi %b, %c7_i32 : i32
    scf.yield %g, %n : !air.async.token, i32
  }
  return
}

// -----

// An scf.for may carry ordinary data alongside its async token; only the token
// is a dependency edge.

// CHECK-LABEL: func.func @scf_for_non_token_iter_arg
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK: air.channel.put async
air.channel @channel_1 [1]
func.func @scf_for_non_token_iter_arg(%arg0: memref<32xi32>, %arg1: memref<32xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = air.wait_all async
  %1:2 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0, %arg4 = %c0_i32) -> (!air.async.token, i32) {
    %2 = air.channel.put async [%arg3] @channel_1[%c0] (%arg0[%c0] [%c4] [%c1]) : (memref<32xi32>)
    %3 = arith.addi %arg4, %c1_i32 : i32
    scf.yield %2, %3 : !air.async.token, i32
  }
  %4 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %1#0) -> (!air.async.token) {
    %5 = air.channel.get async [%arg3] @channel_1[%c0] (%arg1[%c0] [%c4] [%c1]) : (memref<32xi32>)
    scf.yield %5 : !air.async.token
  }
  return
}
