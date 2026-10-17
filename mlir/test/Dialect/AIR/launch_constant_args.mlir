//===- launch_constant_args.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -canonicalize | FileCheck %s

// A constant passed into a launch is rebuilt inside the launch body: the
// launch-level put sees the constant offset, and the launch drops the
// argument. The segment's operand is now that constant, and stays an
// operand.
// CHECK-LABEL: func.func @launch_constant_arg
// CHECK: air.launch {{.*}} args(%[[A:.*]]=%arg0) : memref<64xi32> {
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: air.channel.put  @c[] (%[[A]][8] [16] [1])
// CHECK: air.segment @seg  args(%[[S:.*]]=%[[C8]]) : index
// CHECK: memref.load %{{.*}}[%[[S]]]
air.channel @c [1, 1]
func.func @launch_constant_arg(%arg0: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  air.launch (%l) in (%ls=%c1) args(%a=%arg0, %n=%c8) : memref<64xi32>, index {
    %c16 = arith.constant 16 : index
    %c1_0 = arith.constant 1 : index
    air.channel.put @c[] (%a[%n] [%c16] [%c1_0]) : (memref<64xi32>)
    air.segment @seg args(%sn=%n) : index {
      %buf = memref.alloc() : memref<16xi32, 1>
      air.channel.get @c[] (%buf[] [] []) : (memref<16xi32, 1>)
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%sn] : memref<16xi32, 1>
      memref.store %v, %buf[%c0] : memref<16xi32, 1>
    }
  }
  return
}
