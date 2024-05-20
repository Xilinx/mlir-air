//===- airrt_ops.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s
// CHECK-LABEL: func.func @f0()
// CHECK: %[[VAL_0:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[VAL_1:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[VAL_2:.*]] = airrt.wait_all %[[VAL_0]], %[[VAL_1]] : !airrt.event
// CHECK: airrt.wait_all %[[VAL_2]]
// CHECK: %[[VAL_3:.*]] = airrt.herd_load "herd_0" () : () -> i64
// CHECK: %[[VAL_4:.*]], %[[VAL_5:.*]] = airrt.herd_load "herd_0" () : () -> (i64, !airrt.event)
// CHECK: %[[VAL_6:.*]] = arith.constant 64 : i64
// CHECK: %[[VAL_7:.*]] = arith.constant 42 : i64
// CHECK: %[[VAL_8:.*]], %[[VAL_9:.*]] = airrt.herd_load "herd_1" (%[[VAL_7]]) : (i64) -> (i64, !airrt.event)
// CHECK: %[[VAL_10:.*]] = airrt.herd_load "herd_2" (%[[VAL_7]], %[[VAL_6]]) : (i64, i64) -> i64
module {

func.func @f0() {
    %event1 = airrt.wait_all : !airrt.event
    %event2 = airrt.wait_all : !airrt.event
    %event3 = airrt.wait_all %event1, %event2 : !airrt.event
    airrt.wait_all %event3

    // load herd without runtime parameters
    %herd_load = airrt.herd_load "herd_0" () : () -> i64
    %h0, %e0 = airrt.herd_load "herd_0" () : () -> (i64, !airrt.event)

    // load herd with runtime parameters
    %c64 = arith.constant 64 : i64
    %c42 = arith.constant 42 : i64
    %h1, %e1 = airrt.herd_load "herd_1" (%c42) : (i64) -> (i64, !airrt.event)
    %h2 = airrt.herd_load "herd_2" (%c42, %c64) : (i64, i64) -> (i64)

    return
}

}