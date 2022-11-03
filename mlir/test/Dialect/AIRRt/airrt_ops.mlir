//===- airrt_ops.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s
// CHECK: %[[E0:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[E1:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[E2:.*]] = airrt.wait_all %[[E0]], %[[E1]] : !airrt.event
// CHECK: airrt.wait_all %[[E2]]
module {

func.func @f0() {
    %event1 = airrt.wait_all : !airrt.event
    %event2 = airrt.wait_all : !airrt.event
    %event3 = airrt.wait_all %event1, %event2 : !airrt.event
    airrt.wait_all %event3
    %herd_load = airrt.herd_load "herd_0" : i64
    %h, %e = airrt.herd_load "herd_0" : i64, !airrt.event
    return
}

}