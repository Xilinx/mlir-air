//===- air_pipeline.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s
module  {
  func.func @launch(%arg0: i32) {
    %c4 = arith.constant 4 : index
    air.herd tile (%arg1, %arg2) in (%arg3=%c4, %arg4=%c4) args(%arg5=%arg0, %arg6=%arg0) : i32,i32 {
      %c1_i32 = arith.constant 1 : i32
      // CHECK: air.pipeline attributes {direction = "horiz"} {
      air.pipeline attributes {direction = "horiz"} {
        // CHECK: %{{.*}} = air.pipeline.stage args(%{{.*}}=%{{.*}}) : i32 {
        // CHECK: air.pipeline.yield %{{.*}} : i32
        // CHECK: } : i32
        %output1 = air.pipeline.stage args(%in = %c1_i32) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        } : i32
        %output2 = air.pipeline.stage args(%in = %output1) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        } : i32
        %output3 = air.pipeline.stage args(%in = %output2) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        } : i32
        air.pipeline.stage args(%in = %output3) : i32 {
          %o = arith.addi %in, %c1_i32 : i32
          air.pipeline.yield %o : i32
        }
        air.pipeline.terminator
      }
      air.herd_terminator
    }
    return
  }
}
