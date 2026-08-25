//===- segment_iteration_space.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// An air.segment with an iteration space, containing a herd fed by a channel.
//
// Regression test for two defects that made a multi-instance segment simulate
// as a plausible but wrong number rather than fail:
//
//  1. tokenSpatialFactorForResource stopped its walk at the innermost
//     enclosing hierarchy, so the channel.get inside the herd never saw the
//     segment's iteration space while the matching channel.put at segment
//     level did. The put dispatched two instances and the get expected one,
//     the pairing test in executeOp(ChannelGetOp) could never hold, the get
//     never retired, and the herd was left hanging with its body unexecuted.
//     The run still exited 0, reporting 0.025us instead of 10.5us -- the
//     @nn custom op simply never appeared in the trace.
//
//  2. getResourceCost(air::SegmentOp) ignored the iteration space, so a 2x1
//     segment was billed a single DU and fit an arch that cannot hold it.
//     That half is checked by bad_launch/bad_segment_iteration_space.mlir,
//     which needs an arch too small for the instances; here the arch is big
//     enough and only the first defect is in play.
//
// Both instances are co-resident, so the pair costs the same wall clock as
// one (10480 for @nn plus channel and bookkeeping overhead) but twice the
// DUs.

// RUN: air-runner %s -f test -m %S/custom_op/arch.json | FileCheck %s

// The herd must run its body to completion: the custom op appears, and both
// terminators are reached.
// CHECK: "name": "air.custom",
// CHECK: "name": "HerdTerminator",
// CHECK: "name": "SegmentTerminator",
// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

module {
  air.channel @onchip [1, 1]
  func.func @test() {
    %c1 = arith.constant 1 : index
    %launch = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) attributes {id = 1 : i32} {
      %c2_l = arith.constant 2 : index
      %seg = air.segment async unroll(%l) in (%ls=%c2_l) attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_s = arith.constant 1 : index
        %tok, %buf = air.execute -> (memref<64xi8, 1>) {
          %a = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %a : memref<64xi8, 1>
        }
        %fw = air.channel.put async [%tok] @onchip[] (%buf[] [] []) {id = 20 : i32} : (memref<64xi8, 1>)
        %herd = air.herd @m async [%fw] tile (%tx, %ty) in (%tsx=%c1_s, %tsy=%c1_s) attributes {id = 100 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %tl, %bl = air.execute -> (memref<64xi8, 2>) {
            %a2 = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %a2 : memref<64xi8, 2>
          }
          %g = air.channel.get async [%tl] @onchip[] (%bl[] [] []) {id = 21 : i32} : (memref<64xi8, 2>)
          %x = air.execute [%g] {
            air.custom @nn operands (%bl) : memref<64xi8, 2>
          }
        }
      }
    }
    return
  }
}
