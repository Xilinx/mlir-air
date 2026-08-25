//===- stalled_simulation.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A channel whose put and get disagree on how many spatial instances they
// represent: the put sits in a 4x1 herd, so it dispatches four, while the
// single get at segment level expects one and the channel declares no
// broadcast_shape to reconcile them. This violates the static balance
// condition, and the pairing test in executeOp(ChannelGetOp) can never hold.
//
// The get is therefore never retired and every op behind it is abandoned.
// The scheduling loop stops as soon as no runner node can make progress,
// which is indistinguishable from a normal finish unless the launch
// terminator is checked -- so before this was checked, a stalled run reported
// a plausible latency and exited 0. The failure mode was a wrong number
// rather than an error, which is the expensive kind.
//
// Runs that stop because a hierarchy could not be allocated already report
// their own reason and are expected not to reach the terminator; those must
// stay quiet here, which is what Util/Runner/bad_launch covers.

// RUN: not air-runner %s -f test -m %S/arch.json 2>&1 | FileCheck %s

// CHECK: Error: simulation stalled
// CHECK-SAME: without reaching the launch terminator

module {
  air.channel @unbalanced [1, 1]
  func.func @test() {
    %c1 = arith.constant 1 : index
    %launch = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) attributes {id = 1 : i32} {
      %seg = air.segment async attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %herd = air.herd @m async tile (%tx, %ty) in (%tsx=%c4_s, %tsy=%c1_s) attributes {id = 100 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %tl, %bl = air.execute -> (memref<64xbf16, 2>) {
            %a2 = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %a2 : memref<64xbf16, 2>
          }
          %p = air.channel.put async [%tl] @unbalanced[] (%bl[] [] []) {id = 21 : i32} : (memref<64xbf16, 2>)
        }
        %tok, %buf = air.execute -> (memref<64xbf16, 1>) {
          %a = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %a : memref<64xbf16, 1>
        }
        %rx = air.channel.get async [%tok] @unbalanced[] (%buf[] [] []) {id = 20 : i32} : (memref<64xbf16, 1>)
        // Downstream work, so the stall is on the terminator's dependency
        // path rather than a dangling token the terminator never waited for.
        %sink = air.herd @sink async [%rx] tile (%sx, %sy) in (%ssx=%c1_s, %ssy=%c1_s) attributes {id = 101 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %st, %sb = air.execute -> (memref<64xbf16, 2>) {
            %sa = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %sa : memref<64xbf16, 2>
          }
        }
      }
    }
    return
  }
}
