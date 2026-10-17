//===- ping_pong_core_scratch.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -air-ping-pong-transform | FileCheck %s

// The loop's channel buffer is doubled for ping-pong. The scratch buffer that
// the core writes and reads back within a trip, and that no channel touches,
// has no transfer to overlap with: it moves out of the loop once instead of
// being doubled.
// CHECK-LABEL: func.func @core_scratch
// CHECK-COUNT-3: memref.alloc() : memref<64xf32, 2>
// CHECK-NOT: memref.alloc()
// CHECK: scf.for {{.*}} step %c2

module {
  air.channel @in [1, 1]
  func.func @core_scratch() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %cst = arith.constant 0.0 : f32
          %w = air.wait_all async
          %3 = scf.for %i = %c0 to %c8 step %c1_h iter_args(%t = %w) -> (!air.async.token) {
            %ta, %buf = air.execute [%t] -> (memref<64xf32, 2>) {
              %a = memref.alloc() : memref<64xf32, 2>
              air.execute_terminator %a : memref<64xf32, 2>
            }
            %ts, %scratch = air.execute [%t] -> (memref<64xf32, 2>) {
              %a = memref.alloc() : memref<64xf32, 2>
              air.execute_terminator %a : memref<64xf32, 2>
            }
            %g = air.channel.get async [%ta] @in[] (%buf[] [] []) : (memref<64xf32, 2>)
            %tc = air.execute [%g, %ts] {
              %v = vector.transfer_read %buf[%c0], %cst {in_bounds = [true]} : memref<64xf32, 2>, vector<16xf32>
              vector.transfer_write %v, %scratch[%c0] {in_bounds = [true]} : vector<16xf32>, memref<64xf32, 2>
              %u = vector.transfer_read %scratch[%c0], %cst {in_bounds = [true]} : memref<64xf32, 2>, vector<16xf32>
              vector.transfer_write %u, %buf[%c0] {in_bounds = [true]} : vector<16xf32>, memref<64xf32, 2>
            }
            %d0 = air.execute [%tc] {
              memref.dealloc %buf : memref<64xf32, 2>
            }
            %d1 = air.execute [%tc] {
              memref.dealloc %scratch : memref<64xf32, 2>
            }
            %y = air.wait_all async [%d0, %d1]
            scf.yield %y : !air.async.token
          }
        }
      }
    }
    return
  }
}
