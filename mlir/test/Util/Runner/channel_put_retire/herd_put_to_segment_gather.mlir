//===- herd_put_to_segment_gather.mlir               -*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A herd of four tiles computes, then each tile puts its result on one channel.
// The receiver is an scf.parallel of four gets outside the herd, so the two
// sides agree on the instance count -- four against four -- and the channel
// matches.
//
// The put still has to retire, and it used to ask the channel-wide dispatch
// counter whether it had. That counter is shared with the get, and the get
// zeroes it on completion so the next round can start. Whenever the get was
// executed first the put read zero against its own total of four and could
// never retire, and the simulation stalled instead of returning a latency.
//
// Two things in the design are load-bearing, and both are about ordering
// rather than about counting:
//
//   * the matvec. The put is not ready until it has run, by which time the
//     gets have long since been reached. Take it out and the two sides start
//     together, the put is executed first, and the fault hides.
//   * the herd extent against the port count. At four tiles and four ports
//     every instance dispatches in one round and the put and its get complete
//     in the same step, which is what makes the order decide. Raise the extent
//     past the port count and the transfer takes several rounds and the fault
//     hides again -- measured at five, six and seven.
//
// Stalls at 209 cycles with the retirement change reverted.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// CHECK: Latency (all-iterations mode): 0.211us

module {
  air.channel @part [1, 1]
  func.func @test(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %1 = air.segment async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %cw_s = arith.constant 8 : index
        %2 = air.herd @producer async tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c4_s) attributes {id = 3 : i32} {
          %tok_w, %w = air.execute -> (memref<8x8xi8, 2>) {
            %alloc = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %alloc : memref<8x8xi8, 2>
          }
          %tok_v, %v = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_o, %o = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_c = air.execute [%tok_w, %tok_v, %tok_o] {
            linalg.matvec {air.op_cost = "priced"} ins(%w, %v : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o : memref<8xi8, 2>)
          }
          %4 = air.channel.put async [%tok_c] @part[] (%o[] [] []) {id = 1 : i32} : (memref<8xi8, 2>)
        }
        %tok_p, %parts = air.execute -> (memref<4x8xi8, 1>) {
          %alloc = memref.alloc() : memref<4x8xi8, 1>
          air.execute_terminator %alloc : memref<4x8xi8, 1>
        }
        %3 = scf.parallel (%m) = (%c0_s) to (%c4_s) step (%c1_s) init (%tok_p) -> !air.async.token {
          %4 = air.channel.get async [%tok_p] @part[] (%parts[%m, %c0_s] [%c1_s, %cw_s] [%cw_s, %c1_s]) {id = 2 : i32} : (memref<4x8xi8, 1>)
          scf.reduce(%4 : !air.async.token) {
          ^bb0(%a: !air.async.token, %b: !air.async.token):
            %5 = air.wait_all async [%a, %b]
            scf.reduce.return %5 : !air.async.token
          }
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
