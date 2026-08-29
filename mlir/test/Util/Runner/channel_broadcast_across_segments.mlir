//===- channel_broadcast_across_segments.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A broadcast whose receivers are separate air.segments rather than the tiles
// of one air.herd.
//
// Util/Runner/channel_broadcast.mlir covers the herd-internal shape: one
// air.channel.get inside a 4x4 herd, whose spatial factor already equals the
// declared fanout. That was the only shape the runner handled. Here the
// receivers are 8 independent segments -- 8 separate get ops, each spanning one
// instance -- so no single op's spatial factor equals the fanout. The channel's
// completion test counts the gets the channel expects rather than the ones a
// single op represents; before that, this shape stalled.
//
// The point of modelling it as a broadcast is cost. A producer that fans the
// same buffer out to N consumers over a switched fabric pays for one payload,
// not N. This arch gives a DU 4 outbound ports and there are 8 receivers, so
// written as 8 separate puts the sender needs two rounds and the transfer takes
// twice as long. The companion assertion is the latency below: 0.028us here
// against 0.049us for the same fan-out written point-to-point, measured.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// One put, 8 gets -- begin and end phase for each.
// CHECK: "name": "ChannelPutOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelPutOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK: "name": "ChannelGetOp@bcast
// CHECK-NOT: "name": "ChannelPutOp@bcast

// The run has to reach the terminator: a stalled run stops early and is
// otherwise indistinguishable from a finished one, which is the failure this
// shape used to produce -- see stalled_simulation.mlir.
// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",
// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

// One payload, not 8. The same fan-out written as 8 point-to-point puts needs
// two rounds of the sender's 4 outbound ports and comes to 0.049us; measured,
// not asserted.
// CHECK: Latency (all-iterations mode): 0.028us

module {
  // fanout = prod(broadcast_shape) / prod(size) = 8
  air.channel @bcast [1] {broadcast_shape = [8]}
  func.func @test() {
    %c1 = arith.constant 1 : index
    %launch = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) attributes {id = 1 : i32} {

      // Producer: allocates once, then broadcasts.
      %prod = air.segment async attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %t0, %b0 = air.execute -> (memref<1024xbf16, 1>) {
          %a0 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a0 : memref<1024xbf16, 1>
        }
        %p = air.channel.put async [%t0] @bcast[] (%b0[] [] []) {id = 20 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 0, its own segment, taking its own copy.
      %s0 = air.segment async attributes {id = 11 : i32, x_loc = 1 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %t1, %b1 = air.execute -> (memref<1024xbf16, 1>) {
          %a1 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a1 : memref<1024xbf16, 1>
        }
        %g1 = air.channel.get async [%t1] @bcast[] (%b1[] [] []) {id = 21 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 1, its own segment, taking its own copy.
      %s1 = air.segment async attributes {id = 12 : i32, x_loc = 2 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %t2, %b2 = air.execute -> (memref<1024xbf16, 1>) {
          %a2 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a2 : memref<1024xbf16, 1>
        }
        %g2 = air.channel.get async [%t2] @bcast[] (%b2[] [] []) {id = 22 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 2, its own segment, taking its own copy.
      %s2 = air.segment async attributes {id = 13 : i32, x_loc = 3 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %t3, %b3 = air.execute -> (memref<1024xbf16, 1>) {
          %a3 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a3 : memref<1024xbf16, 1>
        }
        %g3 = air.channel.get async [%t3] @bcast[] (%b3[] [] []) {id = 23 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 3, its own segment, taking its own copy.
      %s3 = air.segment async attributes {id = 14 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 1 : i64, y_size = 1 : i64} {
        %t4, %b4 = air.execute -> (memref<1024xbf16, 1>) {
          %a4 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a4 : memref<1024xbf16, 1>
        }
        %g4 = air.channel.get async [%t4] @bcast[] (%b4[] [] []) {id = 24 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 4, its own segment, taking its own copy.
      %s4 = air.segment async attributes {id = 15 : i32, x_loc = 1 : i64, x_size = 1 : i64, y_loc = 1 : i64, y_size = 1 : i64} {
        %t5, %b5 = air.execute -> (memref<1024xbf16, 1>) {
          %a5 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a5 : memref<1024xbf16, 1>
        }
        %g5 = air.channel.get async [%t5] @bcast[] (%b5[] [] []) {id = 25 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 5, its own segment, taking its own copy.
      %s5 = air.segment async attributes {id = 16 : i32, x_loc = 2 : i64, x_size = 1 : i64, y_loc = 1 : i64, y_size = 1 : i64} {
        %t6, %b6 = air.execute -> (memref<1024xbf16, 1>) {
          %a6 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a6 : memref<1024xbf16, 1>
        }
        %g6 = air.channel.get async [%t6] @bcast[] (%b6[] [] []) {id = 26 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 6, its own segment, taking its own copy.
      %s6 = air.segment async attributes {id = 17 : i32, x_loc = 3 : i64, x_size = 1 : i64, y_loc = 1 : i64, y_size = 1 : i64} {
        %t7, %b7 = air.execute -> (memref<1024xbf16, 1>) {
          %a7 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a7 : memref<1024xbf16, 1>
        }
        %g7 = air.channel.get async [%t7] @bcast[] (%b7[] [] []) {id = 27 : i32} : (memref<1024xbf16, 1>)
      }

      // Consumer 7, its own segment, taking its own copy.
      %s7 = air.segment async attributes {id = 18 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %t8, %b8 = air.execute -> (memref<1024xbf16, 1>) {
          %a8 = memref.alloc() : memref<1024xbf16, 1>
          air.execute_terminator %a8 : memref<1024xbf16, 1>
        }
        %g8 = air.channel.get async [%t8] @bcast[] (%b8[] [] []) {id = 28 : i32} : (memref<1024xbf16, 1>)
      }
    }
    return
  }
}
