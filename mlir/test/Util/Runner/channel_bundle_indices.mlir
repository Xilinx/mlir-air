//===- channel_bundle_indices.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A pipeline seeded and drained on the two ends of one channel bundle, which
// is how a chain of stages is written when the stage index selects the link:
//
//   launch put @io[0] --> stage (10480 cyc) --> launch get @io[1]
//
// The launch's drain becomes ready as soon as its own seed completes, long
// before the stage has produced anything. Keying channel progress on the
// symbol alone lets that drain be satisfied by the seed's put -- the two are
// different entries of the bundle, but indistinguishable without the index.
// The stage's own get then starves and the run stalls.
//
// This is why the entries have to be told apart. A serial chain where every
// stage both receives and sends does not expose it, because there the
// dataflow orders the pairings anyway; it takes a producer and a consumer of
// *different* entries being ready at the same time.
//
// Indices that are not compile-time constants cannot be resolved statically
// and still fall back to the symbol alone, so IR that indexes a bundle by a
// herd induction variable is unaffected.

// RUN: air-runner %s -f test -m %S/custom_op/arch.json | FileCheck %s

// The stage runs and the launch terminates: one @nn, no stall.
// CHECK: "name": "air.custom",
// CHECK: "name": "LaunchTerminator",
// CHECK: Latency (all-iterations mode): 10.

module {
  air.channel @io [2]
  func.func @test(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %launch = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%larg=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %c0_o = arith.constant 0 : index
      %c1_o = arith.constant 1 : index

      // The stage: receives entry 0, works, sends entry 1.
      %seg = air.segment async attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %tok, %buf = air.execute -> (memref<64xi8, 1>) {
          %a = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %a : memref<64xi8, 1>
        }
        %rx = air.channel.get async [%tok] @io[%c0_s] (%buf[] [] []) {id = 20 : i32} : (memref<64xi8, 1>)
        %x = air.execute [%rx] {
          air.custom @nn operands (%buf) : memref<64xi8, 1>
        }
        %tx = air.channel.put async [%x] @io[%c1_s] (%buf[] [] []) {id = 21 : i32} : (memref<64xi8, 1>)
      }

      // Seed then drain, chained: the drain is ready immediately.
      %in = air.channel.put async @io[%c0_o] (%larg[] [] []) {id = 30 : i32} : (memref<64xi8>)
      %out = air.channel.get async [%in] @io[%c1_o] (%larg[] [] []) {id = 31 : i32} : (memref<64xi8>)
    }
    return
  }
}
