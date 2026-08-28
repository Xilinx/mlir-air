//===- fuse_channels_nfl_complementary_arms.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels | FileCheck %s

// Two channels whose puts sit in the two arms of one scf.if must not be fused
// by the NFL (new-for-loop) path.
//
// That path rewrites a pair of channels into a single loop and keeps only the
// destination's ops, on the understanding that the loop's trip count is what
// tells the two apart. Complementary arms break the understanding: the
// condition does not depend on the loop being created, so the surviving arm
// runs on every trip and the erased arm's data is never moved at all.
//
// The failure is silent. Op counts, call counts and channel-op counts all stay
// the same -- only the flows for the dropped channel go missing, four passes
// later, and the kernel returns uncorrelated output. That is how it reached
// hardware: every static comparison the conversion was gated on still matched.
//
// Both puts must survive.

// CHECK-LABEL: @complementary_arms
// CHECK: scf.if
// CHECK: air.channel.put{{.*}}@chan_a
// CHECK: else
// CHECK: air.channel.put{{.*}}@chan_b
// CHECK: air.channel.get{{.*}}@chan_a
// CHECK: air.channel.get{{.*}}@chan_b

air.channel @chan_a [1, 1]
air.channel @chan_b [1, 1]

func.func @complementary_arms(%arg0: memref<64x64xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%tx) in (%sx=%c2) args(%a=%arg0) : memref<64x64xbf16> attributes {id = 1 : i32} {
    %1 = air.segment async args(%b=%tx, %c=%a) : index, memref<64x64xbf16> attributes {id = 2 : i32} {
      %cc0 = arith.constant 0 : index
      %cc1 = arith.constant 1 : index
      %cc2 = arith.constant 2 : index
      %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
      %t0 = air.wait_all async
      %2 = scf.for %i = %cc0 to %cc2 step %cc1 iter_args(%t = %t0) -> (!air.async.token) {
        %pred = arith.cmpi eq, %b, %cc0 : index
        %3 = scf.if %pred -> (!air.async.token) {
          %4 = air.channel.put async [%t] @chan_a[%cc0, %cc0] (%alloc[] [] []) {id = 1 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %4 : !air.async.token
        } else {
          %4 = air.channel.put async [%t] @chan_b[%cc0, %cc0] (%alloc[] [] []) {id = 2 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %4 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
      %5 = air.herd @herd_0 async tile (%hx, %hy) in (%hsx=%cc1, %hsy=%cc1) args(%d=%b) : index attributes {id = 3 : i32} {
        %ch0 = arith.constant 0 : index
        %l1 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %pred2 = arith.cmpi eq, %d, %ch0 : index
        scf.if %pred2 {
          %6 = air.channel.get async @chan_a[%ch0, %ch0] (%l1[] [] []) {id = 3 : i32} : (memref<64x64xbf16, 2 : i32>)
        } else {
          %6 = air.channel.get async @chan_b[%ch0, %ch0] (%l1[] [] []) {id = 4 : i32} : (memref<64x64xbf16, 2 : i32>)
        }
        memref.dealloc %l1 : memref<64x64xbf16, 2 : i32>
      }
      memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
    }
  }
  return
}
