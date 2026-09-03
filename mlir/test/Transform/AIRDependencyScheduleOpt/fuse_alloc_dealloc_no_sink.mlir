//===- fuse_alloc_dealloc_no_sink.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-fuse-alloc-dealloc %s | FileCheck %s
// RUN: air-opt -air-fuse-alloc-dealloc -air-label-scf-for-to-ping-pong %s | FileCheck %s --check-prefix=PP

// `air.disable_alloc_sink` keeps an alloc where the design put it.
//
// Sinking is normally free -- it narrows a live range -- but it is observable
// one pass later. air-label-scf-for-to-ping-pong duplicates exactly the allocs
// in a candidate loop's body, and rejects the WHOLE loop if any of them is not
// dead on entry. A scratch buffer written by an opaque callee cannot be proven
// dead on entry, so sinking one into a loop disqualifies the other allocs in
// that body -- the air.channel.get-filled ones that wanted the ping-pong.

module {

air.channel @chX [1, 1]
air.channel @chW [1, 1]
func.func private @acc(memref<2048xbf16, 2 : i32>, memref<2560xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>)

// Without the attribute the scratch sinks in, and the loop is NOT labeled:
// three allocs in the body, no `unroll`.

// CHECK-LABEL:   @sinks_by_default
// CHECK:   scf.for
// CHECK:   memref.alloc() : memref<8192xbf16, 2 : i32>
// CHECK:   memref.alloc() : memref<2048xbf16, 2 : i32>
// CHECK:   memref.alloc() : memref<2560xbf16, 2 : i32>

// PP-LABEL:   @sinks_by_default
// PP-NOT:   unroll

  func.func @sinks_by_default() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %tok, %ws = air.execute -> (memref<8192xbf16, 2 : i32>) {
      %alloc = memref.alloc() : memref<8192xbf16, 2 : i32>
      air.execute_terminator %alloc : memref<8192xbf16, 2 : i32>
    }
    %l = scf.for %i = %c0 to %c8 step %c1 iter_args(%it = %tok) -> (!air.async.token) {
      %t1, %x = air.execute -> (memref<2048xbf16, 2 : i32>) {
        %a = memref.alloc() : memref<2048xbf16, 2 : i32>
        air.execute_terminator %a : memref<2048xbf16, 2 : i32>
      }
      %g1 = air.channel.get async [%it, %t1] @chX[] (%x[] [] []) : (memref<2048xbf16, 2 : i32>)
      %t2, %w = air.execute -> (memref<2560xbf16, 2 : i32>) {
        %a = memref.alloc() : memref<2560xbf16, 2 : i32>
        air.execute_terminator %a : memref<2560xbf16, 2 : i32>
      }
      %g2 = air.channel.get async [%it, %t2] @chW[] (%w[] [] []) : (memref<2560xbf16, 2 : i32>)
      %t3 = air.execute [%g1, %g2] {
        func.call @acc(%x, %w, %ws) : (memref<2048xbf16, 2 : i32>, memref<2560xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>) -> ()
      }
      %t4 = air.execute [%t3] {
        memref.dealloc %x : memref<2048xbf16, 2 : i32>
      }
      %t5 = air.execute [%t3] {
        memref.dealloc %w : memref<2560xbf16, 2 : i32>
      }
      scf.yield %t3 : !air.async.token
    }
    %td = air.execute [%l] {
      memref.dealloc %ws : memref<8192xbf16, 2 : i32>
    }
    return
  }

// With it the scratch stays above the loop, and the loop IS labeled -- so the
// two channel.get-filled allocs get their ping-pong and the scratch stays
// single. That is the whole point of the attribute; the first CHECK alone
// would pass on a pass that did nothing.

// CHECK-LABEL:   @no_sink_lets_ping_pong_fire
// CHECK:   memref.alloc() {air.disable_alloc_sink} : memref<8192xbf16, 2 : i32>
// CHECK:   scf.for
// CHECK-NOT:   memref.alloc() {{.*}} : memref<8192xbf16, 2 : i32>

// PP-LABEL:   @no_sink_lets_ping_pong_fire
// PP:   unroll = 2 : i32

  func.func @no_sink_lets_ping_pong_fire() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %tok, %ws = air.execute -> (memref<8192xbf16, 2 : i32>) {
      %alloc = memref.alloc() {air.disable_alloc_sink} : memref<8192xbf16, 2 : i32>
      air.execute_terminator %alloc : memref<8192xbf16, 2 : i32>
    }
    %l = scf.for %i = %c0 to %c8 step %c1 iter_args(%it = %tok) -> (!air.async.token) {
      %t1, %x = air.execute -> (memref<2048xbf16, 2 : i32>) {
        %a = memref.alloc() : memref<2048xbf16, 2 : i32>
        air.execute_terminator %a : memref<2048xbf16, 2 : i32>
      }
      %g1 = air.channel.get async [%it, %t1] @chX[] (%x[] [] []) : (memref<2048xbf16, 2 : i32>)
      %t2, %w = air.execute -> (memref<2560xbf16, 2 : i32>) {
        %a = memref.alloc() : memref<2560xbf16, 2 : i32>
        air.execute_terminator %a : memref<2560xbf16, 2 : i32>
      }
      %g2 = air.channel.get async [%it, %t2] @chW[] (%w[] [] []) : (memref<2560xbf16, 2 : i32>)
      %t3 = air.execute [%g1, %g2] {
        func.call @acc(%x, %w, %ws) : (memref<2048xbf16, 2 : i32>, memref<2560xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>) -> ()
      }
      %t4 = air.execute [%t3] {
        memref.dealloc %x : memref<2048xbf16, 2 : i32>
      }
      %t5 = air.execute [%t3] {
        memref.dealloc %w : memref<2560xbf16, 2 : i32>
      }
      scf.yield %t3 : !air.async.token
    }
    %td = air.execute [%l] {
      memref.dealloc %ws : memref<8192xbf16, 2 : i32>
    }
    return
  }

}
