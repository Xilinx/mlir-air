//===- ping_pong_shared_resident_ring.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -air-ping-pong-transform | FileCheck %s

// Shared-resident-ring ping-pong mode (opt-in via `air.shared_resident_ring` on
// the channel decl). Two sibling get-loops in one block re-read the SAME
// resident input stream (one consume pass each). Without the attribute each loop
// gets its own 2-deep ring (4 buffers total); air-to-aie would then fuse them
// into a 4-deep ring whose halves interleave coverage -- numerically wrong. With
// the attribute the second loop is merged onto the first's ring: ONE 2-deep ring
// (here 2 buffers per get x 2 gets = 4 buffers TOTAL, shared by both loops), the
// rotation chained through the first loop's iter-arg results.

// SHARED: the merged form allocates the ring exactly ONCE (4 allocs: x/w ping +
// x/w pong) -- not 8 -- and the second loop chains its iter args from the first
// loop's results.
// CHECK-LABEL: shared_ring
// CHECK-COUNT-4: memref.alloc()
// CHECK-NOT: memref.alloc()
// CHECK: %[[L0:.*]]:4 = scf.for {{.*}} iter_args
// CHECK: %[[L1:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}} = %[[L0]]#0, %{{.*}} = %[[L0]]#1, %{{.*}} = %[[L0]]#2, %{{.*}} = %[[L0]]#3)

module {
  air.channel @inX [1] {air.shared_resident_ring}
  air.channel @inW [1] {air.shared_resident_ring}
  func.func @shared_ring() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%c=%c1, %d=%c1) {
      %1 = air.segment async {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c1s = arith.constant 1 : index
        %2 = air.wait_all async
        %3 = scf.for %v1 = %c0 to %c4 step %c1s iter_args(%t = %2) -> (!air.async.token) {
          %g0 = scf.for %j = %c0 to %c8 step %c1s iter_args(%tt = %t) -> (!air.async.token) {
            %tx, %bx = air.execute [%tt] -> (memref<256xi8, 2>) {
              %al = memref.alloc() : memref<256xi8, 2>
              air.execute_terminator %al : memref<256xi8, 2>
            }
            %gx = air.channel.get async [%tx] @inX[] (%bx[] [] []) : (memref<256xi8, 2>)
            %tw, %bw = air.execute [%tt] -> (memref<2560xi8, 2>) {
              %al = memref.alloc() : memref<2560xi8, 2>
              air.execute_terminator %al : memref<2560xi8, 2>
            }
            %gw = air.channel.get async [%tw] @inW[] (%bw[] [] []) : (memref<2560xi8, 2>)
            %cc = air.execute [%gx, %gw] {
              func.call @acc(%bx, %bw) : (memref<256xi8, 2>, memref<2560xi8, 2>) -> ()
            }
            %dx = air.execute [%cc] { memref.dealloc %bx : memref<256xi8, 2> }
            %dw = air.execute [%cc] { memref.dealloc %bw : memref<2560xi8, 2> }
            %w = air.wait_all async [%dx, %dw]
            scf.yield %w : !air.async.token
          }
          %g1 = scf.for %j = %c0 to %c8 step %c1s iter_args(%tt = %g0) -> (!air.async.token) {
            %tx, %bx = air.execute [%tt] -> (memref<256xi8, 2>) {
              %al = memref.alloc() : memref<256xi8, 2>
              air.execute_terminator %al : memref<256xi8, 2>
            }
            %gx = air.channel.get async [%tx] @inX[] (%bx[] [] []) : (memref<256xi8, 2>)
            %tw, %bw = air.execute [%tt] -> (memref<2560xi8, 2>) {
              %al = memref.alloc() : memref<2560xi8, 2>
              air.execute_terminator %al : memref<2560xi8, 2>
            }
            %gw = air.channel.get async [%tw] @inW[] (%bw[] [] []) : (memref<2560xi8, 2>)
            %cc = air.execute [%gx, %gw] {
              func.call @acc(%bx, %bw) : (memref<256xi8, 2>, memref<2560xi8, 2>) -> ()
            }
            %dx = air.execute [%cc] { memref.dealloc %bx : memref<256xi8, 2> }
            %dw = air.execute [%cc] { memref.dealloc %bw : memref<2560xi8, 2> }
            %w = air.wait_all async [%dx, %dw]
            scf.yield %w : !air.async.token
          }
          scf.yield %g1 : !air.async.token
        }
      }
    }
    return
  }
  func.func private @acc(%a: memref<256xi8, 2>, %b: memref<2560xi8, 2>)
}
