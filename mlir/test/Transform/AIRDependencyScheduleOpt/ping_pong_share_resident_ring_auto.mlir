//===- ping_pong_share_resident_ring_auto.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -air-ping-pong-transform | FileCheck %s

// Auto-detected resident-ring sharing (no attribute). Two sibling get-loops in
// one block re-read the SAME resident streams (@inX + @inW), one consume pass
// each. Each would otherwise build its own 2-deep ring (8 buffers total); air-to-
// aie would then either fuse them into a deeper interleaved ring (numerically
// wrong) or the per-phase rings overrun the tile L1 / BD budget. shareResidentRings
// detects the pattern structurally -- same block, same channel set, transient
// buffers -- and merges the second loop onto the first's ring: ONE shared 2-deep
// ring (4 buffers TOTAL), the rotation chained through the first loop's results.

// CHECK-LABEL: shared_ring
// CHECK-COUNT-4: memref.alloc()
// CHECK-NOT: memref.alloc()
// CHECK: %[[L0:.*]]:4 = scf.for {{.*}} iter_args
// CHECK: %[[L1:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}} = %[[L0]]#0, %{{.*}} = %[[L0]]#1, %{{.*}} = %[[L0]]#2, %{{.*}} = %[[L0]]#3)

module {
  air.channel @inX [1]
  air.channel @inW [1]
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
