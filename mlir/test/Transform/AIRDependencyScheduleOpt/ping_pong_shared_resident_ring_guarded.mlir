//===- ping_pong_shared_resident_ring_guarded.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong -air-ping-pong-transform | FileCheck %s

// The boundary of shared-resident-ring merging: a GUARD ends the chain.
//
// Two sibling get-loops re-reading one marked stream merge onto a single ring
// (see ping_pong_shared_resident_ring.mlir: 4 allocs, the second loop's iter
// args chained from the first's results). Wrap each loop in its own scf.if and
// they no longer merge -- the chain is keyed on (block, channel-set), and a
// guard gives each loop a block of its own. Each still ping-pongs, so the cost
// is not the rotation, it is TWO rings where the declaration asked for one:
// 8 allocs, and the two rings cover the stream independently.
//
// This is conservative, not accidental. Guarded siblings may be mutually
// exclusive arms, and the same reasoning already exempts them from the
// second-consumer FIFO warning in air-enforce-channel-fifo-order. Merging rings
// across arms would additionally require the allocs to be hoisted out of the
// guards so both arms can name the same buffers -- a transform, not a relaxed
// predicate.
//
// It matters because the guard is not always the author's choice. A transfer
// spelled as air.dma_memcpy_nd names BOTH endpoints, so a feed whose far buffer
// differs per tile can only select it with a guard; the ring merge is then off
// the table for reasons that have nothing to do with the data movement.

// CHECK-LABEL: guarded_loops
// Two independent 2-deep rings, not one: 8 allocs, where the unguarded sibling
// pair gets 4.
// CHECK-COUNT-8: memref.alloc()
// CHECK-NOT: memref.alloc()

module {
  air.channel @inX [1] {air.shared_resident_ring}
  air.channel @inW [1] {air.shared_resident_ring}
  func.func @guarded_loops() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%c=%c1, %d=%c1) {
      %1 = air.segment async {
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c1s = arith.constant 1 : index
        %true = arith.constant true
        %2 = air.wait_all async
        %3 = scf.for %v1 = %c0 to %c4 step %c1s iter_args(%t = %2) -> (!air.async.token) {
          %if0 = scf.if %true -> (!air.async.token) {
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
            scf.yield %g0 : !air.async.token
          } else {
            %e0 = air.wait_all async [%t]
            scf.yield %e0 : !air.async.token
          }
          %if1 = scf.if %true -> (!air.async.token) {
            %g1 = scf.for %j = %c0 to %c8 step %c1s iter_args(%tt = %if0) -> (!air.async.token) {
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
          } else {
            %e1 = air.wait_all async [%if0]
            scf.yield %e1 : !air.async.token
          }
          scf.yield %if1 : !air.async.token
        }
      }
    }
    return
  }
  func.func private @acc(%a: memref<256xi8, 2>, %b: memref<2560xi8, 2>)
}
