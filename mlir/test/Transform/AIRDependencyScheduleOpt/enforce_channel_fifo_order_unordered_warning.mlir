//===- enforce_channel_fifo_order_unordered_warning.mlir -------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-enforce-channel-fifo-order can only order same-block siblings. Endpoints
// addressing one channel slot from different blocks are left racing, so report
// them -- unless the declaration says several endpoints are the intent.

// RUN: air-opt %s -air-enforce-channel-fifo-order -split-input-file -verify-diagnostics

// Two producers on a CIRCUIT channel, in sibling blocks, with no token between
// them. This is the case that hangs: air-to-aie's channel specialization keeps
// one producer tile and leaves the other with a DMA program and no flow behind
// it.
air.channel @circuit [2, 2]
func.func @two_producers_circuit(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %t0 = air.wait_all async
  %p0 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%t0) -> !air.async.token {
    // expected-note @+1 {{earlier endpoint here}}
    %e = air.channel.put async @circuit[%i, %j] (%arg0[0, 0] [32, 32] [64, 1]) : (memref<64x64xi32>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  %t1 = air.wait_all async
  %p1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%t1) -> !air.async.token {
    // expected-warning @+1 {{is a second producer on channel @circuit}}
    %e = air.channel.put async @circuit[%i, %j] (%arg1[0, 0] [32, 32] [64, 1]) : (memref<64x64xi32>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  return
}

// -----

// Same shape, but the declaration is npu_dma_packet. Converging several same-id
// producers onto one destination S2MM is exactly what a packet flow is for, so
// this is silent.
air.channel @packet [2, 2] {channel_type = "npu_dma_packet"}
func.func @two_producers_packet(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %t0 = air.wait_all async
  %p0 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%t0) -> !air.async.token {
    %e = air.channel.put async @packet[%i, %j] (%arg0[0, 0] [32, 32] [64, 1]) : (memref<64x64xi32>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  %t1 = air.wait_all async
  %p1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%t1) -> !air.async.token {
    %e = air.channel.put async @packet[%i, %j] (%arg1[0, 0] [32, 32] [64, 1]) : (memref<64x64xi32>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  return
}

// -----

// Two consumers sharing a resident ring: air.shared_resident_ring says
// air-ping-pong-transform is meant to merge the sibling get-loops. Silent.
air.channel @ring [2, 2] {air.shared_resident_ring}
func.func @two_consumers_shared_ring() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %t0 = air.wait_all async
  %p0 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%t0) -> !air.async.token {
    %buf = memref.alloc() : memref<32x32xi32, 2>
    %e = air.channel.get async @ring[%i, %j] (%buf[] [] []) : (memref<32x32xi32, 2>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  %t1 = air.wait_all async
  %p1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%t1) -> !air.async.token {
    %buf = memref.alloc() : memref<32x32xi32, 2>
    %e = air.channel.get async @ring[%i, %j] (%buf[] [] []) : (memref<32x32xi32, 2>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  return
}

// -----

// Ordered explicitly: the second loop's init token comes from the first loop's
// result, so nothing races. Silent.
air.channel @ordered [2, 2]
func.func @two_producers_ordered(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %t0 = air.wait_all async
  %p0 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%t0) -> !air.async.token {
    %e = air.channel.put async @ordered[%i, %j] (%arg0[0, 0] [32, 32] [64, 1]) : (memref<64x64xi32>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  %p1 = scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%p0) -> !air.async.token {
    %e = air.channel.put async @ordered[%i, %j] (%arg1[0, 0] [32, 32] [64, 1]) : (memref<64x64xi32>)
    scf.reduce(%e : !air.async.token) {
    ^bb0(%a: !air.async.token, %b: !air.async.token):
      %w = air.wait_all async [%a, %b]
      scf.reduce.return %w : !air.async.token
    }
  }
  return
}
