//===- dependency_unpaired_channel.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dependency-canonicalize -verify-diagnostics -o /dev/null

// An unpaired channel is a front-end error, and it is diagnosed as one. It has
// to STAY a diagnostic.
//
// The op still exists in the IR, and the dependency-graph builder's caller
// records edges against whatever vertex id comes back. Returning 0 for the
// unpaired case handed it the id of a real, unrelated vertex -- so the graph
// was silently miswired, and DirectedAdjacencyMap::getClosure() then walked it
// forever or indexed out of range and aborted. Either way the message that had
// already named the exact problem scrolled past, and what the user saw was a
// hang or a std::vector assertion inside a graph utility.
//
// This needs BOTH channels to bite: the unpaired one to inject the bad vertex,
// and a paired one crossing into the herd to give the closure a graph big
// enough to walk. With the unpaired get alone the pass finishes by luck.

air.channel @unpaired [1]
air.channel @paired [1]
func.func @unpaired_alongside_paired(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %l2 = memref.alloc() : memref<2048xbf16, 1>
      scf.for %r = %c0 to %c8 step %c1_s {
        // expected-error @+1 {{found channel op not in pairs}}
        air.channel.get @unpaired[%c0] (%l2[] [] []) : (memref<2048xbf16, 1>)
      }
      scf.for %r2 = %c0 to %c8 step %c1_s {
        air.channel.put @paired[%c0] (%l2[0] [256] [1]) : (memref<2048xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c8_h = arith.constant 8 : index
        scf.for %j = %c0_h to %c8_h step %c1_h {
          %a = memref.alloc() : memref<256xbf16, 2>
          air.channel.get @paired[%c0_h] (%a[] [] []) : (memref<256xbf16, 2>)
          memref.dealloc %a : memref<256xbf16, 2>
        }
      }
    }
  }
  return
}
