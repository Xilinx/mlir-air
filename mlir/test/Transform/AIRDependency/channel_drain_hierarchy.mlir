//===- channel_drain_hierarchy.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// A launch-scope air.channel.get that drains data produced on-device (its
// matching air.channel.put lives inside a herd/segment, writing a disjoint L1
// buffer) has an implicit producer->consumer dependency on the producing
// segment. The memref tracer cannot see it (drain writes L3 %aout, the producer
// writes L1), so air-dependency must add an explicit edge from the segment to
// the drain get. Without it the drain floats ahead of the segment and its host
// wait is emitted before the data can exist.

// CHECK: %[[SEG:.*]] = air.segment @seg async
// The drain get depends on the producing segment:
// CHECK: air.channel.get async [%[[SEG]]]  @drain[] (%{{.*}}[] [] []) {{.*}}: (memref<64xi32>)

module {
  air.channel @drain [1]
  air.channel @feed [1]
  func.func @f(%out: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%aout=%out, %ain=%in) : memref<64xi32>, memref<64xi32> {
      air.channel.put @feed[] (%ain[] [] []) : (memref<64xi32>)
      air.segment @seg args(%o=%aout) : memref<64xi32> {
        %c1_1 = arith.constant 1 : index
        air.herd @h tile (%x, %y) in (%sx=%c1_1, %sy=%c1_1) {
          %a = memref.alloc() : memref<64xi32, 2>
          air.channel.get @feed[] (%a[] [] []) : (memref<64xi32, 2>)
          air.channel.put @drain[] (%a[] [] []) : (memref<64xi32, 2>)
          memref.dealloc %a : memref<64xi32, 2>
        }
      }
      air.channel.get @drain[] (%aout[] [] []) : (memref<64xi32>)
      air.launch_terminator
    }
    return
  }
}
