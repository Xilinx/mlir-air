//===- channel_drain_hierarchy.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// A launch-scope air.channel.get that drains data produced on-device (its
// matching air.channel.put lives inside a herd in a segment) must not wait for
// the segment. Both ends are async, so the put finishes only once the get has
// started; the segment's token fires only once the put has finished. An edge
// from the segment to the get would deadlock. The segment takes %aout as an
// arg but never accesses it, so it gives no footprint on %aout either. The host
// wait for the drain is placed at the launch terminator by AIRLowering.

// CHECK: air.segment @seg async
// CHECK: air.channel.put async [%{{.*}}]  @drain[]
// CHECK: air.channel.get async  @drain[] (%{{.*}}[] [] []) {{.*}}: (memref<64xi32>)

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
