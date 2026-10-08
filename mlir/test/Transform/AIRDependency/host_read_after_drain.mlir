//===- host_read_after_drain.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// A launch that reads back from host memory what its own drain wrote: the read
// depends on the drain when their footprints intersect, including a read that
// starts inside the drained rows and a loop over them. A read of another
// column of the same rows, or of rows the drain does not write, does not.

// CHECK: %[[D:.*]] = air.channel.get async  @drain[]
// Starts inside the drained rows 2-4.
// CHECK: air.channel.put async [%[[D]]]  @feed[] {{.*}} {id = 2 : i32}
// Same rows, another column.
// CHECK: air.channel.put async  @feed[] {{.*}} {id = 3 : i32}
// A loop over rows 2-4 starts after the drain.
// CHECK: %[[W:.*]] = air.wait_all async [%[[D]]]
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[W]])
// CHECK: air.channel.put async [%{{.*}}]  @feed[] {{.*}} {id = 4 : i32}
// Row 6, which the drain does not write.
// CHECK: air.channel.put async  @feed[] {{.*}} {id = 5 : i32}

#map = affine_map<()[s0] -> (s0 + 2)>
module {
  air.channel @drain [1]
  air.channel @feed [1]
  func.func @f(%buf: memref<8x4x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%b=%buf) : memref<8x4x16xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c6 = arith.constant 6 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      air.channel.get @drain[] (%b[%c2, %c0, %c0] [%c3, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 1 : i32} : (memref<8x4x16xi32>)
      air.channel.put @feed[] (%b[%c3, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 2 : i32} : (memref<8x4x16xi32>)
      air.channel.put @feed[] (%b[%c3, %c1_0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 3 : i32} : (memref<8x4x16xi32>)
      scf.for %iv = %c0 to %c3 step %c1_0 {
        %r = affine.apply #map()[%iv]
        air.channel.put @feed[] (%b[%r, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 4 : i32} : (memref<8x4x16xi32>)
      }
      air.channel.put @feed[] (%b[%c6, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 5 : i32} : (memref<8x4x16xi32>)
      air.launch_terminator
    }
    return
  }
}
