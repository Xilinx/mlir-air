//===- host_read_after_drain.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -split-input-file | FileCheck %s

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

// -----

// Bounds over a loop nest: the offset is an affine.apply of two induction
// variables, rows 2-4 over the nest, which takes in the drained row 3. The
// outer loop starts after the drain.
// CHECK-LABEL: func.func @nest
// CHECK: %[[D:.*]] = air.channel.get async  @drain[]
// CHECK: %[[W:.*]] = air.wait_all async [%[[D]]]
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[W]])

// A mod in the offset map: evaluated at every iteration, rows 2-3, which take
// in the drained row 3.
// CHECK-LABEL: func.func @wrap
// CHECK: %[[D:.*]] = air.channel.get async  @drain[]
// CHECK: %[[W:.*]] = air.wait_all async [%[[D]]]
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[W]])

// 2 * x - 3 * (x floordiv 2) + 2 over x = 0..4 is 2, 4, 3, 5, 4: row 5, which
// the drain writes, lies between the values at x = 0 and x = 4.
// CHECK-LABEL: func.func @div
// CHECK: %[[D:.*]] = air.channel.get async  @drain[]
// CHECK: %[[W:.*]] = air.wait_all async [%[[D]]]
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[W]])

// A loop nest over rows 5-7: no edge to a drain of row 3.
// CHECK-LABEL: func.func @nest_disjoint
// CHECK: air.channel.get async  @drain[]
// CHECK: %[[W:.*]] = air.wait_all async  {id
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[W]])

// A mod over too many iterations to evaluate: no bound, so no edge.
// CHECK-LABEL: func.func @huge
// CHECK: air.channel.get async  @drain[]
// CHECK: %[[W:.*]] = air.wait_all async  {id
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[W]])

// A row known only at runtime: no bound, so no edge.
// CHECK-LABEL: func.func @runtime
// CHECK: air.channel.get async  @drain[]
// CHECK: air.channel.put async  @feed[]

// Not host memory: an L2 buffer keeps the equal-start test, so a read of row 3
// does not depend on a write of rows 2-4.
// CHECK-LABEL: func.func @l2
// CHECK: %[[T:.*]], %{{.*}} = air.execute -> (memref<8x4x16xi32, 1>)
// CHECK: air.channel.get async [%[[T]]]  @drain[]
// CHECK: air.channel.put async [%[[T]]]  @feed[]

#nest = affine_map<()[s0, s1] -> (s0 + s1 + 2)>
#wrap = affine_map<()[s0] -> (s0 mod 2 + 2)>
#div = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 2) * 3 + 2)>
#far = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
#huge = affine_map<()[s0] -> (s0 mod 4)>
module {
  air.channel @drain [1]
  air.channel @feed [1]
  func.func @nest(%buf: memref<8x4x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%b=%buf) : memref<8x4x16xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      air.channel.get @drain[] (%b[%c3, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 1 : i32} : (memref<8x4x16xi32>)
      scf.for %x = %c0 to %c2 step %c1_0 {
        scf.for %y = %c0 to %c2 step %c1_0 {
          %r = affine.apply #nest()[%x, %y]
          air.channel.put @feed[] (%b[%r, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 2 : i32} : (memref<8x4x16xi32>)
        }
      }
      air.launch_terminator
    }
    return
  }
  func.func @wrap(%buf: memref<8x4x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%b=%buf) : memref<8x4x16xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      air.channel.get @drain[] (%b[%c3, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 3 : i32} : (memref<8x4x16xi32>)
      scf.for %x = %c0 to %c4 step %c1_0 {
        %r = affine.apply #wrap()[%x]
        air.channel.put @feed[] (%b[%r, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 4 : i32} : (memref<8x4x16xi32>)
      }
      air.launch_terminator
    }
    return
  }
  func.func @div(%buf: memref<8x4x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%b=%buf) : memref<8x4x16xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c5 = arith.constant 5 : index
      %cbig = arith.constant 10000 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %crow = arith.constant 5 : index
      air.channel.get @drain[] (%b[%crow, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) : (memref<8x4x16xi32>)
      scf.for %x = %c0 to %c5 step %c1_0 {
        %r = affine.apply #div()[%x]
        air.channel.put @feed[] (%b[%r, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) : (memref<8x4x16xi32>)
      }
      air.launch_terminator
    }
    return
  }
  func.func @nest_disjoint(%buf: memref<8x4x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%b=%buf) : memref<8x4x16xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c5 = arith.constant 5 : index
      %cbig = arith.constant 10000 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %crow = arith.constant 3 : index
      air.channel.get @drain[] (%b[%crow, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) : (memref<8x4x16xi32>)
      scf.for %x = %c0 to %c2 step %c1_0 {
        scf.for %y = %c0 to %c2 step %c1_0 {
          %r = affine.apply #far()[%x, %y]
          air.channel.put @feed[] (%b[%r, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) : (memref<8x4x16xi32>)
        }
      }
      air.launch_terminator
    }
    return
  }
  func.func @huge(%buf: memref<8x4x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%b=%buf) : memref<8x4x16xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c5 = arith.constant 5 : index
      %cbig = arith.constant 10000 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %crow = arith.constant 3 : index
      air.channel.get @drain[] (%b[%crow, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) : (memref<8x4x16xi32>)
      scf.for %x = %c0 to %cbig step %c1_0 {
        %r = affine.apply #huge()[%x]
        air.channel.put @feed[] (%b[%r, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) : (memref<8x4x16xi32>)
      }
      air.launch_terminator
    }
    return
  }
  func.func @runtime(%buf: memref<8x4x16xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) args(%b=%buf, %row=%n) : memref<8x4x16xi32>, index {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      air.channel.get @drain[] (%b[%c2, %c0, %c0] [%c3, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 5 : i32} : (memref<8x4x16xi32>)
      air.channel.put @feed[] (%b[%row, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 6 : i32} : (memref<8x4x16xi32>)
      air.launch_terminator
    }
    return
  }
  func.func @l2(%buf: memref<8x4x16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %m = memref.alloc() : memref<8x4x16xi32, 1>
        air.channel.get @drain[] (%m[%c2, %c0, %c0] [%c3, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 7 : i32} : (memref<8x4x16xi32, 1>)
        air.channel.put @feed[] (%m[%c3, %c0, %c0] [%c1_0, %c1_0, %c16] [%c64, %c16, %c1_0]) {id = 8 : i32} : (memref<8x4x16xi32, 1>)
        memref.dealloc %m : memref<8x4x16xi32, 1>
      }
      air.launch_terminator
    }
    return
  }
}
