//===- isolate_host_read_back.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests="scope=func" | FileCheck %s

// A host read-back through a cast keeps the producing job's other channel
// feeds in the same loop and ahead of the read. The drain token alone does
// not express those producer feeds' ordering constraint.
// CHECK-LABEL: func.func @read_back_view
// CHECK: scf.for
// CHECK: air.channel.put async {{.*}}@a
// CHECK: air.channel.put async {{.*}}@b
// CHECK: air.channel.get async {{.*}}@out
// CHECK: air.channel.put async {{.*}}@a
// CHECK-NOT: scf.for
// CHECK: return

// The same read-back, with the cast inside an air.execute.
// CHECK-LABEL: func.func @read_back_execute
// CHECK: scf.for
// CHECK: air.channel.put async {{.*}}@a
// CHECK: air.channel.put async {{.*}}@b
// CHECK: air.channel.get async {{.*}}@out
// CHECK: air.channel.put async {{.*}}@a
// CHECK-NOT: scf.for
// CHECK: return

// The same read-back, with the write and the read through two launch
// arguments bound to the same host buffer.
// CHECK-LABEL: func.func @read_back_launch_args
// CHECK: air.launch
// CHECK: scf.for
// CHECK: air.channel.put async {{.*}}@a
// CHECK: air.channel.put async {{.*}}@b
// CHECK: air.channel.get async {{.*}}@out
// CHECK: air.channel.put async {{.*}}@a
// CHECK-NOT: scf.for
// CHECK: return

// Independent host inputs and outputs retain ordinary per-channel isolation.
// CHECK-LABEL: func.func @independent
// CHECK: scf.for
// CHECK: air.channel.put async {{.*}}@a
// CHECK: scf.for
// CHECK: air.channel.put async {{.*}}@b
// CHECK: scf.for
// CHECK: air.channel.get async {{.*}}@out
// CHECK: return

module {
  air.channel @a [1]
  air.channel @b [1]
  air.channel @out [1]
  func.func @read_back_view(%x: memref<256xi32>, %w: memref<256xi32>, %y: memref<256xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %a = air.channel.put async @a[%c0] (%x[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %b = air.channel.put async @b[%c0] (%w[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %d = air.channel.get async @out[%c0] (%y[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %view = memref.cast %y : memref<256xi32> to memref<?xi32>
      %read = air.channel.put async [%d] @a[%c0] (%view[%c0] [%c64] [%c1]) : (memref<?xi32>)
    }
    return
  }
  func.func @read_back_execute(%x: memref<256xi32>, %w: memref<256xi32>, %y: memref<256xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %a = air.channel.put async @a[%c0] (%x[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %b = air.channel.put async @b[%c0] (%w[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %d = air.channel.get async @out[%c0] (%y[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %t, %view = air.execute -> (memref<?xi32>) {
        %v = memref.cast %y : memref<256xi32> to memref<?xi32>
        air.execute_terminator %v : memref<?xi32>
      }
      %read = air.channel.put async [%d, %t] @a[%c0] (%view[%c0] [%c64] [%c1]) : (memref<?xi32>)
    }
    return
  }
  func.func @read_back_launch_args(%x: memref<256xi32>, %w: memref<256xi32>, %y: memref<256xi32>) {
    %c1_0 = arith.constant 1 : index
    %l = air.launch async (%lx) in (%ls=%c1_0) args(%ax=%x, %aw=%w, %out=%y, %in=%y) : memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %a = air.channel.put async @a[%c0] (%ax[%c0] [%c64] [%c1]) : (memref<256xi32>)
        %b = air.channel.put async @b[%c0] (%aw[%c0] [%c64] [%c1]) : (memref<256xi32>)
        %d = air.channel.get async @out[%c0] (%out[%c0] [%c64] [%c1]) : (memref<256xi32>)
        %read = air.channel.put async [%d] @a[%c0] (%in[%c0] [%c64] [%c1]) : (memref<256xi32>)
      }
    }
    return
  }
  func.func @independent(%x: memref<256xi32>, %w: memref<256xi32>, %y: memref<256xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %a = air.channel.put async @a[%c0] (%x[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %b = air.channel.put async @b[%c0] (%w[%c0] [%c64] [%c1]) : (memref<256xi32>)
      %d = air.channel.get async @out[%c0] (%y[%c0] [%c64] [%c1]) : (memref<256xi32>)
    }
    return
  }
}
