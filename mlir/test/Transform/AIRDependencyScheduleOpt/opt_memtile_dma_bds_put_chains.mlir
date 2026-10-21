//===- opt_memtile_dma_bds_put_chains.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-memtile-dma-bds="device=npu2" -split-input-file | FileCheck %s

// Consecutive puts of one L2 buffer on one channel, at lattice offsets, fold
// into one put with the lattice as its outer dims.

// A core's six packets, two rounds of every other packet; the unit dims'
// offset moves to the innermost dim, which then splits to fit the BD's wrap.
// CHECK-LABEL: func.func @lattice_2d
// CHECK: air.channel.put async {{.*}}@chan[{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c5120{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c8{{.*}}, %c640{{.*}}] [%c20480{{.*}}, %c40960{{.*}}, %c640{{.*}}, %c1{{.*}}])
// CHECK-NOT: air.channel.put
module {
  air.channel @chan [1, 4]
  func.func @lattice_2d() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %t0, %buf = air.execute -> (memref<1x6x4x5120xi16, 1 : i32>) {
          %a = memref.alloc() : memref<1x6x4x5120xi16, 1 : i32>
          air.execute_terminator %a : memref<1x6x4x5120xi16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 0, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p1 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 2, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 4, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p3 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 1, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p4 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 3, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %p5 = air.channel.put async [%t0] @chan[%c0, %c1] (%buf[0, 5, 1, 0] [1, 1, 1, 5120] [122880, 20480, 5120, 1]) : (memref<1x6x4x5120xi16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2, %p3, %p4, %p5]
      }
    }
    return
  }
}

// -----

// Evenly spaced puts, each depending on the one before: one more dim.
// CHECK-LABEL: func.func @lattice_1d
// CHECK: air.channel.put async {{.*}}@chan[{{.*}}] (%{{.*}}[0, 4, 0] [3, 4, 8] [128, 16, 1])
// CHECK-NOT: air.channel.put
module {
  air.channel @chan [1, 4]
  func.func @lattice_1d() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %t0, %buf = air.execute -> (memref<64x16xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<64x16xbf16, 1 : i32>
          air.execute_terminator %a : memref<64x16xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c2] (%buf[4, 0] [4, 8] [16, 1]) : (memref<64x16xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0, %p0] @chan[%c0, %c2] (%buf[12, 0] [4, 8] [16, 1]) : (memref<64x16xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0, %p1] @chan[%c0, %c2] (%buf[20, 0] [4, 8] [16, 1]) : (memref<64x16xbf16, 1 : i32>)
        %w = air.wait_all async [%p2]
      }
    }
    return
  }
}

// -----

// Descending offsets would need a negative stride: left as they are.
// CHECK-LABEL: func.func @descending
// CHECK-COUNT-3: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @descending() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %t0, %buf = air.execute -> (memref<3x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<3x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<3x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c3] (%buf[128] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t0, %p0] @chan[%c0, %c3] (%buf[64] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0, %p1] @chan[%c0, %c3] (%buf[0] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p2]
      }
    }
    return
  }
}

// -----

// A put of another buffer in between ends the chain.
// CHECK-LABEL: func.func @other_buffer
// CHECK-COUNT-3: air.channel.put async
module {
  air.channel @chan [1, 4]
  func.func @other_buffer() {
    %0 = air.launch async () in () {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %t0, %buf = air.execute -> (memref<3x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<3x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<3x64xbf16, 1 : i32>
        }
        %t1, %buf2 = air.execute -> (memref<3x64xbf16, 1 : i32>) {
          %a = memref.alloc() : memref<3x64xbf16, 1 : i32>
          air.execute_terminator %a : memref<3x64xbf16, 1 : i32>
        }
        %p0 = air.channel.put async [%t0] @chan[%c0, %c3] (%buf[0] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p1 = air.channel.put async [%t1] @chan[%c0, %c3] (%buf2[64] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %p2 = air.channel.put async [%t0] @chan[%c0, %c3] (%buf[128] [64] [1]) : (memref<3x64xbf16, 1 : i32>)
        %w = air.wait_all async [%p0, %p1, %p2]
      }
    }
    return
  }
}
