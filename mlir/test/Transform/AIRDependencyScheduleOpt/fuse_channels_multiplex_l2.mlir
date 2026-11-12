//===- fuse_channels_multiplex_l2.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-fuse-channels="aggressive-mode=L1" -split-input-file | FileCheck %s

// Two shim channels feed two L2 buffers pinned to column 0, one after the
// other. Multiplexing in L1 only, they stay apart.

// CHECK-LABEL: func.func @plain
// CHECK: air.channel.put {{.*}}@ia
// CHECK: air.channel.put {{.*}}@ib
// CHECK: air.channel.get {{.*}}@ia
// CHECK: air.channel.get {{.*}}@ib

module {
  air.channel @ia [1, 1]
  air.channel @ib [1, 1]
  func.func @plain(%arg0: memref<256xi32>, %arg1: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%x0=%arg0, %x1=%arg1) : memref<256xi32>, memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c4 step %c1_l {
        %o = arith.muli %i, %c64 : index
        air.channel.put @ia[] (%x0[%o] [%c64] [%c1_l]) : (memref<256xi32>)
      }
      scf.for %i = %c0 to %c4 step %c1_l {
        %o = arith.muli %i, %c64 : index
        air.channel.put @ib[] (%x1[%o] [%c64] [%c1_l]) : (memref<256xi32>)
      }
      air.segment @seg {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %la = memref.alloc() {air.memtile_col = 0 : i32} : memref<64xi32, 1 : i32>
        %lb = memref.alloc() {air.memtile_col = 0 : i32} : memref<64xi32, 1 : i32>
        scf.for %i = %c0_s to %c4_s step %c1_s {
          air.channel.get @ia[] (%la[] [] []) : (memref<64xi32, 1 : i32>)
        }
        scf.for %i = %c0_s to %c4_s step %c1_s {
          air.channel.get @ib[] (%lb[] [] []) : (memref<64xi32, 1 : i32>)
        }
        air.herd @attn tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
        }
        memref.dealloc %la : memref<64xi32, 1 : i32>
        memref.dealloc %lb : memref<64xi32, 1 : i32>
      }
    }
    return
  }
}

// -----

// The same, with the herd of column 0 marked air.multiplex_l2 (cores whose
// memtile has no channel to spare): the channels to its column's L2 buffers
// are multiplexed in L2 as well, and the two share one.

// CHECK-LABEL: func.func @mux
// CHECK: air.channel.put {{.*}}@ia
// CHECK: air.channel.put {{.*}}@ia
// CHECK: air.channel.get {{.*}}@ia
// CHECK: air.channel.get {{.*}}@ia
// CHECK-NOT: @ib[

module {
  air.channel @ia [1, 1]
  air.channel @ib [1, 1]
  func.func @mux(%arg0: memref<256xi32>, %arg1: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%x0=%arg0, %x1=%arg1) : memref<256xi32>, memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c4 step %c1_l {
        %o = arith.muli %i, %c64 : index
        air.channel.put @ia[] (%x0[%o] [%c64] [%c1_l]) : (memref<256xi32>)
      }
      scf.for %i = %c0 to %c4 step %c1_l {
        %o = arith.muli %i, %c64 : index
        air.channel.put @ib[] (%x1[%o] [%c64] [%c1_l]) : (memref<256xi32>)
      }
      air.segment @seg {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %la = memref.alloc() {air.memtile_col = 0 : i32} : memref<64xi32, 1 : i32>
        %lb = memref.alloc() {air.memtile_col = 0 : i32} : memref<64xi32, 1 : i32>
        scf.for %i = %c0_s to %c4_s step %c1_s {
          air.channel.get @ia[] (%la[] [] []) : (memref<64xi32, 1 : i32>)
        }
        scf.for %i = %c0_s to %c4_s step %c1_s {
          air.channel.get @ib[] (%lb[] [] []) : (memref<64xi32, 1 : i32>)
        }
        air.herd @attn tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64, air.multiplex_l2} {
        }
        memref.dealloc %la : memref<64xi32, 1 : i32>
        memref.dealloc %lb : memref<64xi32, 1 : i32>
      }
    }
    return
  }
}
