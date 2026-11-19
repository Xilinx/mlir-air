//===- shared_l1_no_affinity_opaque_writers.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Every core of a 1x4 herd writes its part of a shared L1 buffer through a
// kernel call, and core 0 sends the assembled buffer out. Each reader also
// writes, but the herd's cross-core dependence is already established, so
// with no tile that neighbours all four rows this is diagnosed rather than
// split into per-core copies (which would lose the other cores' parts).

// RUN: not air-opt %s -air-to-aie='device=npu2 row-offset=2' 2>&1 | FileCheck %s

// CHECK: error: {{.*}}L1 buffer is shared by 4 cores, but no tile has legal memory affinity with all of them
// CHECK: note: cores at: (0, 2), (0, 3), (0, 4), (0, 5)
module {
  air.channel @out_chan [1]
  func.func private @ext_write(memref<8xbf16, 2 : i32>) attributes {llvm.emit_c_interface}
  func.func @opaque_writers(%arg0: memref<128xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) args(%host=%arg0) : memref<128xbf16> {
      %c8 = arith.constant 8 : index
      %c1_g = arith.constant 1 : index
      air.channel.get @out_chan[] (%host[] [%c8] [%c1_g]) : (memref<128xbf16>)
      air.segment @seg {
        %shared = memref.alloc() : memref<8xbf16, 2 : i32>
        %c1_0 = arith.constant 1 : index
        %c4_0 = arith.constant 4 : index
        air.herd @col tile (%tx, %ty) in (%hx=%c1_0, %hy=%c4_0) args(%sbuf=%shared) : memref<8xbf16, 2 : i32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          scf.index_switch %ty
          case 0 {
            func.call @ext_write(%sbuf) : (memref<8xbf16, 2 : i32>) -> ()
            air.channel.put @out_chan[] (%sbuf[] [] []) : (memref<8xbf16, 2 : i32>)
            scf.yield
          }
          default {
            func.call @ext_write(%sbuf) : (memref<8xbf16, 2 : i32>) -> ()
            scf.yield
          }
        }
      }
    }
    return
  }
}
