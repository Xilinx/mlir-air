//===- shared_l1_no_affinity_diag.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A shared L1 buffer that cannot be placed and hands data between cores must
// be a compile error, not a silent miscompile. In a 1x4 herd no tile has
// memory affinity with all four rows, so there is no owner for one shared
// buffer. Core 0 writes the buffer and cores 1..3 read it: giving each core
// its own copy would leave the readers with never-written data.

// RUN: not air-opt %s -air-to-aie='device=npu2 row-offset=2' 2>&1 | FileCheck %s

// CHECK: error: {{.*}}L1 buffer is shared by 4 cores, but no tile has legal memory affinity with all of them
// CHECK-SAME: Route the data through a channel
// CHECK: note: cores at: (0, 2), (0, 3), (0, 4), (0, 5)
module {
  func.func private @sink(memref<16xbf16, 2>) attributes {link_with = "sink.o", llvm.emit_c_interface}
  func.func @single_herd_shared_l1() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @segment_0 {
        // Shared L1 buffer allocated at segment level, passed to the single herd.
        %shared = memref.alloc() : memref<64xbf16, 2>
        %c1_0 = arith.constant 1 : index
        %c4_0 = arith.constant 4 : index
        air.herd @col tile (%tx, %ty) in (%hx=%c1_0, %hy=%c4_0) args(%sbuf=%shared) : memref<64xbf16, 2> attributes {x_loc = 0, y_loc = 2} {
          %c0 = arith.constant 0 : index
          scf.index_switch %ty
          case 0 {
            %cst = arith.constant 1.0 : bf16
            %v = vector.broadcast %cst : bf16 to vector<16xbf16>
            vector.transfer_write %v, %sbuf[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<64xbf16, 2>
            scf.yield
          }
          default {
            %cst0 = arith.constant 0.0 : bf16
            %v = vector.transfer_read %sbuf[%c0], %cst0 {in_bounds = [true]} : memref<64xbf16, 2>, vector<16xbf16>
            %priv = memref.alloc() : memref<16xbf16, 2>
            vector.transfer_write %v, %priv[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, 2>
            func.call @sink(%priv) : (memref<16xbf16, 2>) -> ()
            memref.dealloc %priv : memref<16xbf16, 2>
            scf.yield
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
