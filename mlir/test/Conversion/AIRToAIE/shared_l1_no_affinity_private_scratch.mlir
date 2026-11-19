//===- shared_l1_no_affinity_private_scratch.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Two herds (rows 2-3 and 4-5) name the same L1 allocation and no tile has
// memory affinity with all four rows. Every core writes the buffer before
// reading it, so each core gets its own copy.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// CHECK-COUNT-4: aie.buffer({{.*}}) {sym_name = "{{.*}}"} : memref<64xbf16, 2>
// CHECK-NOT: aie.buffer({{.*}}) {sym_name = "{{.*}}"} : memref<64xbf16, 2>
module {
  func.func private @sink(memref<16xbf16, 2>) attributes {link_with = "sink.o", llvm.emit_c_interface}
  func.func @two_herds_private_scratch() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @segment_0 {
        %shared = memref.alloc() : memref<64xbf16, 2>
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @lo tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2_0) args(%sbuf=%shared) : memref<64xbf16, 2> attributes {x_loc = 0, y_loc = 2} {
          %c0 = arith.constant 0 : index
          %w = arith.constant 2.0 : bf16
          %cst0 = arith.constant 0.0 : bf16
          %wv = vector.broadcast %w : bf16 to vector<16xbf16>
          vector.transfer_write %wv, %sbuf[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<64xbf16, 2>
          %v = vector.transfer_read %sbuf[%c0], %cst0 {in_bounds = [true]} : memref<64xbf16, 2>, vector<16xbf16>
          %priv = memref.alloc() : memref<16xbf16, 2>
          vector.transfer_write %v, %priv[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, 2>
          func.call @sink(%priv) : (memref<16xbf16, 2>) -> ()
          memref.dealloc %priv : memref<16xbf16, 2>
          air.herd_terminator
        }
        air.herd @hi tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2_0) args(%sbuf=%shared) : memref<64xbf16, 2> attributes {x_loc = 0, y_loc = 4} {
          %c0 = arith.constant 0 : index
          %w = arith.constant 3.0 : bf16
          %cst0 = arith.constant 0.0 : bf16
          %wv = vector.broadcast %w : bf16 to vector<16xbf16>
          vector.transfer_write %wv, %sbuf[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<64xbf16, 2>
          %v = vector.transfer_read %sbuf[%c0], %cst0 {in_bounds = [true]} : memref<64xbf16, 2>, vector<16xbf16>
          %priv = memref.alloc() : memref<16xbf16, 2>
          vector.transfer_write %v, %priv[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, 2>
          func.call @sink(%priv) : (memref<16xbf16, 2>) -> ()
          memref.dealloc %priv : memref<16xbf16, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
