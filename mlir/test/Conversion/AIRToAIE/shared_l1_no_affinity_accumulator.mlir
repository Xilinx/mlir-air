//===- shared_l1_no_affinity_accumulator.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// An accumulator allocated outside a 1x4 herd: every core zero-fills it, core
// 0 updates it and the other cores pass it on. No tile has memory affinity
// with all four rows. Each core writes the buffer before reading it, so each
// core gets its own copy.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// CHECK-COUNT-4: aie.buffer({{.*}}) {sym_name = "{{.*}}"} : memref<64xbf16, 2 : i32>
// CHECK-NOT: aie.buffer({{.*}}) {sym_name = "{{.*}}"} : memref<64xbf16, 2 : i32>
module {
  func.func private @zero_fill(memref<64xbf16, 2 : i32>) attributes {link_with = "k.o", llvm.emit_c_interface}
  func.func private @update(memref<16xbf16, 2 : i32>, memref<64xbf16, 2 : i32>) attributes {link_with = "k.o", llvm.emit_c_interface}
  func.func private @pass_on(memref<64xbf16, 2 : i32>, memref<16xbf16, 2 : i32>) attributes {link_with = "k.o", llvm.emit_c_interface}
  func.func @accumulator() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %acc = memref.alloc() : memref<64xbf16, 2 : i32>
        %c1_0 = arith.constant 1 : index
        %c4_0 = arith.constant 4 : index
        air.herd @col tile (%tx, %ty) in (%hx=%c1_0, %hy=%c4_0) args(%sbuf=%acc) : memref<64xbf16, 2 : i32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %priv = memref.alloc() : memref<16xbf16, 2 : i32>
          func.call @zero_fill(%sbuf) : (memref<64xbf16, 2 : i32>) -> ()
          scf.index_switch %ty
          case 0 {
            func.call @update(%priv, %sbuf) : (memref<16xbf16, 2 : i32>, memref<64xbf16, 2 : i32>) -> ()
            scf.yield
          }
          default {
            func.call @pass_on(%sbuf, %priv) : (memref<64xbf16, 2 : i32>, memref<16xbf16, 2 : i32>) -> ()
            scf.yield
          }
          memref.dealloc %priv : memref<16xbf16, 2 : i32>
        }
      }
    }
    return
  }
}
