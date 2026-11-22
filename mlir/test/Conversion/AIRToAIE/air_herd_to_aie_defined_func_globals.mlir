//===- air_herd_to_aie_defined_func_globals.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// A function the herd calls reads a constant memref.global. The global is
// copied into the device once, although two functions read it, in the default
// memory space as the functions' memrefs are. The copy is private and named
// apart from every module symbol (@lut_0 is taken), since aiecc moves the
// device's globals next to the module's; the module's @lut stays for the host.

// CHECK-LABEL: aie.device
// CHECK: aie.core
// CHECK: func.func private @tile_body
// CHECK: memref.get_global @lut_1 : memref<16xi32>
// CHECK: memref.global "private" constant @lut_1 : memref<16xi32> = dense<7>
// CHECK: func.func private @tile_tail
// CHECK: memref.get_global @lut_1 : memref<16xi32>
// CHECK-NOT: memref.global "private" constant @lut_{{[0-9]+}} : memref<16xi32> =
// CHECK: memref.global "public" constant @lut : memref<16xi32, 2> = dense<7>
module {
  memref.global "public" constant @lut : memref<16xi32, 2> = dense<7>
  memref.global "private" constant @lut_0 : memref<4xi32> = dense<1>
  func.func @foo() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %out = memref.alloc() : memref<16xi32, 2>
          func.call @tile_body(%out) : (memref<16xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%b: memref<16xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %g = memref.get_global @lut : memref<16xi32, 2>
    %v = vector.transfer_read %g[%c0], %p {in_bounds = [true]} : memref<16xi32, 2>, vector<16xi32>
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32, 2>
    func.call @tile_tail(%b) : (memref<16xi32, 2>) -> ()
    return
  }
  func.func private @tile_tail(%b: memref<16xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %g = memref.get_global @lut : memref<16xi32, 2>
    %v = vector.transfer_read %g[%c0], %p {in_bounds = [true]} : memref<16xi32, 2>, vector<16xi32>
    %w = vector.transfer_read %b[%c0], %p {in_bounds = [true]} : memref<16xi32, 2>, vector<16xi32>
    %s = arith.addi %v, %w : vector<16xi32>
    vector.transfer_write %s, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32, 2>
    return
  }
}
