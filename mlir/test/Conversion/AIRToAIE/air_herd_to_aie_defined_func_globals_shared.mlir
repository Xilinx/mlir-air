//===- air_herd_to_aie_defined_func_globals_shared.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// Globals a called function reads share a copy in the device when their
// type, value and alignment match (@a and @b, of different memory spaces),
// and get their own when the alignment differs (@c).

// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body
// CHECK: memref.get_global @a_0 : memref<16xi32>
// CHECK: memref.get_global @a_0 : memref<16xi32>
// CHECK: memref.get_global @c_0 : memref<16xi32>
// CHECK: memref.global "private" constant @a_0 : memref<16xi32> = dense<7>{{$}}
// CHECK-NEXT: memref.global "private" constant @c_0 : memref<16xi32> = dense<7> alignment = 64
// CHECK-NEXT: }
module {
  memref.global "private" constant @a : memref<16xi32, 2> = dense<7>
  memref.global "private" constant @b : memref<16xi32> = dense<7>
  "memref.global"() <{alignment = 64 : i64, constant, initial_value = dense<7> : tensor<16xi32>, sym_name = "c", sym_visibility = "private", type = memref<16xi32, 2>}> : () -> ()
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
  func.func private @tile_body(%o: memref<16xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %ga = memref.get_global @a : memref<16xi32, 2>
    %gb = memref.get_global @b : memref<16xi32>
    %gc = memref.get_global @c : memref<16xi32, 2>
    %va = vector.transfer_read %ga[%c0], %p {in_bounds = [true]} : memref<16xi32, 2>, vector<16xi32>
    %vb = vector.transfer_read %gb[%c0], %p {in_bounds = [true]} : memref<16xi32>, vector<16xi32>
    %vc = vector.transfer_read %gc[%c0], %p {in_bounds = [true]} : memref<16xi32, 2>, vector<16xi32>
    %s = arith.addi %va, %vb : vector<16xi32>
    %t = arith.addi %s, %vc : vector<16xi32>
    vector.transfer_write %t, %o[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32, 2>
    return
  }
}
