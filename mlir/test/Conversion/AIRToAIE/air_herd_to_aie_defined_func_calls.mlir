//===- air_herd_to_aie_defined_func_calls.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' --split-input-file | FileCheck %s

// llvm.noalias holds for every call: the first call passes two buffers, the
// second passes one buffer as both arguments, and the function writes one of
// them, so neither argument keeps it.
// CHECK-LABEL: aie.device
// CHECK: call @tile_body
// CHECK: call @tile_body
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32>, %{{.*}}: memref<64xi32>) {
module {
  func.func @two_calls() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %in = memref.alloc() : memref<64xi32, 2>
          %out = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
          func.call @tile_body(%out, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}

// -----

// A defined function the cloned one calls is cloned in the same, default
// memory space, so the nested call stays well typed. It gets no llvm.noalias:
// its arguments are the caller's, which nothing proves disjoint.
// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32> {llvm.noalias}, %[[B:.*]]: memref<64xi32> {llvm.noalias})
// CHECK: call @store_tile(%{{.*}}, %[[B]]) : (vector<16xi32>, memref<64xi32>) -> ()
// CHECK: func.func private @store_tile(%{{.*}}: vector<16xi32>, %{{.*}}: memref<64xi32>) {
module {
  func.func @nested() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %in = memref.alloc() : memref<64xi32, 2>
          %out = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    func.call @store_tile(%v, %b) : (vector<16xi32>, memref<64xi32, 2>) -> ()
    return
  }
  func.func private @store_tile(%v: vector<16xi32>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}


// -----

// The async form the AIR pipeline produces: each buffer is an air.execute
// holding only its allocation, which is still distinct storage, so both
// arguments keep llvm.noalias.
// CHECK-LABEL: aie.device
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32> {llvm.noalias}, %{{.*}}: memref<64xi32> {llvm.noalias})
module {
  func.func @async_allocs() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      %1 = air.segment @seg async {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @h async tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %t0, %in = air.execute -> (memref<64xi32, 2>) {
            %a = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %a : memref<64xi32, 2>
          }
          %t1, %out = air.execute -> (memref<64xi32, 2>) {
            %a = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %a : memref<64xi32, 2>
          }
          %t2 = air.execute [%t0, %t1] {
            func.call @tile_body(%in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>) -> ()
          }
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    vector.transfer_write %v, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
}
