//===- air_herd_to_aie_defined_func.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s
// RUN: air-opt %s -air-to-aie='device=npu1_1col row-offset=2' | FileCheck %s

// A herd calling a function the module defines (a loop nest outlined from the
// herd by transform.loop.outline): the function is cloned into the device with
// its body, in the default memory space, together with what it calls; memref
// arguments the call site proves disjoint get llvm.noalias (two views of one
// buffer that are only read keep it too); and the module-level original keeps
// only its declaration.

// CHECK: aie.device
// CHECK: %[[T:.*]] = aie.tile(0, 2)
// CHECK: %[[IN:.*]] = aie.buffer(%[[T]]) {{.*}} : memref<64xi32, 2>
// CHECK: %[[OUT:.*]] = aie.buffer(%[[T]]) {{.*}} : memref<64xi32, 2>
// CHECK: aie.core(%[[T]])
// CHECK-DAG: %[[A:.*]] = memref.memory_space_cast %[[IN]] : memref<64xi32, 2> to memref<64xi32>
// CHECK-DAG: %[[B:.*]] = memref.memory_space_cast %[[OUT]] : memref<64xi32, 2> to memref<64xi32>
// CHECK: call @tile_body(%[[A]], %{{.*}}, %[[B]])
// CHECK: func.func private @tile_body(%{{.*}}: memref<64xi32> {llvm.noalias}, %{{.*}}: memref<64xi32> {llvm.noalias}, %{{.*}}: memref<64xi32> {llvm.noalias})
// CHECK: call @helper
// CHECK: func.func private @helper(vector<16xi32>, vector<16xi32>) -> vector<16xi32> attributes {llvm.emit_c_interface}
// CHECK: func.func private @tile_body(memref<64xi32, 2>, memref<64xi32, 2>, memref<64xi32, 2>)
module {
  func.func @foo() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %in = memref.alloc() : memref<64xi32, 2>
          %out = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%in, %in, %out) : (memref<64xi32, 2>, memref<64xi32, 2>, memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%a: memref<64xi32, 2>, %a2: memref<64xi32, 2>, %b: memref<64xi32, 2>) {
    %c0 = arith.constant 0 : index
    %p = arith.constant 0 : i32
    %v = vector.transfer_read %a[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    %v2 = vector.transfer_read %a2[%c0], %p {in_bounds = [true]} : memref<64xi32, 2>, vector<16xi32>
    %s = func.call @helper(%v, %v2) : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vector.transfer_write %s, %b[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, 2>
    return
  }
  func.func private @helper(vector<16xi32>, vector<16xi32>) -> vector<16xi32>
}
