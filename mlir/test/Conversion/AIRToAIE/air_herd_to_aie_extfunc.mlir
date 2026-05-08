//===- air_herd_to_aie_extfunc.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie | FileCheck %s
module {

func.func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  // CHECK: aie.device
  // CHECK: %[[VAL_0:.*]] = aie.tile(1, 1)
  // CHECK: %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {{{.*}}} : memref<1024xi32, 2>
  // CHECK: aie.core(%[[VAL_0]])  {
  // CHECK:   call @beefmaker_kernel(%[[VAL_1]]) : (memref<1024xi32, 2>) -> ()
  // CHECK:   aie.end
  // The aie.core attribute dict carries herd metadata persisted at outline
  // time (RFC #1567 Stage C #3) plus link_with. This 1x1 herd gets local id
  // [0, 0] and size [1, 1]. Attributes serialize alphabetically by name.
  // CHECK: } {air.herd_local_id = array<i64: 0, 0>, air.herd_size = array<i64: 1, 1>, link_with = "beefmaker.o"}
  // CHECK: func.func private @beefmaker_kernel(memref<1024xi32, 2>) attributes {link_with = "beefmaker.o", llvm.emit_c_interface}
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with="beefmaker.o"} {
    %src0 = memref.alloc() : memref<1024xi32, 2>
    func.call @beefmaker_kernel(%src0) : (memref<1024xi32, 2>) -> ()
  }
  return
} 

func.func private @beefmaker_kernel(%A: memref<1024xi32, 2>) -> ()

}
