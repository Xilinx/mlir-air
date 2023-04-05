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
  // CHECK-LABEL: module @aie.segment_0
  // CHECK: AIE.core(%0)  {
  // CHECK:   call @beefmaker_kernel(%1) : (memref<1024xi32, 2>) -> ()
  // CHECK:   AIE.end
  // CHECK: } {elf_file = "segment_0_core_1_1.elf", link_with = "beefmaker.o"}
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with="beefmaker.o"} {
    %src0 = memref.alloc() : memref<1024xi32, 2>
    func.call @beefmaker_kernel(%src0) : (memref<1024xi32, 2>) -> ()
    air.herd_terminator
  }
  return
} 

func.func private @beefmaker_kernel(%A: memref<1024xi32, 2>) -> ()

}
