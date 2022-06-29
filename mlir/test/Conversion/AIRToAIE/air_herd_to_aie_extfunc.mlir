// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-to-aie | FileCheck %s
module {

func.func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  // CHECK-LABEL: module @aie.herd_0
  // CHECK: AIE.core(%0)  {
  // CHECK:   call @beefmaker_kernel(%1) : (memref<1024xi32, 2>) -> ()
  // CHECK:   AIE.end
  // CHECK: } {elf_file = "herd_0_core_0_0.elf", link_with = "beefmaker.o"}
  air.launch_herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with="beefmaker.o"} {
    %src0 = memref.alloc() : memref<1024xi32, 2>
    func.call @beefmaker_kernel(%src0) : (memref<1024xi32, 2>) -> ()
    air.herd_terminator
  }
  return
} 

func.func private @beefmaker_kernel(%A: memref<1024xi32, 2>) -> ()

}
