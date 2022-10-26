//===- air_herd_to_aie_extfunc.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie | FileCheck %s
module {

func.func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  // CHECK-LABEL: module @aie.partition_0
  // CHECK: AIE.core(%0)  {
  // CHECK:   call @beefmaker_kernel(%1) : (memref<1024xi32, 2>) -> ()
  // CHECK:   AIE.end
  // CHECK: } {elf_file = "partition_0_core_1_1.elf", link_with = "beefmaker.o"}
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with="beefmaker.o"} {
    %src0 = memref.alloc() : memref<1024xi32, 2>
    func.call @beefmaker_kernel(%src0) : (memref<1024xi32, 2>) -> ()
    air.herd_terminator
  }
  return
} 

func.func private @beefmaker_kernel(%A: memref<1024xi32, 2>) -> ()

}
