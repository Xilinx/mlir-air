//===- air.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  func.func @moo(%arg0: i32) {
    %c1 = arith.constant 1 : index
    air.herd @cowfactory  tile (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {link_with = "beefmaker_kernel.o"} {
      %alloc = memref.alloc() {sym_name = "beef"} : memref<1024xi32, 2>
      func.call @beefmaker_kernel(%alloc) : (memref<1024xi32, 2>) -> ()
      air.herd_terminator
    }
    return
  }
  func.func private @beefmaker_kernel(memref<1024xi32, 2>)
}