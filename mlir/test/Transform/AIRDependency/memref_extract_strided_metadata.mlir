//===- memref_extract_strided_metadata.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// No async air.execute op should be created around memref.ex op.
//      CHECK: } {id = 1 : i32}
// CHECK-NEXT: memref.extract_strided_metadata
// CHECK-NEXT: %[[EVENT0:.*]] = air.execute {
// CHECK-NEXT:        func.call @zero_bf16
// CHECK-NEXT: } {id = 2 : i32}
module {
  func.func private @zero_bf16(memref<bf16, 2 : i32>, index) attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
  func.func @extract_strided_metadata() {
    %c2_8 = arith.constant 2 : index
    %c2_9 = arith.constant 2 : index
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2_8, %arg5=%c2_9) attributes {link_with = "/path/to/mm_microkernel.o"} {
      %alloc_1 = memref.alloc() : memref<16x16x4x4xbf16, 2 : i32>
      %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %alloc_1 : memref<16x16x4x4xbf16, 2 : i32> -> memref<bf16, 2 : i32>, index, index, index, index, index, index, index, index, index
      func.call @zero_bf16(%base_buffer, %offset) : (memref<bf16, 2 : i32>, index) -> ()
      air.herd_terminator
    }
    return
  }
}
