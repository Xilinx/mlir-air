//===- func_call.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// A single async air.execute op should be created around func.call op.
//      CHECK: %[[EVENT0:.*]] = air.execute [
// CHECK-NEXT:        func.call @matmul_scalar_i32_i32
// CHECK-NEXT: } {id = 
module {
  func.func private @matmul_scalar_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
  func.func @generic() {
    %c2_8 = arith.constant 2 : index
    %c2_9 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<8x16xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<16x8xi32, 1 : i32>
    %alloc_3 = memref.alloc() : memref<8x8xi32, 1 : i32>
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2_8, %arg5=%c2_9) args(%arg6=%alloc_3, %arg7=%alloc, %arg8=%alloc_2) : memref<8x8xi32, 1 : i32>, memref<8x16xi32, 1 : i32>, memref<16x8xi32, 1 : i32> attributes {link_with = "/path/to/mm_microkernel.o"} {
      %base_buffer_lhs = memref.alloc() : memref<i32, 2 : i32>
      %base_buffer_rhs = memref.alloc() : memref<i32, 2 : i32>
      %base_buffer = memref.alloc() : memref<i32, 2 : i32>
      %c0_11 = arith.constant 0 : index
      func.call @matmul_scalar_i32_i32(%base_buffer_lhs, %c0_11, %base_buffer_rhs, %c0_11, %base_buffer, %c0_11) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
      air.herd_terminator
    }
    return
  }
}
