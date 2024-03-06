//===- affine_par_to_herd_launch.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd -cse %s | FileCheck %s

// CHECK-LABEL: func.func @par0
// CHECK: %[[C0:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0 tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C0]], {{.*}}=%[[C0]])
func.func @par0()  {
  affine.parallel (%x,%y) = (0,0) to (1,1) {
    %2 = arith.addi %x, %y : index
    affine.yield
  }
  return
}

// -----

func.func @par1()  {
  // expected-error@+1 {{'affine.parallel' op failed conversion to 'air.herd': only 2d loops are supported}}
  affine.parallel (%x,%y,%z) = (0,0,0) to (1,2,3) {
    %2 = arith.addi %x, %y : index
    affine.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @par2
func.func @par2()  {
  // CHECK: %[[C0:.*]] = arith.constant 4 : index
  // CHECK: %[[C1:.*]] = arith.constant 5 : index
  // CHECK: air.herd @herd_0 tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C0]], {{.*}}=%[[C1]])
  affine.parallel (%x,%y) = (0,2) to (4,12) step (1,2) {
    %2 = arith.addi %x, %y : index
    affine.yield
  }
  return
}

// -----

// This test demonstrates that while forming air.herd we look through func.call ops, fetch
// the corresponding function declaration's 'link_with' attribute and attach it to the newly
// formed air.herd op.

// CHECK-LABEL: module {
//       CHECK:  func.func private @matmul_i32_i32
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
//       CHECK:  func.func @matmul_small_dispatch_0_matmul_8x32x16_i32(
//       CHECK:    air.herd @herd_0
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o"} {
//       CHECK:       func.call @matmul_i32_i32
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func private @matmul_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
  func.func @matmul_small_dispatch_0_matmul_8x32x16_i32(%base_buffer: memref<i32, 2 : i32>, %base_buffer_14: memref<i32, 2 : i32>, %base_buffer_18: memref<i32, 2 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%x,%y) = (%c0,%c0) to (%c1,%c1) step (%c1, %c1) {
      %2 = arith.addi %x, %y : index
      func.call @matmul_i32_i32(%base_buffer, %c0, %base_buffer_14, %c0, %base_buffer_18, %c0) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
      scf.reduce
    }
    return
  }
}

// -----

// This test demonstrates the relaying of `link_with` construct to air.herd op even if the
// func.call op is not an immediate child of scf.parallel.

// CHECK-LABEL: module {
//       CHECK:  func.func private @matmul_scalar_i32_i32
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
//       CHECK:  func.func @matmul_small_nested_scf_dispatch_0_matmul_8x32x16_i32(
//       CHECK:    air.herd @herd_0
//  CHECK-SAME:        attributes {link_with = "/path/to/mm_microkernel.o"} {
//       CHECK:       scf.for
//  CHECK-SAME:       {
//       CHECK:           func.call @matmul_scalar_i32_i32
//       CHECK:       }
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func private @matmul_scalar_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
  func.func @matmul_small_nested_scf_dispatch_0_matmul_8x32x16_i32(%base_buffer: memref<i32, 2 : i32>, %base_buffer_14: memref<i32, 2 : i32>, %base_buffer_18: memref<i32, 2 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    scf.parallel (%x,%y) = (%c0,%c0) to (%c1,%c1) step (%c1, %c1) {
      %2 = arith.addi %x, %y : index
      scf.for %arg0 = %c0 to %c32 step %c4 {
        func.call @matmul_scalar_i32_i32(%base_buffer, %c0, %base_buffer_14, %c0, %base_buffer_18, %c0) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
      }
      scf.reduce
    }
    return
  }
}
