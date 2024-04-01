//===- scf_parallel_to_herd_launch.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd -cse %s | FileCheck %s

// CHECK-LABEL: func.func @scf0() {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: air.herd @herd_0  tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C2]], {{.*}}=%[[C2]])
func.func @scf0()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror0(%c0 : index)  {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: lower bound is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror1(%c1 : index)  {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: step is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror2(%c2 : index)  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: upper bound is not a constant}}
  scf.parallel (%x,%y) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

func.func @scferror3()  {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  // expected-error@+2 {{failed to legalize}}
  // expected-error@+1 {{failed to normalize: step '2' does not evenly divide range '7'}}
  scf.parallel (%x,%y) = (%c2, %c2) to (%c9, %c9) step (%c2, %c1) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

// CHECK: #[[M0:.*]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: func.func @scf1() {
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0  tile (%[[A0:.*]], {{.*}}) in ({{.*}}=%[[C4]], {{.*}}=%[[C1]])
// CHECK: affine.apply #[[M0]](%[[A0]])
func.func @scf1()  {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%x) = (%c0) to (%c128) step (%c32) {
    %2 = arith.muli %x, %x : index
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf2() {
// CHECK: %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK: scf.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (%[[VAL_1]], %[[VAL_1]]) to (%[[VAL_0]], %[[VAL_2]]) step (%[[VAL_0]], %[[VAL_0]]) {
// CHECK:   %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:   %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:   air.herd @herd_0  tile (%[[VAL_7:.*]], %[[VAL_8:.*]]) in (%[[VAL_9:.*]]=%[[VAL_5]], %[[VAL_10:.*]]=%[[VAL_6]]) args(%[[VAL_11:.*]]=%[[VAL_3]], %[[VAL_12:.*]]=%[[VAL_4]]) : index, index
func.func @scf2()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.parallel (%a,%b,%x,%y) = (%c0,%c0,%c0,%c0) to (%c1,%c2,%c3,%c4) step (%c1,%c1,%c1,%c1) {
    %2 = arith.muli %x, %y : index
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

// -----

// This test demonstrates how to infer an air.dma_memcpy_nd op between L2 and L1, not within two scf.parallel loop nests, gets inferred with a herd around it. 

// CHECK-LABEL: module {
//       CHECK:  func.func @l2_to_l1_dma_infer_herd(
//       CHECK:    air.herd @herd_0
//       CHECK:       air.dma_memcpy_nd
//       CHECK:       air.herd_terminator
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func @l2_to_l1_dma_infer_herd(%l2_buffer: memref<2x2x32x32xi32, 1>, %l1_buffer: memref<2x2x4x8x4x8xi32, 2>) {
    %c64 = arith.constant 64 : index
    %c16384 = arith.constant 16384 : index
    %c1024 = arith.constant 1024 : index
    %c8192 = arith.constant 8192 : index
    %c512 = arith.constant 512 : index
    %c131072 = arith.constant 131072 : index
    %c2 = arith.constant 2 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c65536 = arith.constant 65536 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.parallel (%x,%y) = (%c0,%c0) to (%c16,%c8) step (%c1, %c1) {
      air.dma_memcpy_nd (%l2_buffer[] [] [], %l1_buffer[%c0, %c0, %c0, %c0, %c0, %c0] [%c2, %c2, %c8, %c4, %c4, %c8] [%c2048, %c1024, %c32, %c8, %c256, %c1]) : (memref<2x2x32x32xi32, 1>, memref<2x2x4x8x4x8xi32, 2>)
    }
    return
  }
}
