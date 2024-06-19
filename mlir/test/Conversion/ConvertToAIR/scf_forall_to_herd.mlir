//===- scf_forall_to_herd_launch.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd -cse %s | FileCheck %s

// CHECK-LABEL: func.func @scf0() {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: air.herd @herd_0  tile ({{.*}}, {{.*}}) in ({{.*}}=%[[C2]], {{.*}}=%[[C2]])
func.func @scf0()  {
  scf.forall (%x,%y) in (2, 2) {
    %2 = arith.addi %x, %y : index
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf1() {
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: air.herd @herd_0  tile (%[[A0:.*]], {{.*}}) in ({{.*}}=%[[C4]], {{.*}}=%[[C1]])
func.func @scf1()  {
  scf.forall (%x) in (4) {
    %2 = arith.muli %x, %x : index
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf2() {
// CHECK-DAG: %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK: scf.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (%[[VAL_0]], %[[VAL_0]]) to (%[[VAL_1]], %[[VAL_2]]) step (%[[VAL_1]], %[[VAL_1]]) {
// CHECK:   %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:   %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:   air.herd @herd_0  tile (%[[VAL_7:.*]], %[[VAL_8:.*]]) in (%[[VAL_9:.*]]=%[[VAL_5]], %[[VAL_10:.*]]=%[[VAL_6]])
func.func @scf2()  {
  scf.forall (%a,%b,%x,%y) in (1,2,3,4) {
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
//       CHECK:    }
//       CHECK:    return
//       CHECK:  }
//       CHECK: }
module {
  func.func private @matmul_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/mm_microkernel.o", llvm.bareptr = true}
  func.func @matmul_small_dispatch_0_matmul_8x32x16_i32(%base_buffer: memref<i32, 2 : i32>, %base_buffer_14: memref<i32, 2 : i32>, %base_buffer_18: memref<i32, 2 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.forall (%x,%y) in (2, 2) {
      %2 = arith.addi %x, %y : index
      func.call @matmul_i32_i32(%base_buffer, %c0, %base_buffer_14, %c0, %base_buffer_18, %c0) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
    }
    return
  }
}
