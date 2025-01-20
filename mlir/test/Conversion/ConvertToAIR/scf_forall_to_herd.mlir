//===- scf_forall_to_herd_launch.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd %s | FileCheck %s
// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd="depth=-1" %s | FileCheck %s --check-prefix=DEPTHM1
// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd="depth=0" %s | FileCheck %s --check-prefix=DEPTH0
// RUN: air-opt -split-input-file -verify-diagnostics -air-par-to-herd="depth=1" %s | FileCheck %s --check-prefix=DEPTH1

// CHECK-LABEL: func.func @scf0() {
// CHECK: air.herd @herd_0  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{.*}}, %{{.*}}=%c2{{.*}})
func.func @scf0()  {
  %src = memref.alloc() : memref<2x2xi32, 2 : i32>
  %dst = memref.alloc() : memref<2x2xi32, 2 : i32>
  scf.forall (%x,%y) in (2, 2) {
    %0 = memref.load %src[%x, %y] : memref<2x2xi32, 2 : i32>
    memref.store %0, %dst[%x, %y] : memref<2x2xi32, 2 : i32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf1() {
// CHECK: air.herd @herd_0  tile (%[[A0:.*]], {{.*}}) in (%{{.*}}=%c4{{.*}}, %{{.*}}=%c1{{.*}}) 
func.func @scf1()  {
  %src = memref.alloc() : memref<4xi32, 2 : i32>
  %dst = memref.alloc() : memref<4xi32, 2 : i32>
  scf.forall (%x) in (4) {
    %0 = memref.load %src[%x] : memref<4xi32, 2 : i32>
    memref.store %0, %dst[%x] : memref<4xi32, 2 : i32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf2() {
// CHECK: scf.parallel (%[[VAL_3:.*]], %[[VAL_4:.*]]) = (%c0{{.*}}, %c0{{.*}}) to (%c1{{.*}}, %c2{{.*}}) step (%c1{{.*}}, %c1{{.*}}) {
// CHECK:   air.herd @herd_0  tile (%[[VAL_7:.*]], %[[VAL_8:.*]]) in (%{{.*}}=%c3{{.*}}, %{{.*}}=%c4{{.*}}) args(%{{.*}}=%[[VAL_4]], %{{.*}}=%[[VAL_3]]) : index, index 
func.func @scf2()  {
  %src = memref.alloc() : memref<1x2x3x4xi32, 2 : i32>
  %dst = memref.alloc() : memref<1x2x3x4xi32, 2 : i32>
  scf.forall (%a,%b,%x,%y) in (1,2,3,4) {
    %0 = memref.load %src[%a,%b,%x,%y] : memref<1x2x3x4xi32, 2 : i32>
    memref.store %0, %dst[%a,%b,%x,%y] : memref<1x2x3x4xi32, 2 : i32>
  }
  return
}

// -----

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: [[$MAP1:#map[0-9]*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL: func.func @scf3() {
// CHECK: air.herd @herd_0  tile (%[[VAL_0:.*]], %[[VAL_1:.*]]) in (%{{.*}}=%c3{{.*}}, %{{.*}}=%c2{{.*}})
// CHECK: affine.apply [[$MAP0]](%[[VAL_1]])
// CHECK: affine.apply [[$MAP1]](%[[VAL_0]])
func.func @scf3()  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %src = memref.alloc() : memref<4x4xi32, 2 : i32>
  %dst = memref.alloc() : memref<4x4xi32, 2 : i32>
  scf.forall (%i, %j) = (%c1, %c0) to (%c4, %c4)
      step (%c1, %c2) {
    %0 = memref.load %src[%i, %j] : memref<4x4xi32, 2 : i32>
    memref.store %0, %dst[%i, %j] : memref<4x4xi32, 2 : i32>
  }
  return
}

// -----

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: [[$MAP1:#map[0-9]*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL: func.func @scf4() {
// CHECK: air.herd @herd_0  tile (%[[VAL_0:.*]], %[[VAL_1:.*]]) in (%{{.*}}=%c3{{.*}}, %{{.*}}=%c2{{.*}})
// CHECK: affine.apply [[$MAP0]](%[[VAL_1]])
// CHECK: affine.apply [[$MAP1]](%[[VAL_0]])
func.func @scf4()  {
  %src = memref.alloc() : memref<4x4xi32, 2 : i32>
  %dst = memref.alloc() : memref<4x4xi32, 2 : i32>
  scf.forall (%i, %j) = (1, 0) to (4, 4) step (1, 2) {
    %0 = memref.load %src[%i, %j] : memref<4x4xi32, 2 : i32>
    memref.store %0, %dst[%i, %j] : memref<4x4xi32, 2 : i32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @scf5() {
// CHECK: air.herd @herd_{{.*}} {
// CHECK: air.herd @herd_{{.*}} {
// CHECK: air.herd @herd_{{.*}} {
// CHECK: }
// CHECK: }
// CHECK: }
// DEPTHM1-LABEL: func.func @scf5() {
// DEPTHM1: scf.forall {{.*}} {
// DEPTHM1: scf.forall {{.*}} {
// DEPTHM1: air.herd @herd_{{.*}} {
// DEPTHM1: }
// DEPTHM1: }
// DEPTHM1: }
// DEPTH0-LABEL: func.func @scf5() {
// DEPTH0: air.herd @herd_{{.*}} {
// DEPTH0: scf.forall {{.*}} {
// DEPTH0: scf.forall {{.*}} {
// DEPTH0: }
// DEPTH0: }
// DEPTH0: }
// DEPTH1-LABEL: func.func @scf5() {
// DEPTH1: scf.forall {{.*}} {
// DEPTH1: air.herd @herd_{{.*}} {
// DEPTH1: scf.forall {{.*}} {
// DEPTH1: }
// DEPTH1: }
// DEPTH1: }
func.func @scf5()  {
  %src = memref.alloc() : memref<4x4x4xi32, 2 : i32>
  %dst = memref.alloc() : memref<4x4x4xi32, 2 : i32>
  scf.forall (%i) = (0) to (4) step (1) {
    scf.forall (%j) = (0) to (4) step (1) {
      scf.forall (%k) = (0) to (4) step (1) {
        %0 = memref.load %src[%i, %j, %k] : memref<4x4x4xi32, 2 : i32>
        memref.store %0, %dst[%i, %j, %k] : memref<4x4x4xi32, 2 : i32>
      }
    }
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
