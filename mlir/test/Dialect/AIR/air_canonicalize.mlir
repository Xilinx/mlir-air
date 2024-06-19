//===- air_herd_launch_canonicalize.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize  %s | FileCheck %s

// CHECK-LABEL: func.func @herd
// CHECK: air.herd tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) {
func.func @herd(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  air.herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  return
}

// CHECK-LABEL: func.func @herd_async
// CHECK: air.herd async [{{.*}}] tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}}) attributes {attr_name = "attrValue"} {
func.func @herd_async(%arg0: i32, %e0 : !air.async.token) {
  %cst2 = arith.constant 2 : index
  %e1 = air.herd async [%e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0, %op2=%arg0, %op3=%arg0) : i32, i32, i32, i32 attributes { attr_name="attrValue" } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
  }
  air.wait_all [%e1]
  return
}

// CHECK-LABEL: herd_async_1
// CHECK: air.execute
// CHECK: memref.alloc
// CHECK: air.execute_terminator
// CHECK: air.herd async [{{.*}}] tile (
// CHECK: air.dma_memcpy_nd
func.func @herd_async_1() {
  %cst2 = arith.constant 2 : index
  %t0, %results = air.execute -> (memref<1xi32>) {
    %1 = memref.alloc() : memref<1xi32>
    air.execute_terminator %1 : memref<1xi32>
  }
  %t1 = air.wait_all async [%t0]
  %e0 = air.wait_all async [%t0, %t1]
  %e1 = air.herd async [%t0, %t1, %e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%results) : memref<1xi32> {
    %d0 = air.dma_memcpy_nd async (%op0[] [] [], %op0[] [] []) : (memref<1xi32>, memref<1xi32>)
  }
  air.wait_all [%e1]
  return
}

// CHECK-LABEL: herd_async_2
// CHECK: air.herd async  tile (
func.func @herd_async_2() {
  %cst2 = arith.constant 2 : index
  %t0 = air.wait_all async
  %t1 = air.wait_all async [%t0]
  %e0 = air.wait_all async [%t0, %t1]
  %e1 = air.herd async [%t0, %t1, %e0] tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
  }
  air.wait_all [%e1]
  return
}

// CHECK-LABEL: wait_all_0
// CHECK-NEXT: return
func.func @wait_all_0() {
  %0 = air.wait_all async
  %1 = air.wait_all async [%0]
  air.wait_all [%0, %1]
  air.wait_all
  return
}

// CHECK-LABEL: wait_all_1
// CHECK-NEXT: return
func.func @wait_all_1() {
  %async_token, %results = air.execute -> (memref<1xi32>) {
    %1 = memref.alloc() : memref<1xi32>
    air.execute_terminator %1 : memref<1xi32>
  }
  %0 = air.wait_all async [%async_token]
  air.wait_all [%0]
  return
}

// CHECK-LABEL: execute_0
// CHECK-NEXT: return
func.func @execute_0() {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (index) {
    air.execute_terminator %c0 : index
  }
  return
}

// CHECK-LABEL: execute_1
// CHECK-NEXT: %{{.*}} = arith.constant 0 : index
// CHECK-NEXT: return %{{.*}} : index
func.func @execute_1(%i : index) -> index {
  %async_token, %results:2 = air.execute -> (index, index) {
    %c0 = arith.constant 0 : index
    air.execute_terminator %i, %c0 : index, index
  }
  return %results#1 : index
}

// CHECK-LABEL: execute_2
// CHECK-NEXT: arith.constant 0 : index
// CHECK-NEXT: air.wait_all async 
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, !air.async.token
func.func @execute_2() -> (index, !air.async.token) {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (index) {
    air.execute_terminator %c0 : index
  }
  %t = air.wait_all async [%async_token]
  return %results, %t : index, !air.async.token
}

// CHECK-LABEL: execute_3
// CHECK-NEXT: arith.constant 0 : index
// CHECK-NEXT: air.wait_all async 
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, !air.async.token
func.func @execute_3() -> (index, !air.async.token) {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (index) {
    %1 = memref.alloc() : memref<1xi32>
    air.dma_memcpy_nd (%1[] [] [], %1[] [] []) : (memref<1xi32>, memref<1xi32>)
    air.execute_terminator %c0 : index
  }
  %t = air.wait_all async [%async_token]
  return %results, %t : index, !air.async.token
}

// CHECK-LABEL: execute_4
// CHECK: air.execute
// CHECK: memref.alloc
// CHECK: air.dma_memcpy_nd
// CHECK: air.execute_terminator
// CHECK: air.execute
// CHECK: memref.dealloc
func.func @execute_4() -> (memref<1xi32>, !air.async.token) {
  %c0 = arith.constant 0 : index
  %async_token, %results = air.execute -> (memref<1xi32>) {
    %1 = memref.alloc() : memref<1xi32>
    air.dma_memcpy_nd (%1[] [] [], %1[] [] []) : (memref<1xi32>, memref<1xi32>)
    air.execute_terminator %1 : memref<1xi32>
  }
  %t = air.wait_all async [%async_token]
  %async_token_0 = air.execute {
    memref.dealloc %results : memref<1xi32>
  }
  return %results, %t : memref<1xi32>, !air.async.token
}
