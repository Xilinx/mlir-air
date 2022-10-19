//===- air_to_async.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-async | FileCheck %s

// CHECK-LABEL: func.func @wait_all_0
// CHECK-NEXT: return
func.func @wait_all_0() -> () {
  air.wait_all
  return
}

// CHECK-LABEL: func.func @wait_all_1
// CHECK: %[[T0:.*]] = async.execute {
// CHECK: %[[T1:.*]] = async.execute {
// CHECK: %[[T2:.*]] = async.execute [%[[T0]], %[[T1]]] {
// CHECK: async.await %[[T0]] : !async.token
// CHECK: async.await %[[T1]] : !async.token
// CHECK: async.await %[[T2]] : !async.token
func.func @wait_all_1() -> () {
  %0 = air.wait_all async
  %1 = air.wait_all async
  %2 = air.wait_all async [%0, %1]
  air.wait_all [%0, %1, %2]
  return
}

// CHECK-LABEL: func.func @execute
// CHECK: %[[T0:.*]], %[[V0:.*]] = async.execute -> !async.value<memref<32xi32>>
// CHECK: async.yield %{{.*}} : memref<32xi32>
// CHECK: %[[V1:.*]] = async.await %[[V0]] : !async.value<memref<32xi32>>
// CHECK: async.await %[[T0]] : !async.token
// CHECK: return %[[V1]] : memref<32xi32>
func.func @execute() -> (memref<32xi32>) {
  %1, %2 = air.execute -> (memref<32xi32>) {
    %3 = memref.alloc() : memref<32xi32>
    air.execute_terminator %3 : memref<32xi32>
  }
  air.wait_all [%1]
  return %2 : memref<32xi32>
}

// CHECK-LABEL: func.func @memcpy_nd
// CHECK: %[[T0:.*]] = async.execute [%token] {
// CHECK: func.call @air_memcpy_nd_I32_M0D2I32_M0D2I32({{.*}}) : (i32, memref<?x?xi32>, memref<?x?xi32>) -> ()
// CHECK:   async.yield
// CHECK: async.await %[[T0]] : !async.token
// CHECK: call @air_memcpy_nd_M0D2I32_M0D2I32({{.*}}) : (memref<?x?xi32>, memref<?x?xi32>) -> ()
func.func @memcpy_nd(%a : memref<64x64xi32>, %b : memref<64x64xi32, 1>) -> () {
  %0 = air.wait_all async
  %1 = air.dma_memcpy_nd async [%0] (%a[] [] [], %b[] [] []) {id = 1 : i32} : (memref<64x64xi32>, memref<64x64xi32, 1>)
  air.dma_memcpy_nd [%1] (%b[] [] [], %a[] [] []) : (memref<64x64xi32, 1>, memref<64x64xi32>)
  return
}

// CHECK-LABEL: func.func @alloc_dealloc
// CHECK-NEXT: %[[A0:.*]] = memref.alloc() : memref<32xi8>
// CHECK-NEXT: %[[A1:.*]] = memref.alloc() : memref<32xi8>
// CHECK: memref.dealloc %[[A0]] : memref<32xi8>
// CHECK: memref.dealloc %[[A1]] : memref<32xi8
func.func @alloc_dealloc() -> () {
  %0 = memref.alloc() : memref<32xi8, 1>
  %1 = memref.alloc() : memref<32xi8, 2>
  air.dma_memcpy_nd (%0[] [] [], %1[] [] []) : (memref<32xi8, 1>, memref<32xi8, 2>)
  memref.dealloc %0 : memref<32xi8, 1>
  memref.dealloc %1 : memref<32xi8, 2>
  return
}

// CHECK-LABEL:   func.func @herd_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32,
// CHECK-SAME:                      %[[VAL_1:.*]]: i32) attributes {llvm.emit_c_interface} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = async.execute {
// CHECK:             async.yield
// CHECK:           }
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = async.execute -> !async.value<memref<32xi32>> {
// CHECK:             %[[VAL_7:.*]] = memref.alloc() : memref<32xi32>
// CHECK:             async.yield %[[VAL_7]] : memref<32xi32>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = async.await %[[VAL_9:.*]] : !async.value<memref<32xi32>>
// CHECK:           %[[VAL_10:.*]] = async.execute {{\[}}%[[VAL_11:.*]], %[[VAL_12:.*]]] {
// CHECK:             %[[VAL_13:.*]] = arith.constant 6 : index
// CHECK:             %[[VAL_14:.*]] = async.create_group %[[VAL_13]] : !async.group
// CHECK:             affine.for %[[VAL_15:.*]] = 0 to 2 {
// CHECK:               affine.for %[[VAL_16:.*]] = 0 to 3 {
// CHECK:                 %[[VAL_17:.*]] = async.execute {
// CHECK:                   %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : index
// CHECK:                   %[[VAL_19:.*]] = arith.muli %[[VAL_2]], %[[VAL_3]] : index
// CHECK:                   %[[VAL_20:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:                   async.yield
// CHECK:                 }
// CHECK:                 %[[VAL_21:.*]] = async.add_to_group %[[VAL_22:.*]], %[[VAL_14]] : !async.token
// CHECK:               } {air.herd = "inner"}
// CHECK:             } {air.herd = "outer"}
// CHECK:             async.await_all %[[VAL_14]]
// CHECK:             async.yield
// CHECK:           }
// CHECK:           async.await %[[VAL_10]] : !async.token
// CHECK:           return
// CHECK:         }
func.func @herd_1(%arg0: i32, %arg1: i32) -> () {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %e0 = air.wait_all async
  %e1, %alloc = air.execute -> (memref<32xi32>) {
    %3 = memref.alloc() : memref<32xi32>
    air.execute_terminator %3 : memref<32xi32>
  }
  %e2 = air.herd async [%e0, %e1] tile (%x, %y) in (%sx=%c2, %sy=%c3) args (%op0=%arg0, %op1=%arg1) : i32, i32 attributes { } {
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    %2 = arith.addi %op0, %op1 : i32
    air.herd_terminator
  }
  air.wait_all [%e2]
  return
}