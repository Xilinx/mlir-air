//===- opt_shim_dma_bds_nonlinear_offset.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" | FileCheck %s

// A loop over four tiles of a 512x512 buffer, as a 1-D grid splits its
// program id: row (iv / 2) * 256, column (iv % 2) * 256. The offset is not
// linear in the loop variable, so the loop cannot become one strided
// dimension; each tile keeps its own transfer.

// CHECK-LABEL: func.func @div_rem_offset
// CHECK: air.channel.put {{.*}}(%arg0[%c0{{.*}}, %c0{{.*}}] [%c256{{.*}}, %c256{{.*}}] [%c512{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put {{.*}}(%arg0[%c0{{.*}}, %c256{{.*}}] [%c256{{.*}}, %c256{{.*}}] [%c512{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put {{.*}}(%arg0[%c256{{.*}}, %c0{{.*}}] [%c256{{.*}}, %c256{{.*}}] [%c512{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put {{.*}}(%arg0[%c256{{.*}}, %c256{{.*}}] [%c256{{.*}}, %c256{{.*}}] [%c512{{.*}}, %c1{{.*}}])
// CHECK-NOT: air.channel.put
func.func @div_rem_offset(%arg0: memref<512x512xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %0 = air.wait_all async
  %1 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t = %0) -> (!air.async.token) {
    %row = arith.divsi %iv, %c2 : index
    %col = arith.remsi %iv, %c2 : index
    %r = arith.muli %row, %c256 : index
    %c = arith.muli %col, %c256 : index
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%r, %c] [%c256, %c256] [%c512, %c1]) {metadata = @airMemcpyId1} : (memref<512x512xbf16>)
    scf.yield %put : !air.async.token
  }
  return
}

// A linear offset still folds: four 128-row blocks, one after another, are
// one transfer of the whole buffer.

// CHECK-LABEL: func.func @linear_offset
// CHECK: air.channel.put {{.*}}(%arg0[] [] [])
// CHECK-NOT: air.channel.put
func.func @linear_offset(%arg0: memref<512x512xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %0 = air.wait_all async
  %1 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t = %0) -> (!air.async.token) {
    %r = arith.muli %iv, %c128 : index
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%r, %c0] [%c128, %c512] [%c512, %c1]) {metadata = @airMemcpyId1} : (memref<512x512xbf16>)
    scf.yield %put : !air.async.token
  }
  return
}

// The IV reaches the row offset through an addi, and the row stride is 512,
// so each iteration moves 512 elements. Rows 8 to 11 are one transfer.

// CHECK-LABEL: func.func @addi_offset
// CHECK: air.channel.put {{.*}}(%arg0[%c4096{{.*}}] [%c2048{{.*}}] [%c1{{.*}}])
// CHECK-NOT: air.channel.put
func.func @addi_offset(%arg0: memref<512x512xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c512 = arith.constant 512 : index
  %0 = air.wait_all async
  %1 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t = %0) -> (!air.async.token) {
    %r = arith.addi %c8, %iv : index
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%r, %c0] [%c1, %c512] [%c512, %c1]) {metadata = @airMemcpyId1} : (memref<512x512xbf16>)
    scf.yield %put : !air.async.token
  }
  return
}

// The row offset is carried in the loop rather than computed from the IV, so
// the fold cannot see how it moves; each iteration keeps its own transfer.

// CHECK-LABEL: func.func @carried_offset
// CHECK: air.channel.put {{.*}}(%arg0[%c0{{.*}}] [%c65536{{.*}}] [%c1{{.*}}])
// CHECK: air.channel.put {{.*}}(%arg0[%c65536{{.*}}] [%c65536{{.*}}] [%c1{{.*}}])
// CHECK: air.channel.put {{.*}}(%arg0[%c131072{{.*}}] [%c65536{{.*}}] [%c1{{.*}}])
// CHECK: air.channel.put {{.*}}(%arg0[%c196608{{.*}}] [%c65536{{.*}}] [%c1{{.*}}])
// CHECK-NOT: air.channel.put
func.func @carried_offset(%arg0: memref<512x512xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %0 = air.wait_all async
  %1:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t = %0, %r = %c0) -> (!air.async.token, index) {
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%r, %c0] [%c128, %c512] [%c512, %c1]) {metadata = @airMemcpyId1} : (memref<512x512xbf16>)
    %next = arith.addi %r, %c128 : index
    scf.yield %put, %next : !air.async.token, index
  }
  return
}

// The offset is an addi of values that do not depend on the IV, so every
// iteration sends the same row: a repeat with stride 0.

// CHECK-LABEL: func.func @addi_without_iv
// CHECK: air.channel.put {{.*}}(%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %{{.*}}] [%c4{{.*}}, %c1{{.*}}, %c1{{.*}}, %c512{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c1{{.*}}])
// CHECK-NOT: air.channel.put
func.func @addi_without_iv(%arg0: memref<512x512xbf16>, %row: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c512 = arith.constant 512 : index
  %0 = air.wait_all async
  %1 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%t = %0) -> (!air.async.token) {
    %r = arith.addi %row, %c8 : index
    %put = air.channel.put async [%t] @channel_0[] (%arg0[%r, %c0] [%c1, %c512] [%c512, %c1]) {metadata = @airMemcpyId1} : (memref<512x512xbf16>)
    scf.yield %put : !air.async.token
  }
  return
}

air.channel @channel_0 [1, 1]
