//===- opt_shim_dma_bds_run_time_contiguous.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2" -split-input-file | FileCheck %s

// A run-time number of records, each a tile of A followed by tiles of B, read
// eight times over. Each record starts where the previous one ended, so the
// eight passes become one put: n records long, repeated eight times.

// CHECK-LABEL: func.func @ab_rounds
// CHECK: %[[N:.*]] = arith.index_cast %arg1 : i32 to index
// CHECK: %[[LEN:.*]] = arith.muli %[[N]], %c16384{{.*}} : index
// CHECK: air.channel.put {{.*}} @ab[] (%arg0[%c0{{.*}}, %c0{{.*}}] [%c8{{.*}}, %[[LEN]]] [%c0{{.*}}, %c1{{.*}}])
// CHECK-NOT: scf.for
// CHECK-NOT: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @ab_rounds(%src: memref<*xbf16>, %n: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c4096 = arith.constant 4096 : index
    %c8192 = arith.constant 8192 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %0 = air.wait_all async
    %1 = scf.for %r = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%t = %0) -> (!air.async.token) : i32 {
      %2 = scf.for %b = %c0_i32 to %n step %c1_i32 iter_args(%u = %t) -> (!air.async.token) : i32 {
        %o = arith.muli %b, %c16384_i32 : i32
        %oi = arith.index_cast %o : i32 to index
        %put_a = air.channel.put async [%u] @ab[] (%src[%c0, %oi] [%c32, %c256] [%c256, %c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
        %off_b = arith.addi %oi, %c8192 : index
        %put_b = air.channel.put async [%put_a, %u] @ab[] (%src[%c0, %c0, %off_b] [%c2, %c32, %c128] [%c4096, %c128, %c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
        scf.yield %put_b : !air.async.token
      }
      scf.yield %2 : !air.async.token
    }
    return
  }
}

// -----

// The records are read once, starting from the third: one put of n - 2
// records.

// CHECK-LABEL: func.func @ab_once
// CHECK: %[[N:.*]] = arith.subi %arg1, %c2{{.*}} : index
// CHECK: %[[LEN:.*]] = arith.muli %[[N]], %c16384{{.*}} : index
// CHECK: air.channel.put {{.*}} @ab[] (%arg0[%c32768{{.*}}] [%[[LEN]]] [%c1{{.*}}])
// CHECK-NOT: scf.for

module {
  air.channel @ab [1, 1]
  func.func @ab_once(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8192 = arith.constant 8192 : index
    %c16384 = arith.constant 16384 : index
    %0 = air.wait_all async
    %1 = scf.for %b = %c2 to %n step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c16384 : index
      %put_a = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      %off_b = arith.addi %o, %c8192 : index
      %put_b = air.channel.put async [%put_a] @ab[] (%src[%off_b] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      scf.yield %put_b : !air.async.token
    }
    return
  }
}

// -----

// Each record has a gap between A and B. The runs of consecutive iterations
// do not join, so the loop stays.

// CHECK-LABEL: func.func @gap
// CHECK: scf.for
// CHECK: air.channel.put
// CHECK: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @gap(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %c8200 = arith.constant 8200 : index
    %c16392 = arith.constant 16392 : index
    %0 = air.wait_all async
    %1 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c16392 : index
      %put_a = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      %off_b = arith.addi %o, %c8200 : index
      %put_b = air.channel.put async [%put_a] @ab[] (%src[%off_b] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      scf.yield %put_b : !air.async.token
    }
    return
  }
}

// -----

// Each iteration advances by more than it reads. The loop stays, with a
// run-time number of iterations of one put.

// CHECK-LABEL: func.func @strided_trips
// CHECK: scf.for
// CHECK: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @strided_trips(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %c16384 = arith.constant 16384 : index
    %0 = air.wait_all async
    %1 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c16384 : index
      %put_a = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      scf.yield %put_a : !air.async.token
    }
    return
  }
}

// -----

// A static loop around a run-time loop that does not fold (a gap between
// iterations). The run-time loop cannot become a dim of the put either, so
// both loops stay.

// CHECK-LABEL: func.func @gap_in_rounds
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @gap_in_rounds(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8192 = arith.constant 8192 : index
    %c16392 = arith.constant 16392 : index
    %0 = air.wait_all async
    %1 = scf.for %r = %c0 to %c4 step %c1 iter_args(%t = %0) -> (!air.async.token) {
      %2 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %t) -> (!air.async.token) {
        %o = arith.muli %b, %c16392 : index
        %p = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
        scf.yield %p : !air.async.token
      }
      scf.yield %2 : !air.async.token
    }
    return
  }
}

// -----

// The put also waits for a token from outside the loop: the loop stays.

// CHECK-LABEL: func.func @outside_token
// CHECK: scf.for
// CHECK: air.channel.put

module {
  air.channel @ab [1, 1]
  func.func @outside_token(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %e = air.wait_all async
    %1 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c8192 : index
      %p = air.channel.put async [%u, %e] @ab[] (%src[%o] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      scf.yield %p : !air.async.token
    }
    return
  }
}

// -----

// The two puts are on different channels: the loop stays.

// CHECK-LABEL: func.func @two_channels
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @a[]
// CHECK: air.channel.put {{.*}} @b[]

module {
  air.channel @a [1, 1]
  air.channel @b [1, 1]
  func.func @two_channels(%src: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8192 = arith.constant 8192 : index
    %c16384 = arith.constant 16384 : index
    %0 = air.wait_all async
    %1 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %0) -> (!air.async.token) {
      %o = arith.muli %b, %c16384 : index
      %pa = air.channel.put async [%u] @a[] (%src[%o] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
      %ob = arith.addi %o, %c8192 : index
      %pb = air.channel.put async [%pa] @b[] (%src[%ob] [%c8192] [%c1]) {metadata = @airMemcpyId2} : (memref<*xbf16>)
      scf.yield %pb : !air.async.token
    }
    return
  }
}

// -----

// The static loop around the run-time one also sends on another channel. The
// run-time loop folds into one put, but the static loop is not just a repeat
// of it, so it gets no repeat dim; it is unrolled instead.

// CHECK-LABEL: func.func @outer_holds_more
// CHECK-COUNT-4: air.channel.put {{.*}} @ab[] (%arg0[%c0{{.*}}] [%{{.*}}] [%c1{{.*}}])
// CHECK-NOT: scf.for

module {
  air.channel @ab [1, 1]
  air.channel @c [1, 1]
  func.func @outer_holds_more(%src: memref<*xbf16>, %other: memref<*xbf16>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
    %0 = air.wait_all async
    %1 = scf.for %r = %c0 to %c4 step %c1 iter_args(%t = %0) -> (!air.async.token) {
      %2 = scf.for %b = %c0 to %n step %c1 iter_args(%u = %t) -> (!air.async.token) {
        %o = arith.muli %b, %c8192 : index
        %p = air.channel.put async [%u] @ab[] (%src[%o] [%c8192] [%c1]) {metadata = @airMemcpyId1} : (memref<*xbf16>)
        scf.yield %p : !air.async.token
      }
      %q = air.channel.put async [%2] @c[] (%other[%c0] [%c64] [%c1]) {metadata = @airMemcpyId2} : (memref<*xbf16>)
      scf.yield %q : !air.async.token
    }
    return
  }
}
