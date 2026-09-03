//===- air_annotate_append_barrier.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-append-barrier --split-input-file | FileCheck %s

// A channel.get lands data in the memref it names, so on an L3 buffer it is the
// shim write; a channel.put sends that memref out, so it is the read. Nothing in
// the runtime sequence orders the two, so mark the writer and the first reader.

// CHECK-LABEL: @append_then_readback
// CHECK: air.channel.get @appendK[] (%arg0[] [] []) {air.append_barrier}
// CHECK: air.channel.put @readback[] (%arg0[] [] []) {air.await_appends}
air.channel @appendK [1]
air.channel @readback [1]
func.func @append_then_readback(%kv: memref<1024xbf16>) {
  air.channel.get @appendK[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.put @readback[] (%kv[] [] []) : (memref<1024xbf16>)
  return
}

// -----

// Both appends feed the one readback, so both are barriered.

// CHECK-LABEL: @two_appends_one_readback
// CHECK: air.channel.get @apK[] (%arg0[] [] []) {air.append_barrier}
// CHECK: air.channel.get @apV[] (%arg0[] [] []) {air.append_barrier}
// CHECK: air.channel.put @rb[] (%arg0[] [] []) {air.await_appends}
air.channel @apK [1]
air.channel @apV [1]
air.channel @rb [1]
func.func @two_appends_one_readback(%kv: memref<1024xbf16>) {
  air.channel.get @apK[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.get @apV[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.put @rb[] (%kv[] [] []) : (memref<1024xbf16>)
  return
}

// -----

// Only the FIRST reader after a write is tagged: the second put consumes the
// same already-barriered data and needs no further await.

// CHECK-LABEL: @only_first_reader
// CHECK: air.channel.get @ap2[] (%arg0[] [] []) {air.append_barrier}
// CHECK: air.channel.put @rbA[] (%arg0[] [] []) {air.await_appends}
// CHECK: air.channel.put @rbB[]
// CHECK-NOT: air.await_appends
air.channel @ap2 [1]
air.channel @rbA [1]
air.channel @rbB [1]
func.func @only_first_reader(%kv: memref<1024xbf16>) {
  air.channel.get @ap2[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.put @rbA[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.put @rbB[] (%kv[] [] []) : (memref<1024xbf16>)
  return
}

// -----

// A second round pairs with its own append, not the first round's: one barrier
// per round, rather than one collapsed barrier at the end.

// CHECK-LABEL: @round_per_append
// CHECK: air.channel.get @ap3[] (%arg0[] [] []) {air.append_barrier}
// CHECK: air.channel.put @rb3[] (%arg0[] [] []) {air.await_appends}
// CHECK: air.channel.get @ap3[] (%arg0[] [] []) {air.append_barrier}
// CHECK: air.channel.put @rb3[] (%arg0[] [] []) {air.await_appends}
air.channel @ap3 [1]
air.channel @rb3 [1]
func.func @round_per_append(%kv: memref<1024xbf16>) {
  air.channel.get @ap3[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.put @rb3[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.get @ap3[] (%kv[] [] []) : (memref<1024xbf16>)
  air.channel.put @rb3[] (%kv[] [] []) : (memref<1024xbf16>)
  return
}

// -----

// A read with no preceding write to that buffer is a plain input feed.

// CHECK-LABEL: @read_only
// CHECK-NOT: air.await_appends
// CHECK-NOT: air.append_barrier
air.channel @feed [1]
func.func @read_only(%x: memref<1024xbf16>) {
  air.channel.put @feed[] (%x[] [] []) : (memref<1024xbf16>)
  return
}

// -----

// Different buffers do not pair with each other.

// CHECK-LABEL: @distinct_buffers
// CHECK-NOT: air.await_appends
// CHECK-NOT: air.append_barrier
air.channel @apX [1]
air.channel @rbY [1]
func.func @distinct_buffers(%a: memref<1024xbf16>, %b: memref<1024xbf16>) {
  air.channel.get @apX[] (%a[] [] []) : (memref<1024xbf16>)
  air.channel.put @rbY[] (%b[] [] []) : (memref<1024xbf16>)
  return
}

// -----

// L1 and L2 traffic is already ordered by the lock protocol; only L3 pairs.

// CHECK-LABEL: @l1_is_left_alone
// CHECK-NOT: air.await_appends
// CHECK-NOT: air.append_barrier
air.channel @apL1 [1]
air.channel @rbL1 [1]
func.func @l1_is_left_alone() {
  %buf = memref.alloc() : memref<64xbf16, 2>
  air.channel.get @apL1[] (%buf[] [] []) : (memref<64xbf16, 2>)
  air.channel.put @rbL1[] (%buf[] [] []) : (memref<64xbf16, 2>)
  memref.dealloc %buf : memref<64xbf16, 2>
  return
}

// -----

// Sibling regions of an scf.index_switch are alternatives that never both run,
// so a write in one and a read in another are not a dependency. Note the walk
// order trap: the op holds its default region first but prints it last.

// CHECK-LABEL: @mutually_exclusive_branches
// CHECK-NOT: air.await_appends
// CHECK-NOT: air.append_barrier
air.channel @apSw [1]
air.channel @rbSw [1]
func.func @mutually_exclusive_branches(%kv: memref<1024xbf16>, %sel: index) {
  scf.index_switch %sel
  case 0 {
    air.channel.put @rbSw[] (%kv[] [] []) : (memref<1024xbf16>)
    scf.yield
  }
  default {
    air.channel.get @apSw[] (%kv[] [] []) : (memref<1024xbf16>)
    scf.yield
  }
  return
}

// -----

// An append spelled as an air.dma_memcpy_nd. The early run of this pass cannot
// see the pair: a DMA names both endpoints in ONE op, so its L3 side is still
// inside the herd and does not share a block with the readback. The two only
// become a pair once air-dma-to-channel has hoisted the external half out to
// launch scope, which is why aircc runs this pass a SECOND time there.
//
// Without that second run the pair is silently unordered and the readback can
// see stale bytes -- the failure this pass exists to prevent, reintroduced by
// changing how the transfer is spelled.

// RUN: air-opt %s -air-annotate-append-barrier -air-dependency \
// RUN:   -air-dma-to-channel -canonicalize -cse -air-annotate-append-barrier \
// RUN:   --split-input-file | FileCheck %s --check-prefix=DMA

// DMA-LABEL: @append_as_dma
// DMA: air.channel.get{{.*}}@appendK{{.*}}{air.append_barrier}
// DMA: air.channel.put{{.*}}@inKV{{.*}}air.await_appends
air.channel @appendK [1]
air.channel @inKV [1]
func.func @append_as_dma(%kv: memref<4096xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%lkv=%kv) : memref<4096xbf16> {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1_l = arith.constant 1 : index
    air.segment @seg args(%skv=%lkv) : memref<4096xbf16> {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%skv) : memref<4096xbf16> {
        %c0_h = arith.constant 0 : index
        %c512_h = arith.constant 512 : index
        %c1_h = arith.constant 1 : index
        %l1 = memref.alloc() : memref<512xbf16, 2>
        // dst is the L3 cache, so the derived external half is a channel.get:
        // the shim WRITE. hoist_before keeps it ahead of the readback.
        air.dma_memcpy_nd (%a[%c0_h] [%c512_h] [%c1_h], %l1[] [] []) {id = 1 : i32, channel = @appendK, channel_indices = array<i64: 0>, hoist_before = @inKV} : (memref<4096xbf16>, memref<512xbf16, 2>)
        memref.dealloc %l1 : memref<512xbf16, 2>
      }
    }
    air.channel.put @inKV[%c0] (%lkv[%c0] [%c512] [%c1_l]) : (memref<4096xbf16>)
  }
  return
}
