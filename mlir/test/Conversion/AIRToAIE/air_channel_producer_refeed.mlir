//===- air_channel_producer_refeed.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// air.refeed_count=N expresses a single-buffer, count-free re-broadcast: the
// producer writes ONE resident buffer and it is re-sent N times without a
// rotation twin. This is the correct lowering for a value produced once and
// re-streamed once per consumer row-block (a stride-0 repeat dim is
// unsupported on AIE2 memtile/core BDs, and a repeat_count BD would re-send
// stale data). The N>1 loop the front-end emits is canonicalized away before
// air-to-aie, so the count is carried on the channel / put / buffer.

// Producer-side (core) re-feed: air.refeed_count=N on the channel makes the
// producing core release its output (data-ready) lock N times, so the
// count-free self-loop MM2S re-sends the one resident buffer N times. The
// consuming core is unaffected (inbound release stays 1). Re-dispatch balance:
// the DMA self-loop frees the producer's buf-free lock once per re-send (N
// times), so the producing core must re-acquire ALL N freed tokens before
// overwriting the resident buffer, and that buf-free lock must INIT to N.
// Otherwise it leaks N-1 per dispatch and stalls every other dispatch.

// CHECK-LABEL: aie.device
// CHECK-DAG: aie.lock(%{{.*}}, {{.*}}) {init = 4 : i32}
// CHECK-DAG: aie.use_lock(%{{.*}}, AcquireGreaterEqual, 4)
// CHECK-DAG: aie.use_lock(%{{.*}}, Release, 4)

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_0 [1, 1] {air.refeed_count = 4 : i32}
func.func @refeed() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// Memtile variant: air.refeed_count=N on an L2 buffer alloc makes ONE fill
// (S2MM) enable N count-free MM2S re-broadcasts. The buffer op carries the
// count (AllocL2BuffersPattern propagation), its write/empty lock inits to N
// (getLockForDMA), and the fill BD acquires/releases that lock xN
// (generateDmaBd), so the count-free MM2S re-reads the resident buffer N times.

// CHECK-LABEL: aie.device
// CHECK: aie.lock(%{{.*}}, {{.*}}) {init = 4 : i32}
// CHECK: aie.buffer(%{{.*}}) {air.refeed_count = 4 : i32
// CHECK: aie.memtile_dma
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, 4)
// CHECK: aie.use_lock(%{{.*}}, Release, 4)

air.channel @cin [1, 1]
air.channel @to_core [1, 1]
air.channel @from_core [1, 1]
air.channel @cout [1, 1]
func.func @mt_refeed(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  air.channel.put @cin[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
  air.segment @seg0 {
    %c1_0 = arith.constant 1 : index
    %l2 = memref.alloc() {air.refeed_count = 4 : i32} : memref<64xi32, 1>
    air.channel.get @cin[] (%l2[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
    air.channel.put @to_core[] (%l2[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    air.herd tile(%tx, %ty) in (%sx = %c1_0, %sy = %c1_0) attributes {sym_name = "herd0"} {
      %buf = memref.alloc() : memref<64xi32, 2>
      %res = memref.alloc() : memref<64xi32, 2>
      air.channel.get @to_core[%tx, %ty] (%buf[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
      air.channel.put @from_core[%tx, %ty] (%res[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
      memref.dealloc %buf : memref<64xi32, 2>
      memref.dealloc %res : memref<64xi32, 2>
    }
    %l2o = memref.alloc() : memref<64xi32, 1>
    air.channel.get @from_core[] (%l2o[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
    air.channel.put @cout[] (%l2o[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
    memref.dealloc %l2 : memref<64xi32, 1>
    memref.dealloc %l2o : memref<64xi32, 1>
  }
  air.channel.get @cout[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
  return
}

// -----

// Channel-carried memtile variant: the count lives on the OUTBOUND (drain)
// channel declaration -- the authoritative carrier -- not on the alloc.
// AllocL2BuffersPattern derives it from the drain put and stamps the memtile
// buffer. Strong invariants, bound to the SAME buf-free lock (the one that
// inits to N): the fill (S2MM) does AcquireGreaterEqual N, while each drain
// (MM2S) re-send releases that lock by exactly 1 -- i.e. ONE fill enables N
// drains and the MM2S side is NOT scaled. N=2 here (distinct from the N=4
// cases above) to catch any hard-coded 4.

// CHECK-LABEL: aie.device
// CHECK: %[[BUFFREE:.*]] = aie.lock(%{{.*}}) {init = 2 : i32}
// CHECK: aie.buffer(%{{.*}}) {air.refeed_count = 2 : i32
// CHECK: aie.memtile_dma
// Drain (MM2S) re-send: releases the buf-free lock by 1 (not scaled).
// CHECK: aie.use_lock(%[[BUFFREE]], Release, 1)
// Fill (S2MM): one fill waits for all N drains, acquiring the buf-free lock N.
// CHECK: aie.use_lock(%[[BUFFREE]], AcquireGreaterEqual, 2)

air.channel @cin2 [1, 1]
air.channel @to_core2 [1, 1] {air.refeed_count = 2 : i32}
air.channel @from_core2 [1, 1]
air.channel @cout2 [1, 1]
func.func @mt_refeed_chan(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
  %c1 = arith.constant 1 : index
  air.channel.put @cin2[] (%arg0[] [] []) {id = 1 : i32} : (memref<64xi32>)
  air.segment @seg0 {
    %c1_0 = arith.constant 1 : index
    %l2 = memref.alloc() : memref<64xi32, 1>
    air.channel.get @cin2[] (%l2[] [] []) {id = 2 : i32} : (memref<64xi32, 1>)
    air.channel.put @to_core2[] (%l2[] [] []) {id = 3 : i32} : (memref<64xi32, 1>)
    air.herd tile(%tx, %ty) in (%sx = %c1_0, %sy = %c1_0) attributes {sym_name = "herd0"} {
      %buf = memref.alloc() : memref<64xi32, 2>
      %res = memref.alloc() : memref<64xi32, 2>
      air.channel.get @to_core2[%tx, %ty] (%buf[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
      air.channel.put @from_core2[%tx, %ty] (%res[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
      memref.dealloc %buf : memref<64xi32, 2>
      memref.dealloc %res : memref<64xi32, 2>
    }
    %l2o = memref.alloc() : memref<64xi32, 1>
    air.channel.get @from_core2[] (%l2o[] [] []) {id = 6 : i32} : (memref<64xi32, 1>)
    air.channel.put @cout2[] (%l2o[] [] []) {id = 7 : i32} : (memref<64xi32, 1>)
    memref.dealloc %l2 : memref<64xi32, 1>
    memref.dealloc %l2o : memref<64xi32, 1>
  }
  air.channel.get @cout2[] (%arg1[] [] []) {id = 8 : i32} : (memref<64xi32>)
  return
}

// -----

// Without air.refeed_count the producing core release stays 1 (no re-feed),
// the buf-free acquire stays 1, and no lock inits to 4. (This section is last
// so its CHECK-NOT window is not entered by the positive sections above.)

// CHECK-LABEL: aie.device
// CHECK-NOT: {init = 4 : i32}
// CHECK-NOT: AcquireGreaterEqual, 4
// CHECK-NOT: Release, 4

#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_1 [1, 1]
func.func @no_refeed() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_1[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_1[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
