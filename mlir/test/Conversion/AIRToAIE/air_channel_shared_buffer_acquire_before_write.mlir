//===- air_channel_shared_buffer_acquire_before_write.mlir -----*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// When several outbound puts share one L1 staging buffer, the buf-free
// (write) lock is acquired per put rather than once at block start, so the
// core cannot overwrite the buffer while the DMA is still reading the
// previous put. That acquire must dominate the code that PRODUCES the put's
// data, not merely sit in front of the put itself: the producing ops (a
// kernel func.call, or a store loop) run BEFORE the put in program order, so
// acquiring at the put leaves those writes unguarded and the DMA streams a
// buffer that is being rewritten underneath it.
//
// The failure is silent data corruption, not a hang, and its severity scales
// with air.refeed_count: with count N the previous put still has up to N
// sends in flight when the core starts overwriting. It is masked for the
// first put only, because the buf-free lock inits to N.

// -----

// One buffer, TWO puts, each preceded by the kernel call that fills it.
// Each buf-free acquire must come before its own producing call.

// CHECK-LABEL: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK-DAG:   %[[WLOCK:.*]] = aie.lock(%[[TILE]], {{[0-9]+}}) {init = 1 : i32}
// CHECK-DAG:   %[[RLOCK:.*]] = aie.lock(%[[TILE]], {{[0-9]+}}) {init = 0 : i32}
// CHECK:       aie.core(%[[TILE]])
// The first producing call must NOT appear before the first acquire.
// CHECK-NOT:     func.call @fill
// CHECK:         aie.use_lock(%[[WLOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:         func.call @fill
// CHECK:         aie.use_lock(%[[RLOCK]], Release, %{{.*}})
// Likewise for the second emission: acquire, then produce, then release.
// CHECK-NOT:     func.call @fill
// CHECK:         aie.use_lock(%[[WLOCK]], AcquireGreaterEqual, %{{.*}})
// CHECK:         func.call @fill
// CHECK:         aie.use_lock(%[[RLOCK]], Release, %{{.*}})

air.channel @chan_w [1, 1]
func.func @shared_buffer_write_then_put() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %async_token_0, %l2_buf = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %3 = air.channel.get async @chan_w[] (%l2_buf[] [] []) : (memref<32x32xbf16, 1>)
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) attributes {link_with = "fill.o"} {
        %async_token_2, %buf = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %tok_a = air.execute [%async_token_2] {
          func.call @fill(%buf) : (memref<32x32xbf16, 2>) -> ()
        }
        %tok_1 = air.channel.put async [%tok_a] @chan_w[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        %tok_b = air.execute [%tok_1] {
          func.call @fill(%buf) : (memref<32x32xbf16, 2>) -> ()
        }
        %tok_2 = air.channel.put async [%tok_b] @chan_w[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        %async_token_3 = air.execute [%tok_2] {
          memref.dealloc %buf : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
func.func private @fill(memref<32x32xbf16, 2>) -> () attributes {link_with = "fill.o"}

// -----

// Same shape with a re-feed channel: air.refeed_count=3 on the channel, and a
// per-emission override of 2 on the second put -- two emissions from one
// buffer with DIFFERENT counts, the shape a fused decode layer's convergent
// activation feed produces. Both acquires are scaled (3 then 2) and both must
// still dominate their producing call. Without the fix the second call runs
// while up to 3 re-sends of the first emission are still in flight.

// CHECK-LABEL: aie.device
// CHECK-DAG:   %[[TILE2:.*]] = aie.tile(2, 3)
// The buf-free lock inits to the channel-level re-feed count.
// CHECK-DAG:   %[[WLOCK2:.*]] = aie.lock(%[[TILE2]], {{[0-9]+}}) {init = 3 : i32}
// CHECK-DAG:   %[[RLOCK2:.*]] = aie.lock(%[[TILE2]], {{[0-9]+}}) {init = 0 : i32}
// CHECK:       aie.core(%[[TILE2]])
// Emission 1 (channel count 3): acquire 3, produce, release 3.
// CHECK-NOT:     func.call @fill2
// CHECK:         aie.use_lock(%[[WLOCK2]], AcquireGreaterEqual, %c3{{.*}})
// CHECK:         func.call @fill2
// CHECK:         aie.use_lock(%[[RLOCK2]], Release, %c3{{.*}})
// Emission 2 (per-put override 2): its acquire must sit BETWEEN emission 1's
// release and its own producing call -- i.e. the acquire neither crosses back
// over the previous put nor slides past the write it is meant to guard.
// CHECK-NOT:     func.call @fill2
// CHECK:         aie.use_lock(%[[WLOCK2]], AcquireGreaterEqual, %c2{{.*}})
// CHECK:         func.call @fill2
// CHECK:         aie.use_lock(%[[RLOCK2]], Release, %c2{{.*}})

air.channel @chan_r [1, 1] {air.refeed_count = 3 : i32}
func.func @shared_buffer_refeed_write_then_put() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %async_token_0, %l2_buf = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      %3 = air.channel.get async @chan_r[] (%l2_buf[] [] []) : (memref<32x32xbf16, 1>)
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) attributes {link_with = "fill2.o"} {
        %async_token_2, %buf = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %tok_a = air.execute [%async_token_2] {
          func.call @fill2(%buf) : (memref<32x32xbf16, 2>) -> ()
        }
        %tok_1 = air.channel.put async [%tok_a] @chan_r[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        %tok_b = air.execute [%tok_1] {
          func.call @fill2(%buf) : (memref<32x32xbf16, 2>) -> ()
        }
        %tok_2 = air.channel.put async [%tok_b] @chan_r[] (%buf[] [] []) {air.refeed_count = 2 : i32} : (memref<32x32xbf16, 2>)
        %async_token_3 = air.execute [%tok_2] {
          memref.dealloc %buf : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
func.func private @fill2(memref<32x32xbf16, 2>) -> () attributes {link_with = "fill2.o"}

// -----

// The producer may also be a plain store loop rather than a kernel call: the
// hoist walks nested regions, so an scf.for body that writes the buffer still
// pulls the acquire above the whole loop.

// CHECK-LABEL: aie.device
// CHECK-DAG:   %[[TILE3:.*]] = aie.tile(2, 3)
// CHECK-DAG:   %[[WLOCK3:.*]] = aie.lock(%[[TILE3]], {{[0-9]+}}) {init = 1 : i32}
// CHECK:       aie.core(%[[TILE3]])
// CHECK-NOT:     scf.for
// CHECK:         aie.use_lock(%[[WLOCK3]], AcquireGreaterEqual, %{{.*}})
// CHECK:         scf.for

air.channel @chan_s [1, 1]
func.func @shared_buffer_store_loop_then_put() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %async_token_0, %l2_buf = air.execute -> (memref<32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32xbf16, 1>
        air.execute_terminator %alloc : memref<32xbf16, 1>
      }
      %3 = air.channel.get async @chan_s[] (%l2_buf[] [] []) : (memref<32xbf16, 1>)
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant 1.000000e+00 : bf16
        %async_token_2, %buf = air.execute -> (memref<32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %alloc : memref<32xbf16, 2>
        }
        %tok_a = air.execute [%async_token_2] {
          scf.for %i = %c0 to %c32 step %c1_h {
            memref.store %cst, %buf[%i] : memref<32xbf16, 2>
          }
        }
        %tok_1 = air.channel.put async [%tok_a] @chan_s[] (%buf[] [] []) : (memref<32xbf16, 2>)
        %tok_b = air.execute [%tok_1] {
          scf.for %i = %c0 to %c32 step %c1_h {
            memref.store %cst, %buf[%i] : memref<32xbf16, 2>
          }
        }
        %tok_2 = air.channel.put async [%tok_b] @chan_s[] (%buf[] [] []) : (memref<32xbf16, 2>)
        %async_token_3 = air.execute [%tok_2] {
          memref.dealloc %buf : memref<32xbf16, 2>
        }
      }
    }
  }
  return
}
