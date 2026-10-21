//===- air_channel_release_ring_order.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// One channel fills K, then V twice into one buffer, with two credits. The
// second V and the next trip's K may come in once the first @pv is done; the
// next trip's first V waits for the second @pv.
// CHECK-LABEL: aie.core
// CHECK: func.call @qk
// CHECK: func.call @pv
// CHECK-NEXT: aie.use_lock(%[[REL:[a-z0-9_]+]], Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%[[REL]], Release, %{{.*}})
// CHECK: func.call @pv
// CHECK-NEXT: aie.use_lock(%[[REL]], Release, %{{.*}})
// CHECK-NEXT: func.call @other

air.channel @kv [1, 1]
func.func private @qk(memref<16x64xbf16, 2>)
func.func private @pv(memref<16x32xbf16, 2>)
func.func private @other()
func.func @k_then_v_twice() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %tk, %k = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %tv, %v = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %4 = air.channel.put async [%tk, %tv]  @kv[] (%k[] [] []) : (memref<16x64xbf16, 2>)
          %5 = air.channel.put async [%4]  @kv[] (%v[] [] []) : (memref<16x32xbf16, 2>)
          %6 = air.channel.put async [%5]  @kv[] (%v[] [] []) : (memref<16x32xbf16, 2>)
          %dk = air.execute [%6] {
            memref.dealloc %k : memref<16x64xbf16, 2>
          }
          %dv = air.execute [%dk] {
            memref.dealloc %v : memref<16x32xbf16, 2>
          }
          scf.yield %dv : !air.async.token
        } else {
          %tk, %k = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %tv, %v = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %4 = air.channel.get async [%tk, %tv]  @kv[] (%k[] [] []) : (memref<16x64xbf16, 2>)
          %q = air.execute [%4] {
            func.call @qk(%k) : (memref<16x64xbf16, 2>) -> ()
          }
          %5 = air.channel.get async [%q]  @kv[] (%v[] [] []) : (memref<16x32xbf16, 2>)
          %p0 = air.execute [%5] {
            func.call @pv(%v) : (memref<16x32xbf16, 2>) -> ()
          }
          %6 = air.channel.get async [%p0]  @kv[] (%v[] [] []) : (memref<16x32xbf16, 2>)
          %p1 = air.execute [%6] {
            func.call @pv(%v) : (memref<16x32xbf16, 2>) -> ()
          }
          %o = air.execute [%p1] {
            func.call @other() : () -> ()
          }
          %dk = air.execute [%o] {
            memref.dealloc %k : memref<16x64xbf16, 2>
          }
          %dv = air.execute [%dk] {
            memref.dealloc %v : memref<16x32xbf16, 2>
          }
          scf.yield %dv : !air.async.token
        }
      }
    }
  }
  return
}
