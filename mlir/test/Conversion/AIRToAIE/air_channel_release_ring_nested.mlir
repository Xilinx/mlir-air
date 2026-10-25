//===- air_channel_release_ring_nested.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// A get before a loop of gets into the same buffer, as a peeled first
// iteration: its release comes before the loop, whose gets need the buffer.
// CHECK-LABEL: aie.core
// CHECK: func.call @w
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: scf.for

air.channel @channel_0 [1, 1]
func.func private @w(memref<32xbf16, 2>)
func.func @peeled() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_i = arith.constant 0 : index
        %c1_i = arith.constant 1 : index
        %c4_i = arith.constant 4 : index
        %cmp = arith.cmpi eq, %arg9, %c0_i : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %ta, %a = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %l = scf.for %i = %c0_i to %c4_i step %c1_i iter_args(%tk = %ta) -> (!air.async.token) {
            %4 = air.channel.put async [%tk]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
            scf.yield %4 : !air.async.token
          }
          %da = air.execute [%l] {
            memref.dealloc %a : memref<32xbf16, 2>
          }
          scf.yield %da : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %4 = air.channel.get async [%ta]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
          %w0 = air.execute [%4] {
            func.call @w(%a) : (memref<32xbf16, 2>) -> ()
          }
          %l = scf.for %i = %c1_i to %c4_i step %c1_i iter_args(%tk = %w0) -> (!air.async.token) {
            %5 = air.channel.get async [%tk]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
            %w1 = air.execute [%5] {
              func.call @w(%a) : (memref<32xbf16, 2>) -> ()
            }
            scf.yield %w1 : !air.async.token
          }
          %6 = air.channel.get async [%l]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
          %w2 = air.execute [%6] {
            func.call @w(%a) : (memref<32xbf16, 2>) -> ()
          }
          %da = air.execute [%w2] {
            memref.dealloc %a : memref<32xbf16, 2>
          }
          scf.yield %da : !air.async.token
        }
      }
    }
  }
  return
}
