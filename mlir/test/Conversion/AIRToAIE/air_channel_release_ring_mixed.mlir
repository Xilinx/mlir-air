//===- air_channel_release_ring_mixed.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Three channels share the core's two inbound DMA channels, each with its own
// lock. Only B's release and A's first come before the second @wa.
// CHECK-LABEL: aie.core
// CHECK: func.call @wa
// CHECK-COUNT-2: Release
// CHECK-NOT: Release
// CHECK: func.call @wa

air.channel @a [1, 1]
air.channel @b [1, 1]
air.channel @c [1, 1]
func.func private @wa(memref<32xbf16, 2>)
func.func private @wb(memref<32xbf16, 2>)
func.func private @wc(memref<32xbf16, 2>)
func.func @mixed_channels() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %ta, %a = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %tc, %c = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %4 = air.channel.put async [%ta]  @a[] (%a[] [] []) : (memref<32xbf16, 2>)
          %5 = air.channel.put async [%tb, %4]  @b[] (%b[] [] []) : (memref<32xbf16, 2>)
          %6 = air.channel.put async [%5]  @a[] (%a[] [] []) : (memref<32xbf16, 2>)
          %7 = air.channel.put async [%tc, %6]  @c[] (%c[] [] []) : (memref<32xbf16, 2>)
          scf.yield %7 : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %tc, %c = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %4 = air.channel.get async [%ta]  @a[] (%a[] [] []) : (memref<32xbf16, 2>)
          %w0 = air.execute [%4] {
            func.call @wa(%a) : (memref<32xbf16, 2>) -> ()
          }
          %5 = air.channel.get async [%tb, %w0]  @b[] (%b[] [] []) : (memref<32xbf16, 2>)
          %w1 = air.execute [%5] {
            func.call @wb(%b) : (memref<32xbf16, 2>) -> ()
          }
          %6 = air.channel.get async [%w1]  @a[] (%a[] [] []) : (memref<32xbf16, 2>)
          %w2 = air.execute [%6] {
            func.call @wa(%a) : (memref<32xbf16, 2>) -> ()
          }
          %7 = air.channel.get async [%tc, %w2]  @c[] (%c[] [] []) : (memref<32xbf16, 2>)
          %w3 = air.execute [%7] {
            func.call @wc(%c) : (memref<32xbf16, 2>) -> ()
          }
          %da = air.execute [%w3] {
            memref.dealloc %a : memref<32xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<32xbf16, 2>
          }
          %dc = air.execute [%db] {
            memref.dealloc %c : memref<32xbf16, 2>
          }
          scf.yield %dc : !air.async.token
        }
      }
    }
  }
  return
}
