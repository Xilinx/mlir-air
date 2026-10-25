//===- air_channel_release_ring_blocks.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Two loops get from one channel into the same two buffers, each trip in the
// DMA's order. The first loop fills each buffer twice per trip; a buffer is
// released after its last reader, not at its next get.
// CHECK-LABEL: aie.core
// CHECK: scf.for
// CHECK: func.call @wa
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: func.call @other
// CHECK: func.call @wb
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: func.call @other
// CHECK: scf.for

air.channel @channel_0 [1, 1]
func.func private @wa(memref<32xbf16, 2>)
func.func private @wb(memref<32xbf16, 2>)
func.func private @other()
func.func @two_loops() {
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
          %tb, %b = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %l = scf.for %i = %c0_i to %c4_i step %c1_i iter_args(%tk = %ta) -> (!air.async.token) {
            %4 = air.channel.put async [%tk, %tb]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
            %5 = air.channel.put async [%4]  @channel_0[] (%b[] [] []) : (memref<32xbf16, 2>)
            scf.yield %5 : !air.async.token
          }
          %da = air.execute [%l] {
            memref.dealloc %a : memref<32xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32xbf16, 2>
            air.execute_terminator %alloc : memref<32xbf16, 2>
          }
          %l1 = scf.for %i = %c0_i to %c4_i step %c1_i iter_args(%tk = %ta) -> (!air.async.token) {
            %4 = air.channel.get async [%tk, %tb]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
            %w0 = air.execute [%4] {
              func.call @wa(%a) : (memref<32xbf16, 2>) -> ()
            }
            %o0 = air.execute [%w0] {
              func.call @other() : () -> ()
            }
            %5 = air.channel.get async [%o0]  @channel_0[] (%b[] [] []) : (memref<32xbf16, 2>)
            %w1 = air.execute [%5] {
              func.call @wb(%b) : (memref<32xbf16, 2>) -> ()
            }
            %o1 = air.execute [%w1] {
              func.call @other() : () -> ()
            }
            %6 = air.channel.get async [%o1]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
            %w2 = air.execute [%6] {
              func.call @wa(%a) : (memref<32xbf16, 2>) -> ()
            }
            %7 = air.channel.get async [%w2]  @channel_0[] (%b[] [] []) : (memref<32xbf16, 2>)
            %w3 = air.execute [%7] {
              func.call @wb(%b) : (memref<32xbf16, 2>) -> ()
            }
            scf.yield %w3 : !air.async.token
          }
          %l2 = scf.for %i = %c0_i to %c4_i step %c1_i iter_args(%tk = %l1) -> (!air.async.token) {
            %4 = air.channel.get async [%tk]  @channel_0[] (%a[] [] []) : (memref<32xbf16, 2>)
            %w0 = air.execute [%4] {
              func.call @wa(%a) : (memref<32xbf16, 2>) -> ()
            }
            %5 = air.channel.get async [%w0]  @channel_0[] (%b[] [] []) : (memref<32xbf16, 2>)
            %w1 = air.execute [%5] {
              func.call @wb(%b) : (memref<32xbf16, 2>) -> ()
            }
            scf.yield %w1 : !air.async.token
          }
          %da = air.execute [%l2] {
            memref.dealloc %a : memref<32xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        }
      }
    }
  }
  return
}
