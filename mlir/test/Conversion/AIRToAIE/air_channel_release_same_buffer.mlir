//===- air_channel_release_same_buffer.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Two gets on one channel fill the same buffer. A release lets the DMA fill
// the channel's next buffer in order, which here is that same buffer again,
// so the first release goes after @work, its last reader, not to the dealloc.
// CHECK: aie.core(%{{.*}})  {
// CHECK-NOT: aie.end
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK: func.call @work
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: func.call @other
// CHECK: aie.end

air.channel @channel_0 [1, 1]
func.func private @work(memref<32x32xbf16, 2>)
func.func private @other()
func.func @same_buffer_twice() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %t, %buf = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %4 = air.channel.put async [%t]  @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
          %5 = air.channel.put async [%4]  @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
          %d = air.execute [%5] {
            memref.dealloc %buf : memref<32x32xbf16, 2>
          }
          scf.yield %d : !air.async.token
        } else {
          %t, %buf = air.execute -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          }
          %4 = air.channel.get async [%t]  @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
          %w = air.execute [%4] {
            func.call @work(%buf) : (memref<32x32xbf16, 2>) -> ()
          }
          %o = air.execute [%w] {
            func.call @other() : () -> ()
          }
          %5 = air.channel.get async [%o]  @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
          %w2 = air.execute [%5] {
            func.call @work(%buf) : (memref<32x32xbf16, 2>) -> ()
          }
          %d = air.execute [%w2] {
            memref.dealloc %buf : memref<32x32xbf16, 2>
          }
          scf.yield %d : !air.async.token
        }
      }
    }
  }
  return
}
