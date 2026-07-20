//===- air_channel_index_switch_shared_ring.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// A single air.channel is consumed by air.channel.get ops in two
// mutually-exclusive arms of an scf.index_switch (RTP-selected mode, carried as
// an i32 herd argument). The two arms read into DIFFERENT local buffers. Because
// only one arm executes per dispatch, the S2MM DMA must be ONE shared count-free
// ring (single dma_start, next_bd self-loop) consumed by both arms, NOT one
// dma_start task per arm (which ahead-blocks the idle arm's task at the head of
// the channel queue and stalls the active arm).

// CHECK: aie.device
// CHECK: aie.mem
// Exactly one S2MM dma_start on channel 0 -> a self-looping count-free ring.
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.next_bd
// CHECK-NOT: aie.dma_start(S2MM, 0

air.channel @xin [1, 1]
func.func @index_switch_shared_ring(%ext: memref<2048xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%e=%ext) : memref<2048xbf16> {
    %p = air.channel.put async @xin[] (%e[] [] []) {id = 1 : i32} : (memref<2048xbf16>)
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %2 = air.herd @rms async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%arm=%c1_i32) : i32 {
        %armi = arith.index_cast %arm : i32 to index
        %c0 = arith.constant 0 : index
        %c1i = arith.constant 1 : index
        %c7 = arith.constant 7 : index
        // Two mutually-exclusive arms consume the SAME channel into DIFFERENT
        // per-arm buffers, with asymmetric read counts (7 vs 1). Without the
        // shared-arm-ring transform this lowers to two S2MM dma_start tasks that
        // ahead-block; the transform must unify them onto one count-free ring.
        scf.index_switch %armi
        case 0 {
          %tokA, %bufA = air.execute -> (memref<2048xbf16, 2>) {
            %a = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %a : memref<2048xbf16, 2>
          }
          %gl = scf.for %iv = %c0 to %c7 step %c1i iter_args(%t = %tokA) -> (!air.async.token) {
            %gi = air.channel.get async [%t] @xin[] (%bufA[] [] []) : (memref<2048xbf16, 2>)
            scf.yield %gi : !air.async.token
          }
          air.execute [%gl] {
            memref.dealloc %bufA : memref<2048xbf16, 2>
          }
          scf.yield
        }
        default {
          %tokB, %bufB = air.execute -> (memref<2048xbf16, 2>) {
            %b = memref.alloc() : memref<2048xbf16, 2>
            air.execute_terminator %b : memref<2048xbf16, 2>
          }
          %g1 = air.channel.get async [%tokB] @xin[] (%bufB[] [] []) : (memref<2048xbf16, 2>)
          air.execute [%g1] {
            memref.dealloc %bufB : memref<2048xbf16, 2>
          }
          scf.yield
        }
      }
    }
  }
  return
}
