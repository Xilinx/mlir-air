//===- air_channel_mmio.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Positive tests for the channel_type="mmio" lowering in air-to-aie:
//   * the L3 put becomes aiex.npu.blockwrite targeting the matching get's
//     L1 aie.buffer, and the get is erased;
//   * no DMA channel, shim allocation, or aie.flow is reserved for an
//     mmio channel;
//   * mmio channels coexist with regular dma_stream channels;
//   * mmio with `broadcast_shape` fans out to N back-to-back blockwrites,
//     one per destination L1 buffer (no hardware fanout).
//
// The negative case (non-constant put source) is in
// `air_channel_mmio_invalid.mlir`. Each split here uses its own check
// prefix so directives don't leak across split boundaries.

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu1" | FileCheck %s --check-prefixes=CHECK-SIMPLE,CHECK-MIXED,CHECK-BCAST,CHECK-INDEXED,CHECK-BF16

// -----

// Simple one-to-one mmio transfer: put at L3 lowers to a single
// aiex.npu.blockwrite into the L1 aie.buffer; the get is erased; no shim
// DMA, no aie.flow.
//
// CHECK-SIMPLE-LABEL: aie.device(npu1)
// CHECK-SIMPLE:         memref.global @const_data : memref<8xi32> = dense<42> {air.mmio_global}
// CHECK-SIMPLE:         %[[TILE:.+]] = aie.tile(0, 2)
// CHECK-SIMPLE:         %[[BUF:.+]] = aie.buffer(%[[TILE]]) {sym_name = "[[BUFNAME:.+]]"} : memref<8xi32, 2>
// CHECK-SIMPLE:         aie.core(%[[TILE]])
// CHECK-SIMPLE-NOT:     air.channel.put @mmio_chan
// CHECK-SIMPLE-NOT:     air.channel.get @mmio_chan
// CHECK-SIMPLE-NOT:     aie.flow
// CHECK-SIMPLE-NOT:     aie.shim_dma_allocation
//
// CHECK-SIMPLE-LABEL: func.func @mmio_simple
// CHECK-SIMPLE:         memref.get_global @const_data
// CHECK-SIMPLE:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @[[BUFNAME]]} : memref<8xi32>
// CHECK-SIMPLE:         air.launch
// CHECK-SIMPLE-NOT:     air.channel.put @mmio_chan
// CHECK-SIMPLE-NOT:     air.channel.get @mmio_chan

memref.global "private" @const_data : memref<8xi32> = dense<42>
air.channel @mmio_chan [] {channel_type = "mmio"}
func.func @mmio_simple() {
  %src = memref.get_global @const_data : memref<8xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<8xi32> {
    air.channel.put @mmio_chan[] (%a[] [] []) : (memref<8xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @mmio_chan[] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}

// -----

// An mmio channel coexists with a regular dma_stream channel without
// interference: the dma_stream side still receives its shim DMA / flow
// allocation, and the mmio side still lowers to a blockwrite. The
// dma_stream put survives in the L3 control func (it lowers later in
// AIRLoweringPass), while the mmio put and get are gone.
//
// CHECK-MIXED-LABEL: func.func @mixed
// CHECK-MIXED:         memref.get_global @mmio_const
// CHECK-MIXED:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<8xi32>
// CHECK-MIXED:         air.launch
// CHECK-MIXED-NOT:     air.channel.put @mmio_chan2
// CHECK-MIXED-NOT:     air.channel.get @mmio_chan2

memref.global "private" @mmio_const : memref<8xi32> = dense<7>
air.channel @mmio_chan2 [] {channel_type = "mmio"}
air.channel @dma_chan [] {channel_type = "dma_stream"}
func.func @mixed(%dma_src: memref<16xi32>) {
  %src = memref.get_global @mmio_const : memref<8xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src, %b = %dma_src) : memref<8xi32>, memref<16xi32> {
    air.channel.put @mmio_chan2[] (%a[] [] []) : (memref<8xi32>)
    air.channel.put @dma_chan[] (%b[] [] []) : (memref<16xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc_m = memref.alloc() : memref<8xi32, 2>
        air.channel.get @mmio_chan2[] (%alloc_m[] [] []) : (memref<8xi32, 2>)
        %alloc_d = memref.alloc() : memref<16xi32, 2>
        air.channel.get @dma_chan[] (%alloc_d[] [] []) : (memref<16xi32, 2>)
        memref.dealloc %alloc_m : memref<8xi32, 2>
        memref.dealloc %alloc_d : memref<16xi32, 2>
      }
    }
  }
  return
}

// -----

// Broadcast mmio: one put with `broadcast_shape` fans out to two distinct
// L1 buffers via two back-to-back blockwrites. There is no hardware
// fanout for MMIO writes (unlike aie.flow), so the controller writes each
// destination separately. Per MMIO_BENCHMARK.md this remains essentially
// free at decode-attention payload sizes.
//
// CHECK-BCAST-LABEL: aie.device(npu1)
// CHECK-BCAST:         aie.tile(0, 2)
// CHECK-BCAST:         aie.tile(0, 3)
// CHECK-BCAST-DAG:     aie.buffer(%{{.+}}) {sym_name = "[[BUF0:.+]]"} : memref<8xi32, 2>
// CHECK-BCAST-DAG:     aie.buffer(%{{.+}}) {sym_name = "[[BUF1:.+]]"} : memref<8xi32, 2>
// CHECK-BCAST-NOT:     air.channel.put @bcast_mmio
// CHECK-BCAST-NOT:     air.channel.get @bcast_mmio
// CHECK-BCAST-NOT:     aie.flow
// CHECK-BCAST-NOT:     aie.shim_dma_allocation
//
// CHECK-BCAST-LABEL: func.func @bcast
// CHECK-BCAST:         memref.get_global @const_q
// CHECK-BCAST:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{buf[01]}}} : memref<8xi32>
// CHECK-BCAST:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{buf[01]}}} : memref<8xi32>
// CHECK-BCAST:         air.launch
// CHECK-BCAST-NOT:     air.channel.put @bcast_mmio
// CHECK-BCAST-NOT:     air.channel.get @bcast_mmio

memref.global "private" @const_q : memref<8xi32> = dense<5>
air.channel @bcast_mmio [1] {channel_type = "mmio", broadcast_shape = [2]}
func.func @bcast() {
  %src = memref.get_global @const_q : memref<8xi32>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<8xi32> {
    %c0 = arith.constant 0 : index
    air.channel.put @bcast_mmio[%c0] (%a[] [] []) : (memref<8xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c2_0) {
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @bcast_mmio[%tx, %ty] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}

// -----

// Non-broadcast indexed mmio with a multi-tile herd: the bundled channel
// `[N]` has N puts at constant indices and N herd-side gets at the
// herd induction var that resolves per tile. After herd outlining each
// tile gets its own constant-indexed get; the lowering matches one put
// to each get and emits N blockwrites, each targeting the matching
// tile's L1 buffer.
//
// This case used to be broken by `specializeChannelBundle` splitting
// the bundle into per-position channels but leaving the host-side puts
// orphaned on the original symbol. The fix makes that pattern skip mmio
// channels and lets the MMIO lowering match across the bundle directly.
//
// CHECK-INDEXED-LABEL: aie.device(npu1)
// CHECK-INDEXED-DAG:     memref.global @c0 : memref<8xi32> = dense<10> {air.mmio_global}
// CHECK-INDEXED-DAG:     memref.global @c1 : memref<8xi32> = dense<20> {air.mmio_global}
// CHECK-INDEXED-DAG:     aie.tile(0, 2)
// CHECK-INDEXED-DAG:     aie.tile(1, 2)
// CHECK-INDEXED-NOT:     air.channel.put @qm
// CHECK-INDEXED-NOT:     air.channel.get @qm
// CHECK-INDEXED-NOT:     aie.shim_dma_allocation
//
// CHECK-INDEXED-LABEL: func.func @indexed
// CHECK-INDEXED:         memref.get_global @c0
// CHECK-INDEXED:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<8xi32>
// CHECK-INDEXED:         memref.get_global @c1
// CHECK-INDEXED:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<8xi32>

memref.global "private" @c0 : memref<8xi32> = dense<10>
memref.global "private" @c1 : memref<8xi32> = dense<20>
air.channel @qm [2] {channel_type = "mmio"}
func.func @indexed() {
  %g0 = memref.get_global @c0 : memref<8xi32>
  %g1 = memref.get_global @c1 : memref<8xi32>
  %c1i = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1i) args(%a0 = %g0, %a1 = %g1) : memref<8xi32>, memref<8xi32> {
    %c0i = arith.constant 0 : index
    %c1ii = arith.constant 1 : index
    air.channel.put @qm[%c0i] (%a0[] [] []) : (memref<8xi32>)
    air.channel.put @qm[%c1ii] (%a1[] [] []) : (memref<8xi32>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c2_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<8xi32, 2>
        air.channel.get @qm[%tx] (%alloc[] [] []) : (memref<8xi32, 2>)
        memref.dealloc %alloc : memref<8xi32, 2>
      }
    }
  }
  return
}

// -----

// Non-i32 element type (here bf16): the `aiex.npu.blockwrite` translator
// only handles i32 so the lowering mirrors the bf16 global into the
// device as a 1-D `memref<Nxi32>` with the same raw bytes (suffixed
// `_mmio_i32` to keep the original undisturbed). Blockwrite carries the
// i32 view; the destination buffer keeps its natural bf16 type since
// `buffer = @sym` is just a symbol ref.
//
// CHECK-BF16-LABEL: aie.device(npu1)
// CHECK-BF16:         memref.global @qbf16_mmio_i32 : memref<2xi32>{{.*}}{air.mmio_global}
//
// CHECK-BF16-LABEL: func.func @bf16_payload
// CHECK-BF16:         memref.get_global @qbf16_mmio_i32 : memref<2xi32>
// CHECK-BF16:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<2xi32>

memref.global "private" @qbf16 : memref<2x2xbf16> = dense<1.5>
air.channel @qbf16_chan [] {channel_type = "mmio"}
func.func @bf16_payload() {
  %src = memref.get_global @qbf16 : memref<2x2xbf16>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<2x2xbf16> {
    air.channel.put @qbf16_chan[] (%a[] [] []) : (memref<2x2xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<2x2xbf16, 2>
        air.channel.get @qbf16_chan[] (%alloc[] [] []) : (memref<2x2xbf16, 2>)
        memref.dealloc %alloc : memref<2x2xbf16, 2>
      }
    }
  }
  return
}
