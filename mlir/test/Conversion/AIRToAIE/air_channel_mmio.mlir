//===- air_channel_mmio.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Positive tests for channel_type="mmio" in air-to-aie. Each split has
// its own CHECK prefix so directives don't leak across boundaries.
// Negative cases live in `air_channel_mmio_invalid.mlir`.

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu1" | FileCheck %s --check-prefixes=CHECK-SIMPLE,CHECK-MIXED,CHECK-BCAST,CHECK-INDEXED,CHECK-BF16,CHECK-BF16NS,CHECK-I8,CHECK-I16

// -----

// One-to-one: L3 put → one blockwrite into the L1 buffer; get erased;
// no shim DMA, no aie.flow.
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

// mmio + dma_stream coexist: dma_stream keeps its shim/flow allocation
// and survives in the L3 control func; mmio lowers to blockwrite.
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

// Broadcast mmio: one put with `broadcast_shape` → N back-to-back
// blockwrites (no hardware fanout for MMIO).
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

// Indexed mmio over a `[N]` bundle on a multi-tile herd: each constant
// index pairs one host-side put with one per-tile get, lowering to N
// blockwrites. Regression: specializeChannelBundle used to split the
// bundle and orphan the host-side puts (now skipped for mmio).
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

// Splat bf16: blockwrite's i32-only translator forces a `memref<Nxi32>`
// mirror (suffixed `_mmio_i32`). bf16(1.5)=0x3FC0 → i32 0x3FC03FC0 =
// 1069563840, splat preserved. Alignment also round-trips.
//
// CHECK-BF16-LABEL: aie.device(npu1)
// CHECK-BF16:         memref.global @qbf16_mmio_i32 : memref<2xi32> = dense<1069563840> {air.mmio_global, alignment = 64 : i64}
//
// CHECK-BF16-LABEL: func.func @bf16_payload
// CHECK-BF16:         memref.get_global @qbf16_mmio_i32 : memref<2xi32>
// CHECK-BF16:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<2xi32>

memref.global "private" @qbf16 : memref<2x2xbf16> = dense<1.5> {alignment = 64 : i64}
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

// -----

// Non-splat bf16: hits the wholesale-copy branch. bf16 bytes
// {3FC0, 4020, 4060, 4090} LE-pack to i32 {0x40203FC0, 0x40904060}
// = {1075855296, 1083195488}.
//
// CHECK-BF16NS-LABEL: aie.device(npu1)
// CHECK-BF16NS:         memref.global @qbf16ns_mmio_i32 : memref<2xi32> = dense<[1075855296, 1083195488]> {air.mmio_global}
//
// CHECK-BF16NS-LABEL: func.func @bf16_nonsplat
// CHECK-BF16NS:         memref.get_global @qbf16ns_mmio_i32 : memref<2xi32>
// CHECK-BF16NS:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<2xi32>

memref.global "private" @qbf16ns : memref<2x2xbf16> = dense<[[1.5, 2.5], [3.5, 4.5]]>
air.channel @qbf16ns_chan [] {channel_type = "mmio"}
func.func @bf16_nonsplat() {
  %src = memref.get_global @qbf16ns : memref<2x2xbf16>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<2x2xbf16> {
    air.channel.put @qbf16ns_chan[] (%a[] [] []) : (memref<2x2xbf16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<2x2xbf16, 2>
        air.channel.get @qbf16ns_chan[] (%alloc[] [] []) : (memref<2x2xbf16, 2>)
        memref.dealloc %alloc : memref<2x2xbf16, 2>
      }
    }
  }
  return
}

// -----

// i8 splat: bytesPerElt=1 stride. dense<66> × 4 → 0x42424242 = 1111638594.
//
// CHECK-I8-LABEL: aie.device(npu1)
// CHECK-I8:         memref.global @c8s_mmio_i32 : memref<1xi32> = dense<1111638594> {air.mmio_global}
//
// CHECK-I8-LABEL: func.func @i8_splat
// CHECK-I8:         memref.get_global @c8s_mmio_i32 : memref<1xi32>
// CHECK-I8:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<1xi32>

memref.global "private" @c8s : memref<4xi8> = dense<66>
air.channel @c8s_chan [] {channel_type = "mmio"}
func.func @i8_splat() {
  %src = memref.get_global @c8s : memref<4xi8>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<4xi8> {
    air.channel.put @c8s_chan[] (%a[] [] []) : (memref<4xi8>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<4xi8, 2>
        air.channel.get @c8s_chan[] (%alloc[] [] []) : (memref<4xi8, 2>)
        memref.dealloc %alloc : memref<4xi8, 2>
      }
    }
  }
  return
}

// -----

// i16 non-splat: pure-int storage path. {0x1234, 0xABCD} LE → i32
// 0xABCD1234 = -1412623820 (signed).
//
// CHECK-I16-LABEL: aie.device(npu1)
// CHECK-I16:         memref.global @c16ns_mmio_i32 : memref<1xi32> = dense<-1412623820> {air.mmio_global}
//
// CHECK-I16-LABEL: func.func @i16_nonsplat
// CHECK-I16:         memref.get_global @c16ns_mmio_i32 : memref<1xi32>
// CHECK-I16:         aiex.npu.blockwrite(%{{.+}}) {address = 0 : ui32, buffer = @{{.+}}} : memref<1xi32>

memref.global "private" @c16ns : memref<2xi16> = dense<[4660, -21555]>
air.channel @c16ns_chan [] {channel_type = "mmio"}
func.func @i16_nonsplat() {
  %src = memref.get_global @c16ns : memref<2xi16>
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%a = %src) : memref<2xi16> {
    air.channel.put @c16ns_chan[] (%a[] [] []) : (memref<2xi16>)
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c1_0) {
        %alloc = memref.alloc() : memref<2xi16, 2>
        air.channel.get @c16ns_chan[] (%alloc[] [] []) : (memref<2xi16, 2>)
        memref.dealloc %alloc : memref<2xi16, 2>
      }
    }
  }
  return
}
