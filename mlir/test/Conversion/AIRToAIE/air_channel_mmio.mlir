//===- air_channel_mmio.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Positive tests for channel_type="mmio" in air-to-aie. Each split has
// its own CHECK prefix so directives don't leak across boundaries.
// Negative cases live in `air_channel_mmio_invalid.mlir`.
//
// The mmio lowering stamps the source memref.global's initializer onto
// the destination L1 aie.buffer's `initial_value` attribute.
// AIERTControl::initBuffers loads this into the tile via
// XAie_DataMemBlockWrite at device-init time — before any core starts —
// which makes the data delivery race-free relative to core execution
// and natively handles any element type (no i32 repack required).

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu1" | FileCheck %s --check-prefixes=CHECK-SIMPLE,CHECK-MIXED,CHECK-BCAST,CHECK-INDEXED,CHECK-BF16,CHECK-BF16NS,CHECK-I8

// -----

// One-to-one: L3 put → destination L1 aie.buffer's initial_value is set;
// get/put erased; no shim DMA, no aie.flow.
//
// CHECK-SIMPLE-LABEL: aie.device(npu1)
// CHECK-SIMPLE:         %[[TILE:.+]] = aie.tile(0, 2)
// CHECK-SIMPLE:         %[[BUF:.+]] = aie.buffer(%[[TILE]]) {sym_name = "[[BUFNAME:.+]]"} : memref<8xi32, 2> = dense<42>
// CHECK-SIMPLE:         aie.core(%[[TILE]])
// CHECK-SIMPLE-NOT:     air.channel.put @mmio_chan
// CHECK-SIMPLE-NOT:     air.channel.get @mmio_chan
// CHECK-SIMPLE-NOT:     aie.flow
// CHECK-SIMPLE-NOT:     aie.shim_dma_allocation
// CHECK-SIMPLE-NOT:     aiex.npu.blockwrite

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
// and survives in the L3 control func; mmio sets buffer initial_value.
//
// CHECK-MIXED-LABEL: aie.device(npu1)
// CHECK-MIXED:         aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<8xi32, 2> = dense<7>
// CHECK-MIXED-NOT:     aiex.npu.blockwrite
//
// CHECK-MIXED-LABEL: func.func @mixed
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

// Broadcast mmio: one put with `broadcast_shape` stamps the same
// initial_value onto each destination L1 buffer.
//
// CHECK-BCAST-LABEL: aie.device(npu1)
// CHECK-BCAST:         aie.tile(0, 2)
// CHECK-BCAST:         aie.tile(0, 3)
// CHECK-BCAST-DAG:     aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<8xi32, 2> = dense<5>
// CHECK-BCAST-DAG:     aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<8xi32, 2> = dense<5>
// CHECK-BCAST-NOT:     air.channel.put @bcast_mmio
// CHECK-BCAST-NOT:     air.channel.get @bcast_mmio
// CHECK-BCAST-NOT:     aie.flow
// CHECK-BCAST-NOT:     aie.shim_dma_allocation
// CHECK-BCAST-NOT:     aiex.npu.blockwrite

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
// index pairs one host-side put with one per-tile get; each destination
// buffer gets its own initial_value. Regression: specializeChannelBundle
// used to split the bundle and orphan the host-side puts (now skipped
// for mmio).
//
// CHECK-INDEXED-LABEL: aie.device(npu1)
// CHECK-INDEXED-DAG:     aie.tile(0, 2)
// CHECK-INDEXED-DAG:     aie.tile(1, 2)
// CHECK-INDEXED-DAG:     aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<8xi32, 2> = dense<10>
// CHECK-INDEXED-DAG:     aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<8xi32, 2> = dense<20>
// CHECK-INDEXED-NOT:     air.channel.put @qm
// CHECK-INDEXED-NOT:     air.channel.get @qm
// CHECK-INDEXED-NOT:     aie.shim_dma_allocation
// CHECK-INDEXED-NOT:     aiex.npu.blockwrite

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

// Splat bf16: AIERTControl::initBuffers handles float types natively
// (XAie_DataMemBlockWrite copies APFloat bytes), so no i32 repack — the
// destination bf16 buffer takes the bf16 splat directly.
//
// CHECK-BF16-LABEL: aie.device(npu1)
// CHECK-BF16:         aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<2x2xbf16, 2> = dense<1.500000e+00>
// CHECK-BF16-NOT:     aiex.npu.blockwrite

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

// -----

// Non-splat bf16: same path, full element list preserved on the
// destination buffer.
//
// CHECK-BF16NS-LABEL: aie.device(npu1)
// CHECK-BF16NS:         aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<2x2xbf16, 2> = dense<{{\[}}{{\[}}1.500000e+00, 2.500000e+00{{\]}}, {{\[}}3.500000e+00, 4.500000e+00{{\]}}{{\]}}>
// CHECK-BF16NS-NOT:     aiex.npu.blockwrite

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

// i8 splat: integer types are also handled natively by initBuffers
// (XAie_DataMemBlockWrite copies APInt bytes), so the destination i8
// buffer takes the i8 splat as-is.
//
// CHECK-I8-LABEL: aie.device(npu1)
// CHECK-I8:         aie.buffer(%{{.+}}) {sym_name = "{{.+}}"} : memref<4xi8, 2> = dense<66>
// CHECK-I8-NOT:     aiex.npu.blockwrite

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
