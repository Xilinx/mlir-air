//===- hop_aliases.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-annotate-packet-ids="assign=true emit-headers=false" -split-input-file | FileCheck %s

// A forwarding hop is recovered by matching the buffer a get lands in against
// the buffer a later put sends. Matching raw SSA identity is not enough: the
// same bytes reach the put through a subview, an air.execute result or a herd
// block argument, and every one of those breaks the match.
//
// Breaking it is SILENT and fatal. The hop keeps its own single id, its
// switchbox filters out the rest, the demux never sees the ids it routes on,
// and the device times out. Nothing looks wrong in the IR.

// A subview between the get and the put.
// CHECK-LABEL: @via_subview
// CHECK: air.channel @hop {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK: air.channel @fanout {{.*}}packet_ids = [31 : i32, 30 : i32]
module @via_subview {
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<2048xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hop[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[] [] []) : (memref<2048xbf16, 1 : i32>)
    %v = memref.subview %hub[0] [1024] [1] : memref<2048xbf16, 1 : i32> to memref<1024xbf16, strided<[1]>, 1 : i32>
    air.channel.put @fanout[%c0, %c0] (%v[] [] []) : (memref<1024xbf16, strided<[1]>, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A THREE-hop chain with the MIDDLE link aliased. This is the worse shape: the
// originating hop's own link is intact, so even a diagnostic that keyed on the
// originator would stay quiet.
// CHECK-LABEL: @middle_link
// CHECK: air.channel @hopA {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK: air.channel @hopB {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK: air.channel @fanout {{.*}}packet_ids = [31 : i32, 30 : i32]
module @middle_link {
  air.channel @hopA [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @hopB [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %mid = memref.alloc() : memref<2048xbf16, 1 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hopA[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hopA[%c0] (%mid[] [] []) : (memref<2048xbf16, 1 : i32>)
    %v = memref.subview %mid[0] [1024] [1] : memref<2048xbf16, 1 : i32> to memref<1024xbf16, strided<[1]>, 1 : i32>
    air.channel.put @hopB[%c0] (%v[] [] []) : (memref<1024xbf16, strided<[1]>, 1 : i32>)
    air.channel.get @hopB[%c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// SEVERAL gets gathering disjoint slices into one buffer, which a single put
// then sends onward. This is the ordinary shape of a memtile hop -- the decode
// designs land four column buffers into one hub buffer before the demux -- and
// the second, third and fourth gets must NOT read as writes that break the
// link between the first get and the put.
// CHECK-LABEL: @gather_then_forward
// CHECK: air.channel @hop {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK: air.channel @fanout {{.*}}packet_ids = [31 : i32, 30 : i32]
module @gather_then_forward {
  air.channel @hop [4] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c768 = arith.constant 768 : index
    %src = memref.alloc() : memref<256xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 1 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hop[%c0] (%src[] [] []) : (memref<256xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[%c0] [%c256] [%c1]) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @hop[%c1] (%hub[%c256] [%c256] [%c1]) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @hop[%c2] (%hub[%c512] [%c256] [%c1]) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @hop[%c3] (%hub[%c768] [%c256] [%c1]) : (memref<1024xbf16, 1 : i32>)
    air.channel.put @fanout[%c0, %c0] (%hub[] [] []) : (memref<1024xbf16, 1 : i32>)
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}

// -----

// A hop crossing a herd boundary: the put inside the herd sees the buffer as a
// BLOCK ARGUMENT, not as the alloc the get named. Resolving it means mapping
// the argument back through the herd's `args(...)` list.
// CHECK-LABEL: @via_herd_arg
// CHECK: air.channel @hop {{.*}}packet_ids = [31 : i32, 30 : i32]
// CHECK: air.channel @fanout {{.*}}packet_ids = [31 : i32, 30 : i32]
module @via_herd_arg {
  air.channel @hop [1] {channel_type = "npu_dma_packet", keep_pkt_header}
  air.channel @fanout [1, 1] {broadcast_shape = [1 : index, 2 : index], channel_type = "npu_dma_packet", keep_pkt_header}
  func.func @f() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %src = memref.alloc() : memref<1024xbf16, 2 : i32>
    %hub = memref.alloc() : memref<1024xbf16, 2 : i32>
    %d0 = memref.alloc() : memref<512xbf16, 2 : i32>
    %d1 = memref.alloc() : memref<512xbf16, 2 : i32>
    air.channel.put @hop[%c0] (%src[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.channel.get @hop[%c0] (%hub[] [] []) : (memref<1024xbf16, 2 : i32>)
    air.herd @h tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%hub) : memref<1024xbf16, 2 : i32> {
      %z = arith.constant 0 : index
      air.channel.put @fanout[%z, %z] (%a[] [] []) : (memref<1024xbf16, 2 : i32>)
    }
    air.channel.get @fanout[%c0, %c0] (%d0[] [] []) : (memref<512xbf16, 2 : i32>)
    air.channel.get @fanout[%c0, %c1] (%d1[] [] []) : (memref<512xbf16, 2 : i32>)
    return
  }
}
