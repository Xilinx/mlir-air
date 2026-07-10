//===- packet_keep_pkt_header.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// A packet channel carrying `keep_pkt_header` means the kernel writes the
// routing header word into the payload itself. air-to-aie must:
//   - emit the aie.packet_flow with {keep_pkt_header = true} so the switchbox
//     keeps (does not strip) the header at the destination, and
//   - NOT statically stamp a pkt_id on the producer BD (a static stamp would
//     prepend a second header word and shift the payload).

// CHECK: aie.dma_bd({{.*}}memref<8xbf16, 2>, 0, 8) {task_id = 0 : i32}
// CHECK-NOT: packet = #aie.packet_info
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT: aie.packet_source<%{{.*}}, DMA : 0>
// CHECK-NEXT: aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK-NEXT: } {keep_pkt_header = true}

air.channel @k [1, 1] {channel_type = "npu_dma_packet", keep_pkt_header}
func.func @f() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      %t, %l2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      air.channel.get @k[] (%l2[] [] []) : (memref<8xbf16, 1>)
      %d_ = air.execute { memref.dealloc %l2 : memref<8xbf16, 1> }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %tok, %l1 = air.execute -> (memref<8xbf16, 2>) {
          %aa = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %aa : memref<8xbf16, 2>
        }
        air.channel.put @k[] (%l1[] [] []) : (memref<8xbf16, 2>)
        %d0 = air.execute {memref.dealloc %l1 : memref<8xbf16, 2>}
      }
    }
  }
  return
}
