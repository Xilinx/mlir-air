//===- insert_trace_packet_flow_tiles.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 insert-trace-packet-flow=true" | FileCheck %s --check-prefix=ALL
// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 insert-trace-packet-flow=true trace-tiles=0:2,0:3" | FileCheck %s --check-prefix=SOME
// RUN: not air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 insert-trace-packet-flow=true trace-tiles=0" 2>&1 | FileCheck %s --check-prefix=BAD

// Four cores of one column each send to a memtile. The four flows take the
// south channels below the bottom core, which is then not traced unless
// trace-tiles names it.

// ALL-NOT: aie.packet_source<%tile_0_2, Trace : 0>
// ALL-DAG: aie.packet_source<%tile_0_3, Trace : 0>
// ALL-DAG: aie.packet_source<%tile_0_4, Trace : 0>
// ALL-DAG: aie.packet_source<%tile_0_5, Trace : 0>
// ALL-NOT: aie.packet_source<%tile_0_2, Trace : 0>

// SOME-NOT: aie.packet_source<%tile_0_{{[45]}}, Trace : 0>
// SOME-DAG: aie.packet_source<%tile_0_2, Trace : 0>
// SOME-DAG: aie.packet_source<%tile_0_3, Trace : 0>
// SOME-NOT: aie.packet_source<%tile_0_{{[45]}}, Trace : 0>

// BAD: error: trace-tiles: expected col:row, got '0'

module {
  air.channel @out [1, 4]
  func.func @foo(%arg0: memref<128xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) {
      air.segment @seg {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c3_s = arith.constant 3 : index
        %c4_s = arith.constant 4 : index
        %l2 = memref.alloc() : memref<128xi32, 1>
        air.channel.get @out[%c0_s, %c0_s] (%l2[0] [32] [1]) : (memref<128xi32, 1>)
        air.channel.get @out[%c0_s, %c1_s] (%l2[32] [32] [1]) : (memref<128xi32, 1>)
        air.channel.get @out[%c0_s, %c2_s] (%l2[64] [32] [1]) : (memref<128xi32, 1>)
        air.channel.get @out[%c0_s, %c3_s] (%l2[96] [32] [1]) : (memref<128xi32, 1>)
        air.herd @herd_0 tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c4_s) {
          %l1 = memref.alloc() : memref<32xi32, 2>
          air.channel.put @out[%hx, %hy] (%l1[] [] []) : (memref<32xi32, 2>)
          memref.dealloc %l1 : memref<32xi32, 2>
        }
        memref.dealloc %l2 : memref<128xi32, 1>
      }
    }
    return
  }
}
