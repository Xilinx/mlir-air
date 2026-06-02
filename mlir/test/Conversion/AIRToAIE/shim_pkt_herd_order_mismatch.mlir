//===- shim_pkt_herd_order_mismatch.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression: when multiple dma_packet shim flows demux into one tile S2MM
// port, the three orderings — shim sender BD task queue (= packet arrival
// FIFO at the receiver), receiver mem BD chain, and core lock-acquire
// sequence — must all match. Receiver and core both follow herd-source
// order; sender pkt_id and BD task dispatch followed launch-decl IR walk
// order. When those diverged (sibling channel.get placed before the
// dominant get in the herd), packets landed in the wrong BD chain →
// wrong-buffer data or core deadlock.
//
// air-to-aie must reorder packet shim flows by their receiver-side
// first-use position so that pkt_id assignment, shim alloc list order,
// the L3 launch puts' IR positions, and the receiver mem BD chain all
// agree with the herd's natural consumption order.
//
// Each module below is compiled independently via --split-input-file so
// pkt_id counters reset between cases.

// RUN: air-opt %s --split-input-file -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// =============================================================================
// Case 1: two packet flows, herd source order is the REVERSE of launch-decl
// order. chan_sib (declared second in launch) is consumed FIRST in herd.
// =============================================================================

// CHECK-LABEL: aie.device(npu2) @basic_seg

// Two packet flows must be emitted, with distinct IDs.
// CHECK-DAG: aie.packet_flow(0)
// CHECK-DAG: aie.packet_flow(1)

// Shim alloc list order reflects the sort: chan_sib must appear BEFORE
// chan_loop. This drives downstream BD ordering on the same physical
// MM2S channel.
// CHECK:     aie.shim_dma_allocation @air_chan_sib(%{{.*}}, MM2S, 0)
// CHECK:     aie.shim_dma_allocation @air_chan_loop(%{{.*}}, MM2S, 0)

// The launch puts must be physically reordered so chan_sib's put appears
// FIRST in the IR (this drives AIRRtToNpuPass's dma_start_task dispatch
// order). Each put must carry the matching pkt_id.
// CHECK-LABEL: func.func @case1_reverse_order
// CHECK:       air.channel.put{{.*}}@chan_sib{{.*}}pkt_id = 0
// CHECK-NEXT:  air.channel.put{{.*}}@chan_loop{{.*}}pkt_id = 1

// Negative checks: the pre-fix pkt_id assignment (driven by launch-decl
// order) must NOT be emitted in this case, and chan_loop's put must not
// precede chan_sib's put in the IR.
// CHECK-NOT:   air.channel.put{{.*}}@chan_loop{{.*}}pkt_id = 0
// CHECK-NOT:   air.channel.put{{.*}}@chan_sib{{.*}}pkt_id = 1

module {
  air.channel @chan_loop [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @chan_sib [1, 1] {channel_type = "npu_dma_packet"}

  func.func @case1_reverse_order(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
    %0 = air.launch async () in () args(%in_loop=%arg0, %in_sib=%arg1) : memref<64xbf16>, memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      // Launch-decl order: chan_loop first, chan_sib second.
      %put_loop = air.channel.put async @chan_loop[%c0, %c0] (%in_loop[] [] []) {id = 1 : i32} : (memref<64xbf16>)
      %put_sib = air.channel.put async @chan_sib[%c0, %c0] (%in_sib[] [] []) {id = 2 : i32} : (memref<64xbf16>)

      %seg = air.segment @basic_seg async [%put_loop, %put_sib] attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_seg = arith.constant 1 : index
        %herd = air.herd @basic_herd async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) attributes {id = 3 : i32} {
          %hc0 = arith.constant 0 : index
          // Herd-source order: chan_sib first (the "sibling" get), then
          // chan_loop. Reverse of launch-decl order.
          %async_sib, %buf_sib = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_sib = air.channel.get async [%async_sib] @chan_sib[%hc0, %hc0] (%buf_sib[] [] []) {id = 3 : i32} : (memref<64xbf16, 2>)

          %async_loop, %buf_loop = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_loop = air.channel.get async [%async_loop] @chan_loop[%hc0, %hc0] (%buf_loop[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)

          %dealloc_sib = air.execute [%get_sib] {
            memref.dealloc %buf_sib : memref<64xbf16, 2>
          }
          %dealloc_loop = air.execute [%get_loop] {
            memref.dealloc %buf_loop : memref<64xbf16, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// =============================================================================
// Case 2: three packet flows; herd source order differs from launch-decl
// order. This exercises the strict-weak-ordering requirement of the
// comparator — a partial comparator that mixes comparable + incomparable
// pairs broke transitivity for n>=3 in an earlier attempt at the fix.
//
// Launch-decl: [chan_a, chan_b, chan_c]
// Herd-source: [chan_b, chan_c, chan_a]
// Expected pkt_id assignment after sort: chan_b=0, chan_c=1, chan_a=2.
// =============================================================================

// CHECK-LABEL: aie.device(npu2) @three_seg
// CHECK-DAG: aie.packet_flow(0)
// CHECK-DAG: aie.packet_flow(1)
// CHECK-DAG: aie.packet_flow(2)

// CHECK:     aie.shim_dma_allocation @air_chan_b(%{{.*}}, MM2S, 0)
// CHECK:     aie.shim_dma_allocation @air_chan_c(%{{.*}}, MM2S, 0)
// CHECK:     aie.shim_dma_allocation @air_chan_a(%{{.*}}, MM2S, 0)

// CHECK-LABEL: func.func @case2_three_flows
// CHECK:       air.channel.put{{.*}}@chan_b{{.*}}pkt_id = 0
// CHECK-NEXT:  air.channel.put{{.*}}@chan_c{{.*}}pkt_id = 1
// CHECK-NEXT:  air.channel.put{{.*}}@chan_a{{.*}}pkt_id = 2

module {
  air.channel @chan_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @chan_b [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @chan_c [1, 1] {channel_type = "npu_dma_packet"}

  func.func @case2_three_flows(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>, %arg2: memref<64xbf16>) {
    %0 = air.launch async () in () args(%in_a=%arg0, %in_b=%arg1, %in_c=%arg2) : memref<64xbf16>, memref<64xbf16>, memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      // Launch-decl order: a, b, c.
      %put_a = air.channel.put async @chan_a[%c0, %c0] (%in_a[] [] []) {id = 1 : i32} : (memref<64xbf16>)
      %put_b = air.channel.put async @chan_b[%c0, %c0] (%in_b[] [] []) {id = 2 : i32} : (memref<64xbf16>)
      %put_c = air.channel.put async @chan_c[%c0, %c0] (%in_c[] [] []) {id = 3 : i32} : (memref<64xbf16>)

      %seg = air.segment @three_seg async [%put_a, %put_b, %put_c] attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_seg = arith.constant 1 : index
        %herd = air.herd @three_herd async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) attributes {id = 3 : i32} {
          %hc0 = arith.constant 0 : index
          // Herd-source order: b, c, a (intentionally reordered).
          %async_b, %buf_b = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_b = air.channel.get async [%async_b] @chan_b[%hc0, %hc0] (%buf_b[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)

          %async_c, %buf_c = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_c = air.channel.get async [%async_c] @chan_c[%hc0, %hc0] (%buf_c[] [] []) {id = 5 : i32} : (memref<64xbf16, 2>)

          %async_a, %buf_a = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_a = air.channel.get async [%async_a] @chan_a[%hc0, %hc0] (%buf_a[] [] []) {id = 6 : i32} : (memref<64xbf16, 2>)

          %dealloc_b = air.execute [%get_b] { memref.dealloc %buf_b : memref<64xbf16, 2> }
          %dealloc_c = air.execute [%get_c] { memref.dealloc %buf_c : memref<64xbf16, 2> }
          %dealloc_a = air.execute [%get_a] { memref.dealloc %buf_a : memref<64xbf16, 2> }
        }
      }
    }
    return
  }
}

// -----

// =============================================================================
// Case 3: the actual HW failure shape from repro_packet_demux. The dominant
// channel chan_loop2 has a multi-iter scf.for around its get inside the
// herd, chan_sib2 has a single get BEFORE the loop. Receiver mem chain
// ends up with two distinct dma_start tasks (single + repeat) on the same
// S2MM port; the FIRST task must match the FIRST-arriving packets.
//
// With the fix: chan_sib2 (early in herd) gets pkt_id=0, chan_loop2 gets
// pkt_id=1; sender dispatches chan_sib2's 1 packet first, then chan_loop2's
// N packets — matches receiver mem chain task order.
// =============================================================================

// CHECK-LABEL: aie.device(npu2) @forloop_seg
// CHECK-DAG: aie.packet_flow(0)
// CHECK-DAG: aie.packet_flow(1)

// CHECK:     aie.shim_dma_allocation @air_chan_sib2(%{{.*}}, MM2S, 0)
// CHECK:     aie.shim_dma_allocation @air_chan_loop2(%{{.*}}, MM2S, 0)

// CHECK-LABEL: func.func @case3_scf_for_pattern
// CHECK:       air.channel.put{{.*}}@chan_sib2{{.*}}pkt_id = 0
// CHECK-NEXT:  air.channel.put{{.*}}@chan_loop2{{.*}}pkt_id = 1

module {
  air.channel @chan_loop2 [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @chan_sib2 [1, 1] {channel_type = "npu_dma_packet"}

  func.func @case3_scf_for_pattern(%arg0: memref<256xbf16>, %arg1: memref<64xbf16>) {
    %0 = air.launch async () in () args(%in_loop=%arg0, %in_sib=%arg1) : memref<256xbf16>, memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      // Launch-decl order: chan_loop2 first (driven by the dominant input
      // being declared first as is natural), chan_sib2 second.
      %put_loop = air.channel.put async @chan_loop2[%c0, %c0] (%in_loop[] [] []) {id = 1 : i32} : (memref<256xbf16>)
      %put_sib = air.channel.put async @chan_sib2[%c0, %c0] (%in_sib[] [] []) {id = 2 : i32} : (memref<64xbf16>)

      %seg = air.segment @forloop_seg async [%put_loop, %put_sib] attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1_seg = arith.constant 1 : index
        %herd = air.herd @forloop_herd async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) attributes {id = 3 : i32} {
          %hc0 = arith.constant 0 : index
          %hc1 = arith.constant 1 : index
          %hc4 = arith.constant 4 : index
          // chan_sib2's get BEFORE the scf.for — the trigger for the bug.
          %async_sib, %buf_sib = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_sib = air.channel.get async [%async_sib] @chan_sib2[%hc0, %hc0] (%buf_sib[] [] []) {id = 3 : i32} : (memref<64xbf16, 2>)

          // chan_loop2's get INSIDE an scf.for — the dominant per-tile get.
          %loop_token = scf.for %it = %hc0 to %hc4 step %hc1 iter_args(%tok = %get_sib) -> (!air.async.token) {
            %async_loop, %buf_loop = air.execute -> (memref<64xbf16, 2>) {
              %alloc = memref.alloc() : memref<64xbf16, 2>
              air.execute_terminator %alloc : memref<64xbf16, 2>
            }
            %get_loop = air.channel.get async [%async_loop, %tok] @chan_loop2[%hc0, %hc0] (%buf_loop[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)
            %dealloc_loop = air.execute [%get_loop] {
              memref.dealloc %buf_loop : memref<64xbf16, 2>
            }
            scf.yield %dealloc_loop : !air.async.token
          }

          %dealloc_sib = air.execute [%loop_token] {
            memref.dealloc %buf_sib : memref<64xbf16, 2>
          }
        }
      }
    }
    return
  }
}
