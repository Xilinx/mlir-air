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

// -----

// =============================================================================
// Case 4: launch interleaves packet PUTs (in0, in1) with unrelated
// non-packet GETs (out0, out1). Herd consumes in1 BEFORE in0 (which is in
// an scf.for). This is the HW-failure shape from repro_packet_demux: an
// unrelated dma sits between the two same-metadata packet puts. The
// reorder must skip past the unrelated gets and still place the packet
// puts in pkt_id order.
// =============================================================================

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK-DAG: aie.packet_flow(0)
// CHECK-DAG: aie.packet_flow(1)
// CHECK:     aie.shim_dma_allocation @air_in1(%{{.*}}, MM2S, 0)
// CHECK:     aie.shim_dma_allocation @air_in0(%{{.*}}, MM2S, 0)

// in1 must precede in0 in the launch; the unrelated out0 / out1 gets are
// not constrained but must continue to coexist with the puts.
// CHECK-LABEL: func.func @case4_interleaved_non_packet
// CHECK:       air.channel.put{{.*}}@in1{{.*}}pkt_id = 0
// CHECK:       air.channel.put{{.*}}@in0{{.*}}pkt_id = 1
// CHECK-NOT:   air.channel.put{{.*}}@in0{{.*}}pkt_id = 0
// CHECK-NOT:   air.channel.put{{.*}}@in1{{.*}}pkt_id = 1

module {
  air.channel @in0 [1] {channel_type = "npu_dma_packet"}
  air.channel @out0 [1]
  air.channel @in1 [1] {channel_type = "npu_dma_packet"}
  air.channel @out1 [1]
  func.func @case4_interleaved_non_packet(%arg0: memref<8x16xi32>, %arg1: memref<16xi32>, %arg2: memref<128xi32>, %arg3: memref<16xi32>) {
    air.launch () in () args(%a0=%arg0, %a1=%arg1, %a2=%arg2, %a3=%arg3) : memref<8x16xi32>, memref<16xi32>, memref<128xi32>, memref<16xi32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c128 = arith.constant 128 : index
      air.channel.put @in0[] (%a0[%c0, %c0] [%c8, %c16] [%c16, %c1]) : (memref<8x16xi32>)
      air.channel.get @out0[] (%a2[%c0] [%c128] [%c1]) : (memref<128xi32>)
      air.channel.put @in1[] (%a1[] [] []) : (memref<16xi32>)
      air.channel.get @out1[] (%a3[] [] []) : (memref<16xi32>)
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_h tile (%tx, %ty) in (%htx=%c1_0, %hty=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %bi1 = memref.alloc() : memref<16xi32, 2>
          %bo1 = memref.alloc() : memref<16xi32, 2>
          air.channel.get @in1[] (%bi1[] [] []) : (memref<16xi32, 2>)
          %c0_h = arith.constant 0 : index
          %c8_h = arith.constant 8 : index
          %c1_h = arith.constant 1 : index
          scf.for %tile = %c0_h to %c8_h step %c1_h {
            %li = memref.alloc() : memref<16xi32, 2>
            %lo = memref.alloc() : memref<16xi32, 2>
            air.channel.get @in0[] (%li[] [] []) : (memref<16xi32, 2>)
            %c0_t = arith.constant 0 : index
            %c1_t = arith.constant 1 : index
            %c16_t = arith.constant 16 : index
            scf.for %i = %c0_t to %c16_t step %c1_t {
              %v = memref.load %li[%i] : memref<16xi32, 2>
              memref.store %v, %lo[%i] : memref<16xi32, 2>
            }
            air.channel.put @out0[] (%lo[] [] []) : (memref<16xi32, 2>)
            memref.dealloc %li : memref<16xi32, 2>
            memref.dealloc %lo : memref<16xi32, 2>
          }
          scf.for %i = %c0_h to %c8_h step %c1_h {
            %v = memref.load %bi1[%c0_h] : memref<16xi32, 2>
            memref.store %v, %bo1[%c0_h] : memref<16xi32, 2>
          }
          air.channel.put @out1[] (%bo1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %bi1 : memref<16xi32, 2>
          memref.dealloc %bo1 : memref<16xi32, 2>
        }
      }
    }
    return
  }
}

// -----

// =============================================================================
// Case 5: two herds, each with its own packet input. Each flow has a unique
// receiver core, so they are NOT a shared-port demux group -- the reorder
// must be a no-op. Regression for dual_herd_packet_switch where over-eager
// reordering corrupted pkt_id assignment across independent herds.
// =============================================================================

// CHECK-LABEL: aie.device(npu2) @dual_seg
// CHECK-DAG: aie.packet_flow(0)
// CHECK-DAG: aie.packet_flow(1)
// CHECK:     aie.shim_dma_allocation @air_in_a(%{{.*}}, MM2S, 0)
// CHECK:     aie.shim_dma_allocation @air_in_b(%{{.*}}, MM2S, 0)

// Launch puts must retain their original declaration order -- no reorder.
// CHECK-LABEL: func.func @case5_dual_herd_independent
// CHECK:       air.channel.put{{.*}}@in_a{{.*}}pkt_id = 0
// CHECK-NEXT:  air.channel.put{{.*}}@in_b{{.*}}pkt_id = 1

module {
  air.channel @in_a [1, 1] {channel_type = "npu_dma_packet"}
  air.channel @in_b [1, 1] {channel_type = "npu_dma_packet"}

  func.func @case5_dual_herd_independent(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
    %0 = air.launch async () in () args(%a=%arg0, %b=%arg1) : memref<64xbf16>, memref<64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %put_a = air.channel.put async @in_a[%c0, %c0] (%a[] [] []) {id = 1 : i32} : (memref<64xbf16>)
      %put_b = air.channel.put async @in_b[%c0, %c0] (%b[] [] []) {id = 2 : i32} : (memref<64xbf16>)
      %seg = air.segment @dual_seg async [%put_a, %put_b] attributes {id = 2 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c1 = arith.constant 1 : index
        %herd_a = air.herd @herd_a async tile (%tx, %ty) in (%htx=%c1, %hty=%c1) attributes {id = 3 : i32} {
          %hc0 = arith.constant 0 : index
          %async_a, %buf_a = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_a = air.channel.get async [%async_a] @in_a[%hc0, %hc0] (%buf_a[] [] []) {id = 3 : i32} : (memref<64xbf16, 2>)
          %dealloc_a = air.execute [%get_a] {
            memref.dealloc %buf_a : memref<64xbf16, 2>
          }
        }
        %herd_b = air.herd @herd_b async tile (%tx, %ty) in (%htx=%c1, %hty=%c1) attributes {id = 4 : i32} {
          %hc0 = arith.constant 0 : index
          %async_b, %buf_b = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %get_b = air.channel.get async [%async_b] @in_b[%hc0, %hc0] (%buf_b[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)
          %dealloc_b = air.execute [%get_b] {
            memref.dealloc %buf_b : memref<64xbf16, 2>
          }
        }
      }
    }
    return
  }
}
