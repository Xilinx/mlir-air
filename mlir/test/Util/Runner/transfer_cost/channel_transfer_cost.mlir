//===- channel_transfer_cost.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Three channels that move the same payload between the same two memory
// spaces, priced three different ways.
//
// Until now the only thing the cost model could key a transfer on was its
// (src, dst) memory-space pair, so every L2 -> L1 transfer cost the same. That
// is wrong whenever a machine prices two transfers differently for a reason
// the memory spaces do not capture. A second interconnect between the same two
// levels is one such reason; another, which needs no extra hardware at all, is
// that one of the two overlaps with compute and never reaches the critical
// path. The runner has no other way to express either.
//
// Deliberately named for the cost model rather than for a wire, because which
// of those it is describing is the arch author's business and not the
// runner's.
//
// An air.channel may now carry an `air.transfer_cost` attribute naming an
// entry in the arch's cost_model.transfer_costs object, which overrides the
// interface's bandwidth, latency, or both.
//
// The three channel names below are just names.
//
// arch.json gives the L2 outbound port 300 cycles of latency and the L1
// inbound port 200, so the default L2 -> L1 interface latency is 500. Both run
// at 1 byte/cycle. The three channels below carry memref<64xi8>:
//
//   @fabric    no attr  -> interface: 64 occupancy + 500 flight = 564
//   @fast      named    -> 4 bytes/cycle and 0 flight           =  16
//   @slowpath  named    -> latency only: interface bandwidth,
//                          900 flight                 = 64 + 900 = 964
//
// @slowpath is the case that matters for backward compatibility: it sets only
// `latency`, so its bandwidth must still come from the interface. A transfer
// cost overrides what it names and nothing else.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// The three transfers interleave in the trace, so the checks below follow
// emission order rather than pairing each begin with its end. Timestamps are
// in microseconds at 1 GHz, so one cycle is 0.001. The deltas are what the
// test is really asserting:
//
//   fabric     0.005 -> 0.569  =  564   64 occupancy + 500 interface flight
//   fast       0.069 -> 0.085  =   16   16 occupancy + 0 flight
//   slowpath   0.085 -> 1.049  =  964   64 occupancy + 900 flight

// CHECK: "name": "ChannelGetOp@fabric(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": 0.005,

// CHECK: "name": "ChannelGetOp@fast(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": 0.069,
// CHECK: "name": "ChannelGetOp@fast(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": 0.085,

// CHECK: "name": "ChannelGetOp@slowpath(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK-NEXT: "ts": 0.085,

// CHECK: "name": "ChannelGetOp@fabric(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": 0.569,

// CHECK: "name": "ChannelGetOp@slowpath(L1<--L2)",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK-NEXT: "ts": 1.049,

module {
  // No attribute: priced by the L2 -> L1 interface, exactly as before this
  // existed. link_latency/channel_latency.mlir is that case.
  air.channel @fabric [1, 1]
  // Overrides both bandwidth and flight time.
  air.channel @fast [1, 1] {air.transfer_cost = "fast"}
  // Overrides flight time only; bandwidth still comes from the interface.
  air.channel @slowpath [1, 1] {air.transfer_cost = "slowpath"}
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_0 = arith.constant 1 : index
        %tf, %bf = air.execute -> (memref<64xi8, 1>) {
          %a = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %a : memref<64xi8, 1>
        }
        %pf = air.channel.put async [%tf] @fabric[] (%bf[] [] []) {id = 3 : i32} : (memref<64xi8, 1>)
        %tr, %br = air.execute -> (memref<64xi8, 1>) {
          %a = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %a : memref<64xi8, 1>
        }
        %pr = air.channel.put async [%tr] @fast[] (%br[] [] []) {id = 4 : i32} : (memref<64xi8, 1>)
        %ts, %bs = air.execute -> (memref<64xi8, 1>) {
          %a = memref.alloc() : memref<64xi8, 1>
          air.execute_terminator %a : memref<64xi8, 1>
        }
        %ps = air.channel.put async [%ts] @slowpath[] (%bs[] [] []) {id = 5 : i32} : (memref<64xi8, 1>)
        %h = air.herd @herd_0 async tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_0) attributes {id = 6 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %gtf, %gbf = air.execute -> (memref<64xi8, 2>) {
            %a = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %a : memref<64xi8, 2>
          }
          %ggf = air.channel.get async [%gtf] @fabric[] (%gbf[] [] []) {id = 7 : i32} : (memref<64xi8, 2>)
          %gtr, %gbr = air.execute -> (memref<64xi8, 2>) {
            %a = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %a : memref<64xi8, 2>
          }
          %ggr = air.channel.get async [%gtr] @fast[] (%gbr[] [] []) {id = 8 : i32} : (memref<64xi8, 2>)
          %gts, %gbs = air.execute -> (memref<64xi8, 2>) {
            %a = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %a : memref<64xi8, 2>
          }
          %ggs = air.channel.get async [%gts] @slowpath[] (%gbs[] [] []) {id = 9 : i32} : (memref<64xi8, 2>)
        }
      }
    }
    return
  }
}
