//===- single_herd_shared_l1_dma_owner.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Owner-tile selection for an intra-herd shared L1 buffer that is read out by
// a DMA. A single air.herd has two cores that both WRITE a segment-scope L1
// buffer via opaque external kernel calls; the lead core (ty==0) additionally
// reads the assembled buffer out through an air.channel.put, which lowers to an
// MM2S DMA BD in that tile's aie.mem. A UseLockOp inside an aie.mem/BD block
// may only access a lock on its OWN tile (AccessesLocalLocks), whereas a
// UseLockOp inside a core may access a neighbor's lock. Therefore the shared
// buffer's owner tile MUST be the tile that hosts the buffer's DMA accessor
// (the channel.put), otherwise the DMA BD references a cross-tile lock and
// verification fails with "can only access a lock in the same tile".
//
// The owner picker must prefer a DMA-accessor tile independently of the order
// the cores are visited. Here the lead (ty==0, tile_0_2) carries the DMA, so
// the shared buffer + its prod/cons locks land on tile_0_2 and the aie.mem BD
// acquires/releases them locally.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir,after-lower-memcpy' | FileCheck %s

// Two writer cores -> prod lock init=2; the DMA readout is the consumer.
// CHECK-DAG: %[[LEAD:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[CONS_LOCK:.*]] = aie.lock(%[[LEAD]], {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CHECK-DAG: %[[PROD_LOCK:.*]] = aie.lock(%[[LEAD]], {{.*}}) {init = 2 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// CHECK-DAG: %[[SHARED_BUF:.*]] = aie.buffer(%[[LEAD]]) {{.*}}sym_name = "shared_l1{{.*}}"{{.*}} : memref<8xbf16, 2 : i32>

// The DMA BD that reads the shared buffer out lives in the OWNER tile's aie.mem
// and acquires/releases the SAME shared locks (all same-tile -> legal).
// CHECK: aie.mem(%[[LEAD]])
// CHECK: aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual
// CHECK: aie.dma_bd(%[[SHARED_BUF]]
// CHECK: aie.use_lock(%[[PROD_LOCK]], Release

module {
  air.channel @out_chan [1]
  func.func private @ext_write(memref<8xbf16, 2 : i32>) attributes {llvm.emit_c_interface}
  func.func @single_herd_shared_l1_dma_owner(%arg0: memref<128xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) args(%host=%arg0) : memref<128xbf16> {
      %c8 = arith.constant 8 : index
      %c1_g = arith.constant 1 : index
      air.channel.get @out_chan[] (%host[] [%c8] [%c1_g]) : (memref<128xbf16>)
      air.segment @seg {
        %shared = memref.alloc() : memref<8xbf16, 2 : i32>
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @col tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2_0) args(%sbuf=%shared) : memref<8xbf16, 2 : i32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          // Both cores write the shared buffer via opaque external kernels, but
          // each write is under its own tile-id guard (reachable on a strict
          // subset of the cores -- the genuine cascade role split). The lead
          // (ty==0) additionally reads the assembled buffer out via a
          // channel.put (-> MM2S DMA in tile_0_2's aie.mem).
          scf.index_switch %ty
          case 0 {
            func.call @ext_write(%sbuf) : (memref<8xbf16, 2 : i32>) -> ()
            air.channel.put @out_chan[] (%sbuf[] [] []) : (memref<8xbf16, 2 : i32>)
            scf.yield
          }
          default {
            func.call @ext_write(%sbuf) : (memref<8xbf16, 2 : i32>) -> ()
            scf.yield
          }
        }
      }
    }
    return
  }
}
