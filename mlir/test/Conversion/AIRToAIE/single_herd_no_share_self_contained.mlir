//===- single_herd_no_share_self_contained.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A herd-operand L1 buffer where EVERY core both writes and reads its own copy
// must NOT be inferred as a cross-core shared buffer. Here each core of a [1,2]
// herd writes the buffer (vector.transfer_write) and then streams its own slot
// out via air.channel.put -- there is no cross-core hand-off (no pure producer
// feeding a consumer on a different tile), so the buffer is per-core private and
// must be cloned into each core, NOT hoisted to a single shared aie.buffer.
//
// This guards the cross-core dependence heuristic against over-sharing: keying
// on set difference (some core reads-but-does-not-write, or writes-but-does-not-
// read) rather than "any reader tile != any writer tile" -- the latter would
// falsely share this buffer (core0 reads / core1 writes are different tiles even
// though each core is self-contained), which is the cascade over-share bug.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir' | FileCheck %s

// No shared buffer / shared locks: the operand is cloned per core instead.
// CHECK-NOT: sym_name = "shared_l1
// CHECK-NOT: _prod_lock
// CHECK-NOT: _cons_lock

module {
  air.channel @out_chan [2]
  func.func @self_contained(%arg0: memref<128xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) args(%host=%arg0) : memref<128xbf16> {
      %c0 = arith.constant 0 : index
      %c1_g = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      air.channel.get @out_chan[%c0] (%host[] [%c8] [%c1_g]) : (memref<128xbf16>)
      air.channel.get @out_chan[%c1_g] (%host[] [%c8] [%c1_g]) : (memref<128xbf16>)
      air.segment @seg {
        %shared = memref.alloc() : memref<8xbf16, 2 : i32>
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @col tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2_0) args(%buf=%shared) : memref<8xbf16, 2 : i32> attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0_b = arith.constant 0 : index
          %cst = arith.constant 1.0 : bf16
          // Every core writes its own copy, then streams that copy out. Each
          // core is self-contained: writer set == reader set -> not shared.
          %v = vector.broadcast %cst : bf16 to vector<8xbf16>
          vector.transfer_write %v, %buf[%c0_b] {in_bounds = [true]} : vector<8xbf16>, memref<8xbf16, 2 : i32>
          air.channel.put @out_chan[%ty] (%buf[] [] []) : (memref<8xbf16, 2 : i32>)
        }
      }
    }
    return
  }
}
