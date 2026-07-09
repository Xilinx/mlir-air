//===- shared_l1_3way_pingpong.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// 3-WAY shared L1 access WITH ping-pong (Part 2 of the shared-L1 primitive).
// Two shared L1 buffers (ping/pong), both owned by tile_0_2,
// each accessed by the same 3-way pattern (helper write + main write +
// mem_0_2 DMA read). Mirrors an N-writer + 1-reader shared L1 buffer.
//
// On top of the Part-1 cap=N prod/cons requirement, Part 2 requires
// PER-BUFFER bracketing in the core body: each buffer's prod-acquire /
// cons-release must immediately bracket the kernel-call that writes THAT
// buffer. The pre-fix asymmetric-scope hoist batches all acquires at loop
// top / all releases at bottom, which holds both ping AND pong prod locks
// simultaneously and defeats double-buffering.
//
// These CHECK lines describe the POST-fix emission; RED until the
// per-buffer bracketing lands.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir'                    | FileCheck %s --check-prefix=CORE
// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir,after-lower-memcpy' | FileCheck %s --check-prefix=DMA

// ---- Two prod/cons pairs (ping + pong), both with prod init=2. ----
// CORE-DAG: aie.lock(%{{.*}}, {{.*}}) {init = 2 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// CORE-DAG: aie.lock(%{{.*}}, {{.*}}) {init = 2 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// CORE-NOT: _mutex_lock

// ---- Per-buffer bracketing in @main: the first buffer's prod-acquire must
// be followed by ITS cons-release BEFORE the second buffer's prod-acquire.
// The CORE-NOT between the first prod-acquire and the first cons-release
// rejects the batch-hoist (acquire ping_prod; acquire pong_prod; ...).
// CORE: aie.core
// CORE: aie.use_lock(%{{.*}}_prod_lock, AcquireGreaterEqual, 1)
// CORE-NOT: aie.use_lock(%{{.*}}_prod_lock, AcquireGreaterEqual, 1)
// CORE: aie.use_lock(%{{.*}}_cons_lock, Release, 1)
// CORE: aie.use_lock(%{{.*}}_prod_lock, AcquireGreaterEqual, 1)
// CORE: aie.use_lock(%{{.*}}_cons_lock, Release, 1)

// ---- DMA-side ping-pong BD chain: two BDs, each acquiring its own
// buffer's cons lock GE 2 / releasing its own prod lock 2. ----
// DMA: aie.mem
// DMA: aie.use_lock(%{{.*}}, AcquireGreaterEqual, 2)
// DMA: aie.dma_bd(%{{.*}}shared_l1{{.*}})
// DMA: aie.use_lock(%{{.*}}, Release, 2)
// DMA: aie.use_lock(%{{.*}}, AcquireGreaterEqual, 2)
// DMA: aie.dma_bd(%{{.*}}shared_l1{{.*}})
// DMA: aie.use_lock(%{{.*}}, Release, 2)

module {
  func.func private @zero_vectorized_bf16(memref<8xbf16, 2 : i32>) attributes {link_with = "mv_int4_q4nx_bf16_v21.o", llvm.emit_c_interface}
  air.channel @out_chan []
  func.func @shared_l1_3way_pingpong(%arg0: memref<128xbf16>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1_0) args(%arg5=%arg0) : memref<128xbf16> {
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c8_1 = arith.constant 8 : index
      %c1_2 = arith.constant 1 : index
      air.channel.get  @out_chan[] (%arg5[] [%c16, %c8] [%c8_1, %c1_2]) : (memref<128xbf16>)
      air.segment @seg  {
        %alloc = memref.alloc() : memref<8xbf16, 2 : i32>
        %alloc_3 = memref.alloc() : memref<8xbf16, 2 : i32>
        %c1_4 = arith.constant 1 : index
        %c1_5 = arith.constant 1 : index
        air.herd @helper  tile (%arg6, %arg7) in (%arg8=%c1_4, %arg9=%c1_5) args(%arg10=%alloc, %arg11=%alloc_3) : memref<8xbf16, 2 : i32>, memref<8xbf16, 2 : i32> attributes {link_with = "mv_int4_q4nx_bf16_v21.o", x_loc = 0 : i64, y_loc = 3 : i64} {
          %c0 = arith.constant 0 : index
          %c1_8 = arith.constant 1 : index
          %c8_9 = arith.constant 8 : index
          scf.for %arg12 = %c0 to %c8_9 step %c1_8 {
            func.call @zero_vectorized_bf16(%arg10) : (memref<8xbf16, 2 : i32>) -> ()
            func.call @zero_vectorized_bf16(%arg11) : (memref<8xbf16, 2 : i32>) -> ()
          }
        }
        %c1_6 = arith.constant 1 : index
        %c1_7 = arith.constant 1 : index
        air.herd @main  tile (%arg6, %arg7) in (%arg8=%c1_6, %arg9=%c1_7) args(%arg10=%alloc, %arg11=%alloc_3) : memref<8xbf16, 2 : i32>, memref<8xbf16, 2 : i32> attributes {link_with = "mv_int4_q4nx_bf16_v21.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          %c1_8 = arith.constant 1 : index
          %c8_9 = arith.constant 8 : index
          scf.for %arg12 = %c0 to %c8_9 step %c1_8 {
            func.call @zero_vectorized_bf16(%arg10) : (memref<8xbf16, 2 : i32>) -> ()
            air.channel.put  @out_chan[] (%arg10[] [] []) : (memref<8xbf16, 2 : i32>)
            func.call @zero_vectorized_bf16(%arg11) : (memref<8xbf16, 2 : i32>) -> ()
            air.channel.put  @out_chan[] (%arg11[] [] []) : (memref<8xbf16, 2 : i32>)
          }
        }
      }
    }
    return
  }
}
