//===- shared_l1_3way.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// 3-WAY shared L1 access (Part 1 of the shared-L1 primitive), no
// ping-pong. A single L1 buffer is accessed by THREE participants:
//   - @helper core (tile_0_3): kernel-call writes the shared buffer
//     (cross-tile L1 write; tile_0_3 is NOT the owner)
//   - @main core (tile_0_2, owner): kernel-call writes the shared buffer
//   - mem_0_2 DMA reads the shared buffer out to the shim (MM2S)
// (@main declared LAST -> AIR owner-picker places the buffer on tile_0_2.)
//
// All three must synchronize on ONE producer/consumer lock pair:
//   prod_lock init=N (N=2 writer cores), cons_lock init=0.
//   each writer core: acquire(prod, 1) / release(cons, 1)
//   DMA reader:       acquire(cons, GE, 2) / release(prod, 2)
//
// Pre-fix, AIR emits a single mutex (init=1) and leaves the DMA BD on
// independent channel-put locks (data race). These CHECK lines describe
// the post-fix shared-L1 emission; they are RED until the shared-L1
// 3-way lock registry lands.

// CORE prefix gates the core-side prod/cons emission + attribute stash
// (stop after createAIEModulesAndOutlineCores).
// DMA prefix gates the shared-lock reuse on the DMA BD + N counts
// (stop after lowerAIRMemcpyOp). NOTE: the DMA stage requires BOTH the
// `to-aie-mlir` and `after-lower-memcpy` keywords in test-patterns.

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir'                    | FileCheck %s --check-prefix=CORE
// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2 test-patterns=to-aie-mlir,after-lower-memcpy' | FileCheck %s --check-prefix=DMA

// ---- Core-side: prod/cons pair (NOT a mutex), N=2 on the prod lock. ----
// CORE-DAG: %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// CORE-DAG: %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 2 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// The channel put that becomes the MM2S DMA BD carries the shared prod/cons
// lock symbol-refs so getLockForDMA reuses the pair later (the op is the
// carrier, not the buffer). N is the prod lock init (=2), so no separate count
// attribute. Attrs print alphabetically.
// CORE-DAG: air.channel.put{{.*}}air.shared_cons_lock = @shared_l1{{.*}}_cons_lock{{.*}}air.shared_prod_lock = @shared_l1{{.*}}_prod_lock
// No mutex anywhere.
// CORE-NOT: _mutex_lock

// Each writer core brackets its kernel-call write with acquire(prod,1) /
// release(cons,1). (main = tile_0_2 prints first, helper = tile_0_3 next.)
// CORE: aie.core
// CORE: aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, %{{.*}})
// CORE: func.call @zero_vectorized_bf16
// CORE: aie.use_lock(%[[CONS_LOCK]], Release, %{{.*}})
// CORE: aie.core
// CORE: aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, %{{.*}})
// CORE: func.call @zero_vectorized_bf16
// CORE: aie.use_lock(%[[CONS_LOCK]], Release, %{{.*}})

// ---- DMA-side: the MM2S BD on the shared buffer must reuse the SAME ----
// prod/cons locks (matched by sym_name) and use count N=2.
// DMA-DAG: %[[DCONS:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 0 : i32, sym_name = "shared_l1{{.*}}_cons_lock"}
// DMA-DAG: %[[DPROD:.*]] = aie.lock(%{{.*}}, {{.*}}) {init = 2 : i32, sym_name = "shared_l1{{.*}}_prod_lock"}
// DMA: aie.mem
// DMA: aie.use_lock(%[[DCONS]], AcquireGreaterEqual, %{{.*}})
// DMA: aie.dma_bd(%{{.*}}shared_l1{{.*}})
// DMA: aie.use_lock(%[[DPROD]], Release, %{{.*}})

module {
  func.func private @zero_vectorized_bf16(memref<8xbf16, 2 : i32>) attributes {link_with = "mv_int4_q4nx_bf16_v21.o", llvm.emit_c_interface}
  air.channel @out_chan []
  func.func @shared_l1_3way(%arg0: memref<128xbf16>) {
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
        %c1_3 = arith.constant 1 : index
        %c1_4 = arith.constant 1 : index
        air.herd @helper  tile (%arg6, %arg7) in (%arg8=%c1_3, %arg9=%c1_4) args(%arg10=%alloc) : memref<8xbf16, 2 : i32> attributes {link_with = "mv_int4_q4nx_bf16_v21.o", x_loc = 0 : i64, y_loc = 3 : i64} {
          %c0 = arith.constant 0 : index
          %c1_7 = arith.constant 1 : index
          %c16_8 = arith.constant 16 : index
          scf.for %arg11 = %c0 to %c16_8 step %c1_7 {
            func.call @zero_vectorized_bf16(%arg10) : (memref<8xbf16, 2 : i32>) -> ()
          }
        }
        %c1_5 = arith.constant 1 : index
        %c1_6 = arith.constant 1 : index
        air.herd @main  tile (%arg6, %arg7) in (%arg8=%c1_5, %arg9=%c1_6) args(%arg10=%alloc) : memref<8xbf16, 2 : i32> attributes {link_with = "mv_int4_q4nx_bf16_v21.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c0 = arith.constant 0 : index
          %c1_7 = arith.constant 1 : index
          %c16_8 = arith.constant 16 : index
          scf.for %arg11 = %c0 to %c16_8 step %c1_7 {
            func.call @zero_vectorized_bf16(%arg10) : (memref<8xbf16, 2 : i32>) -> ()
            air.channel.put  @out_chan[] (%arg10[] [] []) : (memref<8xbf16, 2 : i32>)
          }
        }
      }
    }
    return
  }
}
