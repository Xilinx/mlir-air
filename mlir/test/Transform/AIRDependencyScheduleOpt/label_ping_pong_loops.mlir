//===- label_ping_pong_loops.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-label-scf-for-to-ping-pong | FileCheck %s --check-prefix=DEFAULT
// RUN: air-opt %s -air-label-scf-for-to-ping-pong='omit-memory-space=L1' | FileCheck %s --check-prefix=OMIT_L1
// RUN: air-opt %s -air-label-scf-for-to-ping-pong='omit-memory-space=L2' | FileCheck %s --check-prefix=OMIT_L2

// Label scf.for and memref.alloc as target for ping-pong transformation.
// DEFAULT: memref.alloc() {hoist_alloc = true}
// DEFAULT: scf.yield
// DEFAULT-NEXT: } {unroll = 2 : i32}

module {
  func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
        %c4 = arith.constant 4 : index
        
        // L2 loop (memory space 1) - should be labeled by default, skipped when omit-memory-space=L2
        %c0_seg = arith.constant 0 : index
        %c64_seg = arith.constant 64 : index
        %c256_seg = arith.constant 256 : index
        %async_token_seg = air.wait_all async
        // DEFAULT: scf.for
        // DEFAULT: memref.alloc() {hoist_alloc = true} : memref<64x64xbf16, 1>
        // DEFAULT: } {unroll = 2 : i32}
        // OMIT_L1: scf.for
        // OMIT_L1: memref.alloc() {hoist_alloc = true} : memref<64x64xbf16, 1>
        // OMIT_L1: } {unroll = 2 : i32}
        // OMIT_L2: scf.for
        // OMIT_L2-NOT: hoist_alloc
        // OMIT_L2-NOT: unroll
        %4 = scf.for %arg10 = %c0_seg to %c256_seg step %c64_seg iter_args(%arg11 = %async_token_seg) -> (!air.async.token) {
          %async_token_6, %results_7 = air.execute [%arg11] -> (memref<64x64xbf16, 1>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1>
            air.execute_terminator %alloc : memref<64x64xbf16, 1>
          }
          %async_token_8 = air.execute [%async_token_6] {
            memref.dealloc %results_7 : memref<64x64xbf16, 1>
          }
          scf.yield %async_token_8 : !air.async.token
        }
        
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          // L1 loop (memory space 2) - should be labeled by default, skipped when omit-memory-space=L1
          // DEFAULT: scf.for
          // DEFAULT: memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
          // DEFAULT: } {unroll = 2 : i32}
          // OMIT_L1: scf.for
          // OMIT_L1-NOT: hoist_alloc
          // OMIT_L1-NOT: unroll
          // OMIT_L2: scf.for
          // OMIT_L2: memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
          // OMIT_L2: } {unroll = 2 : i32}
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_3, %results_4 = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc : memref<32x32xbf16, 2>
            }
            %async_token_5 = air.execute [%async_token_3] {
              memref.dealloc %results_4 : memref<32x32xbf16, 2>
            }
            scf.yield %async_token_5 : !air.async.token
          }
        }
      }
    }
    return
  }
}
