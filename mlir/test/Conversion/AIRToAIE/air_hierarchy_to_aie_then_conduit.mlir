//===- air_hierarchy_to_aie_then_conduit.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Verify two-step pipeline:
//   Step 1: --air-hierarchy-to-aie  (creates aie.device + preserves air.channel)
//   Step 2: --air-split-devices     (extracts just the aie.device module)
//   Step 3: --air-channel-to-conduit (converts air.channel → conduit ops)
// The output should contain conduit.create / conduit.put_memref_async /
// conduit.get_memref_async and no residual air.channel ops.

// RUN: air-opt %s \
// RUN:   -air-hierarchy-to-aie="row-offset=3 col-offset=2 device=xcve2802" \
// RUN:   --air-split-devices="output-prefix=%t" \
// RUN: && air-opt %taie.segment_0.mlir \
// RUN:   --air-channel-to-conduit \
// RUN: | FileCheck %s

// After hierarchy + split + Pass B, we expect conduit ops in the device:
// CHECK: aie.device
// Core bodies must contain conduit.put_memref_async or conduit.get_memref_async.
// CHECK-DAG: conduit.put_memref_async
// CHECK-DAG: conduit.get_memref_async
// The conduit.create op appears after the core bodies.
// CHECK: conduit.create
// No residual air.channel ops.
// CHECK-NOT: air.channel.put
// CHECK-NOT: air.channel.get

#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
air.channel @channel_0 [1, 1]
func.func @hierarchy_then_conduit() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) {
        %c0 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %3 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
          %4 = air.channel.put async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        } else {
          %4 = air.channel.get async [%async_token_6]  @channel_0[] (%results_7[] [] []) : (memref<32x32xbf16, 2>)
          affine.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%3] {
          memref.dealloc %results_7 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
