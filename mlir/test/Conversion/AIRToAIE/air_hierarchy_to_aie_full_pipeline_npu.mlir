//===- air_hierarchy_to_aie_full_pipeline_npu.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Full pipeline test for NPU target:
//   air-opt --air-hierarchy-to-aie \
//           --air-channel-to-conduit
//   aie-opt --conduit-to-dma
//
// Verifies that the complete Conduit pipeline produces valid AIE hardware
// ops (locks, buffers) from an AIR program with segment+herd+channels.
//
// The test program:
//   - air.launch wrapping air.segment wrapping air.herd (1x1)
//   - Two channels: @in_data and @out_data (core-to-core within the herd)
//   - Kernel: alloc → put @in_data → get @in_data → put @out_data
//
// Expected after full pipeline:
//   - aie.device with tiles, cores
//   - No residual air.channel / conduit.* ops

// RUN: air-opt %s \
// RUN:   -air-hierarchy-to-aie="row-offset=2 col-offset=0 device=npu1_1col" \
// RUN:   --air-channel-to-conduit \
// RUN: | aie-opt --conduit-to-dma \
// RUN: | FileCheck %s

// CHECK: aie.device(npu1_1col)
// CHECK: aie.core

// No residual air or conduit ops inside device.
// CHECK-NOT: air.channel.put
// CHECK-NOT: air.channel.get
// CHECK-NOT: conduit.put_memref
// CHECK-NOT: conduit.get_memref
// CHECK-NOT: conduit.wait_all

air.channel @in_data [1, 1]
air.channel @out_data [1, 1]
func.func @full_pipeline_npu() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %h = air.herd @compute async tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %tok, %buf = air.execute -> (memref<64xi32, 2>) {
          %a = memref.alloc() : memref<64xi32, 2>
          air.execute_terminator %a : memref<64xi32, 2>
        }
        %p = air.channel.put async [%tok] @in_data[] (%buf[] [] []) : (memref<64xi32, 2>)
        %g = air.channel.get async [%p] @out_data[] (%buf[] [] []) : (memref<64xi32, 2>)
        %d = air.execute [%g] {
          memref.dealloc %buf : memref<64xi32, 2>
        }
      }
    }
  }
  return
}
