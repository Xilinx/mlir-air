//===- air_segment_compute_outside_herd.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A direct memory access in segment scope has no AIE core to execute it.
// Rejected here rather than in air::SegmentOp::verify, because a segment means
// something else on the GPU path, where its body is the workgroup and its
// threads address memory directly.

// RUN: not air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu1" 2>&1 | FileCheck %s

// CHECK: 'memref.store' op is inside 'air.segment' but outside any 'air.herd'
// CHECK-SAME: no AIE core will execute it
func.func @segment_store_outside_herd(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%tx) in (%sx=%c1) args(%a=%arg0) : memref<64xbf16> {
    air.segment @seg args(%b=%a) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %l2 = memref.alloc() : memref<64xbf16, 1>
      memref.store %cst, %l2[%c0] : memref<64xbf16, 1>
      air.dma_memcpy_nd (%b[] [] [], %l2[] [] []) : (memref<64xbf16>, memref<64xbf16, 1>)
      memref.dealloc %l2 : memref<64xbf16, 1>
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
