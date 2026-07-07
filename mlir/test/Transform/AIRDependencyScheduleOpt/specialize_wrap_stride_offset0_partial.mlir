//===- specialize_wrap_stride_offset0_partial.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride="scope=func" | FileCheck %s

// A partial access that canonicalizes down to a single size-1 dim at offset 0
// must NOT have its offsets/sizes/strides list erased to empty. An empty list
// is interpreted downstream (AIRRtToNpu) as a full-volume contiguous access of
// the whole memref, so a 1-element access at offset 0 on a larger memref would
// be silently expanded to the entire memref. canonicalizeWrapAndStrideList must
// keep the innermost dim when the access volume (1) is smaller than the memref.

// The offset/size/stride operand lists must survive as NON-EMPTY (offset 0,
// size 1, stride 1). Without the fix they collapse to `[] [] []`, which means
// a full-memref access.
// CHECK-LABEL: @offset0_partial
// CHECK: air.channel.put async @channel_0[] (%{{.*}}[%{{[a-z0-9_]+}}] [%{{[a-z0-9_]+}}] [%{{[a-z0-9_]+}}]) : (memref<1024xbf16>)

module {
  func.func @offset0_partial(%arg0: memref<1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%x) in (%sx=%c1) args(%a=%arg0) : memref<1024xbf16> {
      %c0i = arith.constant 0 : index
      %c1i = arith.constant 1 : index
      %1 = air.channel.put async @channel_0[] (%a[%c0i] [%c1i] [%c1i]) : (memref<1024xbf16>)
    }
    return
  }
  air.channel @channel_0 [1, 1]
}
