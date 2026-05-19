//===- opt_shim_dma_bds_default.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Smoke test for the default (no shim-dma-tile-sizes) path: pass should
// emit channel ops and not crash. The default tiles each shim loop level
// by 1 (an iteration-count no-op) but still invokes tilePerfectlyNested +
// the post-tile fixup that downstream lowering depends on.
//
// Detailed IR checking lives in opt_shim_dma_bds.mlir for the user-
// override path; this file just exercises the default code path on a
// minimal input that fits within lit's per-test timeout in Assert builds.

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s

module {
  air.channel @ch [1]

  // CHECK-LABEL: func.func @smoke
  // CHECK: air.channel.put
  func.func @smoke(%arg0: memref<512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%i) in (%n=%c1) args(%a=%arg0) : memref<512xbf16> {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %tok0 = air.wait_all async
      %1 = scf.for %j = %c0 to %c8 step %c1_0
          iter_args(%tok = %tok0) -> (!air.async.token) {
        %2 = air.channel.put async [%tok] @ch[]
            (%a[%j, %c0] [%c1_0, %c64] [%c64, %c1_0])
            {metadata = @airMemcpyId0} : (memref<512xbf16>)
        scf.yield %2 : !air.async.token
      }
    }
    return
  }
}
