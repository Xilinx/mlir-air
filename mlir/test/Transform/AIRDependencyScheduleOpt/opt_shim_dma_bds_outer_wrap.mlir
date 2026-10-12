//===- opt_shim_dma_bds_outer_wrap.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s

// The loop folds into 1024 rows at a row stride of 4096. The wrap split puts
// 512 outermost, because 512 rows at that stride is past the stride limit.
// That is over the shim iteration count, so the two outer dims are re-split
// to fit it.

// CHECK-LABEL: func.func @rebalance
// CHECK: air.channel.put {{.*}} @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c32{{.*}}, %c32{{.*}}, %c64{{.*}}] [%c131072{{.*}}, %c4096{{.*}}, %c1{{.*}}])
// CHECK-NOT: air.channel.put

module {
  air.channel @channel_0 [1, 1]
  func.func @rebalance(%arg0: memref<1024x4096xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %0 = air.wait_all async
    %1 = scf.for %i = %c0 to %c1024 step %c1 iter_args(%t = %0) -> (!air.async.token) {
      %put = air.channel.put async [%t] @channel_0[] (%arg0[%i, %c0] [%c1, %c64] [%c4096, %c1]) {metadata = @airMemcpyId7} : (memref<1024x4096xi32>)
      scf.yield %put : !air.async.token
    }
    return
  }
}
