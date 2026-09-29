//===- specialize-channel-wrap-and-stride-symbol.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride="scope=func" | FileCheck %s

// An offset whose affine map carries BOTH a dim and a symbol.
//
// `air::evaluateConstantsInMap` substitutes the constants one input at a time
// and used to declare the wrong shape for the result each time -- 0 dims while
// substituting a symbol, 0 symbols while substituting a dim -- so as soon as a
// map had both kinds, `AffineMap::replace` asserted in `willBeValidAffineMap`.
// Every other test here uses a dims-only map, which is why it went unseen.
//
// A herd whose body carries a second reduction loop produces exactly this
// shape: the index combines the loop's induction variable with the herd tile
// index.

#mapds = affine_map<(d0)[s0] -> (d0 * 8 + s0)>

module {
  air.channel @channel_sym [1, 1]

  // CHECK-LABEL: func.func @both_dim_and_symbol
  // CHECK: air.channel.put {{.*}}@channel_sym
  func.func @both_dim_and_symbol(%arg0: memref<128xf32>, %arg1: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %0 = affine.apply #mapds(%arg2)[%arg1]
      air.channel.put @channel_sym[%c0, %c0] (%arg0[%0] [%c8] [%c1]) : (memref<128xf32>)
    }
    return
  }
}
