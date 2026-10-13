//===- air_herd_to_aie_defined_func_alloc.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' -verify-diagnostics

// A function a herd calls may not allocate: only an allocation directly in
// the herd becomes a tile buffer.
module {
  func.func @callee_allocates() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %out = memref.alloc() : memref<64xi32, 2>
          func.call @tile_body(%out) : (memref<64xi32, 2>) -> ()
        }
      }
    }
    return
  }
  // expected-error @+1 {{is called from a herd and allocates memory}}
  func.func private @tile_body(%b: memref<64xi32, 2>) {
    %tmp = memref.alloc() : memref<64xi32, 2>
    memref.copy %tmp, %b : memref<64xi32, 2> to memref<64xi32, 2>
    return
  }
}
