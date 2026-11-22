//===- air_herd_to_aie_defined_func_globals_invalid.mlir --------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' -split-input-file -verify-diagnostics

// Each core gets its own copy of a global a called function reads, so a
// global that is written, or has no initial value, cannot be copied.

module {
  memref.global "private" @state : memref<16xi32, 2> = dense<0>
  func.func @foo() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %out = memref.alloc() : memref<16xi32, 2>
          func.call @tile_body(%out) : (memref<16xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%b: memref<16xi32, 2>) {
    // expected-error @+1 {{refers to "state", which a core can only have a copy of if it is a constant with an initial value}}
    %g = memref.get_global @state : memref<16xi32, 2>
    memref.copy %g, %b : memref<16xi32, 2> to memref<16xi32, 2>
    return
  }
}

// -----

module {
  memref.global "private" constant @table : memref<16xi32, 2>
  func.func @foo() {
    %c1 = arith.constant 1 : index
    air.launch (%ix, %iy) in (%sx=%c1, %sy=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %out = memref.alloc() : memref<16xi32, 2>
          func.call @tile_body(%out) : (memref<16xi32, 2>) -> ()
        }
      }
    }
    return
  }
  func.func private @tile_body(%b: memref<16xi32, 2>) {
    // expected-error @+1 {{refers to "table", which a core can only have a copy of if it is a constant with an initial value}}
    %g = memref.get_global @table : memref<16xi32, 2>
    memref.copy %g, %b : memref<16xi32, 2> to memref<16xi32, 2>
    return
  }
}
