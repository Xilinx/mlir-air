//===- library_call.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// How air-linalg-to-func names the function it calls. Three cases, because a
// linalg.generic only reports a library call name if it carries the attribute,
// and the OpDSL emitter never sets one.

// RUN: air-opt %s -air-linalg-to-func=link-with=mm.o | FileCheck %s --check-prefix=PLACEHOLDER
// RUN: air-opt %s -air-linalg-to-func='link-with=mm.o derive-library-call=true' | FileCheck %s --check-prefix=DERIVED
// RUN: air-opt %s -air-linalg-to-func='link-with=mm.o derive-library-call=true' | FileCheck %s --check-prefix=EXPLICIT

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Default: MLIR's placeholder, which is the symbol every hand-written kernel in
// this tree exports today. Unchanged so those keep linking.
// PLACEHOLDER-LABEL: func.func @unnamed
// PLACEHOLDER: call @op_has_no_registered_library_name
func.func @unnamed(%a: memref<4x8xbf16>, %b: memref<8x4xbf16>, %c: memref<4x4xbf16>) {
  linalg.generic {indexing_maps = [#map, #map1, #map2],
                  iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%a, %b : memref<4x8xbf16>, memref<8x4xbf16>) outs(%c : memref<4x4xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %0 = arith.mulf %in, %in_0 : bf16
    %1 = arith.addf %out, %0 : bf16
    linalg.yield %1 : bf16
  }
  return
}

// Derived: the name mangles every operand type, so a kernel built for other
// tile dimensions no longer links against the same symbol. The declaration
// carries link_with and the C interface attribute, as before.
// DERIVED-LABEL: func.func @unnamed
// DERIVED: call @linalg_generic_view4x8xbf16_view8x4xbf16_view4x4xbf16
// DERIVED: func.func private @linalg_generic_view4x8xbf16_view8x4xbf16_view4x4xbf16
// DERIVED-SAME: attributes {link_with = "mm.o", llvm.emit_c_interface}

// An explicit library_call always wins, derived or not: deriving only fills in
// a name that is absent.
// EXPLICIT: call @matmul_bf16_bf16_m4k8n4
func.func @named(%a: memref<4x8xbf16>, %b: memref<8x4xbf16>, %c: memref<4x4xbf16>) {
  linalg.generic {indexing_maps = [#map, #map1, #map2],
                  iterator_types = ["parallel", "parallel", "reduction"],
                  library_call = "matmul_bf16_bf16_m4k8n4"}
    ins(%a, %b : memref<4x8xbf16>, memref<8x4xbf16>) outs(%c : memref<4x4xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %0 = arith.mulf %in, %in_0 : bf16
    %1 = arith.addf %out, %0 : bf16
    linalg.yield %1 : bf16
  }
  return
}
