//===- func_signature_mismatch.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-merge-unrolled-devices -verify-diagnostics

// Test: Verify that conflicting function signatures across unrolled devices
// are detected and reported as an error.

module {
  // First device has extern_kernel with memref<32xi32>
  aie.device(npu2_4col) @segment_with_unroll_0_0 {
    func.func private @extern_kernel(memref<32xi32, 2 : i32>)
    %tile_0_2 = aie.tile(0, 2)
    aie.end
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 0 : i64, segment_unroll_y = 0 : i64}

  // Second device has extern_kernel with memref<64xi32> - SIGNATURE MISMATCH!
  aie.device(npu2_4col) @segment_with_unroll_1_0 {
    // expected-error @+1 {{function 'extern_kernel' has conflicting signatures across unrolled devices: '(memref<32xi32, 2 : i32>) -> ()' vs '(memref<64xi32, 2 : i32>) -> ()'}}
    func.func private @extern_kernel(memref<64xi32, 2 : i32>)
    %tile_0_2 = aie.tile(0, 2)
    aie.end
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>, segment_unroll_x = 1 : i64, segment_unroll_y = 0 : i64}
}
