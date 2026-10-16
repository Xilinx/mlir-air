//===- air_transform_loop_wide64_payload.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_loop_wide64.mlir' %s | FileCheck %s

// 0x4300 | q on 64 lanes inside the linearized loop, with the 0x4300
// constant defined outside it (so it reaches the loop reshaped): still the
// two byte interleaves with 0x43.
// CHECK-LABEL: @outer_constant
// CHECK: scf.for
// CHECK: aievec.unpack
// CHECK: call @llvm.aie2p.vshuffle
// CHECK: call @llvm.aie2p.vshuffle
// CHECK-NOT: arith.ori
// CHECK: {linearize}
func.func @outer_constant(%q: memref<4x32xi8>, %w: memref<4x8x8xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pi8 = arith.constant 0 : i8
  %magic = arith.constant dense<17152> : vector<8x8xi16>
  scf.for %i = %c0 to %c4 step %c1 {
    %qv = vector.transfer_read %q[%i, %c0], %pi8 {in_bounds = [true]} : memref<4x32xi8>, vector<32xi8>
    %u = aievec.unpack %qv : vector<32xi8>, vector<64xi8>
    %e = arith.extsi %u : vector<64xi8> to vector<64xi16>
    %e2 = vector.shape_cast %e : vector<64xi16> to vector<8x8xi16>
    %o = arith.ori %e2, %magic : vector<8x8xi16>
    %b = arith.bitcast %o : vector<8x8xi16> to vector<8x8xbf16>
    vector.transfer_write %b, %w[%i, %c0, %c0] {in_bounds = [true, true]} : vector<8x8xbf16>, memref<4x8x8xbf16>
  } {linearize}
  return
}
