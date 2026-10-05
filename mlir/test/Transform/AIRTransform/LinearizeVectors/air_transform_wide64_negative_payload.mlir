//===- air_transform_wide64_negative_payload.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform_wide64.mlir' %s | FileCheck %s

// Cases the f32_lanes = 64 forms must leave alone or handle differently.

// Sign extension equals zero extension only below 0x80, so 0x4300 | sext(q)
// becomes the byte interleave only when q is an unpacked nibble.
// CHECK-LABEL: @sext_of_plain_bytes
// CHECK-NOT: call @llvm.aie2p.vshuffle
// CHECK: arith.ori
func.func @sext_of_plain_bytes(%q: memref<64xi8>, %w: memref<64xi16>) {
  %c0 = arith.constant 0 : index
  %pi8 = arith.constant 0 : i8
  %magic = arith.constant dense<17152> : vector<64xi16>
  %qv = vector.transfer_read %q[%c0], %pi8 {in_bounds = [true]} : memref<64xi8>, vector<64xi8>
  %e = arith.extsi %qv : vector<64xi8> to vector<64xi16>
  %o = arith.ori %e, %magic : vector<64xi16>
  vector.transfer_write %o, %w[%c0] {in_bounds = [true]} : vector<64xi16>, memref<64xi16>
  return
}

// A 64-lane conversion whose result is also used by computation, not only
// stored: two 32-lane srs, put back together for the user.
// CHECK-LABEL: @trunc_feeds_compute
// CHECK: %[[L:.*]] = aievec.srs %{{.*}}, %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: %[[H:.*]] = aievec.srs %{{.*}}, %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: vector.shuffle %[[L]], %[[H]]
// CHECK-NOT: arith.truncf {{.*}} vector<64xf32> to vector<64xbf16>
func.func @trunc_feeds_compute(%a: memref<64xf32>, %w: memref<64xbf16>) {
  %c0 = arith.constant 0 : index
  %pf = arith.constant 0.0 : f32
  %av = vector.transfer_read %a[%c0], %pf {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
  %r = arith.truncf %av : vector<64xf32> to vector<64xbf16>
  %s = arith.addf %r, %r : vector<64xbf16>
  vector.transfer_write %s, %w[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  return
}
