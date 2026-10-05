//===- air_transform_wide64_conflict_payload.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform_wide64.mlir' %s | FileCheck %s

// The module already declares the multiply-accumulate intrinsic's name with
// another type, so calling it would be ill-typed: the 64-lane FMA stays.
// CHECK: func.func private @llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf(vector<64xbf16>) -> vector<64xf32>
// CHECK-LABEL: @fma64
// CHECK-NOT: call @llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf
// CHECK: vector.fma
func.func private @llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf(vector<64xbf16>) -> vector<64xf32>
func.func @fma64(%x: memref<64xbf16>, %y: memref<64xbf16>, %b: memref<64xf32>, %w: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %pbf = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
  %xv = vector.transfer_read %x[%c0], %pbf {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %yv = vector.transfer_read %y[%c0], %pbf {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %bv = vector.transfer_read %b[%c0], %pf {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
  %xf = arith.extf %xv : vector<64xbf16> to vector<64xf32>
  %yf = arith.extf %yv : vector<64xbf16> to vector<64xf32>
  %m = arith.mulf %xf, %yf : vector<64xf32>
  %a = arith.addf %m, %bv : vector<64xf32>
  vector.transfer_write %a, %w[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
  return
}
