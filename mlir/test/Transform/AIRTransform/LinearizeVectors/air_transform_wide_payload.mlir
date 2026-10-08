//===- air_transform_wide_payload.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform.mlir' %s | FileCheck %s --check-prefix=NARROW
// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_contract.mlir' %s | FileCheck %s --check-prefix=CONTRACT
// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_wide.mlir' %s | FileCheck %s --check-prefix=WIDE

// w = bf16(q * s + base) on 64 lanes, q and s bf16 widened to f32 and the
// per-column base replicated from 8 f32 values. The multiply and add carry no
// fast-math flags, so without `contract` they stay separate. With `contract`
// the f32 work is four 16-lane vector.fma; with f32_lanes = 32 as well it
// becomes two 32-lane aievec.mac_elem and two aievec.srs, and the replicated
// base is built at 32 lanes for each. @dequant_fma_flags carries the
// `contract` flag itself and is contracted either way.

// NARROW-LABEL: @dequant_fma
// NARROW-NOT: vector.fma
// NARROW: arith.mulf {{.*}} : vector<16xf32>
// NARROW-LABEL: @dequant_fma_flags
// NARROW-COUNT-4: vector.fma {{.*}} : vector<16xf32>

// CONTRACT-LABEL: @dequant_fma
// CONTRACT-COUNT-4: vector.fma {{.*}} : vector<16xf32>
// CONTRACT-NOT: aievec.mac_elem

// WIDE-LABEL: @dequant_fma
// WIDE: %[[B0:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<8xf32>
// WIDE: %[[M0:.*]] = aievec.mac_elem %{{.*}}, %{{.*}}, %[[B0]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// WIDE: %[[B1:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<8xf32>
// WIDE: %[[M1:.*]] = aievec.mac_elem %{{.*}}, %{{.*}}, %[[B1]] : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
// WIDE: aievec.srs %[[M0]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// WIDE: aievec.srs %[[M1]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// WIDE-NOT: vector.fma
// WIDE-NOT: arith.truncf
func.func @dequant_fma(%q: memref<64xbf16>, %s: memref<64xbf16>,
                       %b: memref<8xf32>, %w: memref<64xbf16>) {
  %c0 = arith.constant 0 : index
  %pbf = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
  %qv = vector.transfer_read %q[%c0], %pbf {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %sv = vector.transfer_read %s[%c0], %pbf {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %bv = vector.transfer_read %b[%c0], %pf {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  %bb = vector.shuffle %bv, %bv [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xf32>, vector<8xf32>
  %qf = arith.extf %qv : vector<64xbf16> to vector<64xf32>
  %sf = arith.extf %sv : vector<64xbf16> to vector<64xf32>
  %m = arith.mulf %qf, %sf : vector<64xf32>
  %a = arith.addf %m, %bb : vector<64xf32>
  %r = arith.truncf %a : vector<64xf32> to vector<64xbf16>
  vector.transfer_write %r, %w[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  return
}

func.func @dequant_fma_flags(%q: memref<64xbf16>, %s: memref<64xbf16>,
                             %b: memref<64xf32>, %w: memref<64xbf16>) {
  %c0 = arith.constant 0 : index
  %pbf = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
  %qv = vector.transfer_read %q[%c0], %pbf {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %sv = vector.transfer_read %s[%c0], %pbf {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %bv = vector.transfer_read %b[%c0], %pf {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
  %qf = arith.extf %qv : vector<64xbf16> to vector<64xf32>
  %sf = arith.extf %sv : vector<64xbf16> to vector<64xf32>
  %m = arith.mulf %qf, %sf fastmath<contract> : vector<64xf32>
  %a = arith.addf %m, %bv fastmath<contract> : vector<64xf32>
  %r = arith.truncf %a : vector<64xf32> to vector<64xbf16>
  vector.transfer_write %r, %w[%c0] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  return
}
