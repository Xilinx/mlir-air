//===- gelu_epilogue_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_wide.mlir' %s | FileCheck %s
// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_wide.mlir' %s | FileCheck %s --check-prefix=RANK1

// g and u are interleaved column pairs of an f32 accumulator, and
// h = g * sigmoid(c2 * (g + k * g^3)) * u is computed in bf16 with the
// reciprocal in f32. Linearized, both slots are read from the same contiguous
// rows and split by shuffles, and every op is rank 1 at the AIE vector width.
// CHECK-LABEL: @gate_up_gelu
// CHECK: %[[R0:.*]] = vector.transfer_read %arg0{{.*}} : memref<8x8xf32>, vector<8x8xf32>
// CHECK: %[[F0:.*]] = vector.shape_cast %[[R0]] : vector<8x8xf32> to vector<64xf32>
// CHECK: vector.shuffle %[[F0]], %{{.*}} [0, 2, 4, 6
// CHECK: %[[R1:.*]] = vector.transfer_read %arg0{{.*}} : memref<8x8xf32>, vector<8x8xf32>
// CHECK: %[[F1:.*]] = vector.shape_cast %[[R1]] : vector<8x8xf32> to vector<64xf32>
// CHECK: vector.shuffle %[[F1]], %{{.*}} [1, 3, 5, 7
// CHECK: math.exp %{{.*}} : vector<32xbf16>
// CHECK: arith.divf %{{.*}} : vector<32xf32>
// CHECK: vector.transfer_write %{{.*}}, %arg1
// RANK1-NOT: {{(arith|math)\.[a-z]+}} {{.*}}vector<8x4x
func.func @gate_up_gelu(%acc: memref<8x8xf32>, %h: memref<8x4xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %p = arith.constant 0.0 : f32
  %c2 = arith.constant dense<1.595700e+00> : vector<8x4xbf16>
  %kg = arith.constant dense<4.467770e-02> : vector<8x4xbf16>
  %one = arith.constant dense<1.000000e+00> : vector<8x4xbf16>
  %onef = arith.constant dense<1.000000e+00> : vector<8x4xf32>
  %e = memref.expand_shape %acc [[0], [1, 2]] output_shape [8, 4, 2] : memref<8x8xf32> into memref<8x4x2xf32>
  %gf = vector.transfer_read %e[%c0, %c0, %c0], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, d1)>} : memref<8x4x2xf32>, vector<8x4xf32>
  %uf = vector.transfer_read %e[%c0, %c0, %c1], %p {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d0, d1)>} : memref<8x4x2xf32>, vector<8x4xf32>
  %g = arith.truncf %gf : vector<8x4xf32> to vector<8x4xbf16>
  %u = arith.truncf %uf : vector<8x4xf32> to vector<8x4xbf16>
  %g2 = arith.mulf %g, %g : vector<8x4xbf16>
  %g3 = arith.mulf %g2, %g : vector<8x4xbf16>
  %kg3 = arith.mulf %kg, %g3 : vector<8x4xbf16>
  %in = arith.addf %g, %kg3 : vector<8x4xbf16>
  %z = arith.mulf %c2, %in : vector<8x4xbf16>
  %nz = arith.negf %z : vector<8x4xbf16>
  %ex = math.exp %nz : vector<8x4xbf16>
  %den = arith.addf %one, %ex : vector<8x4xbf16>
  %denf = arith.extf %den : vector<8x4xbf16> to vector<8x4xf32>
  %sf = arith.divf %onef, %denf : vector<8x4xf32>
  %s = arith.truncf %sf : vector<8x4xf32> to vector<8x4xbf16>
  %gs = arith.mulf %g, %s : vector<8x4xbf16>
  %out = arith.mulf %gs, %u : vector<8x4xbf16>
  vector.transfer_write %out, %h[%c0, %c0] {in_bounds = [true, true]} : vector<8x4xbf16>, memref<8x4xbf16>
  return
}
