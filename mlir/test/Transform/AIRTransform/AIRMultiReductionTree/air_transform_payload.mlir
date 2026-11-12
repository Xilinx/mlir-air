//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// The row maxima of a transposed 8x8 bf16 tile (1024 bits) with 512-bit
// registers: the two halves combined, then the rows rotated within a register
// twice; row 0 then holds the maxima.
// CHECK-LABEL: @rows_max
// CHECK: %[[F:.*]] = vector.shape_cast %arg0 : vector<8x8xbf16> to vector<64xbf16>
// CHECK: %[[LO:.*]] = vector.extract_strided_slice %[[F]] {{.*}}offsets = [0], sizes = [32]
// CHECK: %[[HI:.*]] = vector.extract_strided_slice %[[F]] {{.*}}offsets = [32], sizes = [32]
// CHECK: %[[M0:.*]] = arith.maximumf %[[LO]], %[[HI]] : vector<32xbf16>
// CHECK: %[[R1:.*]] = vector.shuffle %[[M0]], %[[M0]] [16, 17
// CHECK: %[[M1:.*]] = arith.maximumf %[[M0]], %[[R1]] : vector<32xbf16>
// CHECK: %[[R2:.*]] = vector.shuffle %[[M1]], %[[M1]] [8, 9
// CHECK: %[[M2:.*]] = arith.maximumf %[[M1]], %[[R2]] : vector<32xbf16>
// CHECK: vector.shape_cast %[[M2]] : vector<32xbf16> to vector<4x8xbf16>
// CHECK: vector.extract {{.*}}[0]
// CHECK: arith.maximumf {{.*}}, %arg1 : vector<8xbf16>
// CHECK-NOT: vector.multi_reduction
func.func @rows_max(%t: vector<8x8xbf16>, %acc: vector<8xbf16>) -> vector<8xbf16> {
  %r = vector.multi_reduction <maximumf>, %t, %acc [0] : vector<8x8xbf16> to vector<8xbf16>
  return %r : vector<8xbf16>
}

// Not the leading dim, or not a power of two: left.
// CHECK-LABEL: @keep
// CHECK: vector.multi_reduction <add>, %arg0, %arg1 [1]
// CHECK: vector.multi_reduction <add>, %arg2, %arg3 [0]
func.func @keep(%a: vector<8x8xf32>, %b: vector<8xf32>, %c: vector<6x8xf32>, %d: vector<8xf32>) -> (vector<8xf32>, vector<8xf32>) {
  %x = vector.multi_reduction <add>, %a, %b [1] : vector<8x8xf32> to vector<8xf32>
  %y = vector.multi_reduction <add>, %c, %d [0] : vector<6x8xf32> to vector<8xf32>
  return %x, %y : vector<8xf32>, vector<8xf32>
}
