//===- air_transform_wide64_payload.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/Inputs/air_transform_wide64.mlir' %s | FileCheck %s

// A 4-bit dequant tile, w = bf16(bf16(0x4300 | q) * s + base) on 64 lanes, in
// the AIE2P forms a hand-written dequant kernel uses: the 0x4300 | q
// widening as two byte interleaves with 0x43 (vshuffle modes 20/21), one
// 64-lane bf16 multiply-accumulate, and the conversion back to bf16 as two
// 32-lane aievec.srs whose halves are stored separately. The intrinsics are
// called through private func declarations, not LLVM-dialect ops (which
// aie-standard-lowering drops a core for).

// CHECK-DAG: func.func private @llvm.aie2p.vshuffle(vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-DAG: func.func private @llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf(vector<64xbf16>, vector<64xbf16>, vector<64xf32>, i32) -> vector<64xf32>
// CHECK-LABEL: @dequant64
// CHECK: %[[U:.*]] = aievec.unpack
// CHECK: %[[W:.*]] = vector.bitcast %[[U]] : vector<64xi8> to vector<16xi32>
// 0x43434343: the 0x43 high byte of every bf16 lane.
// CHECK: %[[BIAS:.*]] = arith.constant dense<1128481603> : vector<16xi32>
// CHECK: %[[M20:.*]] = arith.constant 20 : i32
// CHECK: %[[LO:.*]] = call @llvm.aie2p.vshuffle(%[[W]], %[[BIAS]], %[[M20]])
// CHECK: %[[M21:.*]] = arith.constant 21 : i32
// CHECK: %[[HI:.*]] = call @llvm.aie2p.vshuffle(%[[W]], %[[BIAS]], %[[M21]])
// CHECK: %[[CAT:.*]] = vector.shuffle %[[LO]], %[[HI]] [0, 1, {{.*}}, 31] : vector<16xi32>, vector<16xi32>
// CHECK: %[[Q16:.*]] = vector.bitcast %[[CAT]] : vector<32xi32> to vector<64xi16>
// CHECK: %[[QB:.*]] = arith.bitcast %[[Q16]] : vector<64xi16> to vector<64xbf16>
// CHECK: %[[S:.*]] = vector.transfer_read %arg1
// CHECK: %[[B:.*]] = vector.transfer_read %arg2
// CHECK: %[[CONF:.*]] = arith.constant 828 : i32
// CHECK: %[[MAC:.*]] = call @llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf(%[[QB]], %[[S]], %[[B]], %[[CONF]])
// CHECK: %[[A0:.*]] = vector.shuffle %[[MAC]], %[[MAC]] [0, 1, 2
// CHECK: %[[B0:.*]] = aievec.srs %[[A0]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: %[[A1:.*]] = vector.shuffle %[[MAC]], %[[MAC]] [32, 33, 34
// CHECK: %[[B1:.*]] = aievec.srs %[[A1]], %{{.*}} : vector<32xf32>, i32, vector<32xbf16>
// CHECK: %[[C0:.*]] = vector.shape_cast %[[B0]] : vector<32xbf16> to vector<4x8xbf16>
// CHECK: vector.transfer_write %[[C0]], %arg3[%c0, %c0]
// CHECK: %[[C1:.*]] = vector.shape_cast %[[B1]] : vector<32xbf16> to vector<4x8xbf16>
// CHECK: vector.transfer_write %[[C1]], %arg3[%c4, %c0]
// CHECK-NOT: arith.ori
// CHECK-NOT: vector.fma
func.func @dequant64(%q: memref<32xi8>, %s: memref<64xbf16>, %b: memref<64xf32>,
                     %w: memref<8x8xbf16>) {
  %c0 = arith.constant 0 : index
  %pi8 = arith.constant 0 : i8
  %pbf = arith.constant 0.0 : bf16
  %pf = arith.constant 0.0 : f32
  %magic = arith.constant dense<17152> : vector<64xi16>
  %qv = vector.transfer_read %q[%c0], %pi8 {in_bounds = [true]} : memref<32xi8>, vector<32xi8>
  %u = aievec.unpack %qv : vector<32xi8>, vector<64xi8>
  %e = arith.extsi %u : vector<64xi8> to vector<64xi16>
  %o = arith.ori %e, %magic : vector<64xi16>
  %qb = arith.bitcast %o : vector<64xi16> to vector<64xbf16>
  %sv = vector.transfer_read %s[%c0], %pbf {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
  %bv = vector.transfer_read %b[%c0], %pf {in_bounds = [true]} : memref<64xf32>, vector<64xf32>
  %qf = arith.extf %qb : vector<64xbf16> to vector<64xf32>
  %sf = arith.extf %sv : vector<64xbf16> to vector<64xf32>
  %m = arith.mulf %qf, %sf : vector<64xf32>
  %a = arith.addf %m, %bv : vector<64xf32>
  %r = arith.truncf %a : vector<64xf32> to vector<64xbf16>
  %r2 = vector.shape_cast %r : vector<64xbf16> to vector<8x8xbf16>
  vector.transfer_write %r2, %w[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xbf16>, memref<8x8xbf16>
  return
}
