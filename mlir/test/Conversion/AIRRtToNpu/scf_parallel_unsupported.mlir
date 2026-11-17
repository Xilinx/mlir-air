//===- scf_parallel_unsupported.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -verify-diagnostics -split-input-file %s | FileCheck %s

// An scf.parallel holding only airrt.wait_all and side-effect-free ops is
// folded into a single airrt.wait_all. Any other op in its body is an error.

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
  } {sym_name = "segment_0"}
  func.func @dma_in_scf_parallel(%arg0: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %z = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c32 = arith.constant 32 : i64
    %id = arith.constant 2 : i32
    // expected-error @+1 {{scf.parallel with side-effecting ops other than airrt.wait_all is not supported by airrt-to-npu}}
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      %ii = arith.index_cast %i : index to i64
      %off = arith.muli %ii, %c32 : i64
      %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%z, %z, %z, %off], [%one, %one, %one, %c32], [%z, %z, %z, %one]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
      scf.reduce
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @pure_ops_in_scf_parallel
// CHECK-NOT: scf.parallel
// CHECK: return
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
  } {sym_name = "segment_0"}
  func.func @pure_ops_in_scf_parallel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %init = airrt.wait_all : !airrt.event
    %r = scf.parallel (%i) = (%c0) to (%c2) step (%c1) init (%init) -> !airrt.event {
      %ub = arith.addi %i, %c1 : index
      scf.parallel (%j) = (%c0) to (%ub) step (%c1) {
        %w = airrt.wait_all : !airrt.event
        scf.reduce
      }
      %e = airrt.wait_all : !airrt.event
      scf.reduce(%e : !airrt.event) {
      ^bb0(%a: !airrt.event, %b: !airrt.event):
        %w = airrt.wait_all %a, %b : !airrt.event
        scf.reduce.return %w : !airrt.event
      }
    }
    airrt.wait_all %r
    return
  }
}

// -----

// CHECK-LABEL: func.func @nested_wait_all_only
// CHECK-NOT: scf.parallel
// CHECK: return
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
  } {sym_name = "segment_0"}
  func.func @nested_wait_all_only() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      scf.parallel (%j) = (%c0) to (%c2) step (%c1) {
        %e = airrt.wait_all : !airrt.event
        scf.reduce
      }
      scf.reduce
    }
    return
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
  } {sym_name = "segment_0"}
  func.func @dma_in_nested_scf_parallel(%arg0: memref<64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %z = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %c32 = arith.constant 32 : i64
    %id = arith.constant 2 : i32
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      // expected-error @+1 {{scf.parallel with side-effecting ops other than airrt.wait_all is not supported by airrt-to-npu}}
      scf.parallel (%j) = (%c0) to (%c2) step (%c1) {
        %jj = arith.index_cast %j : index to i64
        %off = arith.muli %jj, %c32 : i64
        %e = airrt.dma_memcpy_nd(%id, %z, %z, %arg0[%z, %z, %z, %off], [%one, %one, %one, %c32], [%z, %z, %z, %one]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>) : !airrt.event
        scf.reduce
      }
      scf.reduce
    }
    return
  }
}
