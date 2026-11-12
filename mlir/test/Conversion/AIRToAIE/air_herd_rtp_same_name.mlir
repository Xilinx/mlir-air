//===- air_herd_rtp_same_name.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 emit-herd-lock=true" | FileCheck %s
// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 emit-herd-lock=true" -air-to-std -canonicalize -airrt-to-npu | FileCheck %s --check-prefix=NPU
// RUN: air-opt %s -air-to-aie="device=npu2 row-offset=2 col-offset=0 emit-herd-lock=true output-elf=true" | FileCheck %s --check-prefix=ELF

// Two herds of one name run in turn on the same cores, each with a scalar of
// its own. The second is numbered (air.herd_rtp_index = 1) and takes its
// scalar through an RTP buffer of its own; the core reads each herd's from
// its buffer.

// CHECK-DAG: %[[R0:.*]] = aie.buffer(%{{.*}}) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
// CHECK-DAG: %[[R1:.*]] = aie.buffer(%{{.*}}) {sym_name = "__air_herd_rtp1_0_2"} : memref<1xi32>
// CHECK: aie.core
// CHECK: memref.load %[[R0]]
// CHECK: memref.load %[[R1]]
// CHECK: air.herd @h
// CHECK-NOT: air.herd_rtp_index
// CHECK: air.herd @h {{.*}}air.herd_rtp_index = 1

// The runtime sequence writes each herd's scalar into that herd's buffer.
// NPU: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, %arg0)
// NPU: aiex.npu.rtp_write(@__air_herd_rtp1_0_2, 0, %arg1)

// Under output-elf the scalars go through scratchpad parameters instead, one
// for each herd of the name too (see herd_rtp_scratchpad_skip.mlir for the
// runtime sequence's side).
// ELF-DAG: aiex.scratchpad_parameter @__air_param_h_0 : i32
// ELF-DAG: aiex.scratchpad_parameter @__air_param_h_rtp1_0 : i32
// ELF: aie.core
// ELF: aiex.read_scratchpad_parameter @__air_param_h_0 : i32
// ELF: aiex.read_scratchpad_parameter @__air_param_h_rtp1_0 : i32

module {
  func.func @two_herds(%arg0: i32, %arg1: i32) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1) in (%s0=%c1, %s1=%c1) args(%a=%arg0, %b=%arg1) : i32, i32 {
      air.segment @seg args(%sa=%a, %sb=%b) : i32, i32 {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%v=%sa) : i32 {
          %buf = memref.alloc() : memref<1xi32, 2>
          %c0 = arith.constant 0 : index
          memref.store %v, %buf[%c0] : memref<1xi32, 2>
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%v=%sb) : i32 {
          %buf = memref.alloc() : memref<1xi32, 2>
          %c0 = arith.constant 0 : index
          memref.store %v, %buf[%c0] : memref<1xi32, 2>
        }
      }
    }
    return
  }
}
