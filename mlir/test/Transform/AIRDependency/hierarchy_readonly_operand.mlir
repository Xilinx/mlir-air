//===- hierarchy_readonly_operand.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A hierarchy op's memref operands are classified by what its BODY does with
// them. Before this, air.segment / air.herd fell through to the "unknown op"
// case in getAllWriteAccessedMemrefOperandsFromOp, which assumes every operand
// is written -- so passing a read-only buffer into a segment ordered it after
// every prior READER of that buffer, via a WAR edge that is not there.
//
// This matters for air.dma_memcpy_nd specifically: a DMA has to name both
// endpoints in one place, so spelling a feed as a DMA forces the L3 buffer to
// become a hierarchy operand, where the equivalent channel put/get pair never
// passes it in. The spurious edge then forces the enclosing scf.index_switch to
// carry an async token, and air-to-aie cannot legalize the air.wait_all that
// terminates each arm.

// RUN: air-opt %s -air-dependency -split-input-file | FileCheck %s

// The segment only reads %m (its herd puts from it), and the launch-scope put
// also only reads %m. Two readers must not be ordered against each other, so
// the segment carries NO async dependency list.
// CHECK-LABEL: func.func @read_only_segment_operand
// CHECK: air.channel.put async  @feed
// CHECK: air.segment @seg async  args
air.channel @feed [1]
air.channel @inner [1]
func.func @read_only_segment_operand(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @feed[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c0_s = arith.constant 0 : index
      %c32_s = arith.constant 32 : index
      %c64_s = arith.constant 64 : index
      %c1_s = arith.constant 1 : index
      air.channel.put @inner[%c0_s] (%sa[%c0_s, %c0_s] [%c32_s, %c32_s] [%c64_s, %c1_s]) : (memref<64x64xi32>)
    }
  }
  return
}

// -----

// Control: the segment's herd WRITES %m through an air.channel.get, so the WAR
// edge against the launch-scope reader is real and must still be drawn.
// CHECK-LABEL: func.func @written_segment_operand
// CHECK: air.channel.put async  @feed
// CHECK: air.segment @seg async [
air.channel @feed [1]
air.channel @back [1]
func.func @written_segment_operand(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @feed[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c0_s = arith.constant 0 : index
      %c32_s = arith.constant 32 : index
      %c64_s = arith.constant 64 : index
      %c1_s = arith.constant 1 : index
      air.channel.get @back[%c0_s] (%sa[%c0_s, %c0_s] [%c32_s, %c32_s] [%c64_s, %c1_s]) : (memref<64x64xi32>)
    }
  }
  return
}
