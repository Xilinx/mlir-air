//===- multi_segment_herds.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Two air.segments, each containing an air.herd, chained by a channel.
//
// Regression test for the segment runner nodes being built into a std::vector
// while each herd stores a raw pointer to its parent segment: pushing the
// second segment reallocated that vector and left the first segment's herds
// pointing at freed storage, so their tile pool came back empty and the run
// aborted with "failed to reserve resources". A single segment with herds, or
// several segments without them, both happened to avoid the reallocation.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// CHECK: "name": "HerdOp(ma)[1, 1]",
// CHECK: "name": "HerdOp(mb)[1, 1]",
// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

module {
  air.channel @c2c [1, 1]
  air.channel @onchip_a [1, 1]
  air.channel @onchip_b [1, 1]
  func.func @test(%arg0: memref<64xbf16>) {
    %c1 = arith.constant 1 : index
    %launch = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%larg=%arg0) : memref<64xbf16> attributes {id = 1 : i32} {
      %seg_a = air.segment async attributes {id = 10 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_a = arith.constant 1 : index
        %tok_a, %buf_a = air.execute -> (memref<64xbf16, 1>) {
          %al_a = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %al_a : memref<64xbf16, 1>
        }
        %fw_a = air.channel.put async [%tok_a] @onchip_a[] (%buf_a[] [] []) {id = 20 : i32} : (memref<64xbf16, 1>)
        %herd_a = air.herd @ma async [%fw_a] tile (%tax, %tay) in (%tasx=%c1_a, %tasy=%c1_a) attributes {id = 100 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %tl_a, %bl_a = air.execute -> (memref<64xbf16, 2>) {
            %a2 = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %a2 : memref<64xbf16, 2>
          }
          %g_a = air.channel.get async [%tl_a] @onchip_a[] (%bl_a[] [] []) {id = 21 : i32} : (memref<64xbf16, 2>)
          %p_a = air.channel.put async [%g_a] @c2c[] (%bl_a[] [] []) {id = 22 : i32} : (memref<64xbf16, 2>)
        }
      }
      %seg_b = air.segment async attributes {id = 11 : i32, x_loc = 1 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_b = arith.constant 1 : index
        %tok_b, %buf_b = air.execute -> (memref<64xbf16, 1>) {
          %al_b = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %al_b : memref<64xbf16, 1>
        }
        %rx_b = air.channel.get async [%tok_b] @c2c[] (%buf_b[] [] []) {id = 30 : i32} : (memref<64xbf16, 1>)
        %fw_b = air.channel.put async [%rx_b] @onchip_b[] (%buf_b[] [] []) {id = 31 : i32} : (memref<64xbf16, 1>)
        %herd_b = air.herd @mb async [%fw_b] tile (%tbx, %tby) in (%tbsx=%c1_b, %tbsy=%c1_b) attributes {id = 101 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %tl_b, %bl_b = air.execute -> (memref<64xbf16, 2>) {
            %b2 = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %b2 : memref<64xbf16, 2>
          }
          %g_b = air.channel.get async [%tl_b] @onchip_b[] (%bl_b[] [] []) {id = 32 : i32} : (memref<64xbf16, 2>)
        }
      }
    }
    return
  }
}
