//===- dma_to_channel_hoist_index_switch.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// A segment picks which host buffer a get writes with an scf.index_switch on
// the launch index. Each get is hoisted to the launch inside its own copy of
// the switch, in the case it was written in.

// CHECK-LABEL: func.func @pick_buffer
// CHECK: air.launch {{.*}} args(%[[A:.*]]=%arg0, %[[B:.*]]=%arg1)
// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK-NOT: air.channel.get
// CHECK: default {
// CHECK: air.channel.get async {{.*}}@chan[] (%[[B]][
// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK: air.channel.get async {{.*}}@chan[] (%[[A]][
// CHECK: default {
// CHECK: air.segment
// CHECK-NOT: air.channel.get
// CHECK: air.herd

// Three buffers: the switch nests in the default case, and each case joins
// only its own transfers.

// CHECK-LABEL: func.func @pick_of_three
// CHECK: air.launch {{.*}} args(%[[A:.*]]=%arg0, %[[B:.*]]=%arg1, %[[C:.*]]=%arg2)
// CHECK-DAG: air.channel.get async {{.*}}@chan[] (%[[A]][
// CHECK-DAG: air.channel.get async {{.*}}@chan[] (%[[B]][
// CHECK-DAG: air.channel.get async {{.*}}@chan[] (%[[C]][
// CHECK: air.segment

module {
  air.channel @chan [1, 1]
  func.func @pick_buffer(%a: memref<4x16xi32>, %b: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.launch (%lx, %ly) in (%lsx=%c8, %lsy=%c1) args(%la=%a, %lb=%b) : memref<4x16xi32>, memref<4x16xi32> {
      air.segment @seg args(%w=%lx, %sa=%la, %sb=%lb) : index, memref<4x16xi32>, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %later = arith.cmpi sge, %w, %c4 : index
        %k = arith.select %later, %c1_i32, %c0_i32 : i32
        %key = arith.index_cast %k : i32 to index
        scf.index_switch %key
        case 0 {
          air.channel.get @chan[] (%sa[%w, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield
        }
        default {
          %r = arith.subi %w, %c4 : index
          air.channel.get @chan[] (%sb[%r, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) {
          %l1 = memref.alloc() : memref<16xi32, 2>
          air.channel.put @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %l1 : memref<16xi32, 2>
        }
      }
    }
    return
  }
  func.func @pick_of_three(%a: memref<4x16xi32>, %b: memref<4x16xi32>, %c: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    air.launch (%lx, %ly) in (%lsx=%c12, %lsy=%c1) args(%la=%a, %lb=%b, %lc=%c) : memref<4x16xi32>, memref<4x16xi32>, memref<4x16xi32> {
      air.segment @seg3 args(%w=%lx, %sa=%la, %sb=%lb, %sc=%lc) : index, memref<4x16xi32>, memref<4x16xi32>, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %p1 = arith.cmpi sge, %w, %c4 : index
        %k1 = arith.select %p1, %c1_i32, %c0_i32 : i32
        %key1 = arith.index_cast %k1 : i32 to index
        scf.index_switch %key1
        case 0 {
          air.channel.get @chan[] (%sa[%w, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield
        }
        default {
          %p2 = arith.cmpi sge, %w, %c8 : index
          %k2 = arith.select %p2, %c1_i32, %c0_i32 : i32
          %key2 = arith.index_cast %k2 : i32 to index
          scf.index_switch %key2
          case 0 {
            %r = arith.subi %w, %c4 : index
            air.channel.get @chan[] (%sb[%r, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
            scf.yield
          }
          default {
            %r = arith.subi %w, %c8 : index
            air.channel.get @chan[] (%sc[%r, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
            scf.yield
          }
          scf.yield
        }
        air.herd @h3 tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) {
          %l1 = memref.alloc() : memref<16xi32, 2>
          air.channel.put @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %l1 : memref<16xi32, 2>
        }
      }
    }
    return
  }
}
