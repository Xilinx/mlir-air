//===- branch_yields_value.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// A branch that holds an async op and also yields a value: the async branch
// keeps the value as its leading result, every region yields it followed by
// the async token, and its users take the new value result.

// CHECK-LABEL: func.func @scf_if_value
// CHECK: %[[IF:.*]]:2 = scf.if {{.*}} -> (index, !air.async.token)
// CHECK: air.channel.get async
// CHECK: scf.yield %[[C0:.*]], %{{.*}} : index, !air.async.token
// CHECK: } else {
// CHECK: scf.yield %[[C1:.*]], %{{.*}} : index, !air.async.token
// CHECK: air.herd @h async {{.*}} args(%{{.*}}=%[[IF]]#0) : index

// CHECK-LABEL: func.func @affine_if_value
// CHECK: %[[AIF:.*]]:2 = affine.if {{.*}} -> (index, !air.async.token)
// CHECK: air.channel.get async
// CHECK: affine.yield %{{.*}}, %{{.*}} : index, !air.async.token
// CHECK: } else {
// CHECK: affine.yield %{{.*}}, %{{.*}} : index, !air.async.token
// CHECK: air.herd @h async {{.*}} args(%{{.*}}=%[[AIF]]#0) : index

// CHECK-LABEL: func.func @index_switch_value
// CHECK: %[[SW:.*]]:2 = scf.index_switch {{.*}} -> index, !air.async.token
// CHECK: case 0 {
// CHECK: air.channel.get async
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, !air.async.token
// CHECK: default {
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, !air.async.token
// CHECK: air.herd @h async {{.*}} args(%{{.*}}=%[[SW]]#0) : index

#set = affine_set<(d0) : (d0 == 0)>
module {
  air.channel @chan [1, 1]
  func.func @scf_if_value(%a: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.launch (%lx, %ly) in (%lsx=%c8, %lsy=%c1) args(%la=%a) : memref<4x16xi32> {
      air.segment @seg args(%w=%lx, %sa=%la) : index, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %cond = arith.cmpi eq, %w, %c0 : index
        %row = scf.if %cond -> index {
          air.channel.get @chan[] (%sa[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield %c0 : index
        } else {
          scf.yield %c1_s : index
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) args(%r=%row) : index {
          %l1 = memref.alloc() : memref<16xi32, 2>
          air.channel.put @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %l1 : memref<16xi32, 2>
        }
      }
    }
    return
  }
  func.func @affine_if_value(%a: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.launch (%lx, %ly) in (%lsx=%c8, %lsy=%c1) args(%la=%a) : memref<4x16xi32> {
      air.segment @seg args(%w=%lx, %sa=%la) : index, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %row = affine.if #set(%w) -> index {
          air.channel.get @chan[] (%sa[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          affine.yield %c0 : index
        } else {
          affine.yield %c1_s : index
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) args(%r=%row) : index {
          %l1 = memref.alloc() : memref<16xi32, 2>
          air.channel.put @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %l1 : memref<16xi32, 2>
        }
      }
    }
    return
  }
  func.func @index_switch_value(%a: memref<4x16xi32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.launch (%lx, %ly) in (%lsx=%c8, %lsy=%c1) args(%la=%a) : memref<4x16xi32> {
      air.segment @seg args(%w=%lx, %sa=%la) : index, memref<4x16xi32> {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %row = scf.index_switch %w -> index
        case 0 {
          air.channel.get @chan[] (%sa[%c0, %c0] [%c1_s, %c16] [%c16, %c1_s]) : (memref<4x16xi32>)
          scf.yield %c0 : index
        }
        default {
          scf.yield %c1_s : index
        }
        air.herd @h tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) args(%r=%row) : index {
          %l1 = memref.alloc() : memref<16xi32, 2>
          air.channel.put @chan[] (%l1[] [] []) : (memref<16xi32, 2>)
          memref.dealloc %l1 : memref<16xi32, 2>
        }
      }
    }
    return
  }
}
