//===- dma_to_channel_hoist_vector_iter_args.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -canonicalize -cse -split-input-file | FileCheck %s

// The DMA into %buf reads a herd tile id, so its put is hoisted with the
// loops before it in the herd's dependency chain, among them a loop carrying
// a vector besides its token. The hoisted copy of that loop carries only the
// token. It moves no data, so it is erased; the herd keeps the vector loop.

// CHECK-LABEL: func.func @f
// CHECK: air.segment @seg
// CHECK-NOT: vector<16xf32>
// CHECK: scf.parallel
// CHECK-NEXT: air.wait_all
// CHECK-NEXT: scf.for
// CHECK-NEXT: affine.apply
// CHECK-NEXT: air.channel.put {{.*}}@channel_1
// CHECK: air.herd @h
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<16xf32>, !air.async.token)
// CHECK: air.channel.get {{.*}}@channel_1
#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @f(%a: memref<128xf32>, %b: memref<16xf32>) {
    air.launch () in () args(%la=%a, %lb=%b) : memref<128xf32>, memref<16xf32> {
      air.segment @seg args(%sa=%la, %sb=%lb) : memref<128xf32>, memref<16xf32> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<128xf32, 1>
        %l2o = memref.alloc() : memref<16xf32, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<128xf32, 1>, memref<128xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2, %out=%l2o) : memref<128xf32, 1>, memref<16xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c64 = arith.constant 64 : index
          %f0 = arith.constant 0.0 : f32
          %zero = arith.constant dense<0.0> : vector<16xf32>
          %buf = memref.alloc() : memref<16xf32, 2>
          %o = memref.alloc() : memref<16xf32, 2>
          %acc = scf.for %j = %c0 to %c4 step %c1 iter_args(%v = %zero) -> (vector<16xf32>) {
            %r = vector.transfer_read %buf[%c0], %f0 {in_bounds = [true]} : memref<16xf32, 2>, vector<16xf32>
            %s = arith.addf %v, %r : vector<16xf32>
            scf.yield %s : vector<16xf32>
          }
          vector.transfer_write %acc, %o[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
          scf.for %k = %c0 to %c64 step %c16 {
            %off = affine.apply #map(%tx)[%k]
            air.dma_memcpy_nd (%buf[] [] [], %in[%off] [%c16] [%c1]) : (memref<16xf32, 2>, memref<128xf32, 1>)
          }
          air.dma_memcpy_nd (%out[] [] [], %o[] [] []) : (memref<16xf32, 1>, memref<16xf32, 2>)
          memref.dealloc %buf : memref<16xf32, 2>
          memref.dealloc %o : memref<16xf32, 2>
        }
        air.dma_memcpy_nd (%sb[] [] [], %l2o[] [] []) : (memref<16xf32>, memref<16xf32, 1>)
      }
    }
    return
  }
}

// -----

// The DMA's offset comes from a value the loop carries and updates with pure
// ops. The hoisted copy of the loop carries that value too, with the update.

// CHECK-LABEL: func.func @offset_iter_arg
// CHECK: air.segment @seg
// CHECK: scf.for {{.*}} iter_args(%[[BASE:.*]] = %c0{{.*}}, %{{.*}} = %{{.*}}) -> (index, !air.async.token)
// CHECK-NEXT: %[[OFF:.*]] = affine.apply #{{.*}}(%{{.*}})[%[[BASE]]]
// CHECK-NEXT: air.channel.put {{.*}}@channel_1{{.*}}[%[[OFF]]] [16] [1]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[BASE]], %c16{{.*}} : index
// CHECK: scf.yield %[[NEXT]], %{{.*}} : index, !air.async.token
// CHECK: air.herd @h

#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @offset_iter_arg(%a: memref<128xf32>, %b: memref<16xf32>) {
    air.launch () in () args(%la=%a, %lb=%b) : memref<128xf32>, memref<16xf32> {
      air.segment @seg args(%sa=%la, %sb=%lb) : memref<128xf32>, memref<16xf32> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<128xf32, 1>
        %l2o = memref.alloc() : memref<16xf32, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<128xf32, 1>, memref<128xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2, %out=%l2o) : memref<128xf32, 1>, memref<16xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c64 = arith.constant 64 : index
          %f0 = arith.constant 0.0 : f32
          %zero = arith.constant dense<0.0> : vector<16xf32>
          %buf = memref.alloc() : memref<16xf32, 2>
          %o = memref.alloc() : memref<16xf32, 2>
          %acc = scf.for %j = %c0 to %c4 step %c1 iter_args(%v = %zero) -> (vector<16xf32>) {
            %r = vector.transfer_read %buf[%c0], %f0 {in_bounds = [true]} : memref<16xf32, 2>, vector<16xf32>
            %s = arith.addf %v, %r : vector<16xf32>
            scf.yield %s : vector<16xf32>
          }
          vector.transfer_write %acc, %o[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
          %end = scf.for %k = %c0 to %c4 step %c1 iter_args(%base = %c0) -> (index) {
            %off = affine.apply #map(%tx)[%base]
            air.dma_memcpy_nd (%buf[] [] [], %in[%off] [%c16] [%c1]) : (memref<16xf32, 2>, memref<128xf32, 1>)
            %next = arith.addi %base, %c16 : index
            scf.yield %next : index
          }
          air.dma_memcpy_nd (%out[] [] [], %o[] [] []) : (memref<16xf32, 1>, memref<16xf32, 2>)
          memref.dealloc %buf : memref<16xf32, 2>
          memref.dealloc %o : memref<16xf32, 2>
        }
        air.dma_memcpy_nd (%sb[] [] [], %l2o[] [] []) : (memref<16xf32>, memref<16xf32, 1>)
      }
    }
    return
  }
}

// -----

// As above, but the step comes from a value computed in the herd before the
// loop, from the tile id. The hoisted loop computes it again from the hoisted
// tile index.

// CHECK-LABEL: func.func @offset_step_from_herd
// CHECK: air.segment @seg
// CHECK: scf.parallel (%[[TX:.*]]) =
// CHECK: scf.for {{.*}} iter_args(%[[BASE:.*]] = %c0{{.*}}, %{{.*}} = %{{.*}}) -> (index, !air.async.token)
// CHECK: air.channel.put
// CHECK: %[[STEP:.*]] = arith.addi %[[TX]], %c16{{.*}} : index
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[BASE]], %[[STEP]] : index
// CHECK: scf.yield %[[NEXT]], %{{.*}} : index, !air.async.token
// CHECK: air.herd @h

#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @offset_step_from_herd(%a: memref<128xf32>, %b: memref<16xf32>) {
    air.launch () in () args(%la=%a, %lb=%b) : memref<128xf32>, memref<16xf32> {
      air.segment @seg args(%sa=%la, %sb=%lb) : memref<128xf32>, memref<16xf32> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<128xf32, 1>
        %l2o = memref.alloc() : memref<16xf32, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<128xf32, 1>, memref<128xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2, %out=%l2o) : memref<128xf32, 1>, memref<16xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c64 = arith.constant 64 : index
          %f0 = arith.constant 0.0 : f32
          %zero = arith.constant dense<0.0> : vector<16xf32>
          %buf = memref.alloc() : memref<16xf32, 2>
          %o = memref.alloc() : memref<16xf32, 2>
          %acc = scf.for %j = %c0 to %c4 step %c1 iter_args(%v = %zero) -> (vector<16xf32>) {
            %r = vector.transfer_read %buf[%c0], %f0 {in_bounds = [true]} : memref<16xf32, 2>, vector<16xf32>
            %s = arith.addf %v, %r : vector<16xf32>
            scf.yield %s : vector<16xf32>
          }
          vector.transfer_write %acc, %o[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, 2>
          %stride = arith.addi %tx, %c16 : index
          %end = scf.for %k = %c0 to %c4 step %c1 iter_args(%base = %c0) -> (index) {
            %off = affine.apply #map(%tx)[%base]
            air.dma_memcpy_nd (%buf[] [] [], %in[%off] [%c16] [%c1]) : (memref<16xf32, 2>, memref<128xf32, 1>)
            %next = arith.addi %base, %stride : index
            scf.yield %next : index
          }
          air.dma_memcpy_nd (%out[] [] [], %o[] [] []) : (memref<16xf32, 1>, memref<16xf32, 2>)
          memref.dealloc %buf : memref<16xf32, 2>
          memref.dealloc %o : memref<16xf32, 2>
        }
        air.dma_memcpy_nd (%sb[] [] [], %l2o[] [] []) : (memref<16xf32>, memref<16xf32, 1>)
      }
    }
    return
  }
}
