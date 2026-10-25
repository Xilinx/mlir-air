//===- dma_to_channel_hoist_iter_args.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -canonicalize -cse -split-input-file | FileCheck %s

// A loop hoisted with a DMA carries its async tokens, and the iter_args that
// the hoisted ops read, computed by the hoisted ops that update them.

// The DMA into %buf reads a herd tile id, so its put is hoisted with the
// loops before it in the herd's dependency chain, among them a loop carrying
// a vector besides its token. No hoisted op reads the vector, so the hoisted
// copy of that loop carries only the token. It then moves no data and folds
// away; the herd keeps the vector loop.

// CHECK-LABEL: func.func @unused_vector_iter_arg
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
  func.func @unused_vector_iter_arg(%a: memref<128xf32>, %b: memref<16xf32>) {
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

// The DMA's offset comes from an iter_arg the loop updates. The hoisted copy
// of the loop carries it too, with the update.

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
// loop, from the tile id. It is computed again before the hoisted loop, from
// the hoisted tile index.

// CHECK-LABEL: func.func @offset_step_from_herd
// CHECK: air.segment @seg
// CHECK: scf.parallel (%[[TX:.*]]) =
// CHECK-NEXT: %[[STEP:.*]] = arith.addi %[[TX]], %c16{{.*}} : index
// CHECK: scf.for {{.*}} iter_args(%[[BASE:.*]] = %c0{{.*}}, %{{.*}} = %{{.*}}) -> (index, !air.async.token)
// CHECK: air.channel.put
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

// -----

// The DMA reads the iter_arg of an inner loop, whose init value is the outer
// loop's iter_arg, and which the outer loop yields on. Both hoisted loops
// carry it.

// CHECK-LABEL: func.func @nested
// CHECK: air.segment @seg
// CHECK: scf.parallel (%[[TX:.*]]) =
// CHECK: %[[OUTER:.*]]:2 = scf.for {{.*}} iter_args(%[[A:.*]] = %c0{{.*}}, %[[T:.*]] = %{{.*}}) -> (index, !air.async.token)
// CHECK-NEXT: %[[INNER:.*]]:2 = scf.for {{.*}} iter_args(%[[B:.*]] = %[[A]], %{{.*}} = %[[T]]) -> (index, !air.async.token)
// CHECK-NEXT: %[[OFF:.*]] = affine.apply #{{.*}}(%[[TX]])[%[[B]]]
// CHECK-NEXT: air.channel.put {{.*}}@channel_1{{.*}}[%[[OFF]]] [16] [1]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[B]], %c32{{.*}} : index
// CHECK: scf.yield %[[NEXT]], %{{.*}} : index, !air.async.token
// CHECK: scf.yield %[[INNER]]#0, %[[INNER]]#1 : index, !air.async.token
// CHECK: air.herd @h
#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @nested(%a: memref<256xf32>) {
    air.launch () in () args(%la=%a) : memref<256xf32> {
      air.segment @seg args(%sa=%la) : memref<256xf32> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<256xf32, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<256xf32, 1>, memref<256xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2) : memref<256xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c16 = arith.constant 16 : index
          %c32 = arith.constant 32 : index
          %buf = memref.alloc() : memref<16xf32, 2>
          %end = scf.for %i = %c0 to %c2 step %c1 iter_args(%base = %c0) -> (index) {
            %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%b = %base) -> (index) {
              %off = affine.apply #map(%tx)[%b]
              air.dma_memcpy_nd (%buf[] [] [], %in[%off] [%c16] [%c1]) : (memref<16xf32, 2>, memref<256xf32, 1>)
              %nb = arith.addi %b, %c32 : index
              scf.yield %nb : index
            }
            scf.yield %inner : index
          }
          memref.dealloc %buf : memref<16xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// The iter_arg is the upper bound of the inner loop holding the DMA.

// CHECK-LABEL: func.func @bound
// CHECK: air.segment @seg
// CHECK: scf.for {{.*}} iter_args(%[[N:.*]] = %c1{{.*}}, %{{.*}} = %{{.*}}) -> (index, !air.async.token)
// CHECK-NEXT: scf.for %{{.*}} = %c0{{.*}} to %[[N]] step
// CHECK: air.channel.put
// CHECK: %[[NEXT:.*]] = arith.addi %[[N]], %c1{{.*}} : index
// CHECK-NEXT: scf.yield %[[NEXT]], %{{.*}} : index, !air.async.token
// CHECK: air.herd @h
module {
  func.func @bound(%a: memref<256xf32>) {
    air.launch () in () args(%la=%a) : memref<256xf32> {
      air.segment @seg args(%sa=%la) : memref<256xf32> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<256xf32, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<256xf32, 1>, memref<256xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2) : memref<256xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %buf = memref.alloc() : memref<16xf32, 2>
          %end = scf.for %i = %c0 to %c4 step %c1 iter_args(%n = %c1) -> (index) {
            scf.for %j = %c0 to %n step %c1 {
              %off = arith.muli %j, %c16 : index
              air.dma_memcpy_nd (%buf[] [] [], %in[%off] [%c16] [%c1]) : (memref<16xf32, 2>, memref<256xf32, 1>)
            }
            %nn = arith.addi %n, %c1 : index
            scf.yield %nn : index
          }
          memref.dealloc %buf : memref<16xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// The DMA's offset is the result of a loop before it. The hoisted copy of
// the loop carries the value, so that the offset reads its result.

// CHECK-LABEL: func.func @live_result
// CHECK: air.segment @seg
// CHECK: scf.parallel (%[[TX:.*]]) =
// CHECK-NEXT: %[[END:.*]] = scf.for {{.*}} iter_args(%[[B:.*]] = %[[TX]]) -> (index)
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[B]], %c16{{.*}} : index
// CHECK-NEXT: scf.yield %[[NEXT]] : index
// CHECK: air.channel.put {{.*}}@channel_1{{.*}}[%[[END]]] [16] [1]
// CHECK: air.herd @h
module {
  func.func @live_result(%a: memref<256xf32>) {
    air.launch () in () args(%la=%a) : memref<256xf32> {
      air.segment @seg args(%sa=%la) : memref<256xf32> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<256xf32, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<256xf32, 1>, memref<256xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2) : memref<256xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %buf = memref.alloc() : memref<16xf32, 2>
          %end = scf.for %i = %c0 to %c4 step %c1 iter_args(%b = %tx) -> (index) {
            %nb = arith.addi %b, %c16 : index
            scf.yield %nb : index
          }
          air.dma_memcpy_nd (%buf[] [] [], %in[%end] [%c16] [%c1]) : (memref<16xf32, 2>, memref<256xf32, 1>)
          memref.dealloc %buf : memref<16xf32, 2>
        }
      }
    }
    return
  }
}

// -----

// As the first case, but a pure op after the loop reads its result, for a
// write that the hoisted DMA waits on. That write becomes a wait when hoisted,
// so nothing hoisted reads the pure op: it is not hoisted, and the hoisted
// loop does not carry the vector.

// CHECK-LABEL: func.func @dead_result
// CHECK: air.segment @seg
// CHECK-NOT: truncf
// CHECK-NOT: vector<16xf32>
// CHECK: air.herd @h
// CHECK: arith.truncf
#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @dead_result(%a: memref<128xf32>, %b: memref<16xbf16>) {
    air.launch () in () args(%la=%a, %lb=%b) : memref<128xf32>, memref<16xbf16> {
      air.segment @seg args(%sa=%la, %sb=%lb) : memref<128xf32>, memref<16xbf16> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<128xf32, 1>
        %l2o = memref.alloc() : memref<16xbf16, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<128xf32, 1>, memref<128xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2, %out=%l2o) : memref<128xf32, 1>, memref<16xbf16, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %c64 = arith.constant 64 : index
          %f0 = arith.constant 0.0 : f32
          %zero = arith.constant dense<0.0> : vector<16xf32>
          %buf = memref.alloc() : memref<16xf32, 2>
          %o = memref.alloc() : memref<16xbf16, 2>
          %acc = scf.for %j = %c0 to %c4 step %c1 iter_args(%v = %zero) -> (vector<16xf32>) {
            %r = vector.transfer_read %buf[%c0], %f0 {in_bounds = [true]} : memref<16xf32, 2>, vector<16xf32>
            %s = arith.addf %v, %r : vector<16xf32>
            scf.yield %s : vector<16xf32>
          }
          %t = arith.truncf %acc : vector<16xf32> to vector<16xbf16>
          vector.transfer_write %t, %o[%c0] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, 2>
          scf.for %k = %c0 to %c64 step %c16 {
            %off = affine.apply #map(%tx)[%k]
            air.dma_memcpy_nd (%buf[] [] [], %in[%off] [%c16] [%c1]) : (memref<16xf32, 2>, memref<128xf32, 1>)
          }
          air.dma_memcpy_nd (%out[] [] [], %o[] [] []) : (memref<16xbf16, 1>, memref<16xbf16, 2>)
          memref.dealloc %buf : memref<16xf32, 2>
          memref.dealloc %o : memref<16xbf16, 2>
        }
        air.dma_memcpy_nd (%sb[] [] [], %l2o[] [] []) : (memref<16xbf16>, memref<16xbf16, 1>)
      }
    }
    return
  }
}

// -----

// The loop's update of the DMA's offset is the result of an inner loop that
// the DMA does not read. The inner loop is hoisted too, for the outer loop's
// yield.

// CHECK-LABEL: func.func @inner_loop_update
// CHECK: air.segment @seg
// CHECK: scf.for {{.*}} iter_args(%[[BASE:.*]] = %c0{{.*}}, %{{.*}} = %{{.*}}) -> (index, !air.async.token)
// CHECK-NEXT: %[[OFF:.*]] = affine.apply #{{.*}}(%{{.*}})[%[[BASE]]]
// CHECK-NEXT: air.channel.put {{.*}}@channel_1{{.*}}[%[[OFF]]] [16] [1]
// CHECK-NEXT: scf.yield %c32{{.*}}, %{{.*}} : index, !air.async.token
// CHECK: air.herd @h
#map = affine_map<(d0)[s0] -> (d0 * 16 + s0)>
module {
  func.func @inner_loop_update(%a: memref<256xf32>) {
    air.launch () in () args(%la=%a) : memref<256xf32> {
      air.segment @seg args(%sa=%la) : memref<256xf32> {
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %l2 = memref.alloc() : memref<256xf32, 1>
        air.dma_memcpy_nd (%l2[] [] [], %sa[] [] []) : (memref<256xf32, 1>, memref<256xf32>)
        air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%in=%l2) : memref<256xf32, 1> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          %c16 = arith.constant 16 : index
          %buf = memref.alloc() : memref<16xf32, 2>
          %end = scf.for %i = %c0 to %c4 step %c1 iter_args(%base = %c0) -> (index) {
            %off = affine.apply #map(%tx)[%base]
            air.dma_memcpy_nd (%buf[] [] [], %in[%off] [%c16] [%c1]) : (memref<16xf32, 2>, memref<256xf32, 1>)
            %nb = scf.for %j = %c0 to %c2 step %c1 iter_args(%x = %base) -> (index) {
              %c32 = arith.constant 32 : index
              scf.yield %c32 : index
            }
            scf.yield %nb : index
          }
          memref.dealloc %buf : memref<16xf32, 2>
        }
      }
    }
    return
  }
}
