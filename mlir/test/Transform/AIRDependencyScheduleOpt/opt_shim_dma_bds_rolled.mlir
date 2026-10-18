//===- opt_shim_dma_bds_rolled.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2 shim-dma-tile-sizes=2,2" | FileCheck %s
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu2 shim-dma-tile-sizes=2,2 keep-tile-loops-rolled=false" | FileCheck %s --check-prefix=UNROLLED

// A 4x4 launch tiled 2x2. The inter-tile loops are kept. The 2x2 loops of a
// tile fold into the BD dimensions as they do when the tiles are unrolled, and
// the offset adds the tile's base, computed from the inter-tile induction
// variables.

// CHECK-LABEL: func.func @rolled
// CHECK: scf.for %[[X:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: scf.for %[[Y:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK: arith.addi %[[X]]
// CHECK: air.channel.put {{.*}} @chA[] (%{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{[0-9]+}}] [%c2{{.*}}, %c1{{.*}}, %c64{{.*}}, %c512{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c1{{.*}}])
// CHECK: air.channel.put {{.*}} @chA[]
// CHECK: arith.addi %[[Y]]
// CHECK: air.channel.put {{.*}} @chB[] ({{.*}} [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}]
// CHECK: air.channel.get {{.*}} @chC[] ({{.*}} [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c16384{{.*}}, %c64{{.*}}, %c256{{.*}}, %c1{{.*}}])
// CHECK: scf.yield
// CHECK: scf.yield

// UNROLLED-LABEL: func.func @rolled
// UNROLLED-NOT: scf.for
// UNROLLED-COUNT-4: air.channel.get {{.*}} @chC[]
// UNROLLED-NOT: scf.for
// UNROLLED: return

// A loop inside the tile whose induction variable is in the offset. It folds
// into the BD with the intra-tile loops.

// CHECK-LABEL: func.func @nested_loop
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: air.channel.put {{.*}} @chA[] (%{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{[0-9]+}}] [%c2{{.*}}, %c4{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c32768{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}])
// CHECK: scf.yield

// An offset computed inside air.execute.

// CHECK-LABEL: func.func @execute_offset
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @chA[] (%{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{[0-9]+}}] [%c2{{.*}}, %c1{{.*}}, %c64{{.*}}, %c64{{.*}}]
// CHECK: air.channel.put {{.*}} @chA[] (%{{.*}}[%{{.*}}, %{{.*}}, %c64{{.*}}, %{{[0-9]+}}] [%c2{{.*}}, %c1{{.*}}, %c64{{.*}}, %c64{{.*}}]

// A transfer that already uses every BD dimension, so the loops of a tile are
// unrolled into a chain of BDs. Only the loops inside the tile count toward the
// trip count of the chain; counting the inter-tile loops too would leave the
// tile's loops rolled.

// CHECK-LABEL: func.func @bd_chain_in_tile
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK-COUNT-4: air.channel.put {{.*}} @chA[] (%{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [%c2{{.*}}, %c3{{.*}}, %c448{{.*}}, %c64{{.*}}]
// CHECK-NOT: scf.for
// CHECK: scf.yield

// The tiles are unrolled when a channel index depends on a launch induction
// variable, since the index has to be a constant.

// CHECK-LABEL: func.func @bundle_index
// CHECK-NOT: scf.for
// CHECK: air.channel.put {{.*}} @chD[%c0{{.*}}, %c0{{.*}}]
// CHECK-NOT: scf.for
// CHECK: return

// The same, with the index computed inside air.execute.

// CHECK-LABEL: func.func @execute_index
// CHECK-NOT: scf.for
// CHECK: air.channel.put {{.*}} @chD[%c{{[0-9]+}}{{.*}}, %c0{{.*}}]
// CHECK-NOT: scf.for
// CHECK: return

// The tiles are unrolled when an offset is not linear in the launch induction
// variables...

// CHECK-LABEL: func.func @nonlinear
// CHECK-NOT: scf.for
// CHECK: return

// ...when it is computed by an op that is not pure...

// CHECK-LABEL: func.func @not_pure
// CHECK: memref.load
// CHECK-NOT: scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c2
// CHECK: return

// ...or when the transfer has no unit-stride dimension to add the base to.

// CHECK-LABEL: func.func @no_unit_stride
// CHECK-NOT: scf.for
// CHECK: return

// A single tile is unrolled.

// CHECK-LABEL: func.func @single_tile
// CHECK-NOT: scf.for
// CHECK: return

// Loops outside a launch are unrolled.

// CHECK-LABEL: func.func @no_launch
// CHECK-NOT: scf.for
// CHECK: return

module {
  air.channel @chA [1, 1]
  air.channel @chB [1, 1]
  air.channel @chC [1, 1]
  air.channel @chD [4, 1]

  func.func @rolled(%arg0: memref<256x512xbf16>, %arg1: memref<512x256xbf16>, %arg2: memref<256x256xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0, %b=%arg1, %c=%arg2) : memref<256x512xbf16>, memref<512x256xbf16>, memref<256x256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %ox = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%x]
      %oy = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%y]
      %1 = air.channel.put async @chA[] (%a[%ox, %c0] [%c64, %c512] [%c512, %c1]) {id = 2 : i32} : (memref<256x512xbf16>)
      %2 = air.channel.put async @chB[] (%b[%c0, %oy] [%c512, %c64] [%c256, %c1]) {id = 3 : i32} : (memref<512x256xbf16>)
      %3 = air.channel.get async @chC[] (%c[%ox, %oy] [%c64, %c64] [%c256, %c1]) {id = 4 : i32} : (memref<256x256xbf16>)
    }
    return
  }

  func.func @nested_loop(%arg0: memref<256x512xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0) : memref<256x512xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %ox = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%x]
      %t0 = air.wait_all async
      %r = scf.for %k = %c0 to %c2 step %c1 iter_args(%t = %t0) -> (!air.async.token) {
        %ok = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 64)>()[%y, %k]
        %1 = air.channel.put async [%t] @chA[] (%a[%ox, %ok] [%c64, %c64] [%c512, %c1]) {id = 2 : i32} : (memref<256x512xbf16>)
        scf.yield %1 : !air.async.token
      }
    }
    return
  }

  func.func @execute_offset(%arg0: memref<256x512xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0) : memref<256x512xbf16> attributes {id = 3 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %t, %ox = air.execute -> (index) {
        %v = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%x]
        air.execute_terminator %v : index
      }
      %1 = air.channel.put async [%t] @chA[] (%a[%ox, %c0] [%c64, %c64] [%c512, %c1]) {id = 4 : i32} : (memref<256x512xbf16>)
    }
    return
  }

  func.func @bd_chain_in_tile(%arg0: memref<8x8x64x64xbf16>) {
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%x, %y) in (%sx=%c8, %sy=%c8) args(%a=%arg0) : memref<8x8x64x64xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c40000 = arith.constant 40000 : index
      %c7 = arith.constant 7 : index
      %t0 = air.wait_all async
      %r = scf.for %k = %c0 to %c2 step %c1 iter_args(%t = %t0) -> (!air.async.token) {
        %1 = air.channel.put async [%t] @chA[] (%a[%c0, %c0, %c0, %c0] [%c3, %c7, %c64, %c64] [%c40000, %c4096, %c64, %c1]) {id = 2 : i32} : (memref<8x8x64x64xbf16>)
        scf.yield %1 : !air.async.token
      }
    }
    return
  }

  func.func @bundle_index(%arg0: memref<256x512xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0) : memref<256x512xbf16> attributes {id = 5 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %ox = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%x]
      %1 = air.channel.put async @chD[%x, %c0] (%a[%ox, %c0] [%c64, %c512] [%c512, %c1]) {id = 6 : i32} : (memref<256x512xbf16>)
    }
    return
  }

  func.func @execute_index(%arg0: memref<256x512xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0) : memref<256x512xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %t, %idx = air.execute -> (index) {
        %v = arith.addi %x, %c0 : index
        air.execute_terminator %v : index
      }
      %1 = air.channel.put async [%t] @chD[%idx, %c0] (%a[%c0, %c0] [%c64, %c64] [%c512, %c1]) {id = 2 : i32} : (memref<256x512xbf16>)
    }
    return
  }

  func.func @nonlinear(%arg0: memref<256x512xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0) : memref<256x512xbf16> attributes {id = 5 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %ox = affine.apply affine_map<()[s0] -> ((s0 mod 3) * 64)>()[%x]
      %1 = air.channel.put async @chA[] (%a[%ox, %c0] [%c64, %c64] [%c512, %c1]) {id = 6 : i32} : (memref<256x512xbf16>)
    }
    return
  }

  func.func @not_pure(%arg0: memref<256x512xbf16>, %arg1: memref<4xindex>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0, %tbl=%arg1) : memref<256x512xbf16>, memref<4xindex> attributes {id = 7 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %row = memref.load %tbl[%x] : memref<4xindex>
      %ox = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1)>()[%x, %row]
      %1 = air.channel.put async @chA[] (%a[%ox, %c0] [%c64, %c64] [%c512, %c1]) {id = 8 : i32} : (memref<256x512xbf16>)
    }
    return
  }

  func.func @no_unit_stride(%arg0: memref<256x512xbf16>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%x, %y) in (%sx=%c4, %sy=%c4) args(%a=%arg0) : memref<256x512xbf16> attributes {id = 9 : i32} {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %ox = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%x]
      %1 = air.channel.put async @chA[] (%a[%ox, %c0] [%c64, %c32] [%c512, %c2]) {id = 10 : i32} : (memref<256x512xbf16>)
    }
    return
  }

  func.func @single_tile(%arg0: memref<128x512xbf16>) {
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%x, %y) in (%sx=%c2, %sy=%c2) args(%a=%arg0) : memref<128x512xbf16> attributes {id = 7 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %ox = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%x]
      %1 = air.channel.put async @chA[] (%a[%ox, %c0] [%c64, %c512] [%c512, %c1]) {id = 8 : i32} : (memref<128x512xbf16>)
    }
    return
  }

  func.func @no_launch(%arg0: memref<256x512xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %t = air.wait_all async
    %r = scf.for %x = %c0 to %c4 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
      %ox = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%x]
      %1 = air.channel.put async [%tok] @chA[] (%arg0[%ox, %c0] [%c64, %c512] [%c512, %c1]) {id = 9 : i32} : (memref<256x512xbf16>)
      scf.yield %1 : !air.async.token
    }
    return
  }
}
